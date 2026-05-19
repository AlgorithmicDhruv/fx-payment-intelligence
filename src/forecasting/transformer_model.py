"""
src/forecasting/transformer_model.py

Temporal Fusion Transformer-inspired architecture for FX rate forecasting.

Architecture choice rationale (for interviews):
  - Transformers outperform LSTMs on multi-step time-series when sequence length
    is moderate (60 days) because self-attention captures non-local dependencies
    (e.g., month-end patterns 30 days apart)
  - We add positional encoding to preserve temporal order
  - Multi-head attention lets the model attend to different time scales simultaneously
  - Output: point forecast + quantile estimates (10th, 50th, 90th percentile)
    -> this gives us confidence intervals for the dashboard AND the RL agent
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Tuple, Dict, Optional
import joblib
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.settings import (
    SEQ_LEN, PRED_HORIZONS, HIDDEN_DIM, N_HEADS, N_LAYERS,
    DROPOUT, LEARNING_RATE, BATCH_SIZE, MAX_EPOCHS, EARLY_STOP_PAT,
    TRAIN_SPLIT, VAL_SPLIT, MODELS_DIR
)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class FXTransformer(nn.Module):
    """
    Transformer encoder for multi-horizon FX rate forecasting.

    Input:  (batch, seq_len, n_features)
    Output: (batch, n_horizons, 3)  — 3 = [q10, q50, q90] quantiles
            This gives us a full predictive distribution, not just a point estimate.
    """

    def __init__(
        self,
        n_features: int,
        n_horizons: int = 2,
        hidden_dim: int = HIDDEN_DIM,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.n_horizons = n_horizons

        # Project input features to hidden dimension
        self.input_projection = nn.Linear(n_features, hidden_dim)

        self.pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-norm: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Separate head per horizon, outputs 3 quantiles each
        self.forecast_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 3),   # [q10, q50, q90]
            )
            for _ in range(n_horizons)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        x = self.input_projection(x)          # (batch, seq_len, hidden_dim)
        x = self.pos_encoding(x)
        x = self.transformer(x)               # (batch, seq_len, hidden_dim)
        context = x[:, -1, :]                 # use last timestep as summary

        outputs = [head(context) for head in self.forecast_heads]
        return torch.stack(outputs, dim=1)    # (batch, n_horizons, 3)


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: list) -> torch.Tensor:
    """
    Pinball / quantile loss. Asymmetric — penalizes under-prediction of high
    quantiles and over-prediction of low quantiles.
    This is critical for risk management: we want conservative upper bound estimates.
    """
    q_tensor = torch.tensor(quantiles, dtype=torch.float32, device=pred.device)
    losses = []
    for i, q in enumerate(q_tensor):
        error = target - pred[:, i]
        losses.append(torch.max(q * error, (q - 1) * error))
    return torch.stack(losses, dim=1).mean()


class FXDataset(torch.utils.data.Dataset):
    """Sliding window dataset for time-series training."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_len: int = SEQ_LEN,
    ):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_window = self.X[idx : idx + self.seq_len]
        y_target  = self.y[idx + self.seq_len]
        return x_window, y_target


def prepare_data(
    feature_df: pd.DataFrame,
    horizons: list = None,
    feature_cols: list = None,
) -> Tuple[np.ndarray, np.ndarray, list, object]:
    """
    Prepare feature matrix and multi-horizon targets.
    Scales features to zero mean, unit variance (StandardScaler).
    Returns raw arrays + scaler (needed for inverse-transform at inference).
    """
    from sklearn.preprocessing import StandardScaler

    horizons = horizons or PRED_HORIZONS

    # Select feature columns (exclude targets, regime labels are fine as-is)
    exclude = ["price"] + [f"log_ret_{h}d" for h in horizons]
    if feature_cols is None:
        feature_cols = [
            c for c in feature_df.columns
            if c not in exclude
            and feature_df[c].dtype in [np.float64, np.float32, float]
        ]

    X = feature_df[feature_cols].values
    # Target: log returns at each horizon (what we're predicting)
    y = feature_df[[f"log_ret_{h}d" for h in horizons]].values

    # Scale features only (not targets — we keep returns in natural units)
    scaler = StandardScaler()

    n = len(X)
    train_end = int(n * TRAIN_SPLIT)
    val_end   = int(n * (TRAIN_SPLIT + VAL_SPLIT))

    X_train = X[:train_end]
    X_scaled = scaler.fit_transform(X_train)
    X_all_scaled = scaler.transform(X)

    # Replace NaN/inf that can appear in edge features
    X_all_scaled = np.nan_to_num(X_all_scaled, nan=0.0, posinf=3.0, neginf=-3.0)

    return X_all_scaled, y, feature_cols, scaler


def train(
    feature_df: pd.DataFrame,
    horizons: list = None,
    device: str = None,
    save_model: bool = True,
) -> Dict:
    """
    Full training loop with:
      - Train / val / test split (chronological, no leakage)
      - Early stopping on validation loss
      - Quantile loss (q10, q50, q90)
      - Model checkpoint saving

    Returns dict with model, scaler, feature_cols, and test metrics.
    """
    horizons = horizons or PRED_HORIZONS
    quantiles = [0.10, 0.50, 0.90]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    X, y, feature_cols, scaler = prepare_data(feature_df, horizons)

    n = len(X)
    train_end = int(n * TRAIN_SPLIT)
    val_end   = int(n * (TRAIN_SPLIT + VAL_SPLIT))

    train_ds = FXDataset(X[:train_end],  y[:train_end])
    val_ds   = FXDataset(X[:val_end],    y[:val_end])
    test_ds  = FXDataset(X,              y)
    # Test set: only last portion
    test_indices = list(range(val_end - SEQ_LEN, n - SEQ_LEN))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model = FXTransformer(
        n_features=len(feature_cols),
        n_horizons=len(horizons),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Training samples: {len(train_ds)}, Val: {len(val_ds)}")

    for epoch in range(MAX_EPOCHS):
        # ── Train ──
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)  # (batch, n_horizons, 3)
            loss = sum(
                quantile_loss(pred[:, h, :], y_batch[:, h], quantiles)
                for h in range(len(horizons))
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # ── Validate ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                loss = sum(
                    quantile_loss(pred[:, h, :], y_batch[:, h], quantiles)
                    for h in range(len(horizons))
                )
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        train_loss = np.mean(train_losses)

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1:3d} | train: {train_loss:.5f} | val: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PAT:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)

    # ── Evaluate on test set ──
    test_preds, test_targets = [], []
    model.eval()
    with torch.no_grad():
        for idx in test_indices:
            if idx + SEQ_LEN >= n:
                continue
            x = torch.tensor(X[idx:idx+SEQ_LEN], dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(x)
            test_preds.append(pred.cpu().numpy())
            test_targets.append(y[idx + SEQ_LEN])

    test_preds   = np.array(test_preds).squeeze(1)    # (n_test, n_horizons, 3)
    test_targets = np.array(test_targets)               # (n_test, n_horizons)

    metrics = _compute_metrics(test_preds, test_targets, horizons)
    logger.success(f"Test metrics: {metrics}")

    result = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "horizons": horizons,
        "metrics": metrics,
        "best_val_loss": best_val_loss,
    }

    if save_model:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = MODELS_DIR / "forecasting" / "fx_transformer.pt"
        save_path.parent.mkdir(exist_ok=True)
        torch.save({
            "model_state": best_state,
            "n_features": len(feature_cols),
            "n_horizons": len(horizons),
            "feature_cols": feature_cols,
            "scaler": scaler,
            "horizons": horizons,
            "metrics": metrics,
        }, save_path)
        logger.success(f"Model saved to {save_path}")

    return result


def _compute_metrics(preds: np.ndarray, targets: np.ndarray, horizons: list) -> dict:
    """Compute directional accuracy, MAE, and interval coverage per horizon."""
    metrics = {}
    for i, h in enumerate(horizons):
        pred_median = preds[:, i, 1]   # q50
        pred_lo     = preds[:, i, 0]   # q10
        pred_hi     = preds[:, i, 2]   # q90
        target      = targets[:, i]

        mae = np.abs(pred_median - target).mean()
        # Directional accuracy: did we get the sign of the return right?
        dir_acc = (np.sign(pred_median) == np.sign(target)).mean()
        # 80% interval coverage (q10-q90): should be ~80% if calibrated
        coverage = ((target >= pred_lo) & (target <= pred_hi)).mean()

        metrics[f"h{h}d_mae"]      = round(float(mae), 6)
        metrics[f"h{h}d_dir_acc"]  = round(float(dir_acc), 4)
        metrics[f"h{h}d_coverage"] = round(float(coverage), 4)

    return metrics


def load_model(model_path: Optional[Path] = None) -> Dict:
    """Load a saved model checkpoint."""
    if model_path is None:
        model_path = MODELS_DIR / "forecasting" / "fx_transformer.pt"

    checkpoint = torch.load(model_path, map_location="cpu")
    model = FXTransformer(
        n_features=checkpoint["n_features"],
        n_horizons=checkpoint["n_horizons"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return {
        "model": model,
        "scaler": checkpoint["scaler"],
        "feature_cols": checkpoint["feature_cols"],
        "horizons": checkpoint["horizons"],
        "metrics": checkpoint["metrics"],
    }


def predict(
    feature_df: pd.DataFrame,
    model_bundle: Dict,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Run inference on the latest window of feature_df.
    Returns DataFrame with columns: h1d_q10, h1d_q50, h1d_q90, h5d_q10, ...
    """
    model    = model_bundle["model"].to(device)
    scaler   = model_bundle["scaler"]
    feat_cols = model_bundle["feature_cols"]
    horizons  = model_bundle["horizons"]

    X = feature_df[feat_cols].values
    X_scaled = scaler.transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=3.0, neginf=-3.0)

    model.eval()
    results = []
    with torch.no_grad():
        for i in range(SEQ_LEN, len(X_scaled)):
            window = torch.tensor(
                X_scaled[i-SEQ_LEN:i], dtype=torch.float32
            ).unsqueeze(0).to(device)
            pred = model(window).cpu().numpy().squeeze(0)  # (n_horizons, 3)
            row = {}
            for j, h in enumerate(horizons):
                row[f"h{h}d_q10"] = pred[j, 0]
                row[f"h{h}d_q50"] = pred[j, 1]
                row[f"h{h}d_q90"] = pred[j, 2]
            results.append(row)

    pred_df = pd.DataFrame(results, index=feature_df.index[SEQ_LEN:])
    return pred_df


if __name__ == "__main__":
    logger.info("Testing Transformer model with synthetic data...")
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", "2024-12-31", freq="B")
    returns = np.random.normal(0, 0.006, len(dates))
    prices = pd.Series(1.10 * np.exp(np.cumsum(returns)), index=dates, name="EURUSD")

    # Build minimal feature matrix
    sys.path.append(str(Path(__file__).parent.parent))
    from features.engineer import build_feature_matrix
    features = build_feature_matrix(prices, fit_garch=False, save=False)

    result = train(features, save_model=False)
    print("\nMetrics:", result["metrics"])
