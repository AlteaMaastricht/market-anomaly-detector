"""Ensemble anomaly detector (from ``MultiModalAnomalyDetector`` in v1 ``app.py``).

Decision rule (unchanged from the original):
  1. StandardScaler on the detection features.
  2. IsolationForest and EllipticEnvelope ``decision_function`` scores.
  3. Each model's scores are z-scored *within the scored batch*, then averaged.
  4. Rows with averaged z-score < -ANOMALY_THRESHOLD (2.5) are anomalies.

Because of step 3 the ``sensitivity`` -> ``contamination`` mapping hardly
affects which rows are flagged (the threshold is relative to the batch), and
models are fitted and scored on the same window (in-sample). Both are
original design choices, documented as limitations rather than changed.

The core is synchronous and has no FastAPI dependency; the API wraps it.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .config import Config
from .features import feature_matrix
from .models import AnomalyAlert

logger = logging.getLogger(__name__)


class MultiModalAnomalyDetector:
    """IsolationForest + EllipticEnvelope ensemble, one model set per symbol."""

    def __init__(self, threshold: float = Config.ANOMALY_THRESHOLD, random_state: int = 42):
        self.threshold = threshold
        self.random_state = random_state
        self.models: Dict[str, Dict[str, object]] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_columns: Dict[str, List[str]] = {}
        self.is_trained: Dict[str, bool] = {}
        self.last_train_time: Dict[str, datetime] = {}
        self.train_sensitivity: Dict[str, float] = {}

    # ------------------------------------------------------------ training
    def needs_training(self, symbol: str, sensitivity: float, now: Optional[datetime] = None) -> bool:
        """Retrain when no model exists, it is older than MODEL_RETRAIN_INTERVAL,
        or the requested sensitivity changed (the original ignored the latter)."""
        now = now or datetime.now()
        last = self.last_train_time.get(symbol)
        if symbol not in self.models or last is None:
            return True
        if (now - last).total_seconds() >= Config.MODEL_RETRAIN_INTERVAL:
            return True
        return self.train_sensitivity.get(symbol) != sensitivity

    def fit(self, df: pd.DataFrame, symbol: str, sensitivity: float = 0.05, force: bool = False) -> bool:
        """Fit the ensemble on ``df`` (must already contain indicator columns).

        Returns True if models were (re)fitted, False if the cache was reused
        or there were no complete feature rows.
        """
        if not force and not self.needs_training(symbol, sensitivity):
            logger.info("Models for %s still fresh, skipping training", symbol)
            return False
        feats = feature_matrix(df)
        if feats.empty:
            logger.error("No complete feature rows for %s; cannot train", symbol)
            return False

        scaler = StandardScaler()
        X = scaler.fit_transform(feats.values)
        contamination = max(0.01, min(0.2, sensitivity * 2))  # original mapping
        candidates = {
            "isolation_forest": IsolationForest(
                contamination=contamination, random_state=self.random_state, n_estimators=100),
            "elliptic_envelope": EllipticEnvelope(
                contamination=contamination, random_state=self.random_state),
        }
        trained = {}
        for name, model in candidates.items():
            try:
                model.fit(X)
                trained[name] = model
            except Exception as exc:  # e.g. singular covariance on tiny samples
                logger.error("Error training %s for %s: %s", name, symbol, exc)
        if not trained:
            return False

        self.models[symbol] = trained
        self.scalers[symbol] = scaler
        self.feature_columns[symbol] = list(feats.columns)
        self.is_trained[symbol] = True
        self.last_train_time[symbol] = datetime.now()
        self.train_sensitivity[symbol] = sensitivity
        return True

    # ------------------------------------------------------------ scoring
    def score(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        """Averaged batch-z-scored decision values (lower = more anomalous)."""
        if not self.is_trained.get(symbol):
            raise RuntimeError(f"models not trained for {symbol}")
        feats = feature_matrix(df, self.feature_columns[symbol])
        if feats.empty:
            return pd.Series(dtype="float64")
        X = self.scalers[symbol].transform(feats.values)
        ensemble = np.zeros(len(feats))
        for name, model in self.models[symbol].items():
            scores = model.decision_function(X)
            ensemble += (scores - scores.mean()) / (scores.std() + 1e-8)
        ensemble /= len(self.models[symbol])
        return pd.Series(ensemble, index=feats.index, name="ensemble_score")

    def detect(self, df: pd.DataFrame, symbol: str,
               since: Optional[pd.Timestamp] = None) -> List[AnomalyAlert]:
        """Return alerts for rows whose ensemble score is below -threshold.

        ``since`` restricts the *reported* alerts to rows at or after that
        timestamp (scoring still uses the whole frame, as in the original).
        """
        scores = self.score(df, symbol)
        flagged = scores[scores < -self.threshold]
        if since is not None:
            flagged = flagged[flagged.index >= pd.Timestamp(since)]
        alerts = []
        for ts, value in flagged.items():
            row = df.loc[ts]
            anomaly_type, severity = classify_anomaly(row)
            alerts.append(AnomalyAlert(
                symbol=symbol,
                timestamp=pd.Timestamp(ts).to_pydatetime(),
                anomaly_score=float(abs(value)),
                anomaly_type=anomaly_type,
                confidence=float(min(abs(value) / self.threshold, 1.0)),
                current_price=float(row["close"]),
                explanation=generate_explanation(row, anomaly_type),
                severity=severity,
            ))
        return alerts

    # ------------------------------------------------------------ persistence
    def save_models(self, symbol: str, model_dir: Optional[str] = None) -> Optional[str]:
        """Persist the fitted models with joblib (only if a directory is configured)."""
        model_dir = model_dir if model_dir is not None else Config.MODEL_DIR
        if not model_dir or symbol not in self.models:
            return None
        import joblib

        os.makedirs(model_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(model_dir, f"models_{symbol}_{stamp}.joblib")
        joblib.dump({"models": self.models[symbol], "scaler": self.scalers[symbol],
                     "feature_columns": self.feature_columns[symbol]}, path)
        return path


def classify_anomaly(row: pd.Series) -> Tuple[str, str]:
    """Rule-based label and severity (unchanged rules from the original)."""
    def val(key):
        v = row.get(key, np.nan)
        return np.nan if v is None else float(v)

    if val("volume_ratio") > 3:
        return "VOLUME_SPIKE", "HIGH"
    if abs(val("returns")) > 0.05:
        return "PRICE_SHOCK", "CRITICAL" if abs(val("returns")) > 0.10 else "HIGH"
    if val("volatility") > 0.03:
        return "VOLATILITY_SPIKE", "MEDIUM"
    r = val("rsi")
    if r < 20 or r > 80:
        return "TECHNICAL_EXTREME", "LOW"
    return "UNKNOWN", "MEDIUM"


def generate_explanation(row: pd.Series, anomaly_type: str) -> str:
    """Human-readable explanation. Only the needed message is formatted (the
    original formatted all of them eagerly, so a missing column raised)."""
    if anomaly_type == "VOLUME_SPIKE":
        return f"Unusual trading volume detected ({float(row['volume_ratio']):.2f}x normal)"
    if anomaly_type == "PRICE_SHOCK":
        return f"Significant price movement ({float(row['returns']) * 100:.2f}% change)"
    if anomaly_type == "VOLATILITY_SPIKE":
        return f"High volatility detected ({float(row['volatility']) * 100:.2f}%)"
    if anomaly_type == "TECHNICAL_EXTREME":
        return f"Technical indicators at extreme levels (RSI: {float(row['rsi']):.1f})"
    return "Statistical anomaly detected in market behavior"
