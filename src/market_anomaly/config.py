"""Application configuration.

Values mirror the original ``Config`` class in the v1 ``app.py``; the
environment-variable overrides and the warm-up/history settings were added
during repository preparation (see docs/PROVENANCE.md).
"""

import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


class Config:
    """Application configuration (class attributes, read at import time)."""

    DATABASE_PATH = os.getenv("DATABASE_PATH", "market_data.db")
    DEFAULT_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY", "QQQ", "IWM", "GLD"]
    UPDATE_INTERVAL = 300  # seconds between background update/detection runs
    ANOMALY_THRESHOLD = 2.5  # ensemble z-score threshold (original value)
    MODEL_RETRAIN_INTERVAL = 86400  # seconds a fitted model is reused
    MAX_WORKERS = 4

    # "auto" = Yahoo Finance -> Alpha Vantage -> synthetic fallback (original
    # behaviour); "synthetic" = offline deterministic data only.
    DATA_SOURCE = os.getenv("DATA_SOURCE", "auto").strip().lower()

    # Extra calendar days of history fetched before the requested lookback
    # window, so that rolling indicators (up to 30 rows) are warmed up and the
    # models have more than a handful of rows to fit on.
    TRAINING_HISTORY_DAYS = 365

    # Minimum number of rows before indicators are computed (original guard).
    MIN_ROWS = 30

    MODEL_DIR = os.getenv("MODEL_DIR", "")  # empty = do not persist models
    ENABLE_BACKGROUND_TASKS = _env_bool("ENABLE_BACKGROUND_TASKS", True)
