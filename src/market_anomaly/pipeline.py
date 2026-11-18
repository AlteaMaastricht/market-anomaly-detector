"""End-to-end detection pipeline, independent of FastAPI.

``/detect-anomalies`` and the background task in ``api.py`` are thin
wrappers around these functions, so the whole request path can be tested
without a web server.

Window semantics (changed during preparation, see docs/PROVENANCE.md):
the original fetched exactly ``lookback_days`` calendar days (~21 trading
rows for the default 30), which is below the 30-row minimum, so nothing was
ever scored. Now ``lookback_days + Config.TRAINING_HISTORY_DAYS`` calendar
days are fetched; the models are fitted and scored on all complete rows, and
only alerts inside the final ``lookback_days`` are returned.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from .config import Config
from .data_sources import DataSourceManager
from .detectors import MultiModalAnomalyDetector
from .features import calculate_technical_indicators, feature_matrix
from .models import AnomalyAlert
from .storage import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class SymbolResult:
    symbol: str
    alerts: List[AnomalyAlert] = field(default_factory=list)
    n_rows: int = 0
    n_scored: int = 0
    source: Optional[str] = None
    retrained: bool = False
    error: Optional[str] = None


def history_days(lookback_days: int) -> int:
    return int(lookback_days) + Config.TRAINING_HISTORY_DAYS


def analyse_frame(df: pd.DataFrame, symbol: str, detector: MultiModalAnomalyDetector,
                  lookback_days: int = 30, sensitivity: float = 0.05) -> SymbolResult:
    """Indicators -> fit (cached) -> score -> alerts within the lookback window."""
    result = SymbolResult(symbol=symbol, n_rows=len(df))
    feats = calculate_technical_indicators(df)
    result.n_scored = len(feature_matrix(feats))
    if result.n_scored == 0:
        result.error = f"insufficient data ({len(df)} rows; need > {Config.MIN_ROWS} plus warm-up)"
        logger.warning("%s: %s", symbol, result.error)
        return result
    result.retrained = detector.fit(feats, symbol, sensitivity)
    if not detector.is_trained.get(symbol):
        result.error = "model training failed"
        return result
    since = feats.index.max() - pd.Timedelta(days=lookback_days)
    result.alerts = detector.detect(feats, symbol, since=since)
    return result


async def detect_symbols(symbols: List[str], lookback_days: int, sensitivity: float,
                         data_manager: DataSourceManager, detector: MultiModalAnomalyDetector,
                         db: Optional[DatabaseManager] = None) -> List[SymbolResult]:
    """Fetch, store, analyse and persist results for each symbol."""
    results = []
    for symbol in symbols:
        data = await data_manager.get_market_data(symbol, history_days(lookback_days))
        if data is None or data.empty:
            results.append(SymbolResult(symbol=symbol, error="no data"))
            continue
        if db is not None:
            await db.store_market_data(symbol, data)
        try:
            res = analyse_frame(data, symbol, detector, lookback_days, sensitivity)
        except Exception as exc:
            logger.exception("analysis failed for %s", symbol)
            res = SymbolResult(symbol=symbol, n_rows=len(data), error=str(exc))
        res.source = data_manager.last_source
        if db is not None:
            for alert in res.alerts:
                await db.store_anomaly_result(alert)
        if res.retrained:
            detector.save_models(symbol)  # no-op unless MODEL_DIR is set
        results.append(res)
    return results
