"""Feature engineering (from ``AdvancedFeatureEngine`` in the v1 ``app.py``).

Expects the canonical lower-case OHLCV schema produced by
``data_sources.normalize_ohlcv``.

The original computed RSI, MACD, Bollinger Bands, OBV, EMA and the stochastic
oscillator with the ``ta`` package. They are re-implemented here in pandas
using the default parameters of the ``ta`` 0.10.2 classes the original called
(RSI 14 Wilder smoothing, MACD 12/26, Bollinger 20 / 2 std with ddof=0,
stochastic %K 14), so the core package has no ``ta`` dependency. Numerical
equivalence with ``ta`` was NOT verified (``ta`` was not installed during
preparation).
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)

# Features fed to the detectors (unchanged from the original prepare_features).
DETECTION_FEATURES = [
    "returns", "volatility", "rsi", "macd", "bb_position",
    "volume_ratio", "returns_skew", "returns_kurt", "spread",
]


# ---------------------------------------------------------------- indicators
def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, min_periods=window, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0)
    down = (-diff).clip(lower=0)
    avg_up = up.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_up / avg_down
    out = 100 - 100 / (1 + rs)
    # avg_down == 0 -> rs = inf -> RSI 100 (handled by the formula); 0/0 -> 50
    out = out.where(~((avg_up == 0) & (avg_down == 0)), 50.0)
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    return ema(close, fast) - ema(close, slow)


def bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0):
    mavg = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    return mavg + n_std * std, mavg - n_std * std


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    signed = np.where(close < close.shift(1), -volume, volume)
    return pd.Series(signed, index=close.index, dtype="float64").cumsum()


def stochastic_k(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    lowest = low.rolling(window, min_periods=window).min()
    highest = high.rolling(window, min_periods=window).max()
    return 100 * (close - lowest) / (highest - lowest)


# ---------------------------------------------------------------- engine
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add the original indicator set as new columns (returns a copy).

    Rows before the rolling windows are full contain NaN; callers should
    drop them (see ``feature_matrix``) rather than fill them with 0.
    """
    if len(df) < Config.MIN_ROWS:
        logger.warning("Only %d rows (< %d); indicators not computed", len(df), Config.MIN_ROWS)
        return df.copy()

    missing = {"open", "high", "low", "close", "volume"} - set(df.columns)
    if missing:
        # The original swallowed this as a logged KeyError; fail loudly instead.
        raise KeyError(f"missing OHLCV columns {sorted(missing)}; normalise with normalize_ohlcv()")

    df = df.copy()
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"].astype("float64")

    # Price-based features
    df["returns"] = close.pct_change()
    df["log_returns"] = np.log(close / close.shift(1))
    df["price_momentum"] = close / close.shift(10) - 1

    # Volatility measures
    df["volatility"] = df["returns"].rolling(20).std()
    df["parkinson_vol"] = np.sqrt((1 / (4 * np.log(2))) * (np.log(high / low)) ** 2)

    # Technical indicators
    df["rsi"] = rsi(close)
    df["macd"] = macd(close)
    df["bb_upper"], df["bb_lower"] = bollinger(close)
    bb_diff = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = np.where(bb_diff != 0, (close - df["bb_lower"]) / bb_diff, 0.5)
    df.loc[bb_diff.isna(), "bb_position"] = np.nan

    # Volume indicators
    df["volume_sma"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / df["volume_sma"]
    df["obv"] = on_balance_volume(close, volume)

    # Additional technical indicators
    df["ema_12"] = ema(close, 12)
    df["ema_26"] = ema(close, 26)
    df["stoch_k"] = stochastic_k(high, low, close)

    # Rolling distribution features
    df["returns_skew"] = df["returns"].rolling(30).skew()
    df["returns_kurt"] = df["returns"].rolling(30).kurt()
    df["autocorr"] = df["returns"].rolling(30).apply(
        lambda x: pd.Series(x).autocorr() if len(x) > 1 else 0, raw=False
    )

    # Market microstructure
    df["spread"] = (high - low) / close
    df["gap"] = (df["open"] - close.shift(1)) / close.shift(1)
    return df


def feature_matrix(df: pd.DataFrame, columns=DETECTION_FEATURES) -> pd.DataFrame:
    """Rows with a complete detection-feature vector.

    The original used ``fillna(0)``, which mixes placeholder values (e.g.
    RSI = 0, volatility = 0) for the warm-up rows into the training data.
    Incomplete rows are dropped instead. (On the demo series the original
    rule did not flag any warm-up row, so this is a correctness/cleanliness
    change, not a measured accuracy improvement.)
    """
    available = [c for c in columns if c in df.columns]
    if not available:
        return pd.DataFrame(index=df.index[:0])
    feats = df[available].replace([np.inf, -np.inf], np.nan)
    return feats.dropna()


# ---------------------------------------------------------------- statistics
def calculate_statistical_features(df: pd.DataFrame) -> Dict:
    """Stationarity / autocorrelation / normality tests and Hurst exponent.

    Present in the original but never called by the pipeline; kept as a
    stand-alone helper. Requires ``statsmodels`` (optional extra) and was not
    executed during preparation.
    """
    features: Dict = {}
    if len(df) < 50 or "returns" not in df.columns:
        return features
    from scipy import stats
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller

    returns = df["returns"].dropna()
    adf_result = adfuller(returns)
    features["adf_pvalue"] = float(adf_result[1])
    features["is_stationary"] = bool(adf_result[1] < 0.05)
    lb_result = acorr_ljungbox(returns, lags=10, return_df=True)
    features["ljung_box_pvalue"] = float(lb_result["lb_pvalue"].iloc[-1])
    jb_stat, jb_pvalue = stats.jarque_bera(returns)
    features["jarque_bera_pvalue"] = float(jb_pvalue)
    features["is_normal"] = bool(jb_pvalue > 0.05)
    features["hurst_exponent"] = hurst_exponent(df["close"].values)
    return features


def hurst_exponent(time_series: np.ndarray) -> float:
    """Hurst exponent from the scaling of lagged differences (original formula)."""
    time_series = np.asarray(time_series, dtype=float)
    lags = range(2, min(100, len(time_series) // 4))
    if len(lags) < 2:
        return 0.5  # not enough data: random-walk default (as in the original)
    tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return float(poly[0] * 2.0)
