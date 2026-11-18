"""Market data sources with fallback.

Consolidates the two diverging copies of ``DataSourceManager`` in the v1
project (``app.py`` L588-770 and ``data_sources.py``). Changes:

* every source returns the canonical schema via ``normalize_ohlcv``
  (lower-case ``open/high/low/close/volume[/adj_close]``, tz-naive sorted
  DatetimeIndex named ``timestamp``). yfinance returns ``Open``/``Close``
  (and, in recent versions, a (Price, Ticker) column MultiIndex), which the
  original feature code could not read;
* sources take a number of calendar days instead of a yfinance period
  string (the API used to pass ``"30d"``, which the mock source did not
  recognise);
* no custom ``requests.Session`` is passed to yfinance (the author's own
  ``data_sources.py`` had already dropped it; recent yfinance versions
  manage their own session);
* the Alpha Vantage key is sent as a query parameter and never logged;
* the mock source uses the dimensionally consistent generator in
  ``synthetic.py``, seeded per symbol (deterministic).

Network libraries (yfinance, requests) are imported lazily, so the synthetic
path works with only numpy/pandas installed.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from .synthetic import generate_ohlcv

logger = logging.getLogger(__name__)

CANONICAL = ["open", "high", "low", "close", "volume"]


def normalize_ohlcv(df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
    """Return a copy of ``df`` in the canonical lower-case OHLCV schema."""
    if df is None or df.empty:
        return pd.DataFrame(columns=CANONICAL)
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        # yfinance >= 0.2.48 style: levels (Price, Ticker). Keep the level that
        # holds the OHLCV names; select the requested ticker if present.
        names = {"open", "high", "low", "close", "volume", "adj close"}
        price_level = next(
            (i for i in range(out.columns.nlevels)
             if {str(v).lower() for v in out.columns.get_level_values(i)} & names), 0)
        other = [i for i in range(out.columns.nlevels) if i != price_level]
        if other and symbol is not None:
            lvl = other[0]
            if symbol in out.columns.get_level_values(lvl):
                out = out.xs(symbol, axis=1, level=lvl)
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = out.columns.get_level_values(price_level)

    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    out = out.loc[:, ~pd.Index(out.columns).duplicated()]
    missing = set(CANONICAL) - set(out.columns)
    if missing:
        raise KeyError(f"data is missing columns {sorted(missing)}")
    keep = CANONICAL + (["adj_close"] if "adj_close" in out.columns else [])
    out = out[keep].apply(pd.to_numeric, errors="coerce")

    idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out.index = idx
    out.index.name = "timestamp"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(subset=["close"])


class YahooFinanceSource:
    """Yahoo Finance via yfinance, with exponential-backoff retries."""

    name = "yahoo"

    async def get_data(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        import yfinance as yf  # optional dependency (api extra)

        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        retry_delay, max_retries = 2, 3
        for attempt in range(1, max_retries + 1):
            try:
                data = await asyncio.to_thread(
                    yf.download, symbol, start=start, progress=False, auto_adjust=False)
                if data is not None and not data.empty:
                    return normalize_ohlcv(data, symbol)
                raise ValueError("empty data returned")
            except Exception as exc:
                if attempt == max_retries:
                    logger.error("Yahoo: failed to fetch %s after %d attempts: %s",
                                 symbol, max_retries, exc)
                    return None
                wait = retry_delay * (2 ** (attempt - 1))
                logger.info("Yahoo: retry %d/%d for %s in %ss", attempt, max_retries, symbol, wait)
                await asyncio.sleep(wait)
        return None


class AlphaVantageSource:
    """Alpha Vantage TIME_SERIES_DAILY (needs ALPHA_VANTAGE_API_KEY)."""

    name = "alpha_vantage"
    URL = "https://www.alphavantage.co/query"

    def __init__(self):
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")

    async def get_data(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        if not self.api_key or self.api_key == "your_api_key_here":
            logger.warning("Alpha Vantage API key not configured")
            return None
        import requests  # optional dependency (api extra)

        params = {"function": "TIME_SERIES_DAILY", "symbol": symbol,
                  "outputsize": "compact" if days <= 140 else "full", "apikey": self.api_key}
        try:
            response = await asyncio.to_thread(requests.get, self.URL, params=params, timeout=30)
            if response.status_code != 200:
                # Do not log the exception text: it contains the URL incl. the key.
                logger.error("Alpha Vantage HTTP status %s for %s", response.status_code, symbol)
                return None
            data = response.json()
        except Exception as exc:
            logger.error("Alpha Vantage request failed for %s (%s)", symbol, type(exc).__name__)
            return None
        if "Time Series (Daily)" not in data:
            logger.error("Unexpected response format from Alpha Vantage for %s", symbol)
            return None
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
        df = df.rename(columns={"1. open": "open", "2. high": "high", "3. low": "low",
                                "4. close": "close", "5. volume": "volume"})
        df = normalize_ohlcv(df, symbol)
        cutoff = pd.Timestamp(datetime.now() - timedelta(days=days))
        return df[df.index >= cutoff]


class SyntheticDataSource:
    """Deterministic synthetic data (replaces the original MockDataSource).

    Uses the author's original coefficients (0.1 % drift, 1 % noise, 0.7
    momentum, 0.1 mean reversion) but applied to log returns, so the series
    stays bounded. Seeded from the symbol, so the same symbol always yields
    the same series for the same end date.
    """

    name = "synthetic"

    def __init__(self, end: Optional[pd.Timestamp] = None):
        self.end = end

    async def get_data(self, symbol: str, days: int) -> pd.DataFrame:
        return self.generate(symbol, days)

    def generate(self, symbol: str, days: int) -> pd.DataFrame:
        symbol_hash = sum(ord(c) for c in symbol)
        end = self.end if self.end is not None else pd.Timestamp.now().normalize()
        n_days = max(2, len(pd.bdate_range(end=end, periods=None,
                                           start=end - pd.Timedelta(days=days))))
        return generate_ohlcv(
            n_days=n_days,
            seed=symbol_hash,
            start_price=50 + (symbol_hash % 200),
            drift=0.001,
            volatility=0.01,
            momentum=0.7,
            mean_reversion=0.1,
            avg_volume=1_000_000 + (symbol_hash % 5_000_000),
            end=end,
        )


class DataSourceManager:
    """Try each source in order until one returns data.

    ``mode`` (default: env ``DATA_SOURCE``): ``"auto"`` = Yahoo -> Alpha
    Vantage -> synthetic (original order); ``"synthetic"`` = offline only.
    """

    def __init__(self, mode: Optional[str] = None):
        self.mode = (mode or os.getenv("DATA_SOURCE", "auto")).strip().lower()
        if self.mode == "synthetic":
            self.sources = [SyntheticDataSource()]
        elif self.mode == "auto":
            self.sources = [YahooFinanceSource(), AlphaVantageSource(), SyntheticDataSource()]
        else:
            raise ValueError(f"unknown DATA_SOURCE {self.mode!r} (use 'auto' or 'synthetic')")
        self.last_source: Optional[str] = None

    async def get_market_data(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        for source in self.sources:
            try:
                data = await source.get_data(symbol, days)
                if data is not None and not data.empty:
                    self.last_source = source.name
                    if source.name == "synthetic" and self.mode == "auto":
                        logger.warning("All live sources failed for %s; using SYNTHETIC data", symbol)
                    return data
            except Exception as exc:
                logger.warning("Data source %s failed: %s", source.name, exc)
        logger.error("All data sources failed for %s", symbol)
        return None
