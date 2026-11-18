"""SQLite persistence (from ``DatabaseManager`` in v1 ``app.py``).

The schema is unchanged. Changes made during preparation:

* the market-data insert read ``row['timestamp']`` and ``row['adj_close']``
  after ``reset_index()``, but the index column is called ``Date`` (Yahoo) or
  ``index`` (mock data) and mock data has no ``adj_close`` -> every insert
  failed and nothing was stored. Rows are now built from the canonical
  schema; ``adj_close`` is optional (NULL);
* timestamps are bound as ISO strings (``sqlite3`` cannot bind
  ``pandas.Timestamp``, which also broke ``store_anomaly_result``);
* the standard-library ``sqlite3`` module is used for all operations, with
  ``asyncio.to_thread`` wrappers for the API (the original mixed
  ``aiosqlite`` and a thread-pooled ``sqlite3``). This removes the
  ``aiosqlite`` dependency and makes the module testable offline;
* store failures propagate to the caller instead of being logged and
  reported as success.
"""

import asyncio
import logging
import sqlite3
from contextlib import contextmanager
from typing import Optional

import pandas as pd

from .config import Config
from .models import AnomalyAlert

logger = logging.getLogger(__name__)

_TS_FMT = "%Y-%m-%d %H:%M:%S"

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS market_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        open REAL, high REAL, low REAL, close REAL,
        volume INTEGER,
        adj_close REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(symbol, timestamp))""",
    """CREATE TABLE IF NOT EXISTS anomaly_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        anomaly_score REAL, anomaly_type TEXT, confidence REAL,
        severity TEXT, explanation TEXT, model_version TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP)""",
    "CREATE INDEX IF NOT EXISTS idx_market_symbol_time ON market_data(symbol, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_anomaly_symbol_time ON anomaly_results(symbol, timestamp)",
]


def _ts(value) -> str:
    return pd.Timestamp(value).strftime(_TS_FMT)


def _num(value) -> Optional[float]:
    return None if value is None or pd.isna(value) else float(value)


class DatabaseManager:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or Config.DATABASE_PATH

    @contextmanager
    def _connect(self):
        """Connection that commits on success and is always closed."""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------ sync API
    def init_database_sync(self) -> None:
        with self._connect() as conn:
            for stmt in SCHEMA:
                conn.execute(stmt)

    def store_market_data_sync(self, symbol: str, data: pd.DataFrame) -> int:
        """Upsert canonical OHLCV rows; returns the number of rows written."""
        has_adj = "adj_close" in data.columns
        rows = [
            (symbol, _ts(ts), _num(r["open"]), _num(r["high"]), _num(r["low"]),
             _num(r["close"]), None if pd.isna(r["volume"]) else int(r["volume"]),
             _num(r["adj_close"]) if has_adj else None)
            for ts, r in data.iterrows()
        ]
        with self._connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO market_data
                   (symbol, timestamp, open, high, low, close, volume, adj_close)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", rows)
        return len(rows)

    def get_market_data_sync(self, symbol: str, days: int = 30) -> pd.DataFrame:
        query = ("SELECT timestamp, open, high, low, close, volume, adj_close "
                 "FROM market_data WHERE symbol = ? AND timestamp >= datetime('now', ?) "
                 "ORDER BY timestamp")
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=(symbol, f"-{int(days)} days"))
        if df.empty:
            return pd.DataFrame()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        if df["adj_close"].isna().all():
            df = df.drop(columns="adj_close")
        return df

    def store_anomaly_result_sync(self, anomaly: AnomalyAlert, model_version: str = "v1.0") -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO anomaly_results
                   (symbol, timestamp, anomaly_score, anomaly_type, confidence,
                    severity, explanation, model_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (anomaly.symbol, _ts(anomaly.timestamp), anomaly.anomaly_score,
                 anomaly.anomaly_type, anomaly.confidence, anomaly.severity,
                 anomaly.explanation, model_version))

    def count_rows(self, table: str) -> int:
        if table not in {"market_data", "anomaly_results"}:
            raise ValueError(table)
        with self._connect() as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    # ------------------------------------------------------------ async API
    async def init_database(self) -> None:
        await asyncio.to_thread(self.init_database_sync)
        logger.info("Database initialized at %s", self.db_path)

    async def store_market_data(self, symbol: str, data: pd.DataFrame) -> int:
        n = await asyncio.to_thread(self.store_market_data_sync, symbol, data)
        logger.info("Stored %d records for %s", n, symbol)
        return n

    async def get_market_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        return await asyncio.to_thread(self.get_market_data_sync, symbol, days)

    async def store_anomaly_result(self, anomaly: AnomalyAlert, model_version: str = "v1.0") -> None:
        await asyncio.to_thread(self.store_anomaly_result_sync, anomaly, model_version)
