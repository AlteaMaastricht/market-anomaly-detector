import asyncio
import os
import tempfile
import unittest
from datetime import datetime

import pandas as pd

from market_anomaly.data_sources import SyntheticDataSource, normalize_ohlcv
from market_anomaly.models import AnomalyAlert
from market_anomaly.storage import DatabaseManager


class TestStorage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = DatabaseManager(os.path.join(self.tmp.name, "t.db"))
        self.db.init_database_sync()
        # Dates end today so the "last N days" query window contains them.
        self.data = SyntheticDataSource().generate("AAPL", 120)

    def tearDown(self):
        self.tmp.cleanup()

    def test_store_and_read_back_without_adj_close(self):
        # Original: row['timestamp'] / row['adj_close'] KeyError -> nothing stored.
        n = self.db.store_market_data_sync("AAPL", self.data)
        self.assertEqual(n, len(self.data))
        self.assertEqual(self.db.count_rows("market_data"), len(self.data))
        back = self.db.get_market_data_sync("AAPL", days=200)
        self.assertEqual(len(back), len(self.data))
        self.assertNotIn("adj_close", back.columns)
        pd.testing.assert_series_equal(back["close"], self.data["close"], check_freq=False)

    def test_yahoo_shaped_frame_with_adj_close(self):
        yahoo = self.data.rename(columns=str.capitalize)
        yahoo["Adj Close"] = yahoo["Close"] * 0.99
        yahoo.index.name = "Date"
        self.db.store_market_data_sync("MSFT", normalize_ohlcv(yahoo, "MSFT"))
        back = self.db.get_market_data_sync("MSFT", days=200)
        self.assertIn("adj_close", back.columns)
        self.assertAlmostEqual(back["adj_close"].iloc[-1], self.data["close"].iloc[-1] * 0.99)

    def test_upsert_is_idempotent(self):
        self.db.store_market_data_sync("AAPL", self.data)
        self.db.store_market_data_sync("AAPL", self.data)
        self.assertEqual(self.db.count_rows("market_data"), len(self.data))

    def test_window_filter(self):
        self.db.store_market_data_sync("AAPL", self.data)
        self.assertLess(len(self.db.get_market_data_sync("AAPL", days=30)), len(self.data))
        self.assertTrue(self.db.get_market_data_sync("ZZZ", days=30).empty)

    def test_store_anomaly_with_pandas_timestamp(self):
        # Original bound a pandas Timestamp -> sqlite3 "type 'Timestamp' is not supported".
        alert = AnomalyAlert(symbol="AAPL", timestamp=pd.Timestamp("2024-08-05"), anomaly_score=3.0,
                             anomaly_type="PRICE_SHOCK", confidence=1.0, current_price=200.0,
                             explanation="x", severity="HIGH")
        self.db.store_anomaly_result_sync(alert)
        self.assertEqual(self.db.count_rows("anomaly_results"), 1)

    def test_async_wrappers(self):
        async def go():
            await self.db.init_database()
            n = await self.db.store_market_data("AAPL", self.data)
            df = await self.db.get_market_data("AAPL", days=200)
            return n, len(df)
        n, m = asyncio.run(go())
        self.assertEqual(n, m)


if __name__ == "__main__":
    unittest.main()
