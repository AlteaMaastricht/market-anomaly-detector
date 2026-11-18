"""End-to-end request path without FastAPI (what /detect-anomalies calls)."""

import asyncio
import os
import tempfile
import unittest

import pandas as pd

from market_anomaly.data_sources import DataSourceManager
from market_anomaly.detectors import MultiModalAnomalyDetector
from market_anomaly.pipeline import analyse_frame, detect_symbols
from market_anomaly.storage import DatabaseManager
from market_anomaly.synthetic import generate_ohlcv, inject_anomalies


class TestPipeline(unittest.TestCase):
    def test_default_request_is_scored(self):
        # Original: lookback 30 -> ~21 rows < 30-row guard -> always [] .
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = DatabaseManager(os.path.join(tmp.name, "p.db"))
        db.init_database_sync()
        results = asyncio.run(detect_symbols(["AAPL", "BRK-B"], 30, 0.05, DataSourceManager("synthetic"),
                                             MultiModalAnomalyDetector(), db))
        for r in results:
            self.assertIsNone(r.error)
            self.assertEqual(r.source, "synthetic")
            self.assertGreater(r.n_scored, 200)
            for a in r.alerts:
                self.assertGreaterEqual(pd.Timestamp(a.timestamp),
                                        pd.Timestamp.now().normalize() - pd.Timedelta(days=31))
        self.assertGreater(db.count_rows("market_data"), 0)
        self.assertEqual(db.count_rows("anomaly_results"), sum(len(r.alerts) for r in results))

    def test_injected_shock_in_window_is_reported(self):
        data, labels = inject_anomalies(generate_ohlcv(504, seed=42), seed=7)
        last = max(labels, key=lambda l: l.timestamp)
        lookback = (data.index[-1] - last.timestamp).days + 1
        res = analyse_frame(data, "SYN", MultiModalAnomalyDetector(), lookback_days=lookback)
        self.assertIsNone(res.error)
        self.assertTrue(all(pd.Timestamp(a.timestamp) >= last.timestamp - pd.Timedelta(days=1)
                            for a in res.alerts))

    def test_insufficient_data_reported(self):
        res = analyse_frame(generate_ohlcv(25, seed=1), "S", MultiModalAnomalyDetector())
        self.assertIn("insufficient", res.error)
        self.assertEqual(res.alerts, [])


if __name__ == "__main__":
    unittest.main()
