import unittest

import numpy as np
import pandas as pd

from market_anomaly.data_sources import SyntheticDataSource
from market_anomaly.synthetic import detection_summary, generate_ohlcv, inject_anomalies


class TestGenerator(unittest.TestCase):
    def test_deterministic(self):
        pd.testing.assert_frame_equal(generate_ohlcv(100, seed=3), generate_ohlcv(100, seed=3))
        self.assertFalse(generate_ohlcv(100, seed=3).equals(generate_ohlcv(100, seed=4)))

    def test_ohlc_consistency(self):
        df = generate_ohlcv(300, seed=5)
        self.assertTrue((df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all())
        self.assertTrue((df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all())
        self.assertTrue((df[["open", "high", "low", "close", "volume"]] > 0).all().all())
        self.assertEqual(df.index.name, "timestamp")

    def test_gbm_return_volatility(self):
        df = generate_ohlcv(2000, seed=9, volatility=0.012, drift=0.0)
        lr = np.log(df["close"]).diff().dropna()
        self.assertAlmostEqual(lr.std(), 0.012, delta=0.001)

    def test_mock_source_is_bounded(self):
        # Original mock mixed prices into returns and exploded; 5 years must stay finite/bounded.
        df = SyntheticDataSource(end=pd.Timestamp("2025-12-31")).generate("AAPL", 1825)
        self.assertTrue(np.isfinite(df["close"]).all())
        self.assertLess(df["close"].max() / df["close"].min(), 50)


class TestInjection(unittest.TestCase):
    def test_labels_and_effects(self):
        clean = generate_ohlcv(504, seed=42)
        data, labels = inject_anomalies(clean, n_price_shocks=3, n_volume_spikes=2, seed=1)
        self.assertEqual(len(labels), 5)
        self.assertEqual(sorted(l.kind for l in labels).count("price_shock"), 3)
        for l in labels:
            i = data.index.get_loc(l.timestamp)
            if l.kind == "volume_spike":
                self.assertAlmostEqual(data["volume"].iloc[i] / clean["volume"].iloc[i], 5.0, delta=0.01)
                # no price jump on a volume-spike day
                ratio = data["close"] / clean["close"]
                self.assertAlmostEqual(ratio.iloc[i], ratio.iloc[i - 1], places=12)
            else:
                jump = np.log(data["close"].iloc[i] / data["close"].iloc[i - 1]) - \
                    np.log(clean["close"].iloc[i] / clean["close"].iloc[i - 1])
                self.assertAlmostEqual(jump, l.magnitude, places=9)
        self.assertIsNot(data, clean)

    def test_summary_counts(self):
        idx = pd.bdate_range("2024-01-01", periods=10)
        from market_anomaly.synthetic import InjectedAnomaly
        labels = [InjectedAnomaly(idx[2], "price_shock", 0.1), InjectedAnomaly(idx[6], "volume_spike", 5)]
        s = detection_summary(labels, [idx[2], idx[7]], idx, tolerance=0)
        self.assertEqual((s["price_shock_detected"], s["volume_spike_detected"]), (1, 0))
        self.assertEqual(s["precision"], 0.5)
        s1 = detection_summary(labels, [idx[2], idx[7]], idx, tolerance=1)
        self.assertEqual(s1["volume_spike_detected"], 1)


if __name__ == "__main__":
    unittest.main()
