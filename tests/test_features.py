import unittest

import numpy as np
import pandas as pd

from market_anomaly.features import (DETECTION_FEATURES, calculate_technical_indicators,
                                     feature_matrix, hurst_exponent, on_balance_volume, rsi)
from market_anomaly.synthetic import generate_ohlcv


class TestIndicators(unittest.TestCase):
    def setUp(self):
        self.df = generate_ohlcv(n_days=200, seed=1)
        self.f = calculate_technical_indicators(self.df)

    def test_all_detection_features_present(self):
        for col in DETECTION_FEATURES:
            self.assertIn(col, self.f.columns)

    def test_input_not_mutated(self):
        self.assertNotIn("returns", self.df.columns)

    def test_returns_match_pct_change(self):
        np.testing.assert_allclose(self.f["returns"].iloc[1:], self.df["close"].pct_change().iloc[1:])

    def test_bounded_indicators(self):
        r = self.f["rsi"].dropna()
        self.assertTrue(((r >= 0) & (r <= 100)).all())
        k = self.f["stoch_k"].dropna()
        self.assertTrue(((k >= 0) & (k <= 100)).all())
        self.assertTrue((self.f["spread"] >= 0).all())

    def test_rsi_monotonic_series(self):
        up = pd.Series(np.arange(1.0, 40.0))
        self.assertAlmostEqual(rsi(up).iloc[-1], 100.0)
        self.assertAlmostEqual(rsi(up[::-1].reset_index(drop=True)).iloc[-1], 0.0)

    def test_obv_hand_computed(self):
        close = pd.Series([10.0, 11.0, 10.5, 10.5, 12.0])
        vol = pd.Series([100, 200, 300, 400, 500])
        # ta convention: down day subtracts, otherwise adds (incl. first row and flat days)
        self.assertEqual(on_balance_volume(close, vol).tolist(), [100, 300, 0, 400, 900])

    def test_feature_matrix_drops_warmup_rows(self):
        fm = feature_matrix(self.f)
        self.assertFalse(fm.isna().any().any())
        self.assertGreater(len(fm), 0)
        self.assertLess(len(fm), len(self.f))  # warm-up rows removed, not zero-filled
        self.assertGreaterEqual(fm.index.min(), self.f.index[30])

    def test_short_frame_returned_unchanged(self):
        short = self.df.iloc[:20]
        out = calculate_technical_indicators(short)
        self.assertEqual(list(out.columns), list(short.columns))
        self.assertTrue(feature_matrix(out).empty)

    def test_capitalised_columns_raise_instead_of_silently_empty(self):
        # Original bug: yfinance 'Close' vs 'close' -> KeyError swallowed -> []
        yahoo_like = self.df.rename(columns=str.capitalize)
        with self.assertRaises(KeyError):
            calculate_technical_indicators(yahoo_like)


class TestHurst(unittest.TestCase):
    def test_random_walk_near_half(self):
        rw = np.cumsum(np.random.default_rng(0).standard_normal(4000))
        self.assertAlmostEqual(hurst_exponent(rw), 0.5, delta=0.1)

    def test_too_short_returns_default(self):
        self.assertEqual(hurst_exponent(np.arange(5.0)), 0.5)


if __name__ == "__main__":
    unittest.main()
