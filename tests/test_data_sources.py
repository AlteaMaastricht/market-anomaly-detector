import asyncio
import unittest

import numpy as np
import pandas as pd

from market_anomaly.data_sources import DataSourceManager, normalize_ohlcv
from market_anomaly.synthetic import generate_ohlcv


class TestNormalize(unittest.TestCase):
    def setUp(self):
        self.base = generate_ohlcv(60, seed=1)

    def test_yahoo_capitalised_columns(self):
        y = self.base.rename(columns=str.capitalize)
        y["Adj Close"] = y["Close"]
        y.index.name = "Date"
        out = normalize_ohlcv(y)
        self.assertEqual(list(out.columns), ["open", "high", "low", "close", "volume", "adj_close"])
        self.assertEqual(out.index.name, "timestamp")

    def test_yfinance_multiindex(self):
        y = self.base.rename(columns=str.capitalize)
        y.columns = pd.MultiIndex.from_product([y.columns, ["SPY"]], names=["Price", "Ticker"])
        out = normalize_ohlcv(y, "SPY")
        self.assertEqual(list(out.columns), ["open", "high", "low", "close", "volume"])
        np.testing.assert_allclose(out["close"].values, self.base["close"].values)

    def test_alpha_vantage_strings_unsorted_tz(self):
        av = self.base.iloc[::-1].astype(str)
        av.index = av.index.tz_localize("US/Eastern")
        out = normalize_ohlcv(av)
        self.assertTrue(out.index.is_monotonic_increasing)
        self.assertIsNone(out.index.tz)
        self.assertEqual(out["close"].dtype, np.float64)

    def test_missing_column_raises(self):
        with self.assertRaises(KeyError):
            normalize_ohlcv(self.base.drop(columns="volume"))


class TestManager(unittest.TestCase):
    def test_synthetic_mode_is_offline_and_long_enough(self):
        mgr = DataSourceManager("synthetic")
        self.assertEqual([s.name for s in mgr.sources], ["synthetic"])
        df = asyncio.run(mgr.get_market_data("AAPL", 30 + 365))
        self.assertEqual(mgr.last_source, "synthetic")
        self.assertGreater(len(df), 250)

    def test_auto_mode_order(self):
        self.assertEqual([s.name for s in DataSourceManager("auto").sources],
                         ["yahoo", "alpha_vantage", "synthetic"])

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            DataSourceManager("bloomberg")


if __name__ == "__main__":
    unittest.main()
