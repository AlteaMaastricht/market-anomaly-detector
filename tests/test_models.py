import json
import unittest
from datetime import datetime

import pandas as pd
from pydantic import ValidationError

from market_anomaly.models import AnomalyAlert, DetectionRequest, alert_to_message


def make_alert(ts=pd.Timestamp("2024-08-05")):
    return AnomalyAlert(symbol="SPY", timestamp=ts, anomaly_score=3.2, anomaly_type="PRICE_SHOCK",
                        confidence=1.0, current_price=512.3, explanation="x", severity="HIGH")


class TestAlertSerialisation(unittest.TestCase):
    def test_message_is_valid_json_with_iso_timestamp(self):
        msg = json.loads(alert_to_message(make_alert()))
        self.assertEqual(msg["type"], "anomaly_alert")
        self.assertEqual(msg["data"]["timestamp"], "2024-08-05T00:00:00")
        self.assertEqual(msg["data"]["symbol"], "SPY")

    def test_python_datetime_also_serialises(self):
        msg = json.loads(alert_to_message(make_alert(datetime(2020, 3, 16, 15, 30))))
        self.assertEqual(msg["data"]["timestamp"], "2020-03-16T15:30:00")

    def test_original_approach_fails(self):
        """Documents the original bug: json.dumps(alert.dict()) -> TypeError."""
        with self.assertRaises(TypeError):
            json.dumps({"type": "anomaly_alert", "data": make_alert().model_dump()})

    def test_severity_validated(self):
        with self.assertRaises(ValidationError):
            AnomalyAlert(symbol="X", timestamp=datetime(2024, 1, 1), anomaly_score=1, anomaly_type="T",
                         confidence=1, current_price=1, explanation="", severity="EXTREME")


class TestDetectionRequest(unittest.TestCase):
    def test_symbol_validation(self):
        req = DetectionRequest(symbols=["aapl", "BRK-B", "^GSPC", "BRK.B", "bad symbol!", ""])
        rejected = req.validate_symbols()
        self.assertEqual(req.symbols, ["AAPL", "BRK-B", "^GSPC", "BRK.B"])
        self.assertEqual(rejected, ["bad symbol!", ""])

    def test_no_silent_fallback_to_defaults(self):
        req = DetectionRequest(symbols=["???"])
        req.validate_symbols()
        self.assertEqual(req.symbols, [])

    def test_bounds(self):
        with self.assertRaises(ValidationError):
            DetectionRequest(lookback_days=3)
        with self.assertRaises(ValidationError):
            DetectionRequest(sensitivity=0.5)


if __name__ == "__main__":
    unittest.main()
