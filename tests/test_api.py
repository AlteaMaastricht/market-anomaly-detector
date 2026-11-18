"""API tests. Require FastAPI (requirements-api.txt); skipped otherwise.

NOT EXECUTED during repository preparation (FastAPI not installed).
"""

import importlib.util
import json
import os
import tempfile
import unittest

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(HAS_FASTAPI, "needs fastapi (pip install -r requirements-api.txt)")
class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        os.environ["DATA_SOURCE"] = "synthetic"
        os.environ["ENABLE_BACKGROUND_TASKS"] = "0"
        os.environ["DATABASE_PATH"] = os.path.join(cls.tmp.name, "api.db")
        from fastapi.testclient import TestClient
        from market_anomaly import api
        cls.api = api
        cls.client_cm = TestClient(api.app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)
        cls.tmp.cleanup()

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["data_source"], "synthetic")

    def test_detect_anomalies(self):
        r = self.client.post("/detect-anomalies", json={"symbols": ["AAPL"], "lookback_days": 30})
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json(), list)

    def test_invalid_symbols_rejected(self):
        r = self.client.post("/detect-anomalies", json={"symbols": ["???"]})
        self.assertEqual(r.status_code, 422)

    def test_test_data_stores_rows(self):
        r = self.client.get("/test-data")
        self.assertEqual(r.status_code, 200)
        self.assertGreater(r.json()["records"], 200)

    def test_websocket_subscribe_and_broadcast(self):
        from tests.test_models import make_alert
        with self.client.websocket_connect("/ws/c1") as ws:
            ws.send_text(json.dumps({"type": "subscribe", "symbols": ["SPY"]}))
            self.assertEqual(ws.receive_json()["type"], "subscription_confirmed")
            self.client.portal.call(self.api.connection_manager.broadcast_anomaly, make_alert())
            msg = ws.receive_json()
            self.assertEqual(msg["type"], "anomaly_alert")
            self.assertEqual(msg["data"]["timestamp"], "2024-08-05T00:00:00")


if __name__ == "__main__":
    unittest.main()
