"""FastAPI service (from the v1 ``app.py``). Requires the ``api`` extra.

Endpoints (same as the original): ``GET /``, ``GET /health``,
``GET /test-data``, ``POST /detect-anomalies``, ``WebSocket /ws/{client_id}``.

NOTE: this module was NOT executed during repository preparation (FastAPI
was not installed); only the FastAPI-free core (pipeline, detectors,
features, storage, models) is covered by the executed tests.
"""

import asyncio
import json
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import Config
from .data_sources import DataSourceManager, SyntheticDataSource
from .detectors import MultiModalAnomalyDetector
from .models import AnomalyAlert, DetectionRequest, alert_to_message
from .pipeline import detect_symbols, history_days
from .storage import DatabaseManager

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and per-client symbol subscriptions."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, List[str]] = defaultdict(list)

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info("Client %s connected", client_id)

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)
        self.subscriptions.pop(client_id, None)
        logger.info("Client %s disconnected", client_id)

    async def subscribe(self, client_id: str, symbols: List[str]):
        self.subscriptions[client_id] = [s.upper() for s in symbols]
        await self.send_text(client_id, json.dumps(
            {"type": "subscription_confirmed", "symbols": self.subscriptions[client_id]}))

    async def send_text(self, client_id: str, text: str):
        ws = self.active_connections.get(client_id)
        if ws is None:
            return
        try:
            await ws.send_text(text)
        except Exception as exc:
            logger.error("Error sending message to %s: %s", client_id, exc)
            self.disconnect(client_id)

    async def broadcast_anomaly(self, anomaly: AnomalyAlert):
        # Original: json.dumps(anomaly.dict()) -> TypeError on the datetime.
        text = alert_to_message(anomaly)
        for client_id, symbols in list(self.subscriptions.items()):
            if anomaly.symbol in symbols:
                await self.send_text(client_id, text)


# Global instances (as in the original)
db_manager = DatabaseManager()
data_manager = DataSourceManager()
anomaly_detector = MultiModalAnomalyDetector()
connection_manager = ConnectionManager()
background_tasks = set()


async def periodic_data_update():
    await asyncio.sleep(10)
    while True:
        try:
            for symbol in Config.DEFAULT_SYMBOLS:
                data = await data_manager.get_market_data(symbol, history_days(30))
                if data is not None and not data.empty:
                    await db_manager.store_market_data(symbol, data)
                await asyncio.sleep(1)  # rate limiting
        except Exception as exc:
            logger.error("Error in periodic data update: %s", exc)
        await asyncio.sleep(Config.UPDATE_INTERVAL)


async def periodic_anomaly_detection():
    await asyncio.sleep(60)
    from .pipeline import analyse_frame

    while True:
        try:
            for symbol in Config.DEFAULT_SYMBOLS:
                df = await db_manager.get_market_data(symbol, days=history_days(30))
                if df.empty:
                    logger.warning("No stored data for %s, skipping", symbol)
                    continue
                res = await asyncio.to_thread(analyse_frame, df, symbol, anomaly_detector, 30, 0.05)
                for alert in res.alerts:
                    await db_manager.store_anomaly_result(alert)
                    await connection_manager.broadcast_anomaly(alert)
        except Exception as exc:
            logger.error("Error in periodic anomaly detection: %s", exc)
        await asyncio.sleep(Config.UPDATE_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Market Anomaly Detector (DATA_SOURCE=%s)", data_manager.mode)
    await db_manager.init_database()
    if Config.ENABLE_BACKGROUND_TASKS:
        for coro in (periodic_data_update(), periodic_anomaly_detection()):
            background_tasks.add(asyncio.create_task(coro))
    yield
    for task in background_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    background_tasks.clear()


app = FastAPI(
    title="Market Anomaly Detector",
    description="Unsupervised anomaly detection on daily OHLCV data",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Market Anomaly Detector API", "version": app.version,
            "data_source": data_manager.mode}


@app.post("/detect-anomalies", response_model=List[AnomalyAlert])
async def detect_anomalies(request: DetectionRequest):
    rejected = request.validate_symbols()
    if not request.symbols:
        raise HTTPException(status_code=422, detail=f"no valid symbols in request: {rejected}")
    results = await detect_symbols(request.symbols, request.lookback_days, request.sensitivity,
                                   data_manager, anomaly_detector, db_manager)
    for res in results:
        if res.error:
            logger.warning("%s: %s", res.symbol, res.error)
    return [alert for res in results for alert in res.alerts]


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat(),
            "data_source": data_manager.mode, "models_trained": anomaly_detector.is_trained}


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await connection_manager.connect(websocket, client_id)
    try:
        while True:
            message = json.loads(await websocket.receive_text())
            if message.get("type") == "subscribe":
                await connection_manager.subscribe(
                    client_id, message.get("symbols", Config.DEFAULT_SYMBOLS))
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
    except Exception as exc:
        logger.error("WebSocket error for %s: %s", client_id, exc)
        connection_manager.disconnect(client_id)


@app.get("/test-data")
async def generate_test_data(symbol: str = "AAPL", days: int = 400):
    """Store deterministic synthetic data for ``symbol`` (dates end today, so
    the background detector's "last N days" query finds them; the original
    used fixed 2023 dates and its insert always failed)."""
    data = SyntheticDataSource().generate(symbol.upper(), days)
    try:
        n = await db_manager.store_market_data(symbol.upper(), data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"storing test data failed: {exc}")
    return {"message": "Synthetic data stored", "symbol": symbol.upper(), "records": n,
            "date_range": f"{data.index[0].date()} to {data.index[-1].date()}"}
