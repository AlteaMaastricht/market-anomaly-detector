"""Pydantic request/response models and alert serialisation.

Only depends on pydantic, so it can be imported (and tested) without FastAPI.
"""

import json
import re
from datetime import datetime
from typing import List

from pydantic import BaseModel, Field

from .config import Config

# Tickers such as AAPL, BRK-B, BRK.B, ^GSPC, EURUSD=X. The original check
# (``symbol.isalpha() and len(symbol) <= 5``) rejected the last four.
_SYMBOL_RE = re.compile(r"^[\^]?[A-Z0-9][A-Z0-9.\-=]{0,11}$")


class AnomalyAlert(BaseModel):
    symbol: str
    timestamp: datetime
    anomaly_score: float
    anomaly_type: str
    confidence: float
    current_price: float
    explanation: str
    severity: str = Field(..., pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")


class DetectionRequest(BaseModel):
    symbols: List[str] = Field(default_factory=lambda: list(Config.DEFAULT_SYMBOLS))
    lookback_days: int = Field(default=30, ge=7, le=365)
    sensitivity: float = Field(default=0.05, ge=0.01, le=0.1)

    def validate_symbols(self) -> List[str]:
        """Upper-case and keep syntactically valid tickers.

        Returns the list of rejected symbols. Unlike the original, an
        all-invalid request no longer silently falls back to the 8 default
        symbols; the caller decides what to do with an empty list.
        """
        valid, rejected = [], []
        for symbol in self.symbols:
            if isinstance(symbol, str) and _SYMBOL_RE.match(symbol.strip().upper()):
                valid.append(symbol.strip().upper())
            else:
                rejected.append(symbol)
        self.symbols = valid
        return rejected


def alert_to_message(alert: AnomalyAlert) -> str:
    """Serialise an alert as the WebSocket ``anomaly_alert`` JSON message.

    The original called ``json.dumps(anomaly.dict())``, which raises
    ``TypeError`` because the timestamp is a datetime/pandas Timestamp.
    ``model_dump(mode="json")`` converts it to an ISO-8601 string.
    """
    return json.dumps({"type": "anomaly_alert", "data": alert.model_dump(mode="json")})
