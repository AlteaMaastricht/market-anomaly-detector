"""Deterministic synthetic OHLCV data.

Replaces the original ``MockDataSource`` price loop, which computed
``daily_return = 0.001*price + 0.01*rnd + 0.7*(P[t-1]-P[t-2]) + 0.1*(base-P)``
and therefore mixed dollar amounts into a percentage return (the series
explodes for any price above a few dollars). Here every term lives in
log-return space, so the process is a geometric Brownian motion, optionally
with AR(1) momentum on returns and mean reversion of the log price.

``inject_anomalies`` plants labelled price shocks and volume spikes so the
detector can be checked against known ground truth. This is synthetic-only
evaluation; it says nothing about real-market performance.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


def generate_ohlcv(
    n_days: int = 504,
    seed: int = 42,
    start_price: float = 100.0,
    drift: float = 0.0003,
    volatility: float = 0.012,
    momentum: float = 0.0,
    mean_reversion: float = 0.0,
    avg_volume: float = 2_000_000.0,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Generate ``n_days`` business days of OHLCV data.

    Log return: ``r_t = drift + volatility*eps_t + momentum*r_{t-1}
    + mean_reversion*ln(start_price / P_{t-1})``. With momentum = 0 and
    mean_reversion = 0 this is a plain GBM.

    Columns: open, high, low, close, volume (lower case, the canonical
    schema used throughout the package). Index: DatetimeIndex "timestamp".
    """
    if n_days < 2:
        raise ValueError("n_days must be >= 2")
    rng = np.random.default_rng(seed)
    end = pd.Timestamp("2025-12-31") if end is None else pd.Timestamp(end)
    dates = pd.bdate_range(end=end.normalize(), periods=n_days, name="timestamp")

    eps = rng.standard_normal(n_days)
    log_p = np.empty(n_days)
    log_p[0] = np.log(start_price)
    log_base = np.log(start_price)
    prev_r = 0.0
    for t in range(1, n_days):
        r = (drift + volatility * eps[t] + momentum * prev_r
             + mean_reversion * (log_base - log_p[t - 1]))
        log_p[t] = log_p[t - 1] + r
        prev_r = r
    close = np.exp(log_p)

    intraday = 0.6 * volatility
    high = close * (1 + np.abs(rng.normal(0, intraday, n_days)))
    low = close * (1 - np.abs(rng.normal(0, intraday, n_days)))
    open_ = np.empty(n_days)
    open_[0] = close[0]
    open_[1:] = close[:-1] * (1 + rng.normal(0, intraday / 2, n_days - 1))
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])

    # Log-normal volume, mildly correlated with the absolute return.
    abs_ret = np.abs(np.diff(log_p, prepend=log_p[0]))
    volume = avg_volume * np.exp(rng.normal(0, 0.25, n_days)) * (1 + 10 * abs_ret)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": volume.round().astype("int64")},
        index=dates,
    )


@dataclass(frozen=True)
class InjectedAnomaly:
    timestamp: pd.Timestamp
    kind: str  # "price_shock" or "volume_spike"
    magnitude: float  # log-return jump for price shocks, volume multiple for spikes


def inject_anomalies(
    df: pd.DataFrame,
    n_price_shocks: int = 5,
    n_volume_spikes: int = 5,
    shock_size: float = 0.08,
    volume_multiple: float = 5.0,
    seed: int = 7,
    min_index: int = 60,
    min_gap: int = 25,
) -> "tuple[pd.DataFrame, List[InjectedAnomaly]]":
    """Return a copy of ``df`` with planted anomalies and their labels.

    * price shock: permanent log-price jump of +/- ``shock_size`` on one day
      (all later prices are shifted; volume on that day is doubled, which is
      what a real gap usually looks like).
    * volume spike: volume multiplied by ``volume_multiple`` on one day,
      price unchanged.

    Positions are drawn at least ``min_gap`` rows apart and after
    ``min_index`` (so the rolling indicators are warmed up).
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    total = n_price_shocks + n_volume_spikes
    candidates = np.arange(min_index, n)
    positions: List[int] = []
    attempts = 0
    while len(positions) < total:
        attempts += 1
        if attempts > 10_000:
            raise ValueError("could not place anomalies; series too short for min_gap")
        p = int(rng.choice(candidates))
        if all(abs(p - q) >= min_gap for q in positions):
            positions.append(p)
    positions.sort()
    kinds = ["price_shock"] * n_price_shocks + ["volume_spike"] * n_volume_spikes
    rng.shuffle(kinds)

    labels: List[InjectedAnomaly] = []
    price_cols = ["open", "high", "low", "close"]
    for pos, kind in zip(positions, kinds):
        ts = out.index[pos]
        if kind == "price_shock":
            jump = shock_size * (1 if rng.random() < 0.5 else -1)
            factor = float(np.exp(jump))
            out.iloc[pos:, out.columns.get_indexer(price_cols)] *= factor
            # The day's open sits at the pre-jump level (overnight gap).
            prev_close = out["close"].iloc[pos - 1]
            out.iloc[pos, out.columns.get_loc("open")] = prev_close
            o, c = out["open"].iloc[pos], out["close"].iloc[pos]
            out.iloc[pos, out.columns.get_loc("high")] = max(out["high"].iloc[pos], o, c)
            out.iloc[pos, out.columns.get_loc("low")] = min(out["low"].iloc[pos], o, c)
            out.iloc[pos, out.columns.get_loc("volume")] = int(out["volume"].iloc[pos] * 2)
            labels.append(InjectedAnomaly(ts, kind, jump))
        else:
            out.iloc[pos, out.columns.get_loc("volume")] = int(
                out["volume"].iloc[pos] * volume_multiple)
            labels.append(InjectedAnomaly(ts, kind, volume_multiple))
    return out, labels


def detection_summary(
    labels: Sequence[InjectedAnomaly],
    flagged: Sequence[pd.Timestamp],
    index: pd.DatetimeIndex,
    tolerance: int = 0,
) -> dict:
    """Recall per anomaly kind and overall precision against planted labels.

    A label counts as detected if a flagged timestamp lies within
    ``tolerance`` rows of it. Precision = flagged rows that match some label
    / all flagged rows.
    """
    pos = {ts: i for i, ts in enumerate(index)}
    flagged_pos = sorted(pos[pd.Timestamp(t)] for t in flagged if pd.Timestamp(t) in pos)
    label_pos = [(pos[l.timestamp], l.kind) for l in labels]

    def hit(lp: int) -> bool:
        return any(abs(fp - lp) <= tolerance for fp in flagged_pos)

    summary = {"tolerance_rows": tolerance, "n_flagged": len(flagged_pos)}
    for kind in sorted({k for _, k in label_pos}):
        group = [lp for lp, k in label_pos if k == kind]
        detected = sum(hit(lp) for lp in group)
        summary[f"{kind}_detected"] = detected
        summary[f"{kind}_total"] = len(group)
        summary[f"{kind}_recall"] = detected / len(group) if group else None
    matched = sum(any(abs(fp - lp) <= tolerance for lp, _ in label_pos) for fp in flagged_pos)
    summary["n_flagged_matching_a_label"] = matched
    summary["precision"] = matched / len(flagged_pos) if flagged_pos else None
    return summary
