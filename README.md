# Market Anomaly Detector

This project flags unusual days in daily stock-market data (OHLCV: open, high,
low, close, volume). It combines technical-indicator features with an
IsolationForest + EllipticEnvelope ensemble and serves the results through a
FastAPI service with REST, WebSocket and SQLite. An offline mode uses seeded
synthetic data, so the detection pipeline can be run and tested without
network access or API keys.

## 1. Overview

| Status | Feature |
|---|---|
| **Implemented and tested** (offline unit tests) | Feature engineering: returns, rolling volatility, RSI, MACD, Bollinger position, volume ratio, rolling skew and kurtosis, spread, plus extra indicators |
| | IsolationForest + EllipticEnvelope ensemble with rule-based anomaly types and severities |
| | Deterministic synthetic OHLCV generator (GBM) with planted price shocks and volume spikes, plus a recall/precision summary |
| | SQLite storage for market data and alerts |
| | JSON serialisation of alerts for the WebSocket |
| | The `/detect-anomalies` request path as a FastAPI-free pipeline |
| **Implemented and tested** (with `requirements-api.txt`) | FastAPI app: `/`, `/health`, `/test-data`, `/detect-anomalies`, WebSocket `/ws/{client_id}`. Covered by 5 API tests (in-process test client, WebSocket included) and a live REST smoke test against `run.py --demo` |
| **Implemented, not executed here** | Background update and detection tasks over long runs; a live WebSocket connection through uvicorn |
| | Yahoo Finance and Alpha Vantage data sources |
| | `calculate_statistical_features` (ADF, Ljung-Box, Jarque-Bera, Hurst): present, but not wired into the API |
| **Planned** (see [`docs/roadmap.md`](docs/roadmap.md)) | Change-point detection (ruptures/PELT), GARCH, Markov-switching, LSTM autoencoder / Transformer, TimescaleDB/Redis, out-of-sample evaluation, baselines |
| **Not implemented** | Any evaluation on real market data; intraday or real-time streaming (data is daily bars polled every 5 minutes) |

The original version did not work end to end: `/detect-anomalies` always
returned an empty list. Its faults were fixed during repository preparation;
[`docs/PROVENANCE.md`](docs/PROVENANCE.md) lists every fault and every change.

## 2. Problem

The goal is to spot trading days that look unusual for a given ticker
(price shocks, volume spikes, volatility bursts) without labelled examples of
"anomalies", and to push those alerts to clients as they are found.

## 3. Approach

1. **Data.** Daily OHLCV comes from Yahoo Finance, then Alpha Vantage, then
   synthetic data as a last resort (`DATA_SOURCE=auto`). `DATA_SOURCE=synthetic`
   uses synthetic data only. Every source is normalised to lower-case
   `open/high/low/close/volume`.
2. **Features** (`features.py`). Nine detection features: returns, 20-day
   volatility, RSI(14), MACD(12/26), Bollinger-band position(20, 2σ), volume /
   20-day average volume, 30-day rolling skew and kurtosis of returns, and
   high–low spread. Rows whose rolling windows are not yet full are dropped.
3. **Detector** (`detectors.py`). The features are standardised, then an
   IsolationForest and an EllipticEnvelope are fitted. Each model's
   `decision_function` is z-scored within the batch, the two are averaged, and
   rows below **−2.5** are flagged.
4. **Labelling.** Fixed rules assign the anomaly type and severity:
   - volume ratio > 3 → `VOLUME_SPIKE`
   - |return| > 5 % → `PRICE_SHOCK` (`CRITICAL` above 10 %)
   - volatility > 3 % → `VOLATILITY_SPIKE`
   - RSI < 20 or > 80 → `TECHNICAL_EXTREME`
   - anything else → `UNKNOWN`
5. **Window.** A request with `lookback_days = N` fetches N + 365 calendar days.
   The models are fitted and scored on all complete rows, and only alerts from
   the last N days are returned.

## 4. Architecture

```
            ┌──────────────── src/market_anomaly ────────────────┐
 Yahoo ─┐   │ data_sources.py  normalize_ohlcv, fallback manager  │
 AlphaV ├──►│        │                                            │
 Synth ─┘   │        ▼                                            │
 (synthetic.py)  features.py ──► detectors.py ──► models.py       │
            │        ▲             (IF + EE)      AnomalyAlert,    │
            │        │                            alert_to_message │
            │  pipeline.py  (fetch → store → analyse → persist)   │
            │        │                │                            │
            │        ▼                ▼                            │
            │  storage.py (SQLite)   api.py (FastAPI: REST + WS,  │
            │                         background tasks)  [extra]  │
            └─────────────────────────────────────────────────────┘
```

Everything except `api.py` needs only numpy, pandas, scikit-learn, scipy
and pydantic.

## 5. Technologies

- Python
- pandas and numpy
- scikit-learn (IsolationForest, EllipticEnvelope, StandardScaler)
- pydantic v2
- SQLite (`sqlite3`)
- FastAPI and uvicorn (optional extra)
- yfinance and requests (optional)
- statsmodels (optional)

## 6. Installation & running

Core install (pipeline, demo and tests):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

API and live data (optional). The API part of this extra was tested with Python 3.12
(fastapi 0.163.2, httpx 0.28.1, uvicorn 0.54.0). The live data sources were not:

```bash
pip install -r requirements-api.txt
cp .env.example .env            # optional: add ALPHA_VANTAGE_API_KEY
python run.py --demo            # offline: synthetic data only
python run.py                   # DATA_SOURCE=auto: Yahoo -> Alpha Vantage -> synthetic
```

The server listens on `http://127.0.0.1:8000` (`--host` / `--port` to change).
In `auto` mode, a warning is logged whenever the synthetic fallback is used.

## 7. Example usage

Offline demo (no network). It plants 5 price shocks and 5 volume spikes in a
2-year seeded GBM series and reports what the detector finds:

```bash
python examples/demo.py                       # defaults: --seed 42 --inject-seed 7
python examples/demo.py --json summary.json   # also write the summary as JSON
```

Tests:

```bash
python -m unittest discover -s tests -t . -v
```

API calls, once `run.py --demo` is running. These exact commands were run during
preparation and returned HTTP 200. On the synthetic series the 30-day request returned
no alerts; with `"lookback_days": 60` alerts are returned. Invalid symbols or
`lookback_days` < 7 return HTTP 422.

```bash
curl http://127.0.0.1:8000/health
curl "http://127.0.0.1:8000/test-data?symbol=AAPL"
curl -X POST http://127.0.0.1:8000/detect-anomalies \
     -H "Content-Type: application/json" \
     -d '{"symbols": ["AAPL", "BRK-B"], "lookback_days": 30, "sensitivity": 0.05}'
```

WebSocket: connect to `/ws/<client_id>` and send
`{"type": "subscribe", "symbols": ["AAPL"]}`. Each alert then arrives as
`{"type": "anomaly_alert", "data": {...}}`. Live WebSocket connections need the
WebSocket support that `uvicorn[standard]` installs. This path is tested in-process
(`tests/test_api.py`), not over a live connection.

## 8. Results

All numbers below come from the synthetic demo
([`results/demo_output.txt`](results/demo_output.txt),
[`results/demo_summary.json`](results/demo_summary.json)). The demo was
generated by `python examples/demo.py --json results/demo_summary.json` with
GBM seed 42 and injection seed 7: 504 business days, of which 474 rows were
scored.

| Planted anomaly | Detected on the exact day |
|---|---|
| Price shock (log jump of 0.08, ≈ 8 %) | **4 / 5** |
| Volume spike (5 × volume, price unchanged) | **0 / 5** |
| Precision (alerts that match a planted anomaly) | **4 / 10** |
| Same series without planted anomalies: rows flagged | **8 / 474 (1.7 %)** |

What the demo shows:

- **Price gaps are found.** A one-day 8 % gap stands out in several features at
  once, and 4 of the 5 were flagged.
- **Single-day volume spikes are missed.** A 5× spike moves only one of the nine
  standardised features. After batch z-scoring and averaging, the scores stayed
  between −1.52 and +0.15, far from −2.5. The test suite keeps this as an
  *expected failure* (`tests/test_detectors.py`) so the gap stays visible.
- **One label is wrong.** The price shock on 2025-09-19 was detected but labelled
  `VOLUME_SPIKE`. The injector doubles volume on a shock day, and the volume rule
  is checked before the return rule.
- **Alerts cluster after shocks.** Six `UNKNOWN` alerts follow the first shock by
  6–16 trading days. The 20- and 30-day rolling volatility, skew and kurtosis
  windows still contain the shock, so the days after it look unusual too.

Test suite: 56 tests. With `requirements-api.txt` installed (Python 3.12),
55 passed and 1 is an expected failure (the volume-spike recall above). With only
`requirements.txt`, the 5 API tests skip automatically (Python 3.14: 50 passed,
1 expected failure, 5 skipped).

> **Evaluation limits.** These are recall and precision on *synthetic*
> anomalies that we planted ourselves, in a GBM series with Gaussian returns.
> They are a sanity check of the pipeline. They do **not** measure real-world
> performance: there are no labelled market anomalies in this repository, and
> the detector has never been evaluated on real market data.

## 9. Limitations

- **In-sample scoring.** Models are fitted on the same rows they score, and the
  −2.5 threshold applies to batch-z-scored scores. As a result, every batch
  produces roughly the same share of "anomalies", and `sensitivity` (mapped to
  `contamination`) has little effect on which rows are flagged.
- **Anomaly types are fixed rules.** They do not come from the models.
  `UNKNOWN` alerts carry no explanation.
- **Rolling features smear one event across later days.** A single shock makes
  the following 20–30 rows look anomalous (see Results).
- **Daily data only.** "Real-time" means the daily bars are re-polled every
  5 minutes.
- **Indicators re-implemented in pandas.** RSI, MACD, Bollinger, OBV, EMA and
  stochastic %K are re-implemented in pandas following the `ta` 0.10.2 defaults.
  Numerical equivalence with `ta` was not checked.
- **Partly untested paths.** The live data sources (Yahoo, Alpha Vantage), a live
  WebSocket connection through uvicorn, long-running background tasks and
  `calculate_statistical_features` were not executed during preparation.
- **Process-local state.** The model cache lives in memory and SQLite allows a
  single writer, so the server must run as one worker.

## 10. Reproducibility

- **Seeds.** The demo is fully determined by `--seed` (GBM), `--inject-seed`
  (anomaly placement) and a fixed end date (2025-12-31). The models use
  `random_state=42`.
- **Synthetic data source.** `DATA_SOURCE=synthetic` seeds each series from the
  ticker symbol. The series ends today, so the dates move from day to day but
  the shape is fixed.
- **Verified environment.** The results were produced with Python 3.14.0,
  numpy 2.4.4, pandas 3.1.0, scikit-learn 1.8.0 and pydantic 2.14.2. Re-running the
  demo in that environment reproduced `results/demo_summary.json` byte for byte. The API
  tests ran with Python 3.12.16, fastapi 0.163.2, httpx 0.28.1, uvicorn 0.54.0,
  numpy 2.5.5, pandas 3.1.1 and scikit-learn 1.9.1. Other versions may change the
  alert lists slightly.
- **Live data.** It is not reproducible: Yahoo and Alpha Vantage data change and
  are subject to their providers' terms. No market data is included in this
  repository.

## 11. Context, attribution & licence

- **Context.** Solo personal project (2025), extended with a fix-and-test pass
  in 2026. The fixes are listed in [`docs/PROVENANCE.md`](docs/PROVENANCE.md).
- **AI assistance.** The original 2025 version was developed with substantial
  assistance from AI chat assistants (ChatGPT and Claude). The 2026 repair and
  preparation—including bug fixes, tests, the synthetic demo, and documentation—was
  performed with Claude Code under my direction. I reviewed the changes. The demo
  results are reproducible, and the validation details and limitations are
  documented in [docs/PROVENANCE.md](docs/PROVENANCE.md).
- **Data sources.** Yahoo Finance (via the unofficial `yfinance` library) and
  Alpha Vantage (free API key) are used at run time only. No third-party data is
  redistributed here.
- **License.** MIT — see [`LICENSE`](LICENSE).
