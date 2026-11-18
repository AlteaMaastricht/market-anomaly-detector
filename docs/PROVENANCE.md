# Provenance

This repository was prepared from the author's original project folder (v1,
a single-file FastAPI app) in November 2026. The original code was not under
version control. This file records which original code each staged file
comes from and every change made during preparation, so the original work
and the later fixes stay distinguishable.

Line numbers refer to the original v1 files (`app.py` = 1,066 lines,
`data_sources.py` = 182 lines).

## Source → staged file map

| Original | Staged | Nature of change |
|---|---|---|
| `app.py` L53–60 `Config` | `src/market_anomaly/config.py` | Same values; env overrides (`DATA_SOURCE`, `DATABASE_PATH`, `MODEL_DIR`, `ENABLE_BACKGROUND_TASKS`) and `TRAINING_HISTORY_DAYS` / `MIN_ROWS` added |
| `app.py` L62–83 `AnomalyAlert`, `DetectionRequest` | `src/market_anomaly/models.py` | Symbol validation fixed; `alert_to_message()` added (serialisation fix) |
| `app.py` L89–208 `DatabaseManager` | `src/market_anomaly/storage.py` | Same schema; insert/bind bugs fixed; `sqlite3` + `asyncio.to_thread` instead of `aiosqlite` |
| `app.py` L214–313 `AdvancedFeatureEngine` | `src/market_anomaly/features.py` | Same features and windows; `ta` calls re-implemented in pandas; warm-up rows dropped instead of zero-filled |
| `app.py` L319–541 `MultiModalAnomalyDetector` | `src/market_anomaly/detectors.py` | Same models, scaling, z-score ensemble and threshold; synchronous core; cache considers sensitivity |
| `app.py` L549–577 `DataIngestionService` + L885–916 endpoint body | `src/market_anomaly/pipeline.py` | Orchestration moved out of the web layer; window semantics changed (see below) |
| `app.py` L588–770 and `data_sources.py` (duplicate copies) | `src/market_anomaly/data_sources.py` | Consolidated into one copy; column normalisation; days instead of period strings |
| `app.py` L713–770 `MockDataSource` (price loop) | `src/market_anomaly/synthetic.py` + `SyntheticDataSource` | Price process rewritten in log-return space; anomaly injection and evaluation helpers are **new** |
| `app.py` L776–1054 (WebSocket manager, app, endpoints, background tasks) | `src/market_anomaly/api.py` | Same endpoints; uses the core modules; serialisation and test-data fixes |
| `run.py` | `run.py` | `--demo`, `--host`, `--port` flags; optional `.env` loading; binds to 127.0.0.1 by default (was 0.0.0.0) |
| `requirements.txt` | `requirements.txt`, `requirements-api.txt` | Split into core and API extras; `ta`, `ruptures`, `aiosqlite`, `aiofiles`, `joblib`, `curl-cffi`, `gunicorn` pins removed (no longer imported, or come in through scikit-learn) |
| `.env.example` | `.env.example` | Same placeholder key; new variables documented |
| `README.md` | `README.md` | Rewritten; the original "✅ All core functionality implemented" status was not accurate |
| — | `tests/`, `examples/demo.py`, `results/`, `docs/` | **New** |

## Fixes made (each verified against the original code first)

1. **Column case** (`app.py` L225 ff. vs yfinance output): yfinance returns
   `Open/High/Low/Close/Volume`, which recent versions put in a (Price, Ticker)
   MultiIndex, but the feature code read `df['close']`. The resulting KeyError was
   logged and swallowed (L265), so no features were produced and
   `/detect-anomalies` returned `[]`. Fix: `normalize_ohlcv()` at ingestion. A
   missing column now raises an error instead of producing an empty result.
2. **Lookback window** (`app.py` L895, L721–729, L220): the endpoint passed
   `period="30d"`, which the mock source's `days_map` did not contain, so it fell
   back to 30 calendar days (~22 business days). That is below the 30-row guard,
   so nothing was ever scored. Even with a valid period, a 30-day window could
   never pass the guard. Fix: sources take a number of days; the pipeline fetches
   `lookback_days + 365` days, fits on all complete rows, and reports alerts only
   inside the final `lookback_days`. **This changes behaviour**: the model is now
   fitted on about one year of history instead of the lookback window alone.
3. **DB insert** (`app.py` L155–164): `reset_index()` yields a `Date` (Yahoo) or
   `index` (mock) column, not `timestamp`, and mock data has no `adj_close`, so
   every insert raised a KeyError. It was logged, and `/test-data` still reported
   success. Fix: rows are built from the index and the canonical columns, with
   `adj_close` optional. Timestamps are bound as ISO strings: `sqlite3` cannot bind
   `pandas.Timestamp`, which also broke `store_anomaly_result` (new finding).
   Store errors now propagate.
4. **WebSocket serialisation** (`app.py` L810–819, L805): `json.dumps(anomaly.dict())`
   raises TypeError on the timestamp, and the client was then disconnected
   (L806–808). Fix: `alert_to_message()` uses `model_dump(mode="json")`.
5. **Mock price generator** (`app.py` L741–748): the loop mixed dollar amounts into
   a return (`0.001*price`, `0.7*ΔP`, `0.1*(base−P)`). Re-implemented with the same
   seed (`symbol_hash`) and base price, the series went 136 → 156.9 → 2149 →
   2.6e6 and became non-finite after 9 steps. Fix: the author's coefficients
   (0.1 % drift, 1 % noise, 0.7 momentum, 0.1 mean reversion) are applied to
   log returns, seeded per symbol.
6. **Duplicate `DataSourceManager`** (`app.py` L547 import, then L588 redefinition;
   `data_sources.py` also uses `asyncio` without importing it): consolidated into
   one module.
7. **yfinance session** (`app.py` L615, L635): a custom `requests.Session` was passed
   to `yf.download`. The author's own `data_sources.py` (L46) had already removed it
   ("Let yfinance handle session management"), and the staged code follows that
   version. Whether yfinance 0.2.66 rejects a `requests.Session` was not verified:
   no network, and yfinance is not installed.
8. **Alpha Vantage key in logs** (`app.py` L671, L709–710; new finding): the key was
   embedded in the URL, and the logged exception text can contain that URL. Fix:
   the key is sent as a query parameter, and only status codes or exception types
   are logged.
9. **Model cache ignored `sensitivity`** (L355–359): fitted models were reused for
   24 h regardless of the requested sensitivity. Fix: a sensitivity change forces
   a refit. `feature_columns` is now stored per symbol (it was shared).
10. **Symbol validation** (L77–83): `isalpha()` rejected `BRK-B`, `^GSPC`, etc.,
    and a request whose symbols were all invalid silently ran on the 8 default
    symbols. Fix: regex validation; an empty result gives HTTP 422.
11. **Explanation formatting** (L514–520): all messages were formatted eagerly, so a
    missing column raised and dropped the alert. Only the needed message is now
    formatted.
12. **Warm-up rows** (L345 `fillna(0)`): incomplete rows are dropped instead of
    zero-filled. Measured on the demo series, the original rule flagged none of
    the 30 warm-up rows, so this is a correctness change, not an accuracy gain.
13. **`/test-data`** (L947–991): used fixed 2023 dates, which the background task's
    "last 30 days" query could never return. It now stores synthetic data ending
    today.

## Deliberately NOT changed (documented as limitations instead)

- The decision rule: batch z-scoring of `decision_function`, the average of two
  models, threshold −2.5. Because of this, `contamination` / `sensitivity` has
  little effect.
- Fitting and scoring on the same data (in-sample).
- The rule-based anomaly types and severities.
- The feature set (9 detection features) and the extra, unused indicator columns.

## Removed or not carried over

- `import ruptures as rpt` (L31): imported, never used. There was no change-point
  detection in v1, so it is listed as planned work.
- `import aiofiles` (L39): unused.
- `'statistical': None` placeholder model (L385): never implemented.
- `calculate_statistical_features` (L271–301): **kept**, but it is still not called
  by the pipeline. statsmodels is imported lazily; this function was not executed
  during preparation.

## Not copied

- `app_version0.py`: an earlier iteration of `app.py` (TA-Lib/Redis variant)
  that `app.py` supersedes.
- `install.sh`: contained a local absolute path in a comment. It only created a
  venv and ran `pip install`; the README covers this.
- `venv/`, `market_data.db`, `.DS_Store`: environment, local database and OS
  artefact.
- v2 folder (8 files): non-importable skeleton (imports a missing `app.*`
  package). Its design is described in `docs/roadmap.md` as planned work.

## Phase 3 changes (18 Nov 2026) — publication preparation
Documentation and licence only; no code changed.
- `LICENSE` added: MIT, copyright 2026 Arthur J. Maarschalkerweerd (author decision D6).
- `README.md` updated to the Tier 1 evidence:
  - FastAPI endpoints moved to "implemented and tested";
  - the README API commands were executed live against `run.py --demo` (all HTTP 200; the 30-day example returns no alerts on synthetic data; invalid input returns 422);
  - test counts per environment;
  - verified versions;
  - limitations narrowed to the paths that are still unexecuted (live data sources, live WebSocket through uvicorn, long-running background tasks, `calculate_statistical_features`);
  - licence section points to `LICENSE`.
- `requirements-api.txt`: header comment corrected (the API part is tested; live data is not); noted that `uvicorn[standard]` is required for live WebSockets.
- README §11: AI-assistance sentence inserted in the author-approved wording (decision D7, 18 Nov 2026); the `Status: preparation draft` line removed for publication.
