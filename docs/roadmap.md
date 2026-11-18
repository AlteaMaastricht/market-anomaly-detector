# Roadmap (planned, not implemented)

None of the items below exist as working code in this repository.

The author started a redesign ("v2") as a separate set of 8 loose files,
about 500 lines. It was never completed:

- every file imports an `app.config` / `app.database` / `app.services.*`
  package that was never created;
- several helper methods are called but never defined;
- the MF-DFA function is an empty stub;
- there is no Dockerfile or requirements file.

Those files are **not** included here. This page records the intended design
so it can be rebuilt properly.

## Detection

| Planned component | State of the v2 sketch | Notes for a real implementation |
|---|---|---|
| Change-point detection (`ruptures` PELT) | 4-line function calling `rpt.Pelt(model="rbf")` with a fixed penalty, labelled "Bayesian" (it is not Bayesian) | Run on returns / rolling volatility; choose the penalty by BIC or a validation set; expose change-points via an endpoint |
| GARCH(1,1) volatility | Function calling `arch.arch_model` | Flag returns whose standardised residual exceeds a threshold, scored out-of-sample |
| Markov-switching regimes | Function calling statsmodels `MarkovRegression` | Report regime probabilities; alert on regime switches |
| MF-DFA | Empty stub (`pass`) | — |
| LSTM autoencoder | PyTorch module; the attention output is computed and then discarded | Reconstruction-error scoring on rolling windows; needs a train/validation split |
| Transformer encoder | PyTorch module with no input projection | — |
| Statistical tests (ADF, Ljung-Box, Jarque-Bera, Hurst) | Present in v1 as `features.calculate_statistical_features`, not wired in | Expose per-symbol diagnostics, e.g. `GET /stats/{symbol}` |

## Infrastructure

The v2 compose sketch had the API service, TimescaleDB (PostgreSQL 14) and
Redis. The plan was to store market data in a TimescaleDB hypertable and
alerts in PostgreSQL JSONB, with Redis as a cache and pub/sub. None of it was
built.

## Evaluation (the most important missing piece)

- Score with models fitted on a trailing window (out-of-sample), not in-sample.
- Compare against simple baselines: a rolling z-score of returns and of volume.
- On real data, check flagged dates against known market events. This is
  qualitative only: there is no labelled ground truth.
- Make the threshold respond to `sensitivity`: use `predict` / `contamination`
  instead of batch z-scores.
