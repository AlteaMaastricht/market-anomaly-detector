#!/usr/bin/env python3
"""Offline demo: plant anomalies in a seeded GBM series and run the detector.

No network, no FastAPI. Needs numpy, pandas, scikit-learn, pydantic.

    python examples/demo.py                 # defaults below (seed 42 / 7)
    python examples/demo.py --seed 1 --inject-seed 2

What it reports is recall/precision against *synthetic* planted anomalies.
That is a sanity check of the pipeline, not evidence of real-market
performance (there are no labelled real market anomalies in this repo).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from market_anomaly.config import Config  # noqa: E402
from market_anomaly.detectors import MultiModalAnomalyDetector  # noqa: E402
from market_anomaly.features import calculate_technical_indicators, feature_matrix  # noqa: E402
from market_anomaly.models import alert_to_message  # noqa: E402
from market_anomaly.synthetic import detection_summary, generate_ohlcv, inject_anomalies  # noqa: E402


def run(seed: int, inject_seed: int, n_days: int, sensitivity: float) -> dict:
    clean = generate_ohlcv(n_days=n_days, seed=seed)
    data, labels = inject_anomalies(clean, seed=inject_seed)

    out = {"params": {"seed": seed, "inject_seed": inject_seed, "n_days": n_days,
                      "sensitivity": sensitivity, "threshold": Config.ANOMALY_THRESHOLD,
                      "shock_size_log": 0.08, "volume_multiple": 5.0,
                      "n_price_shocks": 5, "n_volume_spikes": 5}}

    # --- injected series
    feats = calculate_technical_indicators(data)
    det = MultiModalAnomalyDetector()
    det.fit(feats, "SYNTH", sensitivity)
    scores = det.score(feats, "SYNTH")
    alerts = det.detect(feats, "SYNTH")
    flagged = [pd.Timestamp(a.timestamp) for a in alerts]
    out["n_rows"] = len(data)
    out["n_scored_rows"] = int(len(scores))
    out["exact_date"] = detection_summary(labels, flagged, data.index, tolerance=0)
    out["within_1_day"] = detection_summary(labels, flagged, data.index, tolerance=1)
    out["labels"] = [{"timestamp": str(l.timestamp.date()), "kind": l.kind,
                      "magnitude": round(l.magnitude, 4),
                      "ensemble_score": round(float(scores.get(l.timestamp, float("nan"))), 3)}
                     for l in labels]
    out["alerts"] = [json.loads(alert_to_message(a))["data"] for a in alerts]

    # --- same series without injections (false-positive check)
    clean_feats = calculate_technical_indicators(clean)
    det_clean = MultiModalAnomalyDetector()
    det_clean.fit(clean_feats, "CLEAN", sensitivity)
    clean_alerts = det_clean.detect(clean_feats, "CLEAN")
    n_clean = len(feature_matrix(clean_feats))
    out["clean_series"] = {"n_scored_rows": n_clean, "n_flagged": len(clean_alerts),
                           "flagged_fraction": len(clean_alerts) / n_clean}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42, help="GBM seed")
    ap.add_argument("--inject-seed", type=int, default=7, help="anomaly placement seed")
    ap.add_argument("--n-days", type=int, default=504, help="business days (~2 years)")
    ap.add_argument("--sensitivity", type=float, default=0.05)
    ap.add_argument("--json", type=str, default=None, help="also write the summary to this JSON file")
    args = ap.parse_args()

    res = run(args.seed, args.inject_seed, args.n_days, args.sensitivity)
    p = res["params"]
    print(f"Synthetic GBM: {res['n_rows']} business days, seed={p['seed']}, "
          f"inject_seed={p['inject_seed']}, sensitivity={p['sensitivity']}, threshold={p['threshold']}")
    print(f"Rows with a complete feature vector (scored): {res['n_scored_rows']}")
    print("\nPlanted anomalies (ensemble score; flagged if < -threshold):")
    for l in res["labels"]:
        print(f"  {l['timestamp']}  {l['kind']:<13} magnitude={l['magnitude']:>7}  score={l['ensemble_score']}")
    print("\nAlerts raised:")
    for a in res["alerts"]:
        print(f"  {a['timestamp'][:10]}  {a['anomaly_type']:<17} {a['severity']:<8} "
              f"score={a['anomaly_score']:.2f}  {a['explanation']}")
    for key in ("exact_date", "within_1_day"):
        s = res[key]
        print(f"\nDetection summary ({key}, tolerance={s['tolerance_rows']} rows):")
        for kind in ("price_shock", "volume_spike"):
            print(f"  {kind:<13} recall {s[kind + '_detected']}/{s[kind + '_total']}")
        prec = "n/a" if s["precision"] is None else f"{s['precision']:.2f}"
        print(f"  precision     {s['n_flagged_matching_a_label']}/{s['n_flagged']} = {prec}")
    c = res["clean_series"]
    print(f"\nSame series without injections: {c['n_flagged']}/{c['n_scored_rows']} rows flagged "
          f"({c['flagged_fraction']:.1%})")
    print("\nNote: synthetic-injection recall is NOT real-world performance.")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
