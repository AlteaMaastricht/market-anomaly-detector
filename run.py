#!/usr/bin/env python3
"""Start the API server.

Usage:
    python run.py            # DATA_SOURCE from the environment (default "auto")
    python run.py --demo     # offline: DATA_SOURCE=synthetic, no network calls
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", action="store_true", help="use deterministic synthetic data only")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:  # optional: read .env (ALPHA_VANTAGE_API_KEY, DATA_SOURCE, ...)
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if args.demo:
        os.environ["DATA_SOURCE"] = "synthetic"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    try:
        import uvicorn
        from market_anomaly.api import app
    except ImportError as exc:
        sys.exit(f"Missing dependency ({exc}). Install with: pip install -r requirements-api.txt")
    # Single worker: SQLite + in-process model cache.
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", workers=1)


if __name__ == "__main__":
    main()
