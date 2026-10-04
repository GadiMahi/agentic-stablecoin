#!/usr/bin/env python3
"""Launcher: start the SthiraSense API (and the built dashboard, if present)."""
from __future__ import annotations

import argparse
import logging

import uvicorn

from sthirasense.config import ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the SthiraSense service")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s")

    built = (ROOT / "frontend" / "dist" / "index.html").exists()
    print(f"\n  SthiraSense  ->  http://{args.host}:{args.port}")
    if not built:
        print("  dashboard not built yet:  cd frontend && npm install && npm run build")
    print()
    uvicorn.run("sthirasense.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
