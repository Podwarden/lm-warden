#!/usr/bin/env python3
"""Refresh app/lens/price_snapshot.json, the price table that ships with a release (#347).

The snapshot is what a fresh or air-gapped install prices with until (or unless) it can
refresh from the network. Run this before cutting a release and commit the result:

    .venv/bin/python scripts/refresh_price_snapshot.py

It reads OpenRouter's public model list (else LiteLLM's price file) through the same code
the app uses (app/lens/price_sources.py), so only the two allow-listed hosts are contacted.
Prices are per million tokens in micro-units of USD (100000 = 0.10); the file records its
source and the time it was read.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.lens import price_sources as src  # noqa: E402
from app.lens.price_store import SNAPSHOT_PATH, snapshot_json  # noqa: E402

NOTE = (
    "Public list prices per million tokens in micro-units of USD (100000 = 0.10), read from "
    "`source` at `fetched_at`. Shipped as the offline fallback; refreshed by "
    "scripts/refresh_price_snapshot.py. Columns: host, model, input, output, cache read, cache write."
)


async def main() -> int:
    rows, source, note = await src.fetch_rows()
    if note:
        print(f"primary skipped: {note}", file=sys.stderr)
    SNAPSHOT_PATH.write_text(snapshot_json(rows, source, src.now_iso(), NOTE), encoding="utf-8")
    print(f"{len(rows)} prices from {source} -> {SNAPSHOT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
