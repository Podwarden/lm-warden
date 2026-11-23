"""The default policy ``lru`` evicts exactly as step 1 did.

The golden file ``fixtures/lru_victims.json`` was recorded on the step 1 store (the
commit before Plan 2 touched ``store.py``) with ``LMW_KV_WRITE_GOLDEN=1``. It lists every
RAM removal (``Store._remove``) and NVMe drop (``Store._drop_nvme``) of one seeded
workload, in order. Any change to the victim order under ``lru`` fails this test.

One segment reclaim (``Store._reclaim_oldest_segment``) drops a set of blocks, and step 1
walks that set in hash order, which changes with ``PYTHONHASHSEED``. So the blocks of
one reclaim are recorded sorted (NVMe drops, then RAM removals): the order of reclaims
and of single RAM evictions is exact, and so is the set each reclaim takes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import pytest

from lmw_kv import store as store_mod
from lmw_kv.format import record_size
from lmw_kv.store import LoadHandle, Store, StoreConfig
from tests.unit.lmw_kv.test_store import GROUPS, ROOT, SCOPE, SLOT, _ns

GOLDEN = Path(__file__).parent / "fixtures" / "lru_victims.json"
STATE_GROUPS = {"state": 300}
BLOCK_ON_DISK = sum(record_size(g, n) for g, n in GROUPS.items())
SEG = 4 * BLOCK_ON_DISK  # four attention blocks per segment file
OPS = 2000
MAX_CHAIN = 8
MAX_HELD = 3


class _FakeTime:
    """Segment ages from a counter, so equal wall-clock readings never tie."""

    def __init__(self) -> None:
        self._now = 1_000_000.0

    def time(self) -> float:
        self._now += 1.0
        return self._now

    @staticmethod
    def monotonic() -> float:
        return time.monotonic()


def record_victims(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record every RAM removal and NVMe drop, in order, into the returned list.

    The blocks of one segment reclaim are recorded sorted (NVMe drops, then RAM removals;
    module docstring). Shared with tests/unit/tools/test_kv_replay.py.
    """
    victims: list[list[str]] = []
    reclaim: list[list[list[str]]] = []  # the open reclaim's records, if any
    real_remove = Store._remove
    real_drop = Store._drop_nvme
    real_reclaim = Store._reclaim_oldest_segment

    def record(v: list[str]) -> None:
        (reclaim[-1] if reclaim else victims).append(v)

    def remove(self: Store, ik: tuple[bytes, bytes], *args: Any, **kwargs: Any) -> Any:
        record(["ram", ik[0].hex(), ik[1].hex()])
        return real_remove(self, ik, *args, **kwargs)

    def drop(self: Store, ns_id: bytes, key: bytes, *args: Any, **kwargs: Any) -> Any:
        record(["nvme", ns_id.hex(), key.hex()])
        return real_drop(self, ns_id, key, *args, **kwargs)

    def reclaim_one(self: Store, *args: Any, **kwargs: Any) -> Any:
        reclaim.append([])
        try:
            return real_reclaim(self, *args, **kwargs)
        finally:
            got = reclaim.pop()
            victims.extend(sorted(got, key=lambda v: (v[0] != "nvme", v)))

    monkeypatch.setattr(Store, "_remove", remove)
    monkeypatch.setattr(Store, "_drop_nvme", drop)
    monkeypatch.setattr(Store, "_reclaim_oldest_segment", reclaim_one)
    return victims


def golden_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Store, list[bytes], list[list[str]]]:
    """The golden workload's store and namespaces, with its victims being recorded."""
    fake = _FakeTime()
    monkeypatch.setattr(store_mod, "SEGMENT_BYTES", SEG)
    monkeypatch.setattr(store_mod, "time", fake)
    victims = record_victims(monkeypatch)
    cfg = StoreConfig(
        ram_path=tmp_path / "ram.pool",
        ram_bytes=12 * SLOT,
        nvme_dir=tmp_path / "nvme",
        nvme_cap_bytes=10 * SEG,
    )
    if "clock" in {f.name for f in dataclasses.fields(StoreConfig)}:
        cfg.clock = fake.time
    s = Store(cfg)
    nss = [
        s.register_namespace(_ns("r1"), GROUPS),
        s.register_namespace(_ns("r2"), GROUPS, nvme=False),
        s.register_namespace(_ns("r3", groups=("state",)), STATE_GROUPS),
    ]
    return s, nss, victims


def _run_workload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    s, nss, victims = golden_store(tmp_path, monkeypatch)
    try:
        rng = random.Random(7)
        chains: list[list[list[tuple[bytes, bytes]]]] = [[] for _ in nss]
        held: list[LoadHandle] = []
        counter = 0
        for _ in range(OPS):
            i = rng.randrange(len(nss))
            ns_id = nss[i]
            op = rng.random()
            if op < 0.55 or not chains[i]:
                if not chains[i] or rng.random() < 0.3:
                    chain: list[tuple[bytes, bytes]] = []
                    chains[i].append(chain)
                else:
                    chain = rng.choice(chains[i])
                    if len(chain) >= MAX_CHAIN:
                        chain = []
                        chains[i].append(chain)
                parent = chain[-1][0] if chain else ROOT
                counter += 1
                key = hashlib.sha256(f"{i}:{counter}".encode()).digest()
                if s.begin_save(ns_id, key, parent, SCOPE) is None:
                    continue
                if rng.random() < 0.85:
                    s.commit_save(ns_id, key)
                    chain.append((key, parent))
                    s.flush()
                else:
                    s.abort_save(ns_id, key)
            elif op < 0.85:
                chain = rng.choice(chains[i])
                h = s.begin_load(ns_id, [k for k, _ in chain])
                if rng.random() < 0.3:
                    held.append(h)
                    if len(held) > MAX_HELD:
                        s.end_load(held.pop(0).id)
                else:
                    s.end_load(h.id)
            elif op < 0.95:
                if held:
                    s.end_load(held.pop(rng.randrange(len(held))).id)
            else:
                chain = rng.choice(chains[i])
                s.lookup(ns_id, [k for k, _ in chain])
        for h in held:
            s.end_load(h.id)
        s.flush()
    finally:
        s.close()
    return victims


def test_lru_victims_match_step1_golden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    victims = _run_workload(tmp_path, monkeypatch)
    assert {v[0] for v in victims} == {"ram", "nvme"}  # the workload exercises both tiers
    if os.environ.get("LMW_KV_WRITE_GOLDEN") == "1":
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text("[\n" + ",\n".join(json.dumps(v) for v in victims) + "\n]\n")
    golden = json.loads(GOLDEN.read_text())
    assert len(victims) == len(golden)
    for n, (got, want) in enumerate(zip(victims, golden, strict=True)):
        assert got == want, f"victim {n} differs"
