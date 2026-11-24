"""N1: no decision reads a reuse counter (counters design 7.2, 7.4; gate G13).

Each test runs a workload with tags and ``used: true`` while a hook sets every
``PosCounters`` field of every position to random values before each store call. The
victims, the segments reclaimed and the refusals must be exactly those of a run without
scrambling (``lru``: the step 1 golden file itself).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmw_kv import store as store_mod
from lmw_kv.fair import Policy, TenantPolicy
from lmw_kv.format import HEADER_LEN
from lmw_kv.reuse import PosCounters
from lmw_kv.scopes import model_tag
from lmw_kv.store import Store, StoreConfig
from tests.unit.lmw_kv.test_store import BLOCK_ON_DISK, GROUPS, ROOT, SCOPE, SLOT, _ns
from tests.unit.lmw_kv.test_store_lru_exact import GOLDEN, _run_workload

STATE_GROUPS = {"state": 300}


def scramble(s: Store, rng: random.Random) -> None:
    """Every counter field of every position set to a random value."""
    with s._cond:
        for c in s._reuse.rows.values():
            _scramble_row(c, rng)


def _scramble_row(c: PosCounters, rng: random.Random) -> None:
    c.first_saved = rng.uniform(0, 2e6)
    c.saver = rng.randbytes(16)
    c.depth = rng.randrange(-1, 100)
    c.loads, c.reloads, c.saver_reloads = (rng.randrange(1000) for _ in range(3))
    c.ram_reloads, c.nvme_reloads = rng.randrange(1000), rng.randrange(1000)
    c.last_reload = rng.uniform(0, 2e6)
    c.d, c.t = rng.uniform(0, 1e4), rng.uniform(0, 2e6)
    c.tenants_seen = tuple(rng.randbytes(16) for _ in range(rng.randrange(5)))
    c.many = rng.random() < 0.5
    c.nvme_written = rng.randrange(1 << 30)


_CALLS = ("begin_save", "commit_save", "abort_save", "begin_load", "end_load", "lookup")


def _with_counters(monkeypatch: pytest.MonkeyPatch, rng: random.Random | None) -> None:
    """Every load carries a tag and is used; ``rng`` set: scramble before each call."""
    tags = random.Random(11)  # its own stream: the workload's draws stay the same

    for name in _CALLS:
        real: Callable[..., Any] = getattr(Store, name)

        def call(self: Store, *a: Any, _real: Any = real, _name: str = name, **kw: Any) -> Any:
            if rng is not None:
                scramble(self, rng)
            if _name == "begin_load":
                kw.setdefault("tag", f"req-{tags.randrange(40)}")
            if _name == "end_load":
                kw.setdefault("used", True)
            return _real(self, *a, **kw)

        monkeypatch.setattr(Store, name, call)


def test_lru_golden_with_counters_scrambled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_counters(monkeypatch, random.Random(3))
    seen: list[int] = []
    real_reloaded = store_mod.ReuseBook.reloaded

    def count(self: Any, *a: Any, **kw: Any) -> None:
        seen.append(1)
        real_reloaded(self, *a, **kw)

    monkeypatch.setattr(store_mod.ReuseBook, "reloaded", count)
    victims = _run_workload(tmp_path, monkeypatch)
    assert seen  # the counters really ran
    golden = json.loads(GOLDEN.read_text())
    assert len(victims) == len(golden)
    for n, (got, want) in enumerate(zip(victims, golden, strict=True)):
        assert got == want, f"victim {n} differs"


# ---- fair ---------------------------------------------------------------------------

MA = model_tag("org/model-a", "rev-1")
MB = model_tag("org/model-b", "rev-1")
F, V, X, C = b"\x0f" * 16, b"\x0e" * 16, b"\x01" * 16, b"\x0c" * 16  # C is capped
TENANTS = (F, F, F, V, X, C)  # the flood saves most
ISO = b"\x7a" * 16  # V's isolation scope, retired mid-run
SEG = HEADER_LEN + 4 * BLOCK_ON_DISK
POLICY = Policy(
    mode="fair",
    owner_weights={"a": 2, "b": 1},
    tenants={C: TenantPolicy(max_ram_pct=25, max_nvme_pct=30)},
    idle_ram_s=3600,
    idle_nvme_s=7200,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def _fair_run(path: Path, monkeypatch: pytest.MonkeyPatch, *, nvme: bool) -> list[Any]:
    """Plan 2's fair shapes in one seeded workload; every decision, in order."""
    events: list[Any] = []
    real_remove, real_drop, real_delete = Store._remove, Store._drop_nvme, Store._delete_segment

    def remove(self: Store, ik: Any, *a: Any, **kw: Any) -> Any:
        events.append(("ram", ik[0].hex(), ik[1].hex()))
        return real_remove(self, ik, *a, **kw)

    def drop(self: Store, ns_id: bytes, key: bytes, *a: Any, **kw: Any) -> Any:
        events.append(("nvme", ns_id.hex(), key.hex()))
        return real_drop(self, ns_id, key, *a, **kw)

    def delete(self: Store, ns_id: bytes, seg: int, *a: Any, **kw: Any) -> Any:
        events.append(("segment", ns_id.hex(), seg))
        return real_delete(self, ns_id, seg, *a, **kw)

    with monkeypatch.context() as m:
        m.setattr(store_mod, "SEGMENT_BYTES", SEG)
        m.setattr(Store, "_remove", remove)
        m.setattr(Store, "_drop_nvme", drop)
        m.setattr(Store, "_delete_segment", delete)
        clock = _Clock()
        path.mkdir(parents=True)
        cfg = StoreConfig(
            ram_path=path / "ram.pool", ram_bytes=16 * SLOT,
            nvme_dir=path / "nvme" if nvme else None, nvme_cap_bytes=12 * SEG if nvme else 0,
            clock=clock,
        )  # fmt: skip
        s = Store(cfg)
        try:
            s.set_policy(POLICY)
            _fair_ops(s, clock, events)
        finally:
            s.close()
    return events


def _fair_ops(s: Store, clock: _Clock, events: list[Any]) -> None:
    attn = {
        MA: s.register_namespace(_ns("a1"), GROUPS, owner="a", model=MA),
        MB: s.register_namespace(_ns("b1"), GROUPS, owner="b", model=MB),
    }
    state = s.register_namespace(_ns("a2", groups=("state",)), STATE_GROUPS, owner="a", model=MA)
    rng = random.Random(5)
    chains: list[tuple[bytes, bytes, bytes, list[bytes]]] = []  # model, tenant, tag, keys
    held: list[Any] = []
    n = 0
    for step in range(900):
        op = rng.random()
        if step == 600:
            events.append(("retired", s.retire_scopes([ISO])))
        if op < 0.04:
            clock.now += rng.choice((60.0, 1800.0, 5000.0))
            continue
        if op < 0.55 or not chains:
            if not chains or rng.random() < 0.3:
                model = rng.choice((MA, MA, MB))
                tenant = rng.choice(TENANTS)
                tag = ISO if tenant == V and rng.random() < 0.5 else b""
                chains.append((model, tenant, tag, []))
            model, tenant, tag, keys = rng.choice(chains)
            if len(keys) >= 6:
                continue
            n += 1
            key = hashlib.sha256(f"k{n}".encode()).digest()
            parent = keys[-1] if keys else ROOT
            ns_id = attn[model]
            got = s.begin_save(ns_id, key, parent, SCOPE, tenant=tenant, scope_tag=tag)
            events.append(("save", key.hex(), got is not None))
            if got is None:
                continue
            s.commit_save(ns_id, key)
            s.flush()  # the writer is idle at every decision: one order, every run
            keys.append(key)
            if model == MA and rng.random() < 0.4:  # a second member of the position
                sk = hashlib.sha256(f"s{n}".encode()).digest()
                ok = s.begin_save(state, sk, ROOT, SCOPE, tenant=tenant, scope_tag=tag, pos=key)
                events.append(("state", sk.hex(), ok is not None))
                if ok is not None:
                    s.commit_save(state, sk)
                    s.flush()
        elif op < 0.85:
            model, tenant, _, keys = rng.choice(chains)
            loader = tenant if rng.random() < 0.7 else rng.choice(TENANTS)
            h = s.begin_load(attn[model], keys, tenant=loader)
            events.append(("load", len(h.slots), h.nvme_blocks))
            if rng.random() < 0.3:
                held.append(h)
                if len(held) > 3:
                    s.end_load(held.pop(0).id)
            else:
                s.end_load(h.id)
        elif held:
            s.end_load(held.pop(rng.randrange(len(held))).id)
        else:
            model, _, _, keys = rng.choice(chains)
            events.append(("lookup", s.lookup(attn[model], keys)))
    for h in held:
        s.end_load(h.id)
    s.flush()


@pytest.mark.parametrize("nvme", [False, True], ids=["ram", "nvme"])
def test_fair_decisions_with_counters_scrambled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nvme: bool,
) -> None:
    _with_counters(monkeypatch, None)
    plain = _fair_run(tmp_path / "plain", monkeypatch, nvme=nvme)
    monkeypatch.undo()
    _with_counters(monkeypatch, random.Random(9))
    scrambled = _fair_run(tmp_path / "scrambled", monkeypatch, nvme=nvme)
    kinds = {e[0] for e in plain}
    assert {"ram", "save", "load"} <= kinds
    assert any(e[0] == "save" and not e[2] for e in plain)  # some refusals
    if nvme:
        assert {"nvme", "segment"} <= kinds
    assert len(scrambled) == len(plain)
    for i, (got, want) in enumerate(zip(scrambled, plain, strict=True)):
        assert got == want, f"decision {i} differs"
