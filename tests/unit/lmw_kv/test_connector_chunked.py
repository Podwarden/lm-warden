"""The connector's `nvme` tier and its chunked loads (spec 4.3, 4.6, 4.8, 5 and 6).

Never imports ``lmw_kv.connector`` (that needs vLLM).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lmw_kv import connector_logic as cl
from lmw_kv import scopes
from lmw_kv.store import Store
from tests.unit.lmw_kv.test_connector_logic import _WHO as WHO
from tests.unit.lmw_kv.test_connector_logic import (
    BS,
    CFG,
    GROUPS,
    MODEL_ID,
    MODEL_NS,
    NAMES,
    SECRET,
    SIZES,
    STORE_IDS,
    AnyGroup,
    FakeClient,
    FreshDaemon,
    S,
    T,
    _store,
    _worker_namespaces,
)
from tests.unit.lmw_kv.test_store import _cfg as _store_cfg
from tests.unit.lmw_kv.test_store import opened  # noqa: F401  (fixture)

ATTN = 1  # GROUPS[1] is the attention group


class SlotFake:
    """Hands out at most ``free`` slots per ``begin_load`` and records every call.

    ``empty_first`` calls get no slot at all (another rank holds the pool);
    ``always_empty`` never gives one; ``bad_call`` (1-based) answers that call with an
    offset outside the mapping; ``nvme`` slots of each reply count as promoted.
    """

    def __init__(
        self,
        free: int = 6,
        *,
        empty_first: int = 0,
        always_empty: bool = False,
        bad_call: int | None = None,
        nvme: int = 0,
    ) -> None:
        self.free = free
        self.empty_first = empty_first
        self.always_empty = always_empty
        self.lookup_answer: int | None = 1
        self.lookups: list[tuple[bytes, list[bytes]]] = []
        self.bad_call = bad_call
        self.nvme = nvme
        self.calls: list[tuple[Any, ...]] = []
        self._next = 0

    def begin_load(
        self, ns_id: bytes, keys: list[bytes], *, tenant: bytes = b"", tag: str = ""
    ) -> tuple[int, list[dict[str, int]], int]:
        self._next += 1
        h = 100 + self._next
        self.calls.append(("begin_load", h, list(keys), tenant, tag))
        if self.always_empty or self._next <= self.empty_first:
            return h, [], 0
        n = min(self.free, len(keys))
        off = 1 << 40 if self._next == self.bad_call else 0
        return h, [AnyGroup({"*": off}) for _ in range(n)], min(self.nvme, n)

    def lookup(self, ns_id: bytes, keys: list[bytes]) -> int | None:
        self.lookups.append((ns_id, list(keys)))
        return self.lookup_answer

    def end_load(self, handle: int, *, used: bool | None = None) -> bool:
        self.calls.append(("end_load", handle, used))
        return True

    def begins(self) -> list[tuple[Any, ...]]:
        return [c for c in self.calls if c[0] == "begin_load"]

    def ends(self) -> list[tuple[Any, ...]]:
        return [c for c in self.calls if c[0] == "end_load"]


def _job(n: int = 20) -> cl.LoadJob:
    keys = [bytes([i]) * 32 for i in range(n)]
    return cl.LoadJob(
        req_id="r",
        tag="req-tag",
        tokens=n * BS,
        entries=[cl.LoadEntry(ATTN, keys, list(range(500, 500 + n)))],
        tenant=T,
    )


def _run(
    client: Any,
    job: cl.LoadJob,
    window: int,
    same_file: Any = None,
    limit: int = 1 << 20,
) -> tuple[list[int] | None, list[tuple[int, int, int]], list[float]]:
    copies: list[tuple[int, int, int]] = []
    sleeps: list[float] = []
    got = cl.run_load_job(
        client,
        job,
        STORE_IDS,
        NAMES,
        SIZES,
        limit,
        lambda g, b, o: copies.append((g, b, o)),
        lambda: None,
        same_file or (lambda: True),
        window=window,
        sleep=sleeps.append,
    )
    return got, copies, sleeps


# ---- chunked loads ------------------------------------------------------------------


def test_load_runs_in_windows_and_releases_each() -> None:
    job = _job(20)
    fc = SlotFake(free=6, nvme=2)
    got, copies, sleeps = _run(fc, job, window=6)
    keys = job.entries[0].keys
    assert [c[2] for c in fc.begins()] == [keys[0:6], keys[6:12], keys[12:18], keys[18:20]]
    # each window is ended, used, before the next one begins
    kinds = [(c[0], c[2] if c[0] == "end_load" else None) for c in fc.calls]
    assert kinds == [("begin_load", None), ("end_load", True)] * 4
    assert [c[1] for c in fc.ends()] == [c[1] for c in fc.begins()]
    assert len(copies) == 20
    assert sorted(b for _g, b, _o in copies) == list(range(500, 520))
    assert got == [8]  # 2 promoted per window, summed
    assert sleeps == []
    # the request's tag and tenant reach every window
    assert {(c[3], c[4]) for c in fc.begins()} == {(T, "req-tag")}


def test_a_short_window_continues_from_where_it_ended() -> None:
    job = _job(20)
    fc = SlotFake(free=4)
    got, copies, _ = _run(fc, job, window=6)
    keys = job.entries[0].keys
    begins = [c[2] for c in fc.begins()]
    assert begins[0] == keys[0:6]
    assert begins[1] == keys[4:10]  # 4 of 6 came back: progress, not a failure
    assert got == [0] and len(copies) == 20
    assert all(c[2] is True for c in fc.ends())


def test_windows_wait_for_slots_held_elsewhere() -> None:
    """Review Focus 1: an empty reply is retried, the job still succeeds."""
    job = _job(20)
    fc = SlotFake(free=6, empty_first=5)
    got, copies, sleeps = _run(fc, job, window=6)
    assert got == [0] and len(copies) == 20
    assert sleeps == [cl.LOAD_RETRY_S] * 5
    ends = fc.ends()
    assert len(ends) == len(fc.begins())  # the empty handles are ended too
    assert [c[2] for c in ends[:5]] == [None] * 5  # nothing copied: not used
    assert [c[2] for c in ends[5:]] == [True] * 4


def test_an_empty_window_after_the_retries_fails_the_job() -> None:
    job = _job(20)
    fc = SlotFake(always_empty=True)
    got, copies, sleeps = _run(fc, job, window=6)
    assert got is None and copies == []
    assert sleeps == [cl.LOAD_RETRY_S] * cl.LOAD_EMPTY_RETRIES
    assert len(fc.begins()) == cl.LOAD_EMPTY_RETRIES + 1
    assert len(fc.lookups) == 1  # once per window, not per retry
    assert sorted(c[1] for c in fc.ends()) == sorted(c[1] for c in fc.begins())
    assert all(c[2] is not True for c in fc.ends())


def test_a_vanished_block_falls_back_at_once() -> None:
    job = _job(20)
    keys = job.entries[0].keys
    for answer in (0, None):
        fc = SlotFake(always_empty=True)
        fc.lookup_answer = answer
        got, copies, sleeps = _run(fc, job, window=6)
        assert got is None and copies == [] and sleeps == []
        assert fc.lookups == [(STORE_IDS[ATTN], keys[0:1])]
        assert len(fc.begins()) == 1
        assert sorted(c[1] for c in fc.ends()) == sorted(c[1] for c in fc.begins())
        assert all(c[2] is not True for c in fc.ends())

    class Raises(SlotFake):
        def lookup(self, ns_id: bytes, keys: list[bytes]) -> int | None:
            raise OSError("daemon gone")

    fc = Raises(always_empty=True)
    got, _copies, sleeps = _run(fc, job, window=6)
    assert got is None and sleeps == [] and len(fc.begins()) == 1
    assert all(c[2] is not True for c in fc.ends())


def test_contention_waits_within_the_budget() -> None:
    job = _job(20)
    fc = SlotFake(free=6, empty_first=3)
    got, copies, sleeps = _run(fc, job, window=6)
    assert got == [0] and len(copies) == 20
    assert sleeps == [cl.LOAD_RETRY_S] * 3
    assert len(fc.lookups) == 1


def test_retry_budget_is_small() -> None:
    assert cl.LOAD_RETRY_BUDGET_S <= 0.2
    assert cl.LOAD_RETRY_BUDGET_S == cl.LOAD_EMPTY_RETRIES * cl.LOAD_RETRY_S


def test_a_failed_window_fails_the_job_after_releasing() -> None:
    job = _job(20)
    fc = SlotFake(free=6, bad_call=2)
    got, copies, _ = _run(fc, job, window=6)
    assert got is None
    assert len(copies) == 6  # the second window is checked before anything is copied
    b1, b2 = fc.begins()
    assert [(c[1], c[2]) for c in fc.ends()] == [(b1[1], True), (b2[1], None)]


def test_same_file_checked_per_window() -> None:
    job = _job(20)
    fc = SlotFake(free=6)
    answers = iter([True, False])
    got, _copies, _ = _run(fc, job, window=6, same_file=lambda: next(answers, False))
    assert got is None
    assert [c[2] for c in fc.ends()] == [True, None]


def test_a_daemon_failure_mid_job_fails_it_and_releases() -> None:
    class DiesSecond(SlotFake):
        def begin_load(self, *a: Any, **kw: Any) -> Any:
            got = super().begin_load(*a, **kw)
            return None if self._next == 2 else got

    job = _job(20)
    fc = DiesSecond(free=6)
    got, copies, sleeps = _run(fc, job, window=6)
    assert got is None and len(copies) == 6 and sleeps == []
    assert [c[2] for c in fc.ends()] == [True]  # the first window; the second had no handle


# ---- pipelined loads (Task 13): the next window's begin_load overlaps this copy ----


def _run_pipelined(
    client: Any,
    job: cl.LoadJob,
    window: int,
    same_file: Any = None,
    prefetch: Any = None,
) -> tuple[list[int] | None, list[tuple[int, int, int]], list[float]]:
    from concurrent.futures import ThreadPoolExecutor

    copies: list[tuple[int, int, int]] = []
    sleeps: list[float] = []
    with ThreadPoolExecutor(1) as ex:
        got = cl.run_load_job(
            client,
            job,
            STORE_IDS,
            NAMES,
            SIZES,
            1 << 20,
            lambda g, b, o: copies.append((g, b, o)),
            lambda: None,
            same_file or (lambda: True),
            window=window,
            sleep=sleeps.append,
            prefetch=prefetch or ex.submit,
        )
    return got, copies, sleeps


def _pinned_peak(fc: SlotFake) -> int:
    """Most blocks pinned at once, replaying the calls in the order they happened."""
    held: dict[int, int] = {}
    peak = 0
    sizes = {c[1]: min(fc.free, len(c[2])) for c in fc.begins()}
    for c in fc.calls:
        if c[0] == "begin_load":
            held[c[1]] = sizes[c[1]]
        else:
            held.pop(c[1], None)
        peak = max(peak, sum(held.values()))
    return peak


def _each_handle_ended_once(fc: SlotFake) -> bool:
    return sorted(c[1] for c in fc.ends()) == sorted(c[1] for c in fc.begins())


def test_a_pipelined_load_uses_half_windows_in_order() -> None:
    job = _job(20)
    fc = SlotFake(free=6, nvme=1)
    got, copies, sleeps = _run_pipelined(fc, job, window=6)
    keys = job.entries[0].keys
    assert [c[2] for c in fc.begins()] == [keys[i : i + 3] for i in range(0, 20, 3)]
    assert [b for _g, b, _o in copies] == list(range(500, 520))  # in order
    assert got == [7] and sleeps == []
    assert all(c[2] is True for c in fc.ends()) and _each_handle_ended_once(fc)
    assert _pinned_peak(fc) <= 6  # two half windows: the window budget holds


def test_a_pipelined_load_covers_every_entry() -> None:
    keys = [bytes([i]) * 32 for i in range(10)]
    job = cl.LoadJob(
        req_id="r",
        tag="t",
        tokens=10 * BS,
        entries=[
            cl.LoadEntry(ATTN, keys, list(range(500, 510))),
            cl.LoadEntry(0, keys[4:5], [900], state=True),
        ],
        tenant=T,
    )
    fc = SlotFake(free=6, nvme=6)
    got, copies, _ = _run_pipelined(fc, job, window=8)
    assert got == [10, 1]
    assert [(g, b) for g, b, _o in copies] == [(ATTN, b) for b in range(500, 510)] + [(0, 900)]
    assert _each_handle_ended_once(fc)


def test_a_short_pipelined_window_releases_the_prefetch_and_goes_on_serially() -> None:
    job = _job(20)
    fc = SlotFake(free=2)  # every half window of 3 comes back with 2
    got, copies, _ = _run_pipelined(fc, job, window=6)
    keys = job.entries[0].keys
    begins = [c[2] for c in fc.begins()]
    assert begins[0] == keys[0:3] and begins[1] == keys[3:6]  # the prefetch
    assert begins[2] == keys[2:8]  # serial from where the short reply ended
    assert [b for _g, b, _o in copies] == list(range(500, 520))
    assert got == [0] and _each_handle_ended_once(fc)
    prefetched = fc.begins()[1][1]
    assert [c[2] for c in fc.ends() if c[1] == prefetched] == [None]  # never copied


def test_a_bad_pipelined_window_fails_the_job_and_releases_both() -> None:
    job = _job(20)
    fc = SlotFake(free=6, bad_call=2)
    got, copies, _ = _run_pipelined(fc, job, window=6)
    assert got is None and len(copies) == 3  # the bad window is checked before copying
    assert _each_handle_ended_once(fc)
    assert [c[2] for c in fc.ends()][0] is True
    assert all(c[2] is not True for c in fc.ends()[1:])


def test_a_raising_prefetch_fails_the_job_and_releases() -> None:
    from concurrent.futures import Future

    calls: list[int] = []

    def prefetch(fn: Any) -> Future[Any]:
        f: Future[Any] = Future()
        calls.append(1)
        if len(calls) == 2:
            f.set_exception(OSError("daemon gone"))
        else:
            f.set_result(fn())
        return f

    job = _job(20)
    fc = SlotFake(free=6)
    got, copies, _ = _run_pipelined(fc, job, window=6, prefetch=prefetch)
    assert got is None and len(copies) == 3
    assert _each_handle_ended_once(fc)


def test_a_replaced_ram_file_stops_the_pipeline_and_releases() -> None:
    job = _job(20)
    fc = SlotFake(free=6)
    answers = iter([True, False])
    got, _copies, _ = _run_pipelined(fc, job, window=6, same_file=lambda: next(answers, False))
    assert got is None
    assert _each_handle_ended_once(fc)
    assert [c[2] for c in fc.ends()].count(True) == 1


class ThreadConnFake(SlotFake):
    """A per-thread connection, like ``Client``: records the thread of each call."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.began_on: dict[int, int] = {}
        self.ended_on: dict[int, list[int]] = {}

    def begin_load(self, *a: Any, **kw: Any) -> Any:
        import threading

        got = super().begin_load(*a, **kw)
        self.began_on[got[0]] = threading.get_ident()
        return got

    def end_load(self, handle: int, *, used: bool | None = None) -> bool:
        import threading

        self.ended_on.setdefault(handle, []).append(threading.get_ident())
        return super().end_load(handle, used=used)

    def same_connection(self) -> bool:
        return set(self.ended_on) == set(self.began_on) and all(
            ends == [self.began_on[h]] for h, ends in self.ended_on.items()
        )


@pytest.mark.parametrize(
    "fake, same_file",
    [
        (lambda: ThreadConnFake(free=6), None),  # every window copied
        (lambda: ThreadConnFake(free=2), None),  # short reply: prefetch released unused
        (lambda: ThreadConnFake(free=6, bad_call=2), None),  # bad window: both released
        (lambda: ThreadConnFake(free=6), "second"),  # replaced RAM file
    ],
)
def test_each_pipelined_handle_ends_on_the_connection_that_began_it(
    fake: Any, same_file: Any
) -> None:
    job = _job(20)
    fc = fake()
    answers = iter([True, False])
    check = (lambda: next(answers, False)) if same_file else None
    _run_pipelined(fc, job, window=6, same_file=check)
    # Every handle, prefetched or serial, ends once and on the thread that began it.
    assert fc.same_connection()
    assert len(set(fc.began_on.values())) >= 1


def test_a_prefetch_that_cannot_be_submitted_releases_the_current_window() -> None:
    from concurrent.futures import Future

    calls: list[int] = []

    def prefetch(fn: Any) -> Future[Any]:
        calls.append(1)
        if len(calls) == 2:  # the next window's begin_load cannot be queued
            raise RuntimeError("cannot schedule new futures after shutdown")
        f: Future[Any] = Future()
        f.set_result(fn())
        return f

    job = _job(20)
    fc = SlotFake(free=6)
    got, copies, _ = _run_pipelined(fc, job, window=6, prefetch=prefetch)
    assert got is None and copies == []
    assert _each_handle_ended_once(fc)  # window 0 was ended, from here, unused
    assert [c[2] for c in fc.ends()] == [None]


def test_window_one_never_pipelines() -> None:
    job = _job(5)
    fc = SlotFake(free=6)
    got, copies, _ = _run_pipelined(fc, job, window=1)
    kinds = [(c[0], c[2] if c[0] == "end_load" else None) for c in fc.calls]
    assert kinds == [("begin_load", None), ("end_load", True)] * 5
    assert got == [0] and len(copies) == 5


def test_window_blocks() -> None:
    assert cl.window_blocks(1 << 30, 14_000_000, 4) == 9
    assert cl.window_blocks(1 << 30, 2 << 30, 4) == 1
    assert cl.window_blocks(16 << 30, 14_000_000, 1) == 613
    assert cl.window_blocks(1 << 30, 14_000_000, 0) == 38  # ranks floor at 1


def test_constants() -> None:
    assert (cl.LOAD_EMPTY_RETRIES, cl.LOAD_RETRY_S) == (8, 0.025)
    assert cl.TIERS == ("ram", "ram+nvme", "nvme")
    assert cl.STAGING_MIN_MINOR == 2


# ---- granularity: the engine's GPU hit can end inside a block (spec 4.6) -------------


def test_gpu_hit_inside_a_block_loads_that_whole_block() -> None:
    """Review Focus 4."""
    b = 1584
    fc = FakeClient()
    s = cl.SchedulerLogic(fc, SECRET, CFG, MODEL_ID, GROUPS, STORE_IDS, b)
    toks = list(range(60_000))
    keys = cl.block_keys_for(SECRET, b"", MODEL_ID, toks, b)
    assert len(keys) == 37
    _store(fc, ATTN, keys[:37])
    _store(fc, 0, [keys[36]])
    assert s.match("r", toks, {}, 50_000) == (37 * b - 50_000, True)
    blocks: list[list[int | None]] = [[None] * 36 + [900], list(range(1000, 1038))]
    s.after_alloc("r", blocks, 37 * b - 50_000)
    (job,) = s.build([], lambda rid: blocks).loads
    assert not job.doomed
    ents = {e.gid: (e.keys, e.block_ids) for e in job.entries}
    assert ents[ATTN] == (keys[31:37], list(range(1031, 1037)))
    assert ents[0] == ([keys[36]], [900])


# ---- saves: depth for the prefix chain only (spec 5) ------------------------------


def test_saves_send_depth_for_chain_groups_only() -> None:
    fc = FakeClient()
    s = cl.SchedulerLogic(fc, SECRET, CFG, MODEL_ID, GROUPS, STORE_IDS, BS)
    s.match("r", list(range(13)), {}, 0)
    plan = s.build([("r", 0, 12)], lambda rid: [[None, None, 10], [20, 21, 22]])
    attn = [e for e in plan.saves if e.gid == ATTN]
    state = [e for e in plan.saves if e.gid == 0]
    assert [e.depth for e in attn] == [0, 1, 2]
    assert [e.depth for e in state] == [None]
    fc.offsets_override = AnyGroup({"*": 0})
    n = cl.run_saves(
        fc,
        plan.saves,
        STORE_IDS,
        NAMES,
        SIZES,
        1 << 20,
        lambda g, b, o: None,
        lambda: None,
        lambda: True,
    )
    assert n == 4
    sent = fc.depths
    assert len(sent) == len(fc.ops("begin_save")) == 4
    assert sorted(v for (ns, _k), v in sent.items() if ns == STORE_IDS[ATTN]) == [0, 1, 2]
    assert [v for (ns, _k), v in sent.items() if ns == STORE_IDS[0]] == [None]


# ---- the `nvme` tier --------------------------------------------------------------


def test_parse_config_accepts_nvme() -> None:
    c = cl.parse_config({"socket": "/s", "tiers": "nvme"})
    assert c is not None and c.tiers == "nvme"
    assert cl.uses_nvme("nvme") and cl.uses_nvme("ram+nvme") and not cl.uses_nvme("ram")
    assert cl.uses_staging("nvme")
    assert not cl.uses_staging("ram+nvme") and not cl.uses_staging("ram")


@pytest.mark.parametrize(("tiers", "staging"), [("nvme", True), ("ram+nvme", False)])
def test_attach_registers_staging_for_nvme(tmp_path: Path, tiers: str, staging: bool) -> None:
    d = FreshDaemon(tmp_path / "lmw-kv.pool")
    cl.attach_store(
        d,
        _worker_namespaces(),
        lambda: None,
        lambda: True,
        nvme=cl.uses_nvme(tiers),
        staging=cl.uses_staging(tiers),
        **WHO,
    )
    assert d.nvme == [True] * (2 * len(GROUPS))
    assert d.staging == [staging] * (2 * len(GROUPS))


def test_attach_registers_no_staging_by_default(tmp_path: Path) -> None:
    d = FreshDaemon(tmp_path / "lmw-kv.pool")
    cl.attach_store(d, _worker_namespaces(), lambda: None, lambda: True, **WHO)
    assert d.staging == [False] * (2 * len(GROUPS))


def test_disk_only_turns_off_against_a_minor1_daemon() -> None:
    """Review Focus 3."""
    minor1 = {"protocol": 1, "minor": 1}
    assert cl.store_supports("nvme", minor1) is False
    assert cl.store_supports("ram+nvme", minor1) is True
    assert cl.store_supports("ram", minor1) is True
    assert cl.store_supports("nvme", {"protocol": 1}) is False  # minor 0
    assert cl.store_supports("nvme", {"protocol": 1, "minor": 2}) is True


# ---- isolation holds across windows (Plan 2) ------------------------------------------


class _LoadingStore:
    """The in-process store behind the client calls a worker makes (no daemon)."""

    def __init__(self, store: Store) -> None:
        self.s = store

    def register_ns(self, ns: Any, groups: dict[str, int], nvme: bool = True, **kw: Any) -> bytes:
        return self.s.register_namespace(ns, groups, nvme=nvme, **kw)

    def begin_save(
        self, ns_id: bytes, key: bytes, parent: bytes, scope: bytes = b"", **kw: Any
    ) -> Any:
        return self.s.begin_save(ns_id, key, parent, scope, **kw)

    def commit_save(self, ns_id: bytes, key: bytes) -> bool:
        self.s.commit_save(ns_id, key)
        return True

    def abort_save(self, ns_id: bytes, key: bytes) -> bool:
        self.s.abort_save(ns_id, key)
        return True

    def lookup(self, ns_id: bytes, keys: list[bytes]) -> int:
        return self.s.lookup(ns_id, keys)

    def begin_load(self, ns_id: bytes, keys: list[bytes], **kw: Any) -> Any:
        h = self.s.begin_load(ns_id, keys, **kw)
        return h.id, h.slots, h.nvme_blocks

    def end_load(self, handle: int, **kw: Any) -> bool:
        self.s.end_load(handle, **kw)
        return True


def test_an_isolated_keys_blocks_stay_private_across_windows(
    tmp_path: Path,
    opened: list[Any],  # noqa: F811
) -> None:
    store = Store(_store_cfg(tmp_path, slots=32, nvme=False))
    opened.append(store)
    c = _LoadingStore(store)
    nss = [(g.gid, cl.store_namespace(MODEL_NS, g, 0), {NAMES[g.gid]: 1000}) for g in GROUPS]
    ids = cl.attach_store(c, nss, lambda: None, lambda: True, nvme=False, **WHO)
    assert ids == STORE_IDS
    s = cl.SchedulerLogic(c, SECRET, CFG, MODEL_ID, GROUPS, ids, BS)
    toks = list(range(41))  # 10 whole blocks, all loadable
    blocks: list[list[int | None]] = [[None] * 9 + [90], list(range(10, 21))]
    mine = {"lmw_scope": S.hex(), "lmw_tenant": T.hex()}
    s.match("w", toks, mine, 0)
    plan = s.build([("w", 0, 40)], lambda rid: blocks)
    limit = store.cfg.ram_bytes
    noop = (lambda g, b, o: None, lambda: None, lambda: True)
    assert cl.run_saves(c, plan.saves, ids, NAMES, SIZES, limit, *noop) == 11
    # the same key resumes in windows of 3
    assert s.match("r", toks, mine, 0) == (40, True)
    s.after_alloc("r", blocks, 40)
    (job,) = s.build([], lambda rid: blocks).loads
    copies: list[int] = []
    got = cl.run_load_job(
        c,
        job,
        ids,
        NAMES,
        SIZES,
        limit,
        lambda g, b, o: copies.append(b),
        lambda: None,
        lambda: True,
        window=3,
        sleep=lambda _s: None,
    )
    assert got == [0, 0] and len(copies) == 11
    # another scope finds nothing, and a retired scope stays hidden
    assert s.match("x", toks, {"lmw_scope": "bb"}, 0) == (0, False)
    store.retire_scopes([scopes.scope_tag(S)])
    copies.clear()
    got = cl.run_load_job(
        c,
        job,
        ids,
        NAMES,
        SIZES,
        limit,
        lambda g, b, o: copies.append(b),
        lambda: None,
        lambda: True,
        window=3,
        sleep=lambda _s: None,
    )
    assert got is None and copies == []
