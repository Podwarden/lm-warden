"""``ReuseBook``: the reuse counters of positions, pure (counters design 3.2 to 3.5, 6.1)."""

from __future__ import annotations

from typing import Any

import pytest

from lmw_kv import reuse
from lmw_kv.reuse import (
    HIST_BUCKETS,
    HOLDER_TOP,
    REGRET_WINDOW_S,
    TAGS_MAX,
    TENANTS_SEEN_MAX,
    PosCounters,
    ReuseBook,
    decayed,
    label,
)

M = b"\x09" * 16
A, B, C = b"\x0a" * 16, b"\x0b" * 16, b"\x0c" * 16
KEY = b"\x42" * 32


def _pos(n: int) -> tuple[bytes, bytes]:
    return (M, n.to_bytes(32, "big"))


P = _pos(1)


def _live(*pos: tuple[bytes, bytes], tiers: tuple[bool, bool] = (True, False)) -> dict[Any, Any]:
    return dict.fromkeys(pos, tiers)


def _report(book: ReuseBook, live: dict[Any, Any], charged: bytes = A) -> dict[str, Any]:
    return book.report(live, lambda _p: False, lambda _p: charged)


def test_constants_are_the_spec_values() -> None:
    assert reuse.HALF_LIFE_S == 21600
    assert (reuse.TAGS_MAX, reuse.REGRET_RING, reuse.REGRET_WINDOW_S) == (10_000, 16_384, 86_400)
    assert (reuse.HOLDER_TOP, reuse.TENANTS_SEEN_MAX, reuse.TOP_MAX) == (64, 4, 50)
    assert HIST_BUCKETS == ("0", "1", "2-3", "4-7", "8+")


def test_decay_follows_the_pair_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reuse, "HALF_LIFE_S", 100.0)
    c = PosCounters(first_saved=0.0, saver=A, depth=0, d=2.0, t=0.0)
    assert decayed(c, 100.0, 100.0) == 1.0
    book = ReuseBook()
    book.saved(P, A, 0, 0.0)
    row = book.rows[P]
    row.d, row.t = 2.0, 0.0
    book.reloaded(P, "t1", A, "ram", 100.0)  # reads the patched half-life
    assert row.d == 2.0 and row.t == 100.0
    assert row.last_reload == 100.0


def test_a_save_starts_at_one_sighting() -> None:
    book = ReuseBook()
    book.saved(P, A, -1, 50.0)
    row = book.rows[P]
    assert (row.d, row.t, row.first_saved, row.saver, row.depth) == (1.0, 50.0, 50.0, A, -1)
    book.saved(P, B, 3, 60.0)  # a later member: never a new row; sets an unknown depth
    assert book.rows[P] is row and row.saver == A and row.depth == 3 and row.first_saved == 50.0
    book.saved(P, B, 7, 70.0)
    assert row.depth == 3


def test_reload_counted_once_per_tag() -> None:
    book = ReuseBook()
    book.saved(P, A, 0, 0.0)
    for _ in range(12):  # four ranks, three groups: one request
        book.loaded(P, "req-1")
        book.reloaded(P, "req-1", A, "ram", 1.0)
    row = book.rows[P]
    assert (row.loads, row.reloads, row.saver_reloads, row.ram_reloads) == (1, 1, 1, 1)
    book.loaded(P, "req-2")
    book.reloaded(P, "req-2", A, "nvme", 2.0)
    assert (row.loads, row.reloads, row.nvme_reloads) == (2, 2, 1)


def test_load_without_tag_never_counts_a_reload() -> None:
    book = ReuseBook()
    book.saved(P, A, 0, 0.0)
    book.loaded(P, "")
    book.loaded(P, "")
    book.reloaded(P, "", A, "ram", 1.0)
    row = book.rows[P]
    assert row.loads == 2 and row.reloads == 0 and row.d == 1.0 and row.last_reload == 0.0


def test_tags_forgotten_after_TAGS_MAX() -> None:
    book = ReuseBook()
    book.saved(P, A, 0, 0.0)
    other = _pos(2)
    book.saved(other, A, 0, 0.0)
    book.reloaded(P, "first", A, "ram", 1.0)
    book.reloaded(P, "first", A, "ram", 1.0)
    assert book.rows[P].reloads == 1
    for i in range(TAGS_MAX):
        book.reloaded(other, f"t{i}", A, "ram", 1.0)
    book.reloaded(P, "first", A, "ram", 2.0)  # "first" is no longer remembered
    assert book.rows[P].reloads == 2


def test_reload_of_an_unknown_position_is_ignored() -> None:
    book = ReuseBook()
    book.loaded(P, "t")
    book.reloaded(P, "t", A, "ram", 1.0)
    assert book.rows == {}


def _evict(book: ReuseBook, pos: Any, tier: Any = "ram", *, gone: bool = True, **kw: Any) -> None:
    book.left(pos, tier, A, evicted=True, gone=gone, now=kw.pop("now", 10.0), **kw)


def test_histogram_buckets() -> None:
    book = ReuseBook()
    for i, n in enumerate((0, 1, 3, 5, 9)):
        pos = _pos(100 + i)
        book.saved(pos, A, 0, 0.0)
        for j in range(n):
            book.reloaded(pos, f"r{i}-{j}", A, "ram", 1.0)
        _evict(book, pos, nbytes=10 * (i + 1))
    r = _report(book, {})["by_tenant"][A.hex()]
    for view in ("charged", "saver"):
        hist = r[view]["evicted_hist"]["ram"]
        assert {b: hist[b]["positions"] for b in HIST_BUCKETS} == dict.fromkeys(HIST_BUCKETS, 1)
        assert [hist[b]["bytes"] for b in HIST_BUCKETS] == [10, 20, 30, 40, 50]
        assert all(c["positions"] == 0 for c in r[view]["evicted_hist"]["nvme"].values())


def test_saver_histogram_uses_its_own_reloads_and_charged_uses_all() -> None:
    book = ReuseBook()
    book.saved(P, A, 0, 0.0)
    book.reloaded(P, "b1", B, "ram", 1.0)
    book.reloaded(P, "b2", B, "ram", 1.0)
    book.left(P, "ram", B, evicted=True, gone=True, now=5.0, nbytes=8)
    rep = _report(book, {})["by_tenant"]
    assert rep[B.hex()]["charged"]["evicted_hist"]["ram"]["2-3"] == {"positions": 1, "bytes": 8}
    assert rep[A.hex()]["saver"]["evicted_hist"]["ram"]["0"] == {"positions": 1, "bytes": 8}


def test_a_release_is_not_counted() -> None:
    book = ReuseBook()
    book.saved(P, A, 0, 0.0)
    book.left(P, "ram", A, evicted=False, gone=False, now=1.0, nbytes=8)
    hist = _report(book, _live(P, tiers=(False, True)))["by_tenant"][A.hex()]["charged"]
    assert all(c["positions"] == 0 for t in hist["evicted_hist"].values() for c in t.values())
    assert P in book.rows


def test_wasted_counts_only_never_reloaded_nvme_bytes() -> None:
    book = ReuseBook()
    used, unused, ram_only = _pos(1), _pos(2), _pos(3)
    for pos in (used, unused, ram_only):
        book.saved(pos, A, 0, 0.0)
    book.nvme_written(used, 100, A)
    book.nvme_written(unused, 40, A)
    book.nvme_written(unused, 60, A)
    book.reloaded(used, "t", A, "nvme", 1.0)
    book.nvme_read(used, 100, A)
    _evict(book, used, "nvme")
    _evict(book, unused, "nvme", gone=False)  # still in RAM: wasted all the same
    _evict(book, ram_only, "ram")
    sv = _report(book, _live(unused))["by_tenant"][A.hex()]["saver"]
    assert sv["wasted_nvme_bytes"] == 100
    assert (sv["nvme_bytes_written"], sv["nvme_bytes_read"]) == (200, 100)
    st = _report(book, {})["store"]
    assert (st["nvme_bytes_written"], st["nvme_bytes_read"]) == (200, 100)
    _evict(book, unused, "ram")  # leaving RAM later never counts it again
    assert _report(book, {})["by_tenant"][A.hex()]["saver"]["wasted_nvme_bytes"] == 100


def test_wasted_ignores_reloads_by_other_keys() -> None:
    # Ruling R1: in the saver view "reused" means reused by the saver.
    book = ReuseBook()
    book.saved(P, A, 0, 0.0)
    book.nvme_written(P, 70, A)
    book.reloaded(P, "t", B, "nvme", 1.0)
    _evict(book, P, "nvme")
    assert _report(book, {})["by_tenant"][A.hex()]["saver"]["wasted_nvme_bytes"] == 70


def test_wasted_counts_only_the_savers_own_members() -> None:
    # Another key filling a member of A's position writes bytes on its own account.
    book = ReuseBook()
    book.saved(P, A, 0, 0.0)
    book.nvme_written(P, 70, A)
    book.nvme_written(P, 30, B)
    _evict(book, P, "nvme")
    by = _report(book, {})["by_tenant"]
    assert by[A.hex()]["saver"]["wasted_nvme_bytes"] == 70
    assert by[A.hex()]["saver"]["nvme_bytes_written"] == 70
    assert by[B.hex()]["saver"]["nvme_bytes_written"] == 30
    assert by[B.hex()]["saver"]["wasted_nvme_bytes"] == 0
    assert _report(book, {})["store"]["nvme_bytes_written"] == 100


def test_regret_within_the_window_only() -> None:
    book = ReuseBook()
    early, late = _pos(1), _pos(2)
    for pos in (early, late):
        book.saved(pos, A, 0, 0.0)
        _evict(book, pos, now=1000.0)
    assert book.rows == {}
    book.saved(early, B, 0, 1000.0 + REGRET_WINDOW_S - 1)
    book.saved(late, B, 0, 1000.0 + REGRET_WINDOW_S + 1)
    rep = _report(book, {})["by_tenant"]
    assert rep[B.hex()]["saver"]["regret"] == 1
    assert rep[A.hex()]["saver"]["regret"] == 0
    _evict(book, early, now=1.0e6)
    book.saved(early, B, 0, 1.0e6 + 1)  # counted once per departure
    _evict(book, early, now=2.0e6)
    assert _report(book, {})["by_tenant"][B.hex()]["saver"]["regret"] == 2


def test_regret_ring_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reuse, "REGRET_RING", 4)
    book = ReuseBook()
    for i in range(6):
        book.saved(_pos(i), A, 0, 0.0)
        _evict(book, _pos(i), now=1.0)
    book.saved(_pos(0), A, 0, 2.0)  # pushed out of the ring
    book.saved(_pos(5), A, 0, 2.0)
    assert _report(book, {})["by_tenant"][A.hex()]["saver"]["regret"] == 1


def test_tenants_seen_caps_at_four_then_many() -> None:
    book = ReuseBook()
    book.saved(P, A, 0, 0.0)
    tenants = [bytes([i]) * 16 for i in range(1, TENANTS_SEEN_MAX + 1)]
    for i, t in enumerate(tenants * 2):
        book.reloaded(P, f"t{i}", t, "ram", 1.0)
    row = book.rows[P]
    assert row.tenants_seen == tuple(tenants) and row.many is False
    book.reloaded(P, "fifth", b"\x77" * 16, "ram", 1.0)
    assert len(row.tenants_seen) == TENANTS_SEEN_MAX and row.many is True
    assert "tenants_seen=" not in repr(row) and "saver=" not in repr(row)


def _top(book: ReuseBook, live: dict[Any, Any], **kw: Any) -> list[dict[str, Any]]:
    args: dict[str, Any] = {"model": None, "tenant": None, "view": "admin", "n": 10}
    args.update(kw)
    return book.top(
        args["model"], args["tenant"], args["view"], args["n"], KEY, live,
        lambda _p: False, lambda _p: A,
    )  # fmt: skip


def test_holder_top_ranks_by_own_reloads_only() -> None:
    book = ReuseBook()
    hot, mine = _pos(1), _pos(2)
    book.saved(hot, A, 0, 0.0)
    book.saved(mine, A, 1, 0.0)
    for i in range(10):
        book.reloaded(hot, f"b{i}", B, "ram", 1.0)
    book.reloaded(hot, "a0", A, "ram", 1.0)
    book.reloaded(mine, "a1", A, "ram", 1.0)
    book.reloaded(mine, "a2", A, "ram", 1.0)
    live = _live(hot, mine)
    rows = _top(book, live, view="holder", tenant=A)
    assert [r["reloads"] for r in rows] == [2, 1]
    assert rows[1]["label"] == label(KEY, A, M, hot[1])
    assert set(rows[0]) == {"label", "model", "depth", "reloads"}
    assert rows[0]["model"] == M.hex() and rows[0]["depth"] == 1
    admin = _top(book, live)
    assert [r["reloads"] for r in admin] == [11, 2]
    assert admin[0]["tenants_seen"] == 2 and admin[0]["many"] is False
    assert admin[0]["charged"] == A.hex() and admin[0]["tiers"] == ["ram"]
    assert admin[0]["label"] == label(KEY, b"", M, hot[1])
    assert {"decayed", "last_reload", "first_saved"} <= set(admin[0])


def test_holder_sketch_keeps_at_most_HOLDER_TOP() -> None:
    book = ReuseBook()
    n = HOLDER_TOP + 10
    for i in range(n):
        book.saved(_pos(i), A, 0, 0.0)
        book.reloaded(_pos(i), f"t{i}", A, "ram", 1.0)
    rows = _top(book, _live(*[_pos(i) for i in range(n)]), view="holder", tenant=A, n=50)
    assert len(rows) == 50
    assert len(book._sketch[A]) == HOLDER_TOP


def test_top_skips_dead_unstored_and_other_models() -> None:
    book = ReuseBook()
    stored, gone, dead, other = _pos(1), _pos(2), _pos(3), (b"\x08" * 16, b"\x01" * 32)
    for i, pos in enumerate((stored, gone, dead, other)):
        book.saved(pos, A, 0, 0.0)
        book.reloaded(pos, f"t{i}", A, "ram", 1.0)
    live = _live(stored, dead, other)
    for view in ("admin", "holder"):
        rows = book.top(M, A, view, 10, KEY, live, lambda p: p == dead, lambda _p: A)
        viewer = A if view == "holder" else b""
        assert [r["label"] for r in rows] == [label(KEY, viewer, M, stored[1])]


def test_labels_are_bound_to_the_viewer_and_stable() -> None:
    pos = _pos(5)[1]
    one = label(KEY, A, M, pos)
    assert one == label(KEY, A, M, pos)
    assert len(one) == 12 and int(one, 16) >= 0
    assert label(KEY, B, M, pos) != one
    assert label(KEY, b"", M, pos) != one
    assert label(b"\x01" * 32, A, M, pos) != one
    assert one != pos.hex()[:12]


def test_report_counts_stored_positions_in_both_views() -> None:
    book = ReuseBook()
    p1, p2 = _pos(1), _pos(2)
    book.saved(p1, A, 0, 0.0)
    book.saved(p2, A, 1, 0.0)
    book.reloaded(p1, "t", A, "ram", 1.0)
    book.reloaded(p2, "u", B, "nvme", 1.0)
    rep = book.report(
        {p1: (True, True), p2: (False, True)}, lambda _p: False, lambda p: A if p == p1 else B
    )
    a, b = rep["by_tenant"][A.hex()], rep["by_tenant"][B.hex()]
    assert a["charged"]["positions"] == {"ram": 1, "nvme": 1}
    assert a["charged"]["positions_reloaded"] == {"ram": 1, "nvme": 1}
    assert b["charged"]["positions"] == {"ram": 0, "nvme": 1}
    assert (b["charged"]["reloads"], b["charged"]["nvme_reloads"]) == (1, 1)
    assert a["saver"]["positions"] == 2 and a["saver"]["positions_reloaded"] == 1
    assert a["saver"]["saver_reloads"] == 1
    assert rep["by_model_tenant"][(M, A)]["saver"]["positions"] == 2
    assert rep["store"] == {
        "reuse_persist_degraded": False, "half_life_s": 21600.0, "order": "lru",
        "nvme_bytes_written": 0, "nvme_bytes_read": 0,
    }  # fmt: skip
    assert a["saver"]["admission_skipped"] == 0 and a["saver"]["ghost_admitted"] == 0


def test_report_leaves_out_dead_positions() -> None:
    book = ReuseBook()
    book.saved(P, A, 0, 0.0)
    rep = book.report(_live(P), lambda _p: True, lambda _p: A)
    assert rep["by_tenant"] == {} and rep["by_model_tenant"] == {}
