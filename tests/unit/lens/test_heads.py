"""Prompt Lens recent heads: per-key last exact sys id (lens spec 4.6)."""

from app.lens.heads import MAX_DIGESTS, HeadEntry, RecentHeads

A = b"scope-a"
B = b"scope-b"
D1 = (b"\x01" * 8, b"\x02" * 8)
D2 = (b"\x01" * 8, b"\x03" * 8)


def test_lookup_returns_the_last_exact_sys_id() -> None:
    h = RecentHeads()
    h.remember(A, "m", "ls1", "t1", "sys-1", D1, now=0.0)
    assert h.lookup(A, "m", "ls1", "t1", now=1.0) == HeadEntry("sys-1", D1)
    h.remember(A, "m", "ls1", "t1", "sys-2", D2, now=2.0)
    assert h.lookup(A, "m", "ls1", "t1", now=3.0) == HeadEntry("sys-2", D2)


def test_lookup_needs_every_part_of_the_key() -> None:
    h = RecentHeads()
    h.remember(A, "m", "ls1", "t1", "sys-1", D1, now=0.0)
    assert h.lookup(A, "other-model", "ls1", "t1", now=1.0) is None
    assert h.lookup(A, "m", "ls2", "t1", now=1.0) is None
    assert h.lookup(A, "m", "ls1", "t2", now=1.0) is None


def test_65th_entry_evicts_the_oldest() -> None:
    h = RecentHeads()
    for i in range(65):
        h.remember(A, "m", f"ls{i}", "t", f"sys-{i}", D1, now=float(i))
    assert h.lookup(A, "m", "ls0", "t", now=100.0) is None
    assert h.lookup(A, "m", "ls1", "t", now=100.0) == HeadEntry("sys-1", D1)
    assert h.lookup(A, "m", "ls64", "t", now=100.0) == HeadEntry("sys-64", D1)


def test_refresh_keeps_an_entry_from_being_evicted() -> None:
    h = RecentHeads(max_per_key=2)
    h.remember(A, "m", "ls0", "t", "sys-0", D1, now=0.0)
    h.remember(A, "m", "ls1", "t", "sys-1", D1, now=1.0)
    h.remember(A, "m", "ls0", "t", "sys-0b", D1, now=2.0)
    h.remember(A, "m", "ls2", "t", "sys-2", D1, now=3.0)
    assert h.lookup(A, "m", "ls1", "t", now=4.0) is None
    assert h.lookup(A, "m", "ls0", "t", now=4.0) == HeadEntry("sys-0b", D1)


def test_expired_after_ttl() -> None:
    h = RecentHeads(ttl_s=100.0)
    h.remember(A, "m", "ls", "t", "sys-1", D1, now=0.0)
    assert h.lookup(A, "m", "ls", "t", now=100.0) == HeadEntry("sys-1", D1)
    assert h.lookup(A, "m", "ls", "t", now=100.5) is None
    # and stays gone
    assert h.lookup(A, "m", "ls", "t", now=1.0) is None


def test_default_ttl_is_six_hours() -> None:
    h = RecentHeads()
    h.remember(A, "m", "ls", "t", "sys-1", D1, now=0.0)
    assert h.lookup(A, "m", "ls", "t", now=21_600.0) is not None
    assert h.lookup(A, "m", "ls", "t", now=21_601.0) is None


def test_another_scope_never_matches() -> None:
    h = RecentHeads()
    h.remember(A, "m", "ls", "t", "sys-1", D1, now=0.0)
    assert h.lookup(B, "m", "ls", "t", now=1.0) is None
    h.remember(B, "m", "ls", "t", "sys-b", D2, now=2.0)
    assert h.lookup(A, "m", "ls", "t", now=3.0) == HeadEntry("sys-1", D1)
    assert h.lookup(B, "m", "ls", "t", now=3.0) == HeadEntry("sys-b", D2)


def test_entries_are_bounded_across_scopes() -> None:
    h = RecentHeads(max_entries=3)
    for i in range(4):
        h.remember(bytes([i]), "m", "ls", "t", f"sys-{i}", D1, now=float(i))
    assert len(h) == 3
    assert h.lookup(bytes([0]), "m", "ls", "t", now=5.0) is None
    assert h.lookup(bytes([3]), "m", "ls", "t", now=5.0) == HeadEntry("sys-3", D1)


def test_digests_per_entry_are_capped() -> None:
    h = RecentHeads()
    h.remember(A, "m", "ls", "t", "sys-1", (b"\x00" * 8,) * 10_000, now=0.0)
    e = h.lookup(A, "m", "ls", "t", now=1.0)
    assert e is not None and len(e.digests) == MAX_DIGESTS


def test_per_scope_eviction_keeps_the_global_count() -> None:
    h = RecentHeads(max_per_key=2)
    for i in range(5):
        h.remember(A, "m", f"ls{i}", "t", "s", D1, now=float(i))
    assert len(h) == 2
