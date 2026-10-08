"""Place a session on first sight, then stick (DP router)."""

import asyncio

from app.proxy.dp_affinity import (
    KV_WEIGHT,
    DpRoutingState,
    SessionPlacement,
    home_rank,
)
from app.proxy.routes_dp import RankScrapeCache
from app.runtime.backends.vllm.metrics import RankReading


def route(st, key, *, dp=4, threshold=8, epoch="e1", kv=None, **kw):
    args = dict(dp=dp, key=key, threshold=threshold, pinned=None, affinity_enabled=True)
    args.update(kw)
    return st.route("m", epoch=epoch, kv=kv, **args)


def test_new_sessions_spread_to_idle_ranks_instead_of_hashing():
    st = DpRoutingState()
    placed = [route(st, f"s{i}") for i in range(4)]
    assert [d for _, d in placed] == ["placed"] * 4
    assert sorted(r for r, _ in placed) == [0, 1, 2, 3]  # one each, none left idle


def test_a_known_session_stays_even_when_its_rank_is_busier():
    st = DpRoutingState()
    home, _ = route(st, "a")
    for _ in range(5):  # others pile onto the same rank via pinned traffic
        st.route("m", dp=4, key=None, threshold=99, pinned=home, affinity_enabled=True)
    assert route(st, "a") == (home, "sticky")


def test_kv_breaks_an_in_flight_tie():
    st = DpRoutingState()
    kv = {0: 0.9, 1: 0.1, 2: 0.5, 3: 0.95}
    assert route(st, "new", kv=kv) == (1, "placed")
    # one request in flight is worth KV_WEIGHT x kv: rank 1 now scores 1.8, rank 2 4.0
    assert KV_WEIGHT == 8.0
    assert route(st, "new2", kv=kv)[0] == 1


def test_in_flight_dominates_when_kv_is_missing_or_partial():
    st = DpRoutingState()
    st.route("m", dp=4, key=None, threshold=99, pinned=0, affinity_enabled=True)
    assert route(st, "x", kv=None)[0] == 1
    assert route(st, "y", kv={0: 0.0})[0] in (1, 2)  # rank 0 has 1 in flight


def test_ties_go_to_the_rank_with_fewest_live_sessions_then_lowest_index():
    clock = [0.0]
    sp = SessionPlacement(clock=lambda: clock[0])
    sp.assign("m", "e", "a", 0)
    sp.assign("m", "e", "b", 0)
    sp.assign("m", "e", "c", 1)
    assert sp.place("m", "e", 4, {}, None) == 2  # 2 and 3 empty: lowest index
    sp.assign("m", "e", "d", 2)
    assert sp.place("m", "e", 4, {}, None) == 3
    # sessions idle for > 5 minutes no longer count as live
    clock[0] = 301.0
    assert sp.live_counts("m", "e") == {}
    assert sp.place("m", "e", 4, {}, None) == 0


def test_stale_scrape_means_in_flight_only():
    t = [100.0]

    async def fetch(host, port):
        return "x", None

    cache = RankScrapeCache(clock=lambda: t[0], fetch=fetch)
    cache._entries["m"] = (100.0, "iso", {0: RankReading(kv_cache_usage_perc=0.9)}, None)
    assert cache.recent_kv("m", 15.0) == {0: 0.9}
    t[0] = 116.0
    assert cache.recent_kv("m", 15.0) is None
    t[0] = 100.0
    cache._entries["m"] = (100.0, "iso", {}, "ConnectError")  # failed scrape
    assert cache.recent_kv("m", 15.0) is None
    assert cache.recent_kv("other", 15.0) is None


def test_ttl_lru_and_epoch_reset():
    clock = [0.0]
    sp = SessionPlacement(max_keys=3, ttl_s=100.0, clock=lambda: clock[0])
    sp.assign("m", "e1", "a", 2)
    assert sp.lookup("m", "e1", "a", 4) == 2
    assert sp.lookup("m", "e2", "a", 4) is None  # new engine run forgets it
    clock[0] = 99.0
    assert sp.lookup("m", "e1", "a", 4) == 2  # use refreshes the TTL
    clock[0] = 198.0
    assert sp.lookup("m", "e1", "a", 4) == 2
    clock[0] = 400.0
    assert sp.lookup("m", "e1", "a", 4) is None  # expired
    for k in "bcde":
        sp.assign("m", "e1", k, 1)
    assert sp.lookup("m", "e1", "b", 4) is None  # LRU evicted the oldest
    assert sp.lookup("m", "e1", "e", 4) == 1
    assert (
        sp.lookup("m", "e1", "e", 2) == 1 and sp.lookup("m", "e1", "e", 1) is None
    )  # invalid rank


def test_keys_are_stored_hashed_only():
    sp = SessionPlacement()
    sp.assign("m", "e", "user_secret_account_xyz", 1)
    for epoch, digest in sp._m["m"]:
        assert len(digest) == 16 and b"secret" not in digest and epoch == "e"


def test_a_new_epoch_replaces_placements_in_the_router():
    st = DpRoutingState()
    r1, d1 = route(st, "a", epoch="e1")
    for _ in range(3):
        st.route("m", dp=4, key=None, threshold=99, pinned=r1, affinity_enabled=True)
    r2, d2 = route(st, "a", epoch="e2")  # engine restarted: placed afresh, on a quieter rank
    assert (d1, d2) == ("placed", "placed") and r2 != r1


def test_spill_pinned_unrouted_and_balanced_paths_are_unchanged():
    st = DpRoutingState()
    home, _ = route(st, "a", threshold=2)
    route(st, "a", threshold=2)  # in flight on home: 2 == threshold
    rank, decision = route(st, "a", threshold=2)
    assert decision == "spilled" and rank != home
    assert st.route("m", dp=4, key="a", threshold=2, pinned=3, affinity_enabled=True) == (
        3,
        "client_pinned",
    )
    assert st.route("m", dp=4, key="a", threshold=2, pinned=None, affinity_enabled=False) == (
        None,
        "unrouted",
    )
    assert route(st, None)[1] == "balanced"
    snap = st.snapshot("m", 4)
    assert snap["totals"]["placed"] == 1 and snap["totals"]["spilled"] == 1


def test_a_failing_placement_falls_back_to_the_hash_home():
    st = DpRoutingState()

    class Boom(SessionPlacement):
        def lookup(self, *a, **k):
            raise RuntimeError("boom")

    st.placement = Boom()
    assert route(st, "a") == (home_rank("a", 4), "sticky")


def test_snapshot_reports_assigned_sessions_and_placed():
    st = DpRoutingState()
    for i in range(6):
        route(st, f"s{i}")
    snap = st.snapshot("m", 4)
    assert sum(r["assigned_sessions"] for r in snap["ranks"]) == 6
    assert sum(r["placed"] for r in snap["ranks"]) == 6
    assert snap["totals"]["placed"] == 6
    st.reset("m")
    assert sum(r["assigned_sessions"] for r in st.snapshot("m", 4)["ranks"]) == 0


def test_the_background_scraper_survives_failures_and_warms_the_cache(monkeypatch):
    monkeypatch.setenv("VW_DP_RANK_SCRAPER", "1")
    from types import SimpleNamespace

    from app.proxy.routes_dp import run_rank_scraper

    state = SimpleNamespace(dp_rank_scrape_cache=None, supervisor=None)

    async def go():
        task = asyncio.create_task(run_rank_scraper(state, interval=0.01))
        await asyncio.sleep(0.05)
        assert not task.done()  # a pass with nothing to scrape (or failing) never kills it
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(go())


def test_the_scraper_can_be_switched_off(monkeypatch):
    from types import SimpleNamespace

    from app.proxy.routes_dp import run_rank_scraper

    monkeypatch.setenv("VW_DP_RANK_SCRAPER", "0")
    asyncio.run(asyncio.wait_for(run_rank_scraper(SimpleNamespace(), interval=0.01), 1))


# ---- review fixes: live sessions in the score, epoch-scoped counts and KV ----


def test_live_sessions_weigh_in_the_score_not_just_the_tie_break():
    from app.proxy.dp_affinity import LIVE_WEIGHT

    assert LIVE_WEIGHT == 1.0
    sp = SessionPlacement()
    for i in range(5):  # rank 0 holds five idle (cached) sessions, nothing running
        sp.assign("m", "e", f"idle{i}", 0)
    # rank 1 has one request in flight and no sessions: 1 < 5, so it wins
    assert sp.place("m", "e", 2, {0: 0, 1: 1}, None) == 1


def test_a_burst_of_new_keys_spreads_with_no_in_flight_change():
    st = DpRoutingState()
    ranks = [route(st, f"k{i}", dp=4)[0] for i in range(8)]
    for r in range(4):
        assert ranks.count(r) == 2  # just-placed sessions count as live at once
    # even through the placement object alone, with nothing in flight
    sp = SessionPlacement()
    picks = []
    for i in range(12):
        r = sp.place("m", "e", 3, {}, None)
        sp.assign("m", "e", f"b{i}", r)
        picks.append(r)
    assert [picks.count(r) for r in range(3)] == [4, 4, 4]


def test_assigned_sessions_counts_only_the_current_engine_run():
    st = DpRoutingState()
    for i in range(3):
        route(st, f"old{i}", epoch="e1")
    for i in range(2):
        route(st, f"new{i}", epoch="e2")
    assert sum(r["assigned_sessions"] for r in st.snapshot("m", 4, "e2")["ranks"]) == 2
    assert sum(r["assigned_sessions"] for r in st.snapshot("m", 4, "e1")["ranks"]) == 3
    assert sum(r["assigned_sessions"] for r in st.snapshot("m", 4)["ranks"]) == 5  # no filter


def test_kv_is_ignored_unless_every_rank_has_a_reading():
    sp = SessionPlacement()
    # rank 0 is read as saturated; ranks 1 and 2 are unread. Used as-is that
    # would steer everything to the unread ranks; a partial scrape is ignored.
    partial = {0: 1.0, 1: 0.0}
    assert sp.place("m", "e", 3, {0: 0, 1: 1, 2: 1}, partial) == 0  # in-flight only
    full = {0: 1.0, 1: 0.0, 2: 0.0}
    assert sp.place("m", "e", 3, {0: 0, 1: 1, 2: 1}, full) == 1  # kv now counts


def test_a_reload_does_not_reuse_the_previous_engines_kv():
    t = [100.0]
    fetched = []

    async def fetch(host, port):
        fetched.append(1)
        return "x", None

    cache = RankScrapeCache(clock=lambda: t[0], fetch=fetch)
    cache._entries["m"] = (100.0, "iso", {0: RankReading(kv_cache_usage_perc=0.9)}, None)
    cache._epochs["m"] = "v:1"
    assert cache.recent_kv("m", 15.0, "v:1") == {0: 0.9}
    assert cache.recent_kv("m", 15.0, "v:2") is None  # engine reloaded
    assert cache.recent_kv("m", 15.0) == {0: 0.9}  # legacy callers: no epoch check


def test_get_refetches_when_the_epoch_changed_within_the_ttl():
    import asyncio as _aio

    fetched = []

    async def fetch(host, port):
        fetched.append(1)
        return None, "down"

    cache = RankScrapeCache(clock=lambda: 100.0, fetch=fetch)

    async def go():
        await cache.get("m", "h", 1, "v:1")
        await cache.get("m", "h", 1, "v:1")  # TTL hit
        await cache.get("m", "h", 1, "v:2")  # new engine run: fresh scrape

    _aio.run(go())
    assert len(fetched) == 2
