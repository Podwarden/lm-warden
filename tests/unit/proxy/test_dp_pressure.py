"""Spill on live-context pressure against the replica's KV pool."""

from app.proxy.dp_affinity import SPILL_FILL, SessionPlacement, choose_rank

POOL = 682_704  # 1584 x 431, lmw48's per-replica pool


def _c(live, *, in_flight=None, own=150_000, pool=POOL, threshold=8, dp=3):
    return choose_rank(
        "k",
        dp,
        in_flight or {},
        threshold,
        home=0,
        live_ctx=live,
        pool_tokens=pool,
        own_tokens=own,
    )


def test_spills_on_live_context_pressure_before_the_request_threshold():
    live = {0: 1_200_000, 1: 100_000, 2: 100_000}
    assert _c(live, in_flight={0: 2}) == (1, "spilled")


def test_pressure_target_prefers_fewer_in_flight_then_lower_context_then_lower_index():
    live = {0: 1_200_000, 1: 300_000, 2: 100_000}
    assert _c(live, in_flight={1: 1, 2: 1}) == (2, "spilled")
    assert _c(live, in_flight={1: 1}) == (2, "spilled")
    assert _c({0: 1_200_000, 1: 100_000, 2: 100_000}, in_flight={}) == (1, "spilled")


def test_no_pressure_spill_when_no_rank_has_room():
    live = {0: 1_200_000, 1: 900_000, 2: 900_000}
    assert _c(live) == (0, "sticky")


def test_no_pressure_spill_without_pool_reading():
    live = {0: 1_200_000, 1: 100_000, 2: 100_000}
    assert _c(live, pool=None) == (0, "sticky")


def test_own_session_alone_never_pushes_itself_off():
    assert _c({0: 600_000}, own=600_000) == (0, "sticky")


def test_request_threshold_still_spills_without_pressure():
    assert _c({}, in_flight={0: 8}, own=0)[1] == "spilled"


def _sp():
    clock = [0.0]
    return SessionPlacement(clock=lambda: clock[0]), clock


def test_note_prompt_feeds_live_context_and_expires():
    sp, clock = _sp()
    sp.assign("m", "e", "a", 1)
    sp.assign("m", "e", "b", 1)
    sp.note_prompt("m", "e", "a", 100_000)
    sp.note_prompt("m", "e", "b", 50_000)
    sp.note_prompt("m", "e", "unknown", 5)
    assert sp.live_context("m") == {1: 150_000}
    assert sp.tokens_of("m", "e", "a") == 100_000
    assert sp.tokens_of("m", "e", "zzz") == 0
    sp.assign("m", "e", "a", 2)  # re-home keeps the size
    assert sp.tokens_of("m", "e", "a") == 100_000
    clock[0] = 301.0
    assert sp.live_context("m") == {}


def test_live_context_is_scoped_to_the_epoch():
    sp, _ = _sp()
    sp.assign("m", "a", "k", 0)
    sp.note_prompt("m", "a", "k", 10)
    assert sp.live_context("m", "a") == {0: 10}
    assert sp.live_context("m", "b") == {}


def test_new_session_avoids_a_rank_full_of_live_context():
    sp, _ = _sp()
    kv = {0: 0.0, 1: 0.0, 2: 0.0}
    assert sp.place("m", "e", 3, {}, kv, pressure={0: 1.7, 1: 0.1, 2: 0.2}) == 1
    assert SPILL_FILL == 0.85


# cap = 0.85 * 682_704 = 580_298.4 tokens per rank.


def test_home_under_the_cap_is_sticky_even_with_other_live_context():
    # home holds 400k (< cap) of which 100k is this session's: not hot
    assert _c({0: 400_000, 1: 0, 2: 0}, own=100_000) == (0, "sticky")


def test_in_flight_outranks_context_when_picking_the_target():
    # own 150k: rank 1 -> 250k, rank 2 -> 450k, both fit. Rank 1 has MORE
    # in-flight, so rank 2 wins although it holds more context.
    live = {0: 1_200_000, 1: 100_000, 2: 300_000}
    assert _c(live, in_flight={1: 1}, own=150_000) == (2, "spilled")


def test_a_rank_without_room_is_excluded_even_if_least_loaded():
    # own 150k: rank 1 -> 650k > cap (no room); rank 2 -> 250k fits. Rank 1 has
    # fewer in-flight, so only the room filter sends this to rank 2.
    live = {0: 1_200_000, 1: 500_000, 2: 100_000}
    assert _c(live, in_flight={2: 1}, own=150_000) == (2, "spilled")


def test_recent_pool_tokens_respects_age_and_epoch():
    from app.proxy.routes_dp import RankScrapeCache
    from app.runtime.backends.vllm.metrics import RankReading

    t = [100.0]

    async def fetch(host, port):
        return "x", None

    cache = RankScrapeCache(clock=lambda: t[0], fetch=fetch)
    ok = {0: RankReading(kv_pool_tokens=POOL), 1: RankReading(kv_pool_tokens=POOL)}
    cache._entries["m"] = (100.0, "iso", ok, None)
    cache._epochs["m"] = "e1"
    assert cache.recent_pool_tokens("m", 15.0, "e1") == POOL
    assert cache.recent_pool_tokens("m", 15.0, "other") is None  # other epoch
    t[0] = 116.0
    assert cache.recent_pool_tokens("m", 15.0, "e1") is None  # stale
    t[0] = 100.0
    cache._entries["m"] = (100.0, "iso", ok, "ConnectError")  # error scrape
    assert cache.recent_pool_tokens("m", 15.0, "e1") is None
    assert cache.recent_pool_tokens("nope", 15.0) is None


def test_route_pressure_spill_rehomes_and_keeps_tokens():
    from app.proxy.dp_affinity import DpRoutingState

    st = DpRoutingState()
    pl = st.placement
    for k in ("a", "b"):
        pl.assign("m", "e", k, 0)
    pl.note_prompt("m", "e", "a", 300_000)
    pl.note_prompt("m", "e", "b", 400_000)  # rank 0 live: 700k > cap
    kw = dict(dp=3, threshold=8, pinned=None, affinity_enabled=True, epoch="e", pool_tokens=POOL)
    assert st.route("m", key="a", **kw) == (1, "spilled")
    assert pl.tokens_of("m", "e", "a") == 300_000
    assert pl.live_context("m", "e") == {0: 400_000, 1: 300_000}
    assert st.route("m", key="a", **kw) == (1, "sticky")


def _state(assignments):
    from app.proxy.dp_affinity import DpRoutingState

    st = DpRoutingState()
    for rank, key, tokens in assignments:
        st.placement.assign("m", "e", key, rank)
        st.placement.note_prompt("m", "e", key, tokens)
    return st


_KW = dict(dp=3, threshold=8, pinned=None, affinity_enabled=True, epoch="e", pool_tokens=POOL)


def test_no_ping_pong_between_a_pressured_and_a_busy_rank():
    # rank 0: A=200k + B=450k (over cap). rank 1: room but 8 in flight (= threshold).
    # rank 2: full. Before the fix A alternated 1, 0, 1, 0 with a cold prefill each turn.
    st = _state([(0, "A", 200_000), (0, "B", 450_000), (2, "C", 600_000)])
    for _ in range(8):
        st.route("m", dp=3, key=None, threshold=99, pinned=1, affinity_enabled=True)
    ranks = []
    for _ in range(5):
        rank, _decision = st.route("m", key="A", **_KW)
        st.release("m", rank)
        ranks.append(rank)
    moves = sum(1 for x, y in zip(ranks, ranks[1:], strict=False) if x != y)
    assert moves <= 1, ranks
    assert ranks == [0] * 5


def test_in_flight_spill_prefers_a_rank_with_room():
    live = {0: 100_000, 1: 590_000, 2: 300_000}
    # home 0 at the threshold; rank 1 is least loaded but has no room for 150k
    assert _c(live, in_flight={0: 8, 2: 2}, own=150_000) == (2, "spilled")
    # no rank has room: plain least-loaded
    live = {0: 100_000, 1: 590_000, 2: 590_000}
    assert _c(live, in_flight={0: 8, 2: 2}, own=150_000) == (1, "spilled")


def test_the_largest_session_is_not_moved_by_pressure():
    st = _state([(0, "A", 350_000), (0, "B", 250_000)])  # 600k > cap 580_298
    assert st.route("m", key="A", **_KW) == (0, "sticky")
    st.release("m", 0)
    rank, decision = st.route("m", key="B", **_KW)
    assert decision == "spilled" and rank != 0


def test_spill_reason_says_which_rule_fired():
    from app.proxy.dp_affinity import choose_rank_why

    live = {0: 1_200_000, 1: 100_000, 2: 100_000}
    kw = dict(home=0, live_ctx=live, pool_tokens=POOL, own_tokens=150_000)
    assert choose_rank_why("k", 3, {}, 8, **kw) == (1, "spilled", "pressure")
    assert choose_rank_why("k", 3, {0: 8, 1: 3}, 8, home=0)[2] == "in_flight"
    assert choose_rank_why("k", 3, {}, 8, home=0)[2] is None

    st = _state([(0, "A", 200_000), (0, "B", 450_000)])
    st.route("m", key="A", **_KW)
    assert st.last_reason == "pressure"
    st2 = _state([(0, "A", 1_000)])
    for _ in range(8):
        st2.route("m", dp=3, key=None, threshold=99, pinned=0, affinity_enabled=True)
    assert st2.route("m", key="A", **_KW)[1] == "spilled"
    assert st2.last_reason == "in_flight"


def test_in_flight_spill_never_targets_a_rank_over_the_threshold():
    # Home 0 is at the threshold (8). own 50k: rank 1 (590k) has no room
    # (640k > cap 580_298); rank 2 (100k) has room but 12 in flight >= 8, so it
    # is filtered out. The room list is empty -> plain least-loaded:
    # in_flight {0: 8, 1: 0, 2: 12} -> rank 1, "spilled". Never rank 2.
    live = {0: 100_000, 1: 590_000, 2: 100_000}
    got = _c(live, in_flight={0: 8, 1: 0, 2: 12}, own=50_000)
    assert got != (2, "spilled")
    assert got == (1, "spilled")


def test_no_ping_pong_through_the_in_flight_rule():
    # rank 0: A (50k) + B (50k), 8 in flight (= threshold). rank 1: C 590k, no
    # room for A, idle. rank 2: D 100k with room but 12 in flight. Before the
    # fix A alternated 2, 0, 2, 0 through the in-flight rule.
    st = _state([(0, "A", 50_000), (0, "B", 50_000), (1, "C", 590_000), (2, "D", 100_000)])
    for _ in range(8):
        st.route("m", dp=3, key=None, threshold=99, pinned=0, affinity_enabled=True)
    for _ in range(12):
        st.route("m", dp=3, key=None, threshold=99, pinned=2, affinity_enabled=True)
    ranks = []
    for _ in range(6):
        rank, _decision = st.route("m", key="A", **_KW)
        st.release("m", rank)
        ranks.append(rank)
    moves = sum(1 for x, y in zip(ranks, ranks[1:], strict=False) if x != y)
    assert moves <= 1, ranks
    assert 2 not in ranks, ranks
    seen: list[int] = []
    for r in ranks:
        if not seen or seen[-1] != r:
            assert r not in seen, ranks  # never returns to a rank it left
            seen.append(r)
