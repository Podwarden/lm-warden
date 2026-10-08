from app.stats import forest as fo


def _row(
    t,
    *,
    tok="t1",
    name="farm",
    sk=None,
    psk=None,
    batch=None,
    p=1000,
    g=50,
    cached=None,
    fin="tool_calls",
    tin=None,
    tout=None,
    dur=2.0,
    variant="v1",
):
    return {
        "finished_at": t,
        "duration_s": dur,
        "token_id": tok,
        "token_name": name,
        "model_id": "m",
        "variant_id": variant,
        "prompt_tokens": p,
        "completion_tokens": g,
        "cached_tokens": cached,
        "finish_reason": fin,
        "session_key": sk,
        "parent_session_key": psk,
        "batch_id": batch,
        "tools_in": tin or [],
        "tools_out": tout if tout is not None else [["read", "x"]],
    }


def test_sessions_by_key_and_children_attach_to_parent():
    rows = [
        _row(10, sk="A"),
        _row(20, sk="A", p=1500),
        _row(25, sk="C", psk="A"),
        _row(30, sk="A", p=2000),
    ]
    f = fo.build_forest(rows, t0=0)
    (tree,) = f["trees"]
    (ses,) = tree["sessions"]
    assert ses["id"] == "A" and len(ses["turns"]) == 3
    (child,) = ses["children"]
    assert child["id"] == "C" and child["at"] == 1


def test_heuristic_sessions_without_keys():
    rows = [_row(10, p=1000), _row(20, p=1400), _row(25, p=300), _row(40, p=1800)]
    f = fo.build_forest(rows, t0=0)
    sess = f["trees"][0]["sessions"]
    assert sorted(len(s["turns"]) for s in sess) == [1, 3]
    assert all(s["id"].startswith("h:") for s in sess)


def test_batch_then_spell_boundaries():
    b = [
        _row(100, sk="B1", batch="farm-A27"),
        _row(9000, sk="B2", batch="farm-A27"),
        _row(9100, sk="B3", batch="farm-A28"),
    ]
    f = fo.build_forest(b, t0=0)
    assert sorted(t["id"] for t in f["trees"]) == ["t1:batch:farm-A27", "t1:batch:farm-A28"]
    busy = [
        _row(h * 3600 + 60, sk=f"S{h}") for h in range(14)
    ]  # a busy day stays as spells (no day tree)
    f = fo.build_forest(busy, t0=0)
    assert all(t["id"].startswith("t1:spell:") for t in f["trees"])
    spells = [_row(100, sk="X"), _row(200, sk="Y"), _row(100 + 4 * 3600, sk="Z")]
    f = fo.build_forest(spells, t0=0)
    assert len(f["trees"]) == 2 and all(t["id"].startswith("t1:spell:") for t in f["trees"])


def test_tree_ids_are_stable_as_sessions_arrive():
    rows = [_row(h * 1800 + 60 + 5 * i, sk=f"S{h}") for h in range(40) for i in range(3)]
    seen: dict[str, set[str]] = {}
    for n in range(1, len(rows) + 1):
        for t in fo.build_forest(rows[:n], t0=0)["trees"]:
            ids = {s["id"] for s in t["sessions"]}
            # a tree id never loses a session it once had, and no session moves to another tree
            assert seen.get(t["id"], set()) <= ids
            seen[t["id"]] = ids
    owners = [i for ids in seen.values() for i in ids]
    assert len(owners) == len(set(owners))


def test_single_turn_without_tools_is_a_flower():
    rows = [_row(10, sk="Q", fin="stop", tout=[]), _row(20, sk="A"), _row(30, sk="A")]
    f = fo.build_forest(rows, t0=0)
    assert len(f["flowers"]) == 1 and f["flowers"][0][0] == 8.0  # start = finished - duration
    assert len(f["trees"]) == 1


def test_merge_turns_caps_and_counts():
    turns = [[i, 1000 + i, 10, 0, [], [], "tool_calls", 1] for i in range(200)]
    m = fo.merge_turns(turns, cap=90, block=8)
    # 90 single turns, 13 full blocks of 8 (104 turns), then the 6 turns of the open block one by one
    assert len(m) == 90 + 13 + 6 and sum(t[7] for t in m) == 200
    assert m[90][0] == 97  # a block carries the time of its last request
    assert m[-1][0] == 199


def test_merge_turns_is_prefix_stable():
    turns = [[i, 1000 + i, 10, 0, [], [], "tool_calls", 1] for i in range(300)]
    prev = fo.merge_turns(turns[:150], cap=90, block=8)
    for n in range(151, 301):
        cur = fo.merge_turns(turns[:n], cap=90, block=8)
        closed = len(prev) - (n - 1 - 90) % 8  # entries before the open block never change
        assert cur[:closed] == prev[:closed]
        prev = cur


def test_traits():
    rows = [
        _row(10, sk="A", p=40000, g=100, cached=0),
        _row(20, sk="A", p=41000, g=120, cached=39000, tin=[["shell", 10, True]]),
    ]
    tr = fo.build_forest(rows, t0=0)["trees"][0]["traits"]
    assert (
        tr["sys0"] == 40000 and tr["mean_gen"] == 110 and tr["fail"] == 1.0 and tr["thrash"] == 0.0
    )


def test_tree_last_is_latest_finish():
    rows = [_row(100, sk="X"), _row(250, sk="X")]
    assert fo.build_forest(rows, t0=0)["trees"][0]["last"] == 250.0


def test_ids_are_stable_when_rows_are_added():
    base = [_row(100, sk="X"), _row(200, sk="X")]
    a = fo.build_forest(base, t0=0)["trees"][0]["id"]
    b = fo.build_forest(base + [_row(300, sk="X")], t0=0)["trees"][0]["id"]
    assert a == b


def test_model_throughput_p75():
    rows = [_row(i, p=10000, cached=0, g=500, dur=3.0) | {"ttft_s": 1.0} for i in range(1, 9)]
    m = fo.model_throughput(rows)
    assert m["v1"]["prefill_tps"] == 10000.0 and m["v1"]["decode_tps"] == 250.0


def test_child_anchor_maps_into_merged_space():
    rows = [_row(i + 1, sk="P", dur=0.5) for i in range(200)]
    rows.append(_row(151.2, sk="C", psk="P", dur=0.5))
    (tree,) = fo.build_forest(rows, t0=0)["trees"]
    (child,) = tree["sessions"][0]["children"]
    assert child["at"] == 150  # under LOD_CAP every turn is sent, so the index is the row index


def test_merged_index_matches_merge_turns():
    for n in (50, 90, 97, 98, 140, 141):
        turns = [[i, 1, 1, 0, [], [], "stop", 1] for i in range(n)]
        m = fo.merge_turns(turns, cap=90, block=8)
        for i in range(n):
            k = fo.merged_index(i, n, cap=90, block=8)
            assert m[k][0] >= i and (k == 0 or m[k - 1][0] < i)


def test_traits_include_child_rows():
    rows = [_row(10, sk="A"), _row(20, sk="C", psk="A", tin=[["shell", 10, True]])]
    tr = fo.build_forest(rows, t0=0)["trees"][0]["traits"]
    assert tr["fail"] == 1.0 and tr["fanout"] == 0.5


def test_chaining_is_bounded_and_correct():
    import time

    rows = [_row(i * 0.1 + 1, p=1000 + (i * 7919) % 500_000) for i in range(5000)]
    t = time.perf_counter()
    f = fo.build_forest(rows, t0=0)
    assert time.perf_counter() - t < 2.0
    assert sum(len(tr["sessions"]) for tr in f["trees"]) + len(f["flowers"]) > 0
    mixed = [_row(10, p=1000), _row(20, p=1400), _row(25, p=300), _row(40, p=1800)]
    sess = fo.build_forest(mixed, t0=0)["trees"][0]["sessions"]
    assert sorted(len(s["turns"]) for s in sess) == [1, 3]


def test_parent_cycle_yields_one_tree_with_both():
    rows = [_row(10, sk="A", psk="B"), _row(20, sk="B", psk="A")]
    f = fo.build_forest(rows, t0=0)
    assert len(f["trees"]) == 1
    (root,) = f["trees"][0]["sessions"]
    assert root["id"] == "A" and [c["id"] for c in root["children"]] == ["B"]


def test_parent_with_child_is_not_a_flower():
    rows = [_row(10, sk="P", fin="stop", tout=[]), _row(12, sk="C", psk="P")]
    f = fo.build_forest(rows, t0=0)
    assert len(f["trees"]) == 1 and f["flowers"] == []


def test_tree_last_at_is_exact_max_finished_at():
    rows = [_row(1791270449.123456, sk="X"), _row(1791270459.654321, sk="X")]
    assert fo.build_forest(rows, t0=1791270000.7)["trees"][0]["last_at"] == 1791270459.654321


def test_parallel_unkeyed_chains_get_distinct_stable_ids():
    # Two agents fire their first unkeyed call in the same 0.1 s; prompts too far apart to chain.
    rows = [
        _row(10.02, p=1000) | {"id": "aaaaaaaa-1111"},
        _row(10.04, p=200_000) | {"id": "bbbbbbbb-2222"},
        _row(20, p=1200) | {"id": "cccccccc-3333"},
        _row(21, p=201_000) | {"id": "dddddddd-4444"},
    ]
    ids = [s["id"] for s in fo.build_forest(rows, t0=0)["trees"][0]["sessions"]]
    assert len(ids) == 2 and len(set(ids)) == 2
    assert sorted(ids) == ["h:8.0:aaaaaaaa", "h:8.0:bbbbbbbb"]
    more = rows + [_row(30, p=1400) | {"id": "eeeeeeee-5555"}]
    again = [s["id"] for s in fo.build_forest(more, t0=0)["trees"][0]["sessions"]]
    assert sorted(again) == sorted(ids)  # stable as the chains grow


def test_same_first_row_prefix_still_gets_distinct_ids():
    rows = [_row(10.02, p=1000), _row(10.04, p=200_000)]  # no ids at all
    ids = [s["id"] for s in fo.build_forest(rows, t0=0)["trees"][0]["sessions"]]
    assert len(set(ids)) == 2


def test_traits_ignore_non_list_tools_in_entries():
    rows = [_row(10, sk="A", tin=[1, "x", ["read", 3, False], ["shell", 2, True]])]
    assert fo.build_forest(rows, t0=0)["trees"][0]["traits"]["fail"] == 0.5


def test_model_throughput_reports_sample_count():
    rows = [_row(i, p=10000, cached=0, g=500, dur=3.0) | {"ttft_s": 1.0} for i in range(1, 9)]
    rows.append(_row(9, p=10000, cached=0, g=500, dur=3.0) | {"ttft_s": None})  # no sample
    assert fo.model_throughput(rows)["v1"]["n"] == 8
