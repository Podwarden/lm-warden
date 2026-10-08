"""Review N3: the forest is kept up to date row by row (``forest.TokenForest``).

- Correctness: after any sequence of appends (some committed late) and prunes, the incremental
  forest equals a from-scratch ``build_parts`` over the rows it holds.
- Cost: a refresh with one new row rebuilds one tree, through the real SQLite path, in time
  proportional to that tree, not to the window (a scaled-down benchmark with a proportional bound;
  the 200k-row numbers are in the report).
"""

import json
import random
import sqlite3
import time

import pytest

from app.stats import forest as fo
from app.stats import routes_forest as rf


def _row(
    i, t, *, sk, psk=None, batch=None, p=1000, fin="tool_calls", tout=True, dur=2.0, variant="v1"
):
    return {
        "id": f"r{i}",
        "finished_at": t,
        "duration_s": dur,
        "token_id": "t1",
        "token_name": "farm",
        "model_id": "m",
        "variant_id": variant,
        "prompt_tokens": p,
        "completion_tokens": 50 + i % 7,
        "cached_tokens": (p // 2) if i % 3 else None,
        "finish_reason": fin,
        "session_key": sk,
        "parent_session_key": psk,
        "batch_id": batch,
        "tools_in": [["read", 10 + i % 5, i % 11 == 0]],
        "tools_out": [["read", "x"]] if tout else [],
    }


def _stream(seed: int, n: int = 900):
    """Rows of a busy key over ~30 h: keyed sessions (some subagents, some batches), unkeyed chains, flowers."""
    rnd = random.Random(seed)
    rows, t = [], 1_000_000.0
    open_s: list[str] = []
    for i in range(n):
        t += rnd.expovariate(1 / 120)
        kind = rnd.random()
        if kind < 0.05:  # a flower: single request, no tools
            rows.append(_row(i, t, sk=f"f{i}", fin="stop", tout=False))
            continue
        if kind < 0.12:  # an unkeyed chain (the prompt-chain heuristic)
            rows.append(_row(i, t, sk=None, p=1000 + (i % 20) * 300))
            continue
        if not open_s or rnd.random() < 0.08:
            open_s.append(f"s{i}")
            open_s = open_s[-4:]
        sk = rnd.choice(open_s)
        psk = open_s[0] if sk != open_s[0] and rnd.random() < 0.3 else None
        batch = "B1" if sk.endswith("7") else None
        rows.append(_row(i, t, sk=sk, psk=psk, batch=batch, p=1000 + i * 37 % 30000))
    return rows


def _plain(b):
    """The trees as their cached JSON says (a reused tree comes back as (None, json))."""
    return {"trees": [json.loads(m) for _, m in b["trees"]], "flowers": b["flowers"]}


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_incremental_equals_from_scratch_after_appends_and_prunes(seed):
    rnd = random.Random(100 + seed)
    rows = _stream(seed)
    t0 = rows[0]["finished_at"] - 10
    tf = fo.TokenForest("t1", t0)
    held: list[dict] = []
    k = 0
    ws = -1.0
    steps = 0
    while k < len(rows):
        chunk = rows[k : k + rnd.randint(1, 40)]
        k += len(chunk)
        # a commit can land late: shuffle within the chunk (rows are loaded ORDER BY finished_at anyway)
        rnd.shuffle(chunk)
        tf.add(chunk)
        held += chunk
        if rnd.random() < 0.3:  # the window slides
            ws = held[-1]["finished_at"] - rnd.uniform(3 * 3600, 10 * 3600)
            tf.prune(ws)
            held = [r for r in held if r["finished_at"] >= ws]
        got = _plain(tf.build(make=lambda t: json.dumps(t)))
        want = fo.build_parts(held, t0=t0).get("t1", {"trees": [], "flowers": []})
        assert got == want, f"seed {seed}, step {steps}"
        steps += 1
    assert steps > 20


def test_build_reuses_every_unchanged_tree(monkeypatch):
    rows = _stream(7)
    tf = fo.TokenForest("t1", rows[0]["finished_at"])
    tf.add(rows)
    first = tf.build(make=lambda t: object())
    built = []
    real = fo._build_tree
    monkeypatch.setattr(
        fo, "_build_tree", lambda tok, tid, *a: built.append(tid) or real(tok, tid, *a)
    )
    again = tf.build(make=lambda t: object())
    assert built == [] and [m for _, m in again["trees"]] == [m for _, m in first["trees"]]
    last = rows[-1]
    tf.add([{**last, "id": "new", "finished_at": last["finished_at"] + 30}])
    tf.build(make=lambda t: object())
    assert len(built) == 1


COLS = (
    "id, finished_at, model_id, model, token_name, token_id, client_ip, prompt_tokens, completion_tokens, "
    "duration_s, ttft_s, finish_reason, orphan, started_iso, variant_id, cached_tokens, session_key, batch_id, tools_out"
)


def _bench_db(tmp_path, n, now):
    """n rows over 7 d through the real schema: 4 keys, key A with 70 %, sessions of 150 turns."""
    import asyncio

    from app.db.database import open_db
    from app.db.migrations import apply_migrations

    db_path = tmp_path / "bench.db"

    async def mk():
        async with open_db(db_path) as db:
            await apply_migrations(db)

    asyncio.run(mk())
    rows = []
    for tok, share in (("A", 0.7), ("B", 0.1), ("C", 0.1), ("D", 0.1)):
        m = int(n * share)
        per = 7 * 86400 / m
        for k in range(m):
            rows.append(
                (
                    f"{tok}{k}",
                    now - 7 * 86400 + k * per + 1,
                    "id-m",
                    "m",
                    tok,
                    tok,
                    "x",
                    4000,
                    200,
                    2.0,
                    0.5,
                    "tool_calls",
                    0,
                    "z",
                    "v1",
                    3000,
                    f"{tok}-{k // 150}",
                    None,
                    '[["read","ab"]]',
                )
            )
    with sqlite3.connect(db_path) as db:
        db.executemany(f"INSERT INTO request_history({COLS}) VALUES ({','.join('?' * 19)})", rows)
        db.commit()
    return db_path


def _state(db_path, store, now, prev=None, token_id=None):
    rf._store_refresh(db_path, store, now)
    return rf._compose(store.view, token_id, prev, {})


def _insert(db_path, rows):
    with sqlite3.connect(db_path) as db:
        db.executemany(f"INSERT INTO request_history({COLS}) VALUES ({','.join('?' * 19)})", rows)
        db.commit()


def _hrow(rid, at, tok, sk):
    return (
        rid,
        at,
        "id-m",
        "m",
        tok,
        tok,
        "x",
        4000,
        200,
        2.0,
        0.5,
        "tool_calls",
        0,
        "z",
        "v1",
        3000,
        sk,
        None,
        "[]",
    )


def test_refresh_with_one_new_row_costs_its_tree_not_the_window(tmp_path):
    """20k rows / 7 d (a tenth of the 200k target): a full build vs a refresh with one new row of key A. The refresh
    must be ≤ 50 ms at 200k rows; it does not grow with the window, so at 20k it must be both ≤ 50 ms and well under
    a tenth of the full build."""
    now = time.time()
    db_path = _bench_db(tmp_path, 20_000, now)
    store = rf._Store()
    t = time.perf_counter()
    _state(db_path, store, now)
    full = time.perf_counter() - t
    _insert(db_path, [_hrow("new", now + 1, "A", f"A-{14_000 // 150}")])
    t = time.perf_counter()
    st2 = _state(db_path, store, now + 2)
    inc = time.perf_counter() - t
    print(f"20k rows: full {full * 1000:.0f} ms, one-row refresh {inc * 1000:.1f} ms")
    # A got the new row; the window slid 2 s, so the oldest tree of each key lost a row too
    assert "A" in st2.rebuilt
    assert inc <= 0.05
    assert inc <= full / 10
    # and the incremental state renders what a from-scratch state renders
    fresh = _state(db_path, rf._Store(), now + 2)
    a = json.loads(rf._render(st2, "7d", None, False, now + 2))
    b = json.loads(rf._render(fresh, "7d", None, False, now + 2))
    assert a["ids"] == b["ids"]
    assert _absolute(a) == _absolute(b)


def _absolute(body):
    """Trees with absolute times. Two states anchor their relative times differently and the wire rounds them to
    0.1 s against its anchor, so the times are compared within 0.15 s (``_Near``)."""
    t0 = body["t0"]

    def ses(s):
        return {
            **s,
            "turns": [[_Near(x[0] + t0), *x[1:]] for x in s["turns"]],
            "children": [ses(c) for c in s["children"]],
        }

    return [
        {
            **t,
            "start": _Near(t["start"] + t0),
            "end": _Near(t["end"] + t0),
            "last": _Near(t["last"] + t0),
            "sessions": [ses(x) for x in t["sessions"]],
        }
        for t in sorted(
            body["trees"], key=lambda t: t["id"]
        )  # equal starts may sort either way after rounding
    ]


class _Near(float):
    def __eq__(self, other):
        return isinstance(other, float) and abs(float(self) - float(other)) <= 0.15

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = float.__hash__


def _render_abs(st, now):
    return _absolute(json.loads(rf._render(st, "7d", None, False, now)))


# ---- re-review 2: transactional refresh (R1), pruned rows (R3), one store for every viewer (R2) ----------------


def test_a_failed_refresh_leaves_no_rows_behind(tmp_path, monkeypatch):
    """R1: a refresh that fails after adding rows must not keep them: the retry equals a from-scratch build."""
    now = time.time()
    db_path = _bench_db(tmp_path, 2_000, now)
    store = rf._Store()
    _state(db_path, store, now)
    _insert(db_path, [_hrow("n1", now + 1, "A", "A-new"), _hrow("n2", now + 1.5, "B", "B-new")])
    real = rf._build_part
    calls = {"n": 0}

    def boom(tf):
        calls["n"] += 1
        raise RuntimeError("build failed")

    monkeypatch.setattr(rf, "_build_part", boom)
    before = store.view
    with pytest.raises(RuntimeError):
        rf._store_refresh(db_path, store, now + 2)
    assert calls["n"] >= 1
    assert store.view is before  # nothing half-applied was published
    monkeypatch.setattr(rf, "_build_part", real)
    st = _state(db_path, store, now + 3)
    fresh = _state(db_path, rf._Store(), now + 3)
    assert _render_abs(st, now + 3) == _render_abs(fresh, now + 3)
    held = store.held_rows()
    with sqlite3.connect(db_path) as db:
        (n,) = db.execute(
            "SELECT COUNT(*) FROM request_history WHERE finished_at >= ?", (now + 3 - 7 * 86400,)
        ).fetchone()
    assert held == n  # no duplicated row


def test_rows_the_pruner_deletes_inside_the_window_leave_within_one_refresh(tmp_path):
    """R3: the pruner's row cap deletes rows inside the 7 d window; its generation bump makes the store rebuild."""
    import asyncio

    from app.runtime import stats_pruner

    now = time.time()
    db_path = _bench_db(tmp_path, 4_000, now)
    store = rf._Store()
    _state(db_path, store, now)

    class S:
        pass

    settings = S()
    settings.db_path = db_path
    settings.request_history_retention_days = 30
    settings.request_history_max_rows = 3_000
    asyncio.run(stats_pruner.prune_once(settings))
    st = _state(db_path, store, now + 1)
    fresh = _state(db_path, rf._Store(), now + 1)
    assert _render_abs(st, now + 1) == _render_abs(fresh, now + 1)
    assert store.held_rows() == 3_000


def test_admin_and_key_holder_share_one_store(tmp_path):
    """R2: rows are held once: a key holder's state reads the same token part the admin's state composes."""
    now = time.time()
    db_path = _bench_db(tmp_path, 2_000, now)
    store = rf._Store()
    admin = _state(db_path, store, now)
    holder = rf._compose(store.view, "A", None, {})
    assert set(holder.parts) == {"A"} and holder.parts["A"] is admin.parts["A"]
    assert set(admin.parts) == {"A", "B", "C", "D"}
    assert store.held_rows() == 2_000


def test_over_budget_a_token_is_truncated_stable_and_never_rebuilt_per_request(
    tmp_path, monkeypatch
):
    """Re-review 3, N1: with the budget at 10k and key A holding 14k of 20k rows, the full build holds every token's
    fair share (B, C, D whole; A its newest 4k), never more than the budget, during or after the build; A's response
    says from when it is truncated; and polls with new rows load only the new rows (no rebuild per request), while
    the rows held stay within the budget plus its slack."""
    now = time.time()
    db_path = _bench_db(tmp_path, 20_000, now)
    monkeypatch.setattr(rf, "MAX_HELD_ROWS", 10_000)
    loads: list[int] = []
    real_load = rf._load

    def load(*a, **k):
        rows = real_load(*a, **k)
        loads.append(len(rows))
        return rows

    monkeypatch.setattr(rf, "_load", load)
    store = rf._Store()
    st = _state(db_path, store, now)
    # every row the full build loaded is held (one token at a time): the peak is the budget
    assert sum(loads) == store.held_rows() == 10_000
    assert len(store.toks["A"].tf) == 4_000 and store.toks["A"].floor is not None
    assert store.toks["B"].floor is None and len(store.toks["B"].tf) == 2_000
    assert st.truncated_before == store.toks["A"].floor
    assert rf._compose(store.view, "B", None, {}).truncated_before is None
    body = json.loads(rf._render(st, "7d", None, False, now))
    assert body["truncated_before"] == st.truncated_before
    n_full = len(loads)
    took = []
    for i in range(20):
        _insert(db_path, [_hrow(f"n{i}", now + 1 + i, "A", f"A-{14_000 // 150}")])
        t = time.perf_counter()
        _state(db_path, store, now + 2 + i)
        took.append(time.perf_counter() - t)
    # the median, not every run: a loaded CI box can stall one refresh (GC, scheduler); the load count below is the
    # deterministic guard that no refresh reloads the window
    assert sorted(took)[len(took) // 2] < 0.05
    assert len(loads) == n_full  # not one window or token load: only the new rows
    assert store.held_rows() <= 10_000 * (1 + rf.HELD_SLACK)


def test_rebalance_keeps_the_budget_as_new_rows_arrive(tmp_path, monkeypatch):
    """New rows past the budget and its slack move the largest token's floor up: held rows come back under budget."""
    now = time.time()
    db_path = _bench_db(tmp_path, 4_000, now)
    monkeypatch.setattr(rf, "MAX_HELD_ROWS", 4_000)
    store = rf._Store()
    _state(db_path, store, now)
    assert store.held_rows() == 4_000
    _insert(db_path, [_hrow(f"m{i}", now + 1 + i * 0.01, "A", "A-m") for i in range(600)])
    _state(db_path, store, now + 10)
    assert store.held_rows() <= 4_000
    assert store.toks["A"].floor is not None


def test_route_level_random_refreshes_equal_from_scratch(tmp_path):
    """Seeds x refreshes through the real path: 30 % of the commits late (finished up to 3 h before the newest), the
    window sliding, the pruner's row cap: after every refresh the incremental state renders what a from-scratch store
    renders."""
    import asyncio

    from app.runtime import stats_pruner

    for seed in (1, 2, 3):
        rnd = random.Random(seed)
        now = 1_800_000_000.0
        db_path = (
            _bench_db(tmp_path / f"s{seed}", 1_500, now)
            if (tmp_path / f"s{seed}").mkdir() is None
            else None
        )
        store = rf._Store()
        _state(db_path, store, now)
        k = 0
        for step in range(40):
            now += rnd.uniform(60, 4 * 3600)  # the window slides by up to 4 h a step
            rows = []
            for _ in range(rnd.randint(0, 25)):
                k += 1
                tok = rnd.choice("ABCD")
                late = rnd.random() < 0.3
                at = now - (rnd.uniform(0, 3 * 3600) if late else rnd.uniform(0, 30))
                rows.append(_hrow(f"x{seed}-{k}", at, tok, f"{tok}-r{rnd.randint(0, 30)}"))
            _insert(db_path, rows)
            if rnd.random() < 0.1:

                class S:
                    pass

                st_ = S()
                st_.db_path = db_path
                st_.request_history_retention_days = 30
                st_.request_history_max_rows = rnd.randint(800, 1_400)
                asyncio.run(stats_pruner.prune_once(st_))
            st = _state(db_path, store, now)
            fresh = _state(db_path, rf._Store(), now)
            assert _render_abs(st, now) == _render_abs(fresh, now), f"seed {seed}, step {step}"


def test_compose_never_sees_a_half_applied_refresh(tmp_path):
    """Re-review 3, N2: 4 compose threads against 2 refresh threads (incremental refreshes and full rebuilds forced by
    the pruner's generation), with no new data: every composed admin state has exactly the trees of a from-scratch
    build, and nothing raises."""
    import threading

    from app.stats import request_history

    now = time.time()
    db_path = _bench_db(tmp_path, 6_000, now)
    expect = len(_state(db_path, rf._Store(), now).ids)
    store = rf._Store()
    _state(db_path, store, now)
    stop = threading.Event()
    sizes: list[int] = []
    errors: list[str] = []
    refreshes = [0]
    lock = (
        threading.Lock()
    )  # the cache's lock: one refresh at a time (as ForestCache._refresh_store)

    def composer():
        while not stop.is_set():
            try:
                v = store.view
                sizes.append(len(rf._compose(v, None, None, {}).ids))
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))
            time.sleep(0.0005)  # leave the refreshes some of the GIL

    def refresher(k):
        t = now
        for i in range(6):
            if (i + k) % 2:
                request_history.bump_prune_generation()  # the next refresh is a full rebuild
            t += 1
            with lock:
                rf._store_refresh(db_path, store, t)
                refreshes[0] += 1

    cs = [threading.Thread(target=composer) for _ in range(4)]
    rs = [threading.Thread(target=refresher, args=(k,)) for k in range(2)]
    for th in cs + rs:
        th.start()
    for th in rs:
        th.join()
    stop.set()
    for th in cs:
        th.join()
    print(
        f"N2: {len(sizes)} composes during {refreshes[0]} refreshes; min trees {min(sizes)}, expected {expect}"
    )
    assert errors == []
    assert refreshes[0] == 12 and len(sizes) > 100
    assert min(sizes) == max(sizes) == expect


def test_the_pruner_bumps_the_generation_only_for_rows_inside_7_days(tmp_path):
    """Re-review 3, N5: a row-cap prune that deletes only rows older than 7 d leaves the forest alone."""
    import asyncio

    from app.runtime import stats_pruner
    from app.stats import request_history

    now = time.time()
    db_path = _bench_db(tmp_path, 1_000, now)
    _insert(db_path, [_hrow(f"old{i}", now - 10 * 86400 + i, "A", "A-old") for i in range(100)])

    class S:
        pass

    st = S()
    st.db_path = db_path
    st.request_history_retention_days = 30
    st.request_history_max_rows = 1_000  # drops exactly the 100 old rows
    g = request_history.prune_generation()
    asyncio.run(stats_pruner.prune_once(st))
    assert request_history.prune_generation() == g
    st.request_history_max_rows = 900  # now 100 in-window rows go too
    asyncio.run(stats_pruner.prune_once(st))
    assert request_history.prune_generation() == g + 1
