"""Grouped sessions list: views, histogram, filters, sort, paging (sessions redesign, task 7)."""

from __future__ import annotations

from typing import Any

import pytest

from app.sessions.listing import apply_list
from app.sessions.pauses import PauseMark

T0 = 1_000_000.0 * 3600  # an hour boundary


def item(
    sid,
    *,
    tok="t1",
    name=None,
    start=T0 + 100,
    dur=600.0,
    prompt=1000,
    cached=0.9,
    dw=0.0,
    measured=True,
    known=True,
    active=False,
    evicted=0,
    lost=None,
    where="local",
    harness="claude_code",
    marks=(),
):
    c = round(prompt * cached)
    return {
        "session_id": "s_" + sid,
        "token_id": tok,
        "token_name": name or ("name-" + tok),
        "models": ["qwen"],
        "harness": harness,
        "where": where,
        "started_at": start,
        "last_at": start + dur,
        "duration_s": dur,
        "requests": 3,
        "prompt_tokens": prompt,
        "cached_share": cached,
        "cache_share": cached if known else None,
        "_ck": (c, prompt) if known else (0, 0),
        "anatomy": {
            "cached": c if known else 0,
            "dead": 0,
            "recomputed": 0,
            "new": prompt - c if known else 0,
            "unknown": 0 if known else prompt,
        },
        "dead_weight_share": dw if measured else None,
        "dead_weight_measured": measured,
        "cache_known": known,
        "active": active,
        "evicted_resumes": evicted,
        "evicted_lost_s": lost,
        "saved_vs_vanilla_s": 1.0,
        "_marks": list(marks),
    }


def apply(items: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    args: dict[str, Any] = dict(
        now=T0 + 7200,
        range_start=T0,
        view="all",
        where="all",
        q=None,
        model=None,
        since=None,
        until=None,
        group="key",
        group_key=None,
        sort="newest",
        offset=0,
        limit=50,
        per_group=3,
        tz_offset_min=0,
    )
    args.update(kw)
    return apply_list(items, **args)


def test_group_by_key_totals_and_first_page():
    out = apply(
        items=[item(f"s{i}", tok="t1", start=T0 + i) for i in range(5)] + [item("x", tok="t2")]
    )
    g = {g["key"]: g for g in out["groups"]}
    assert g["t1"]["sessions"] == 5 and len(g["t1"]["items"]) == 3 and g["t1"]["more"] == 2
    assert g["t1"]["prompt_tokens"] == 5 * 1000 and g["t1"]["label"] == "name-t1"
    assert g["t1"]["where"] == {"local": 5, "relayed": 0, "hybrid": 0}


def test_group_key_pages_within_the_group():
    out = apply(items=[item(f"s{i}", start=T0 + i) for i in range(5)], group_key="t1", offset=3)
    (g,) = out["groups"]
    assert len(g["items"]) == 2 and g["more"] == 0


def test_group_by_harness_and_day_with_tz():
    late = T0 - (T0 % 86400) + 23.5 * 3600  # 23:30 UTC
    out = apply(
        items=[item("a", start=late)],
        group="day",
        tz_offset_min=60,
        range_start=late - 10,
        now=late + 10,
    )
    import datetime as dt

    want = (dt.datetime.fromtimestamp(late, dt.UTC) + dt.timedelta(minutes=60)).strftime("%Y-%m-%d")
    assert out["groups"][0]["key"] == want
    out = apply(items=[item("a", harness="codex"), item("b", harness="codex")], group="harness")
    assert [g["key"] for g in out["groups"]] == ["codex"]


def test_views_count_before_the_view_filter():
    out = apply(
        items=[
            item("a", dw=0.25),
            item("b", dw=0.05, active=True),
            item("c", cached=0.5),
            item("d", dur=4000),
            item("e", evicted=2),
        ],
        view="dead_weight",
    )
    assert out["views"] == {
        "all": 5,
        "running": 1,
        "biggest": 5,
        "dead_weight": 1,
        "cache_trouble": 1,
        "long_running": 1,
        "evicted": 1,
    }
    assert out["total"] == 1


def test_cache_trouble_skips_unknown_cache():
    out = apply(items=[item("a", known=False)], view="cache_trouble")
    assert out["views"]["cache_trouble"] == 0 and out["total"] == 0


def test_dead_weight_view_skips_unmeasured():
    assert apply(items=[item("a", measured=False)])["views"]["dead_weight"] == 0


def test_where_hybrid_and_search():
    items = [
        item("aa", where="hybrid"),
        item("bb", where="local", name="alice"),
        item("cc", where="relayed"),
    ]
    assert apply(items=items, where="hybrid")["total"] == 1
    assert apply(items=items, q="ALI")["total"] == 1
    assert apply(items=items, q="s_cc")["total"] == 1


def test_histogram_bins_and_over_dw_ignore_the_brush():
    out = apply(
        items=[item("a", start=T0 + 10, dw=0.2), item("b", start=T0 + 3700)],
        since=T0 + 3600,
        until=T0 + 7200,
    )
    bins = out["histogram"]["bins"]
    assert [b["sessions"] for b in bins[:2]] == [1, 1] and bins[0]["over_dw"] == 1
    assert out["total"] == 1 and out["histogram"]["bin_s"] == 3600


def test_empty_brush_is_empty_not_an_error():
    out = apply(items=[item("a", start=T0 + 10)], since=T0 + 5000, until=T0 + 5000)
    assert out["total"] == 0 and out["groups"] == []
    out = apply(items=[item("a", start=T0 + 10)], since=T0 - 99999, until=T0 + 99999)  # clamped
    assert out["total"] == 1


def test_sort_by_evicted_lost_puts_none_last():
    out = apply(
        items=[item("a", lost=None), item("b", lost=5.0), item("c", lost=9.0)],
        group="none",
        sort="evicted_lost",
    )
    assert [s["session_id"] for s in out["sessions"]] == ["s_c", "s_b", "s_a"] and out[
        "groups"
    ] is None


def test_pause_summary_sums_filtered_items_and_marks_are_stripped():
    m = PauseMark(600.0, True, True, False, 5000, 5.0)
    out = apply(items=[item("a", marks=[m]), item("b", marks=[m], where="relayed")], where="local")
    assert out["pause_summary"]["evicted"] == 1
    assert all("_marks" not in s for g in out["groups"] for s in g["items"])


def test_none_group_pages_by_offset():
    out = apply(
        items=[item(f"s{i}", start=T0 + i) for i in range(5)], group="none", offset=2, limit=2
    )
    assert [s["session_id"] for s in out["sessions"]] == ["s_s2", "s_s1"]


@pytest.mark.parametrize("view", ["biggest"])
def test_biggest_is_top_by_prompt(view):
    out = apply(items=[item("a", prompt=10), item("b", prompt=99)], view=view, group="none")
    assert out["sessions"][0]["session_id"] == "s_b"


def test_cache_trouble_uses_the_real_share_not_the_dead_weight_scaled_anatomy():
    # 90% from cache, but the anatomy's cached part is cut by heavy dead weight
    it = item("a", cached=0.9)
    it["anatomy"] = {"cached": 540, "dead": 400, "recomputed": 0, "new": 60, "unknown": 0}
    out = apply(items=[it, item("b", cached=0.5)], view="cache_trouble", group="none")
    assert out["views"]["cache_trouble"] == 1 and [s["session_id"] for s in out["sessions"]] == [
        "s_b"
    ]


def test_group_cached_share_is_the_weighted_real_share_and_unknown_when_no_cache_is_known():
    out = apply(items=[item("a", cached=0.9), item("b", cached=0.5, prompt=3000)], group="key")
    (g,) = out["groups"]
    assert g["cached_share"] == pytest.approx((900 + 1500) / 4000)
    out = apply(items=[item("c", known=False)], group="key")
    assert out["groups"][0]["cached_share"] is None
    assert "_ck" not in out["groups"][0]["items"][0]
