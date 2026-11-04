"""The grouped sessions list: filters, ready views with counts, an hourly activity
histogram, grouping (key / harness / day / none), sort and paging, and the
cross-session pause summary. Pure: it runs on the cached scan summary, so a
filter change never rescans."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable
from typing import Any

from app.sessions.costs import Anatomy, add_anatomy, anatomy_json, share_of
from app.sessions.pauses import summarize_pauses

GROUPS = ("key", "harness", "day", "none")
VIEWS = ("all", "running", "biggest", "dead_weight", "cache_trouble", "long_running", "evicted")
WHERES = ("all", "local", "relayed", "hybrid")
SORTS = ("newest", "prompt_tokens", "dead_weight", "duration", "evicted_lost")
GROUP_PAGE = 3
GROUP_MORE = 50
BIGGEST_N = 50
HIST_BIN_S = 3600
DW_VIEW = 0.20
DW_HIST = 0.10
CACHE_TROUBLE = 0.60
LONG_S = 3600.0

_SORT_KEY: dict[str, Callable[[dict[str, Any]], float | None]] = {
    "newest": lambda i: i["started_at"],
    "prompt_tokens": lambda i: i.get("prompt_tokens"),
    "dead_weight": lambda i: i.get("dead_weight_share"),
    "duration": lambda i: i.get("duration_s"),
    "evicted_lost": lambda i: i.get("evicted_lost_s"),
}


def in_view(item: dict[str, Any], view: str, *, biggest_ids: set[str]) -> bool:
    if view == "all":
        return True
    if view == "running":
        return bool(item.get("active"))
    if view == "biggest":
        return item["session_id"] in biggest_ids
    if view == "dead_weight":
        dw = item.get("dead_weight_share")
        return dw is not None and dw > DW_VIEW
    if view == "cache_trouble":
        # the real share served from cache, not the anatomy's (dead weight taken out)
        share = item.get("cache_share")
        return bool(item.get("cache_known")) and share is not None and share < CACHE_TROUBLE
    if view == "long_running":
        return (item.get("duration_s") or 0) > LONG_S
    if view == "evicted":
        return (item.get("evicted_resumes") or 0) > 0
    return False


def _sorted(items: list[dict[str, Any]], sort: str) -> list[dict[str, Any]]:
    key = _SORT_KEY[sort]
    known = [i for i in items if key(i) is not None]
    unknown = [i for i in items if key(i) is None]
    known.sort(key=lambda i: (key(i), i["session_id"]), reverse=True)
    return known + unknown


def _group_key(item: dict[str, Any], group: str, tz_offset_min: int) -> tuple[str, str]:
    if group == "key":
        return str(item.get("token_id") or "-"), str(
            item.get("token_name") or item.get("token_id") or "-"
        )
    if group == "harness":
        h = str(item.get("harness") or "other")
        return h, h
    day = (
        dt.datetime.fromtimestamp(item["started_at"], dt.UTC) + dt.timedelta(minutes=tz_offset_min)
    ).strftime("%Y-%m-%d")
    return day, day


def _sum_opt(vals: Iterable[float | None]) -> float | None:
    out: float | None = None
    for v in vals:
        if v is not None:
            out = (out or 0.0) + v
    return out


def _public(item: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in item.items() if not k.startswith("_")}


def _group(
    key: str, label: str, members: list[dict[str, Any]], items: list[dict[str, Any]], more: int
) -> dict[str, Any]:
    prompt = sum(m.get("prompt_tokens") or 0 for m in members)
    ck_c = sum((m.get("_ck") or (0, 0))[0] for m in members)
    ck_p = sum((m.get("_ck") or (0, 0))[1] for m in members)
    an = Anatomy(0.0, 0.0, 0.0, 0.0, 0.0)
    for m in members:
        a = m.get("anatomy") or {}
        an = add_anatomy(
            an,
            Anatomy(
                a.get("cached", 0),
                a.get("dead", 0),
                a.get("recomputed", 0),
                a.get("new", 0),
                a.get("unknown", 0),
            ),
        )
    measured = [
        m
        for m in members
        if m.get("dead_weight_measured") and m.get("dead_weight_share") is not None
    ]
    mp = sum(m.get("prompt_tokens") or 0 for m in measured)
    dw_share = (
        (sum((m["dead_weight_share"] or 0) * (m.get("prompt_tokens") or 0) for m in measured) / mp)
        if mp
        else None
    )
    where = {"local": 0, "relayed": 0, "hybrid": 0}
    for m in members:
        w = m.get("where") or "local"
        where[w] = where.get(w, 0) + 1
    harnesses = sorted({str(m.get("harness") or "other") for m in members})
    return {
        "key": key,
        "label": label,
        "harnesses": harnesses,
        "where": where,
        "sessions": len(members),
        "duration_s": sum(m.get("duration_s") or 0 for m in members),
        "prompt_tokens": prompt,
        "cached_share": share_of((ck_c, ck_p)),
        "anatomy": anatomy_json(an),
        "dead_weight_share": dw_share,
        "saved_vs_vanilla_s": _sum_opt(m.get("saved_vs_vanilla_s") for m in members),
        "evicted_resumes": sum(m.get("evicted_resumes") or 0 for m in members),
        "evicted_lost_s": _sum_opt(m.get("evicted_lost_s") for m in members),
        "items": [_public(i) for i in items],
        "more": more,
    }


def apply_list(
    items: list[dict[str, Any]],
    *,
    now: float,
    range_start: float,
    view: str,
    where: str,
    q: str | None,
    model: str | None,
    since: float | None,
    until: float | None,
    group: str,
    group_key: str | None,
    sort: str,
    offset: int,
    limit: int,
    per_group: int,
    tz_offset_min: int,
) -> dict[str, Any]:
    # 1. filters (the brush comes after the histogram)
    base = items
    if model:
        base = [i for i in base if model in (i.get("models") or [])]
    if where != "all":
        base = [i for i in base if i.get("where") == where]
    if q:
        ql = q.lower()
        base = [
            i
            for i in base
            if i["session_id"].startswith(q) or ql in str(i.get("token_name") or "").lower()
        ]
    n_bins = max(1, int((now - range_start) // HIST_BIN_S) + 1)
    bins = [{"t": range_start + k * HIST_BIN_S, "sessions": 0, "over_dw": 0} for k in range(n_bins)]
    for i in base:
        k = int((i["started_at"] - range_start) // HIST_BIN_S)
        if 0 <= k < n_bins:
            bins[k]["sessions"] += 1
            dw = i.get("dead_weight_share")
            if dw is not None and dw > DW_HIST:
                bins[k]["over_dw"] += 1
    lo = max(since if since is not None else range_start, range_start)
    hi = min(until if until is not None else now, now)
    if since is not None or until is not None:
        base = [i for i in base if lo <= i["started_at"] <= hi]
    # 2. views, counted before the view filter
    biggest = {
        i["session_id"]
        for i in sorted(base, key=lambda i: -(i.get("prompt_tokens") or 0))[:BIGGEST_N]
    }
    views = {v: sum(1 for i in base if in_view(i, v, biggest_ids=biggest)) for v in VIEWS}
    # 3. view, then sort
    chosen = _sorted([i for i in base if in_view(i, view, biggest_ids=biggest)], sort)
    pause_summary = summarize_pauses(m for i in chosen for m in i.get("_marks") or ())
    out: dict[str, Any] = {
        "total": len(chosen),
        "views": views,
        "histogram": {"bin_s": HIST_BIN_S, "start": range_start, "bins": bins},
        "pause_summary": pause_summary,
    }
    # 4. group
    if group == "none":
        out["groups"] = None
        out["sessions"] = [_public(i) for i in chosen[offset : offset + limit]]
        return out
    grouped: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    for i in chosen:
        key, label = _group_key(i, group, tz_offset_min)
        grouped.setdefault(key, (label, []))[1].append(i)
    groups = []
    for key, (label, members) in grouped.items():  # dict keeps the sorted order of first members
        if group_key is not None and key != group_key:
            continue
        if group_key is not None:
            page = members[offset : offset + limit]
            more = max(0, len(members) - offset - len(page))
        else:
            page = members[:per_group]
            more = max(0, len(members) - len(page))
        groups.append(_group(key, label, members, page, more))
    out["groups"] = groups
    out["sessions"] = []
    return out
