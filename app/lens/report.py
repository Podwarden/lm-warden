"""The downloadable session report (analysis Task 4.4, spec 12.3).

``report_data`` turns a scorecard (``service.scorecard``'s answer) into a document that can be
passed on: scores with their ranges and confidence, the top 3 fixes, every finding, the cost,
the ``score_version``. It uses report-local labels (``T4`` for turn 4, "system prompt line 3",
"tool #5") and carries NO key name, key id, session id, request id, kvsig id or client address,
and never prompt or reply text (A-2, A-10). Tool names appear only when the caller asks
(``tool_names``). Two reports of different sessions therefore share no id. ``report_html`` makes
one self-contained page: inline CSS, no script, no link, nothing fetched. ``report.schema.json``
(JSON Schema 2020-12) describes the JSON so CI can diff two reports.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from fastapi import Response

from app.lens.catalogue import RULES

SCHEMA_PATH = Path(__file__).with_name("report.schema.json")
REPORT_VERSION = 1
_PLACEHOLDER = re.compile(r"\{[^{}]*\}")
_SCORE_KEYS = ("overall", "E", "C", "R")
_LAYER_NAMES = {"E": "Efficiency", "C": "Construction", "R": "Response", "overall": "Overall"}


def schema() -> dict[str, Any]:
    out: dict[str, Any] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return out


class _Labels:
    """Report-local tool labels: ``tool #1`` in order of first appearance."""

    def __init__(self, names: Mapping[str, str] | None) -> None:
        self._names = names
        self._ix: dict[str, int] = {}

    def tool(self, h: str | None) -> tuple[str | None, str | None]:
        if not h:
            return None, None
        n = self._ix.setdefault(h, len(self._ix) + 1)
        name = (self._names or {}).get(h) if self._names is not None else None
        return f"tool #{n}", name


def _message(rule: str, counts: Mapping[str, int], tool: str | None, turns: list[int]) -> str:
    """The rule's message with what the report can say filled in, the rest as an ellipsis."""
    fill = {
        "n": counts.get("line"),
        "m": counts.get("after"),
        "k": counts.get("k", len(turns) or None),
        "i": counts.get("msg"),
        "name/#k": tool,
        "name": tool,
    }
    return _PLACEHOLDER.sub(
        lambda m: str(fill.get(m.group(0)[1:-1])) if fill.get(m.group(0)[1:-1]) else "…",
        RULES[rule].message,
    )


def _finding(f: Mapping[str, Any], labels: _Labels) -> dict[str, Any]:
    rule = str(f["rule"])
    r = RULES.get(rule)
    tool, name = labels.tool(f.get("tool"))
    turns = [int(t) for t in f.get("turns", [])]
    counts = {str(k): int(v) for k, v in (f.get("counts") or {}).items()}
    out: dict[str, Any] = {
        "rule": rule,
        "layer": r.layer if r else str(f.get("layer", "")),
        "message": _message(rule, counts, tool, turns) if r else rule,
        "fix": r.fix if r else "",
        "leanprompt": r.lp if r else None,
        "turns": [f"T{t}" for t in turns],
        "counts": counts,
        "confidence": f.get("confidence"),
        "saving": {k: v for k, v in (f.get("saving") or {}).items() if v is not None},
        "muted": bool(f.get("muted")),
    }
    if "line" in counts:
        out["where"] = f"system prompt line {counts['line']}"
    elif tool:
        out["where"] = tool
    if tool:
        out["tool"] = tool
        if name:
            out["tool_name"] = name
    return out


def _score(js: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"scored": bool(js.get("scored")), "turns": int(js.get("turns", 0))}
    if js.get("scored"):
        for k in ("value", "lo", "hi", "confidence"):
            out[k] = js.get(k)
        inputs = js.get("inputs")
        if isinstance(inputs, dict) and "with_marks" in inputs:
            out["with_marks"] = inputs["with_marks"]
        if js.get("excluded_rules"):
            out["excluded_rules"] = list(js["excluded_rules"])
    else:
        out["reason"] = js.get("reason")
    return out


def report_data(
    card: Mapping[str, Any],
    *,
    tool_names: Mapping[str, str] | None = None,
    view: Literal["key", "admin"] = "key",
) -> dict[str, Any]:
    """The report's JSON. ``tool_names`` (hash -> name) is passed only when the caller ticked
    "include tool names"; without it a tool is "tool #n" only."""
    labels = _Labels(tool_names)
    s = card["session"]
    cost = card["cost"]
    findings = [_finding(f, labels) for f in card["findings"]]
    out: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "view": view,
        "session": {
            "harness": s.get("harness"),
            "model": s.get("model"),
            "route": s.get("route"),
            "turns": int(s.get("turns", 0)),
            "duration_s": round(float(s.get("duration_s", 0.0)), 1),
            "prompt_version": s.get("version"),
        },
        "score_version": int(card["score_version"]),
        "weights": dict(card["weights"]),
        "scores": {k: _score(card["scores"][k]) for k in _SCORE_KEYS if k in card["scores"]},
        "not_scored": {k: v for k, v in card.get("not_scored", {}).items() if v},
        "top": [_finding(f, labels) for f in card["top"]],
        "notice": _finding(card["notice"], labels) if card.get("notice") else None,
        "findings": findings,
        "turns": [
            {
                "turn": f"T{t['n']}",
                "reused": t["reused"],
                "recomputed": t["recomputed"],
                "new": t["new"],
                "markers": list(t.get("markers", [])),
            }
            for t in card["turns"]
        ],
        "cost": {
            "prompt_tokens": cost["prompt_tokens"],
            "reused_tokens": cost["reused_tokens"],
            "recomputed_tokens": cost["recomputed_tokens"],
            "fixable_lost_tokens": cost["fixable_lost_tokens"],
            "price_card": (
                {
                    "provider": cost["card"]["provider"],
                    "model": cost["card"]["model"],
                    "currency": cost["card"]["currency"],
                    "effective_from": cost["card"]["effective_from"],
                    "actual_input": cost.get("actual_input"),
                    "saving": cost.get("saving"),
                    "saving_pct": cost.get("saving_pct"),
                }
                if cost.get("card")
                else None
            ),
        },
        # Rubric scores only, never the review's text (A-8, A-10).
        "review": _review(card.get("review")),
    }
    return out


def _review(r: Any) -> dict[str, Any] | None:
    if not isinstance(r, dict):
        return None
    out: dict[str, Any] = {}
    for k in ("score", "scale", "rubric"):
        if k in r and isinstance(r[k], int | float | dict):
            out[k] = r[k]
    return out or None


# -- HTML -------------------------------------------------------------------------------


def _e(v: Any) -> str:
    return html.escape("" if v is None else str(v))


_CSS = (
    ":root{color-scheme:light dark;--fg:#1b1f24;--bg:#fff;--mut:#59636e;--line:#d0d7de;--ac:#0b5cad}"
    "@media(prefers-color-scheme:dark){:root{--fg:#e6edf3;--bg:#0d1117;--mut:#8b949e;--line:#30363d;--ac:#58a6ff}}"
    "body{font:15px/1.5 system-ui,sans-serif;color:var(--fg);background:var(--bg);margin:0 auto;"
    "max-width:60rem;padding:1.25rem 1rem}h1{font-size:1.4rem}h2{font-size:1.1rem;margin-top:1.6rem}"
    "table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid var(--line);"
    "padding:.35rem .5rem;text-align:left;vertical-align:top}th{color:var(--mut);font-weight:600}"
    ".mut{color:var(--mut)}.num{text-align:right;font-variant-numeric:tabular-nums}"
    ".big{font-size:1.6rem;font-weight:700}.tag{border:1px solid var(--line);border-radius:4px;"
    "padding:0 .35rem;font-size:.8rem;color:var(--mut)}.bar{display:flex;height:.7rem;min-width:8rem}"
    ".bar i{display:block;height:100%}.r{background:#2da44e}.c{background:#cf222e}.n{background:#8b949e}"
    "@media(max-width:36rem){.wide{display:none}}"
)


def _scores_html(d: Mapping[str, Any]) -> str:
    rows = []
    for k in _SCORE_KEYS:
        s = d["scores"].get(k)
        if s is None:
            continue
        if s["scored"]:
            rng = f"{s['lo']}-{s['hi']}" if s.get("lo") is not None else ""
            cell = (
                f"<span class=big>{_e(s['value'])}</span> "
                f"<span class=mut>range {_e(rng)} · {_e(s['confidence'])} confidence · "
                f"{_e(s['turns'])} turns</span>"
            )
            if "with_marks" in s:
                cell += f" <span class=mut>(with won't-fix findings counted: {_e(s['with_marks'])})</span>"
        else:
            cell = f"<span class=mut>not scored: {_e(s.get('reason'))}</span>"
        rows.append(f"<tr><th>{_e(_LAYER_NAMES[k])}</th><td>{cell}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def _finding_html(f: Mapping[str, Any]) -> str:
    where = f" · {_e(f['where'])}" if f.get("where") else ""
    name = f" ({_e(f['tool_name'])})" if f.get("tool_name") else ""
    lp = f" · leanprompt {_e(f['leanprompt'])}" if f.get("leanprompt") else ""
    turns = ", ".join(f["turns"][:12]) + (" …" if len(f["turns"]) > 12 else "")
    muted = " <span class=tag>won't fix</span>" if f.get("muted") else ""
    return (
        f"<tr><td><b>{_e(f['rule'])}</b>{muted}</td><td>{_e(f['message'])}{name}"
        f"<div class=mut>Fix: {_e(f['fix'])}</div>"
        f"<div class=mut>Turns {_e(turns)}{where}{lp} · {_e(f.get('confidence'))} confidence</div></td></tr>"
    )


def report_html(data: Mapping[str, Any]) -> str:
    """One self-contained page: inline CSS, no script, no link, no external fetch."""
    s = data["session"]
    admin = (
        "<p class=mut>admin view</p>" if data.get("view") == "admin" else ""
    )  # first line of the body only when the admin chose it
    top = (
        "".join(_finding_html(f) for f in data["top"])
        or "<tr><td class=mut>No fixes found.</td></tr>"
    )
    allf = "".join(_finding_html(f) for f in data["findings"]) or (
        "<tr><td class=mut>No findings.</td></tr>"
    )
    strip = "".join(
        f"<tr><td>{_e(t['turn'])}</td><td class='num'>{t['reused']:,}</td>"
        f"<td class='num'>{t['recomputed']:,}</td><td class='num'>{t['new']:,}</td>"
        f"<td class=wide>{_e(', '.join(t['markers']))}</td></tr>"
        for t in data["turns"]
    )
    cost = data["cost"]
    card = cost.get("price_card")
    money = (
        f"<p>Price card: {_e(card['provider'])} {_e(card['model'])}, from {_e(card['effective_from'])}. "
        f"Input cost {_e(card.get('actual_input'))} {_e(card['currency'])}; "
        f"avoidable {_e(card.get('saving'))} {_e(card['currency'])}.</p>"
        if card
        else "<p class=mut>No price card for this model: savings are in tokens.</p>"
    )
    notice = (
        f"<h2>Notice</h2><table>{_finding_html(data['notice'])}</table>"
        if data.get("notice")
        else ""
    )
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>Prompt Lens session report</title>"
        f"<style>{_CSS}</style></head><body>{admin}"
        "<h1>Prompt Lens session report</h1>"
        f"<p class=mut>{_e(s['harness'])} · {_e(s['model'])} · {_e(s['route'])} · "
        f"{_e(s['turns'])} turns · {_e(s['duration_s'])} s"
        + (f" · prompt version {_e(s['prompt_version'])}" if s.get("prompt_version") else "")
        + f" · score version {_e(data['score_version'])}</p>"
        f"<h2>Scores</h2>{_scores_html(data)}{notice}"
        f"<h2>Top fixes</h2><table>{top}</table>"
        f"<h2>All findings</h2><table>{allf}</table>"
        "<h2>Turns</h2><table><tr><th>Turn</th><th class=num>Reused</th>"
        "<th class=num>Recomputed</th><th class=num>New</th><th class=wide>Markers</th></tr>"
        f"{strip}</table>"
        f"<h2>Cost</h2><p>{cost['prompt_tokens']:,} prompt tokens: {cost['reused_tokens']:,} reused, "
        f"{cost['recomputed_tokens']:,} recomputed, {cost['fixable_lost_tokens']:,} of them lost to "
        f"causes you control.</p>{money}"
        "<p class=mut>Made by LM Warden Prompt Lens from numbers and codes only: no prompt, reply "
        "or tool text, no key or session identifier.</p></body></html>"
    )


def response(
    card: Mapping[str, Any], *, fmt: Literal["json", "html"], tool_names: bool, admin: bool
) -> Response:
    """The download: the report as an attachment, in the format asked for."""
    data = report_data(
        card, tool_names=card["names"] if tool_names else None, view="admin" if admin else "key"
    )
    if fmt == "html":
        return Response(
            report_html(data),
            media_type="text/html; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="prompt-lens-report.html"'},
        )
    return Response(
        json.dumps(data, indent=2) + "\n",
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="prompt-lens-report.json"'},
    )
