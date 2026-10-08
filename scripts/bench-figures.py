#!/usr/bin/env python3
"""Router-hero figure generator and slot gate (stdlib only).

Reads assets/bench/dp-affinity-ab.json (the A/B data file, `null` until measured) and

  --check    list every `null` path in the data file; exit 1 while any remain.
  --render   write the figures: placeholders ("measurement pending") while any value is
             null (exit 1), the real charts and the T1/T2/T3 tables once it is complete.
             D2 (the mechanism drawing) and T2 (the issue #286 table) need no A/B data
             and are always written.
  --fill     render, then substitute `{{SLOT:<path>}}` in the shipped surfaces (README.md,
             documents/ROUTING.md, app/landing/*); refuses, touching nothing, while any
             value is null or any slot is unknown.

Filling the numbers after the live A/B is one command. Pass the two load-test run
directories (agent_ramp.py: summary.json + dp_snapshots.jsonl + requests.jsonl) and the
router_smoke JSON; they are ingested into the data file, then checked, rendered, filled:

  python scripts/bench-figures.py --fill --off RUNS/affinity-off --on RUNS/affinity-on \\
      --smoke RUNS/smoke.json --set box.model_hf=ORG/MODEL --set box.served_name=NAME \\
      --set box.quant=Q --set box.lm_warden=vYYYY.MM.DD.N --set box.max_num_seqs=N \\
      --set box.max_model_len=N

`--set path=value` (JSON or bare string) fills what no run directory carries (the box
spec plate, router_smoke cases that router_smoke.py does not measure). Never invent a
number: anything not measured stays null and blocks --render/--fill.

Mapping from the harness: tasks_per_h = sessions_per_h; gpus_busy_avg = replicas_busy
(replicas with requests_running > 0, sampled each poll); prefix_hit_pct = 100 * sum of
per-replica hit deltas / sum of query deltas over the step (raw counters, not a mean of
rates); tokens_per_min = 60 * output_tok_per_s; sticky/spilled_pct = share of
sticky+spilled routing decisions in the step. vLLM's lifetime per-replica counters are
not used: the arms ran back to back on one engine without a restart, so the second
arm's lifetime counters would include the first arm's traffic.

Series colours (dataviz validate_palette.js, run 2026-10-04, every check PASS): affinity
on is the console's amber, vLLM's balancer a blue. Light `#2A78D6,#A86400` on #EFECE7 and
on the plate #E3DFD8 (CVD dE 26.7, normal-vision dE 28.8, both >= 3:1); dark
`#3987E5,#C98500` on #1C1409 and #2C251A (CVD dE 27.4, normal-vision dE 30.7, both
>= 3:1). The landing page carries the same values as --series-on / --series-off. Text
stays in the ink tokens; the series are also told apart by marker shape (circle, square),
direct end labels and a legend. Every landing chart is drawn twice, wide and narrow
(stacked, for a phone column), so its text never shrinks below about 12px on screen.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_DATA = "assets/bench/dp-affinity-ab.json"
SURFACES = (
    "README.md",
    "documents/ROUTING.md",
    "app/landing/landing.html",
    "app/landing/faq.json",
    "app/landing/llms.txt",
    "app/landing/llms-full.txt",
)
SLOT_RE = re.compile(r"\{\{SLOT:([^}]+)\}\}")
ARROW = " → "
STATIC_FIELDS = {"route", "status"}  # fixed by the schema; a smoke case must agree


class BenchError(Exception):
    """Bad input: exit 2."""


# --------------------------------------------------------------------------- data


def find_nulls(obj: Any, prefix: str = "") -> list[str]:
    out: list[str] = []
    if obj is None:
        return [prefix]
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += find_nulls(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += find_nulls(v, f"{prefix}[{i}]")
    return out


def _tokens(path: str) -> list[str | int]:
    toks: list[str | int] = []
    for part in path.split("."):
        m = re.fullmatch(r"([^\[\]]*)((?:\[\d+\])*)", part)
        if not m or (not m.group(1) and not m.group(2)):
            raise BenchError(f"bad slot path {path!r}")
        if m.group(1):
            toks.append(m.group(1))
        toks += [int(i) for i in re.findall(r"\[(\d+)\]", m.group(2))]
    return toks


def peak_row(data: dict) -> dict:
    rows = data["ramp"]
    vals = [r["on"]["tasks_per_h"] for r in rows]
    if not rows or any(v is None for v in vals):
        raise BenchError("slot peak.*: ramp[].on.tasks_per_h has nulls")
    return rows[vals.index(max(vals))]


def fmt(v: Any) -> str:
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{v:g}"
    return str(v)


def resolve_slot(data: dict, path: str) -> str:
    path = path.strip()
    if path == "peak.workers":
        cur: Any = peak_row(data)["workers"]
    elif path.startswith("peak."):
        arm, _, field = path[5:].partition(".")
        if arm not in ("on", "off") or not field:
            raise BenchError(f"unknown slot {{{{SLOT:{path}}}}}")
        cur = peak_row(data)[arm].get(field)
    else:
        cur = data
        for t in _tokens(path):
            try:
                cur = cur[t]  # type: ignore[index]
            except (KeyError, IndexError, TypeError):
                raise BenchError(f"unknown slot {{{{SLOT:{path}}}}}") from None
    if cur is None:
        raise BenchError(f"slot {{{{SLOT:{path}}}}} is null in the data file")
    if isinstance(cur, dict | list):
        raise BenchError(f"slot {{{{SLOT:{path}}}}} is not a scalar")
    return fmt(cur)


# ------------------------------------------------------------------------- ingest


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise BenchError(f"cannot read {p}: {e}") from None


def _arm_row(step: dict, label: str, on: bool) -> dict:
    hits, qs = step.get("prefix_hits_by_replica"), step.get("prefix_queries_by_replica")
    if not hits or not qs:
        raise BenchError(
            f"{label} step {step['concurrency']}: no raw prefix_hits_by_replica/"
            "prefix_queries_by_replica (re-run with the patched agent_ramp.py and "
            "--admin-token-file/--model-id)"
        )
    row: dict[str, Any] = {
        "tasks_per_h": round(step["sessions_per_h"]),
        "ttfb_p50_s": None if step["ttfb_p50_s"] is None else round(step["ttfb_p50_s"], 2),
        "ttfb_p95_s": None if step["ttfb_p95_s"] is None else round(step["ttfb_p95_s"], 2),
        "gpus_busy_avg": None if step["replicas_busy"] is None else round(step["replicas_busy"], 2),
        "prefix_hit_pct": round(100 * sum(hits) / sum(qs), 1) if sum(qs) > 0 else None,
        "tokens_per_min": round(60 * step["output_tok_per_s"]),
    }
    if on:
        st, sp = step.get("sticky_delta"), step.get("spilled_delta")
        tot = (st or 0) + (sp or 0)
        row["sticky_pct"] = round(100 * st / tot, 1) if tot and st is not None else None
        row["spilled_pct"] = round(100 * sp / tot, 1) if tot and sp is not None else None
    return row


def _first_prompt_range(run: Path) -> list[int] | None:
    p = run / "requests.jsonl"
    if not p.exists():
        return None
    first: dict[int, tuple[float, int]] = {}
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r.get("error") or not r.get("in"):
            continue
        cur = first.get(r["session"])
        if cur is None or r["t"] < cur[0]:
            first[r["session"]] = (r["t"], r["in"])
    ins = [v[1] for v in first.values()]
    return [min(ins), max(ins)] if ins else None


def _set_path(data: dict, path: str, value: Any) -> None:
    toks = _tokens(path)
    cur: Any = data
    for t in toks[:-1]:
        try:
            cur = cur[t]
        except (KeyError, IndexError, TypeError):
            raise BenchError(f"--set {path}: no such path") from None
    last = toks[-1]
    if isinstance(cur, dict) and last not in cur:
        raise BenchError(f"--set {path}: not a field of the data file")
    try:
        cur[last] = value
    except (IndexError, TypeError):
        raise BenchError(f"--set {path}: no such path") from None


def ingest(
    data: dict,
    off_dir: Path | None = None,
    on_dir: Path | None = None,
    smoke: Path | None = None,
    sets: list[str] | None = None,
) -> dict:
    data = copy.deepcopy(data)
    if bool(off_dir) != bool(on_dir):
        raise BenchError("--off and --on go together")
    if off_dir and on_dir:
        off, on = _read_json(off_dir / "summary.json"), _read_json(on_dir / "summary.json")
        so = {s["concurrency"]: s for s in off["steps"]}
        sn = {s["concurrency"]: s for s in on["steps"]}
        if sorted(so) != sorted(sn):
            raise BenchError(f"ramp steps differ: off {sorted(so)} vs on {sorted(sn)}")
        data["ramp"] = [
            {
                "workers": w,
                "off": _arm_row(so[w], "off", False),
                "on": _arm_row(sn[w], "on", True),
            }
            for w in sorted(so)
        ]
        dp_on, dp_off = on.get("dp") or {}, off.get("dp") or {}
        if dp_on.get("affinity_enabled") is False or dp_off.get("affinity_enabled") is True:
            raise BenchError("the --off/--on directories look swapped (affinity_enabled)")
        if dp_on.get("spill_threshold") is not None:
            data["spill_threshold"] = dp_on["spill_threshold"]
        c_on, c_off = on.get("config", {}), off.get("config", {})
        if c_on.get("step_seconds") != c_off.get("step_seconds"):
            raise BenchError("the two arms used different step_seconds")
        if c_on.get("step_seconds") is not None:
            data["workload"]["step_minutes"] = round(c_on["step_seconds"] / 60, 2)
        if c_on:
            data["workload"]["description"] = (
                f"Synthetic load from our own harness, not Claude Code itself: "
                f"closed-loop Claude Code-style sessions over streaming /v1/messages, "
                f"{c_on.get('turns')} turns each, a shared system prompt of about "
                f"{c_on.get('system_tokens')} tokens, a session-unique first message "
                f"({c_on.get('first_user_tokens')} tokens nominal), about "
                f"{c_on.get('turn_tokens')} tokens of new input per follow-up turn, "
                f"max_tokens {c_on.get('max_tokens')}, "
                f"{fmt(float(c_on.get('step_seconds') or 0))} s per step"
            )
        rng = _first_prompt_range(on_dir)
        if rng:
            data["workload"]["first_prompt_tokens_range"] = rng
        mt = (on_dir / "summary.json").stat().st_mtime
        data["captured"] = dt.datetime.fromtimestamp(mt).date().isoformat()
    if smoke:
        cases = _read_json(smoke).get("cases", {})
        for name, vals in cases.items():
            if name not in data["router_smoke"]:
                raise BenchError(f"smoke case {name!r} is not in the data file")
            tgt = data["router_smoke"][name]
            for k, v in vals.items():
                if k not in tgt:
                    raise BenchError(f"smoke case {name}.{k} is not in the data file")
                if k in STATIC_FIELDS and tgt[k] != v:
                    raise BenchError(f"smoke case {name}.{k}: expected {tgt[k]!r}, got {v!r}")
                tgt[k] = v
    for s in sets or []:
        path, sep, raw = s.partition("=")
        if not sep:
            raise BenchError(f"--set wants path=value, got {s!r}")
        try:
            val = json.loads(raw)
        except ValueError:
            val = raw
        _set_path(data, path.strip(), val)
    return data


# ---------------------------------------------------------------------------- SVG


class Theme:
    def __init__(
        self,
        name: str,
        ink: str,
        ink2: str,
        rule: str,
        paper: str,
        on: str,
        off: str,
        font: str,
        bg: bool,
    ):
        self.name, self.ink, self.ink2, self.rule, self.paper = name, ink, ink2, rule, paper
        self.on, self.off = on, off
        self.font, self.bg = font, bg


TOKENS = Theme(
    "tokens",
    "var(--ink)",
    "var(--ink-2)",
    "var(--rule)",
    "var(--paper)",
    "var(--series-on)",
    "var(--series-off)",
    "var(--sans, ui-sans-serif, system-ui, sans-serif)",
    False,
)
LIGHT = Theme(
    "light", "#1D1913", "#565049", "#CBC7BF", "#EFECE7", "#A86400", "#2A78D6",
    "system-ui, sans-serif", True,
)  # fmt: skip
DARK = Theme(
    "dark", "#EAE5DC", "#ACA397", "#433C31", "#1C1409", "#C98500", "#3987E5",
    "system-ui, sans-serif", True,
)  # fmt: skip
FILE_THEMES = (LIGHT, DARK)

SERIES_ON = "affinity on"
SERIES_OFF = "vLLM balancer"
X_LABEL = "concurrent sessions"


class Layout:
    """Geometry and type size of one rendering. `wide` is the GitHub file and the
    landing page at desktop width; `narrow` is the landing page on a phone, drawn
    for a ~320px column so its text stays at about 12px on screen."""

    def __init__(self, name: str, fs: float, width: int, stacked: bool):
        self.name, self.fs, self.width, self.stacked = name, fs, width, stacked


WIDE = Layout("wide", 15, 640, False)
NARROW = Layout("narrow", 13, 340, True)


def _x(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def _svg(t: Theme, w: int, h: int, title: str, desc: str, body: str, fs: float = 15) -> str:
    bg = f'<rect width="{w}" height="{h}" style="fill:{t.paper}"/>' if t.bg else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'role="img" style="font-family:{t.font};font-size:{_x(fs)}px">'
        f"<title>{html.escape(title)}</title><desc>{html.escape(desc)}</desc>{bg}{body}</svg>\n"
    )


def _text(
    t: Theme,
    x: float,
    y: float,
    s: str,
    anchor: str = "start",
    color: str | None = None,
    weight: int | None = None,
) -> str:
    w = f";font-weight:{weight}" if weight else ""
    return (
        f'<text x="{_x(x)}" y="{_x(y)}" text-anchor="{anchor}" style="fill:{color or t.ink2}{w}">'
        f"{html.escape(s)}</text>"
    )


def _nice(vmax: float) -> tuple[float, float]:
    if vmax <= 0:
        return 1.0, 0.25
    import math

    mag = 10 ** math.floor(math.log10(vmax))
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * mag >= vmax:
            top = m * mag
            break
    step = top / 4
    return top, step


def _marker(t: Theme, shape: str, cx: float, cy: float, color: str, tip: str = "") -> str:
    ring = f"stroke:{t.paper};stroke-width:2"
    tip_el = f"<title>{html.escape(tip)}</title>" if tip else ""
    if shape == "square":
        return (
            f'<rect x="{_x(cx - 4.5)}" y="{_x(cy - 4.5)}" width="9" height="9" '
            f'style="fill:{color};{ring}">{tip_el}</rect>'
        )
    return (
        f'<circle cx="{_x(cx)}" cy="{_x(cy)}" r="5" style="fill:{color};{ring}">{tip_el}</circle>'
    )


class Scale:
    """A y scale: linear from 0, or log10 between two bounds, with its tick values."""

    def __init__(self, kind: str, lo: float, hi: float, ticks: list[float]):
        self.kind, self.lo, self.hi, self.ticks = kind, lo, hi, ticks

    def frac(self, v: float) -> float:
        import math

        if self.kind == "log":
            return (math.log10(v) - math.log10(self.lo)) / (
                math.log10(self.hi) - math.log10(self.lo)
            )
        return (v - self.lo) / (self.hi - self.lo)


def _linear(top: float, step: float) -> Scale:
    n = int(round(top / step))
    return Scale("lin", 0.0, top, [round(i * step, 4) for i in range(n + 1)])


LOG_SECONDS = Scale("log", 0.3, 100.0, [0.3, 1, 3, 10, 30, 100])


def _spread(ys: dict[str, float], gap: float) -> dict[str, float]:
    """Direct end labels must not collide: push two labels apart to at least `gap`."""
    (a, ya), (b, yb) = sorted(ys.items(), key=lambda kv: kv[1])
    short = gap - (yb - ya)
    if short > 0:
        ya, yb = ya - short / 2, yb + short / 2
    return {a: ya, b: yb}


def _line_panel(
    t: Theme,
    lay: Layout,
    x0: float,
    y0: float,
    w: float,
    h: float,
    cats: list,
    off: list,
    on: list,
    scale: Scale,
    yunit: str,
    title: str,
    vfmt: Any = None,
) -> str:
    fs = lay.fs
    left, right, bot = fs * 2.9, fs * 7.4, fs * 3.1
    px0, px1, py0, py1 = x0 + left, x0 + w - right, y0 + fs * 3.4, y0 + h - bot
    vfmt = vfmt or fmt
    out = [_text(t, x0, y0 + fs, title, color=t.ink, weight=600)]
    for v in scale.ticks:
        y = py1 - (py1 - py0) * scale.frac(v)
        out.append(
            f'<line x1="{_x(px0)}" x2="{_x(px1)}" y1="{_x(y)}" y2="{_x(y)}" style="stroke:{t.rule};stroke-width:1"/>'
        )
        out.append(_text(t, px0 - 8, y + fs * 0.33, fmt(v), "end"))
    out.append(_text(t, x0, py0 - fs * 0.9, yunit))
    xs = [px0 + (px1 - px0) * (i + 0.5) / len(cats) for i in range(len(cats))]
    for x, c in zip(xs, cats, strict=True):
        out.append(_text(t, x, py1 + fs * 1.35, fmt(c), "middle"))
    out.append(_text(t, (px0 + px1) / 2, y0 + h - fs * 0.25, X_LABEL, "middle"))

    def ypos(v: float) -> float:
        return py1 - (py1 - py0) * scale.frac(v)

    ends = _spread({SERIES_OFF: ypos(off[-1]), SERIES_ON: ypos(on[-1])}, fs * 1.25)
    for vals, color, shape, name in (
        (off, t.off, "square", SERIES_OFF),
        (on, t.on, "circle", SERIES_ON),
    ):
        pts = [(x, ypos(v)) for x, v in zip(xs, vals, strict=True)]
        d = " ".join(f"{_x(px)},{_x(py)}" for px, py in pts)
        out.append(
            f'<polyline points="{d}" style="fill:none;stroke:{color};stroke-width:2.5;stroke-linejoin:round"/>'
        )
        out += [
            _marker(t, shape, px, py, color, f"{name}, {fmt(c)} sessions: {vfmt(v)}")
            for (px, py), c, v in zip(pts, cats, vals, strict=True)
        ]
        out.append(_text(t, pts[-1][0] + 12, ends[name] + fs * 0.33, name, color=t.ink))
    return "".join(out)


def _legend(t: Theme, lay: Layout, x: float, y: float) -> str:
    out = []
    step = lay.fs * 10
    for i, (name, color, shape) in enumerate(
        ((SERIES_ON, t.on, "circle"), (SERIES_OFF, t.off, "square"))
    ):
        lx = x + i * step
        out.append(
            f'<line x1="{_x(lx)}" x2="{_x(lx + 26)}" y1="{_x(y)}" y2="{_x(y)}" style="stroke:{color};stroke-width:2.5"/>'
        )
        out.append(_marker(t, shape, lx + 13, y, color))
        out.append(_text(t, lx + 34, y + lay.fs * 0.33, name, color=t.ink))
    return "".join(out)


def pending_svg(t: Theme, title: str) -> str:
    body = _text(t, 320, 175, "measurement pending", "middle") + _text(t, 320, 198, title, "middle")
    return _svg(
        t,
        640,
        360,
        f"{title} (measurement pending)",
        "The A/B measurement has not been taken yet.",
        body,
    )


def _chart_cats(data: dict) -> list:
    return [r["workers"] for r in data["ramp"]]


def _series(data: dict, field: str) -> tuple[list, list]:
    return [r["off"][field] for r in data["ramp"]], [r["on"][field] for r in data["ramp"]]


def _seconds(v: float) -> str:
    return f"{fmt(v)} s"


def _one_panel_chart(
    t: Theme,
    lay: Layout,
    data: dict,
    field: str,
    scale: Scale,
    yunit: str,
    title: str,
    svg_title: str,
    desc: str,
    vfmt: Any = None,
) -> str:
    off, on = _series(data, field)
    w = lay.width
    ph = 330 if lay.stacked else 360
    body = _legend(t, lay, 0, lay.fs) + _line_panel(
        t, lay, 0, lay.fs * 2.6, w, ph, _chart_cats(data), off, on, scale, yunit, title, vfmt
    )
    return _svg(t, w, int(lay.fs * 2.6 + ph + 4), svg_title, desc, body, lay.fs)


def chart_c1(t: Theme, data: dict, lay: Layout = WIDE) -> str:
    """Time to first byte, p50 and p95, on a log axis: the 0.4 s and the 83 s both read."""
    cats = _chart_cats(data)
    o50, n50 = _series(data, "ttfb_p50_s")
    o95, n95 = _series(data, "ttfb_p95_s")
    fs = lay.fs
    unit = "seconds, log scale"
    if lay.stacked:
        w, ph = lay.width, 300
        body = _legend(t, lay, 0, fs)
        body += _line_panel(
            t, lay, 0, fs * 2.6, w, ph, cats, o50, n50, LOG_SECONDS, unit,
            "Time to first byte, median", _seconds,
        )  # fmt: skip
        body += _line_panel(
            t, lay, 0, fs * 2.6 + ph + fs * 2, w, ph, cats, o95, n95, LOG_SECONDS, unit,
            "Time to first byte, p95", _seconds,
        )  # fmt: skip
        h = int(fs * 4.6 + 2 * ph + 4)
    else:
        w, ph = 960, 380
        body = _legend(t, lay, 0, fs)
        body += _line_panel(
            t, lay, 0, fs * 2.6, 470, ph, cats, o50, n50, LOG_SECONDS, unit,
            "Time to first byte, median", _seconds,
        )  # fmt: skip
        body += _line_panel(
            t, lay, 490, fs * 2.6, 470, ph, cats, o95, n95, LOG_SECONDS, unit,
            "Time to first byte, p95", _seconds,
        )  # fmt: skip
        h = int(fs * 2.6 + ph + 4)
    return _svg(
        t,
        w,
        h,
        "Time to first byte by concurrent sessions",
        "Median and 95th-percentile time to first byte on the four-card box at 4 to 32 "
        "concurrent sessions, affinity on against vLLM’s balancer, on a log scale from "
        "0.3 to 100 seconds. "
        + "; ".join(
            f"{fmt(c)} sessions: median {fmt(a)} s off, {fmt(b)} s on, p95 {fmt(x)} s off, {fmt(y)} s on"
            for c, a, b, x, y in zip(cats, o50, n50, o95, n95, strict=True)
        )
        + ".",
        body,
        fs,
    )


def chart_c2(t: Theme, data: dict, lay: Layout = WIDE) -> str:
    off, on = _series(data, "gpus_busy_avg")
    return _one_panel_chart(
        t,
        lay,
        data,
        "gpus_busy_avg",
        _linear(4, 1),
        "replicas busy, of 4",
        "Replicas busy",
        "Replicas busy by concurrent sessions",
        "Average replicas with running requests, of four, affinity on against vLLM’s balancer: "
        + "; ".join(
            f"{fmt(c)} sessions {fmt(a)} off, {fmt(b)} on"
            for c, a, b in zip(_chart_cats(data), off, on, strict=True)
        )
        + ".",
    )


def chart_c3(t: Theme, data: dict, lay: Layout = WIDE) -> str:
    """Prefix-cache hit rate per step, pooled over the four replicas (raw counter deltas)."""
    off, on = _series(data, "prefix_hit_pct")
    return _one_panel_chart(
        t,
        lay,
        data,
        "prefix_hit_pct",
        _linear(100, 25),
        "% of prompt tokens",
        "Prefix-cache hit rate",
        "Prefix-cache hit rate by concurrent sessions",
        "Share of prompt tokens served from the prefix cache in each step, all four replicas "
        "pooled, affinity on against vLLM’s balancer: "
        + "; ".join(
            f"{fmt(c)} sessions {fmt(a)}% off, {fmt(b)}% on"
            for c, a, b in zip(_chart_cats(data), off, on, strict=True)
        )
        + ".",
        lambda v: f"{fmt(v)}%",
    )


def chart_c4(t: Theme, data: dict, lay: Layout = WIDE) -> str:
    off, on = _series(data, "tasks_per_h")
    top, step = _nice(max(v for v in off + on if v is not None))
    return _one_panel_chart(
        t,
        lay,
        data,
        "tasks_per_h",
        _linear(top, step),
        "completed sessions per hour",
        "Completed sessions per hour",
        "Completed sessions per hour by concurrent sessions",
        "Completed six-turn sessions per hour, affinity on against vLLM’s balancer. A step "
        "lasts three minutes, so one more or one fewer completed session moves a point by 20.",
        lambda v: fmt(round(v)),
    )


def diagram_d2(t: Theme, lay: Layout = WIDE) -> str:
    """Mechanism, not a measurement: no numbers. Four turns of one conversation, per-replica bands.

    Wide: the two panels side by side. Narrow: stacked, for a phone column."""
    fs = 14 if not lay.stacked else 13
    if lay.stacked:
        panel_w, band_h, sub, bar_h, lx0 = 340.0, 62.0, 13.0, 9.0, 68.0
    else:
        panel_w, band_h, sub, bar_h, lx0 = 470.0, 74.0, 16.0, 11.0, 70.0
    lens = [0.40, 0.55, 0.72, 0.92]
    stub = 0.10
    full = panel_w - lx0 - fs * 2  # room for the "t4" label after the longest bar
    titles = (
        "vLLM’s balancer: a turn lands anywhere",
        "LM Warden: a conversation stays on its replica",
    )
    homes = ([2, 0, 3, 1], [2, 2, 2, 2])
    caps: tuple[list[str], list[str]] = (
        ["prefilled again: about the whole history, every turn"],
        ["prefilled again: only the new turn"],
    )
    if lay.stacked:
        caps = (["prefilled again: about the whole history,", "every turn"], caps[1])
    legend_y = fs
    out = [
        f'<rect x="0" y="{_x(legend_y - 9)}" width="14" height="10" style="fill:{t.ink2};opacity:.35"/>',
        _text(t, 22, legend_y, "served from the prefix cache"),
    ]
    if lay.stacked:
        legend_y2 = legend_y + fs * 1.5
        out += [
            f'<rect x="0" y="{_x(legend_y2 - 9)}" width="14" height="10" style="fill:{t.ink}"/>',
            _text(t, 22, legend_y2, "prefilled from scratch"),
        ]
        top0 = legend_y2 + fs * 2.4
    else:
        out += [
            f'<rect x="240" y="{_x(legend_y - 9)}" width="14" height="10" style="fill:{t.ink}"/>',
            _text(t, 262, legend_y, "prefilled from scratch"),
        ]
        top0 = legend_y + fs * 2.6
    panel_h = fs * 1.4 + 4 * band_h + fs * 1.6 * len(caps[0]) + fs
    for p in range(2):
        ox, oy = (
            (0.0, top0 + p * (panel_h + fs * 1.5)) if lay.stacked else (p * (panel_w + 40), top0)
        )
        out.append(_text(t, ox, oy, titles[p], color=t.ink, weight=600))
        b0 = oy + fs * 1.0
        for r in range(4):
            by = b0 + r * band_h
            out.append(_text(t, ox, by + band_h / 2 + fs * 0.33, f"replica {r}"))
            out.append(
                f'<line x1="{_x(ox + lx0)}" x2="{_x(ox + panel_w)}" y1="{_x(by + band_h - 4)}" y2="{_x(by + band_h - 4)}" style="stroke:{t.rule};stroke-width:1"/>'
            )
        for k in range(4):
            r = homes[p][k]
            y = b0 + r * band_h + 4 + k * sub
            total = lens[k] * full
            cached = stub * full if (p == 0 or k == 0) else lens[k - 1] * full
            out.append(
                f'<rect x="{_x(ox + lx0)}" y="{_x(y)}" width="{_x(cached)}" height="{_x(bar_h)}" style="fill:{t.ink2};opacity:.35"/>'
            )
            out.append(
                f'<rect x="{_x(ox + lx0 + cached + 2)}" y="{_x(y)}" width="{_x(total - cached - 2)}" height="{_x(bar_h)}" style="fill:{t.ink}"/>'
            )
            out.append(_text(t, ox + lx0 + total + 6, y + bar_h - 1, f"t{k + 1}"))
        for i, line in enumerate(caps[p]):
            out.append(_text(t, ox, b0 + 4 * band_h + fs * 1.4 + i * fs * 1.35, line))
    if lay.stacked:
        w, h = (
            int(panel_w),
            int(top0 + 2 * panel_h + fs * 1.5 - fs * 1.6 * (len(caps[0]) - len(caps[1]))),
        )
    else:
        w, h = int(2 * panel_w + 40), int(top0 + panel_h)
    return _svg(
        t,
        w,
        h,
        "Why affinity: one conversation over four turns",
        "Two panels of four replicas. First: each of four turns of one conversation lands on a different replica and is almost entirely prefilled from scratch. Second: all four turns stay on one replica and only the new turn is prefilled; the rest is served from the prefix cache.",
        "".join(out),
        fs,
    )


# ----------------------------------------------------------------------- tables


def _cell(a: Any, b: Any) -> str:
    return f"{fmt(a)}{ARROW}{fmt(b)}"


def t1_rows(data: dict) -> list[list[str]]:
    rows = []
    for r in data["ramp"]:
        o, n = r["off"], r["on"]
        rows.append(
            [
                fmt(r["workers"]),
                _cell(round(o["tasks_per_h"]), round(n["tasks_per_h"])),
                _cell(f"{o['tokens_per_min']:,}", f"{n['tokens_per_min']:,}"),
                _cell(o["ttfb_p50_s"], n["ttfb_p50_s"]),
                _cell(o["ttfb_p95_s"], n["ttfb_p95_s"]),
                _cell(o["gpus_busy_avg"], n["gpus_busy_avg"]),
                _cell(o["prefix_hit_pct"], n["prefix_hit_pct"]),
                fmt(n["sticky_pct"]),
            ]
        )
    return rows


T1_HEAD = [
    "Sessions",
    "Sessions/h off → on",
    "Output tokens/min off → on",
    "TTFB p50 off → on (s)",
    "TTFB p95 off → on (s)",
    "Replicas busy off → on",
    "Prefix hits off → on (%)",
    "Stayed on replica (%)",
]


def md_table(head: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def html_table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f'<th scope="col">{html.escape(h)}</th>' for h in head)
    body = []
    for r in rows:
        cells = [f'<th scope="row">{html.escape(r[0])}</th>'] + [
            f'<td class="n" data-h="{html.escape(head[i])}">{html.escape(c)}</td>'
            for i, c in enumerate(r)
            if i
        ]
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f'<table class="readings"><thead><tr>{th}</tr></thead><tbody>\n'
        + "\n".join(body)
        + "\n</tbody></table>\n"
    )


def t2_rows(data: dict) -> tuple[list[str], list[list[str]]]:
    head = ["Workers", "Tasks/h", "TTFB p50 / p95 (s)", "GPUs busy (of 7)", "Avg GPU util (%)"]
    rows = [
        [
            fmt(r["workers"]),
            fmt(r["tasks_per_h"]),
            f"{fmt(r['ttfb_p50_s'])} / {fmt(r['ttfb_p95_s'])}",
            fmt(r["gpus_busy_avg"]),
            fmt(r["gpu_util_avg_pct"]),
        ]
        for r in data["before_7gpu"]["rows"]
    ]
    return head, rows


SMOKE = "router smoke test"
LIVE = "live Claude Code session"


def t3_rows(data: dict) -> list[list[str]]:
    """Two rows come from router_smoke.py, two from the live Claude Code session (passed
    in with --set); the last column says which, so neither is mistaken for the other."""
    s = data["router_smoke"]
    return [
        ["Haiku, matched by a rule", "local", "200", fmt(s["haiku_local"]["ttfb_p50_s"]), SMOKE],
        [
            "A model with no rule",
            "passthrough",
            "200",
            fmt(s["opus_passthrough"]["ttfb_p50_s"]),
            LIVE,
        ],
        [
            "Haiku, local leg failed before the first byte, fall back on",
            "fallback",
            fmt(s["fallback_forced"]["reason"]),
            "–",
            LIVE,
        ],
        [
            "Opus, key without “May relay”",
            "refused",
            f"{s['refused_no_relay']['status']} {s['refused_no_relay']['reason']}",
            "–",
            SMOKE,
        ],
    ]


T3_HEAD = ["Asked for", "Where it went", "Status or reason", "TTFB p50 (s)", "Measured by"]


# ----------------------------------------------------------------------- driver


def _w(root: Path, rel: str, text: str) -> str:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return rel


def render(root: Path, data: dict, complete: bool) -> list[str]:
    written: list[str] = []
    for t in FILE_THEMES:
        written.append(_w(root, f"assets/diagrams/replica-affinity-{t.name}.svg", diagram_d2(t)))
    written.append(_w(root, "app/landing/figures/d2.svg", diagram_d2(TOKENS)))
    written.append(_w(root, "app/landing/figures/d2-narrow.svg", diagram_d2(TOKENS, NARROW)))
    h, rows = t2_rows(data)
    written.append(_w(root, "assets/bench/t2-before-7gpu.md", md_table(h, rows)))
    charts = (
        ("c1-ttfb", "app/landing/figures/c1.svg", chart_c1, "Time to first byte"),
        ("c2-busy", "app/landing/figures/c2.svg", chart_c2, "Replicas busy"),
        ("c3-hits", "app/landing/figures/c3.svg", chart_c3, "Prefix-cache hit rate"),
        ("c4-tasks", None, chart_c4, "Completed agent tasks per hour"),
    )
    for name, landing, fn, title in charts:
        for t in FILE_THEMES:
            svg = fn(t, data) if complete else pending_svg(t, title)
            written.append(_w(root, f"assets/bench/{name}-{t.name}.svg", svg))
        if landing:
            written.append(
                _w(root, landing, fn(TOKENS, data) if complete else pending_svg(TOKENS, title))
            )
            narrow = landing.replace(".svg", "-narrow.svg")
            written.append(
                _w(
                    root,
                    narrow,
                    fn(TOKENS, data, NARROW) if complete else pending_svg(TOKENS, title),
                )
            )
    if complete:
        written.append(_w(root, "assets/bench/t1-ramp.md", md_table(T1_HEAD, t1_rows(data))))
        written.append(
            _w(root, "app/landing/figures/t1-ramp.html", html_table(T1_HEAD, t1_rows(data)))
        )
        written.append(
            _w(root, "assets/bench/t3-router-smoke.md", md_table(T3_HEAD, t3_rows(data)))
        )
    return written


def fill(root: Path, data: dict) -> list[str]:
    """Resolve every slot first; write only if all resolve (all or nothing)."""
    new: dict[str, str] = {}
    for rel in SURFACES:
        p = root / rel
        if not p.exists():
            continue
        text = p.read_text()
        if SLOT_RE.search(text):
            new[rel] = SLOT_RE.sub(lambda m: resolve_slot(data, m.group(1)), text)
    for rel, text in new.items():
        (root / rel).write_text(text)
    return sorted(new)


def surfaces_with_slots(root: Path) -> list[str]:
    return [r for r in SURFACES if (root / r).exists() and SLOT_RE.search((root / r).read_text())]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--check", action="store_true", help="list null paths; exit 1 if any")
    ap.add_argument("--render", action="store_true", help="write figures and table fragments")
    ap.add_argument(
        "--fill",
        action="store_true",
        help="render, then substitute {{SLOT:..}} in the shipped surfaces",
    )
    ap.add_argument("--off", type=Path, help="agent_ramp.py run dir, affinity off")
    ap.add_argument("--on", type=Path, help="agent_ramp.py run dir, affinity on")
    ap.add_argument("--smoke", type=Path, help="router_smoke.py --json-out file")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="set a data-file field (JSON or string)",
    )
    ap.add_argument("--data", type=Path, help=f"data file (default {DEFAULT_DATA})")
    ap.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent.parent, help="repo root"
    )
    a = ap.parse_args(argv)
    root = a.root
    data_path = a.data or root / DEFAULT_DATA
    try:
        data = _read_json(data_path)
        if a.off or a.on or a.smoke or a.set:
            data = ingest(data, a.off, a.on, a.smoke, a.set)
            data_path.write_text(json.dumps(data, indent=2) + "\n")
            print(f"ingested into {data_path}")
        nulls = find_nulls(data)
        if not (a.check or a.render or a.fill):
            ap.error("nothing to do: pass --check, --render, --fill or ingest inputs")
        if a.render or a.fill:  # --fill renders first, so the figures and the prose agree
            for rel in render(root, data, not nulls):
                print(f"wrote {rel}")
        if nulls:
            print(f"{len(nulls)} value(s) still null in {data_path}:", file=sys.stderr)
            for n in nulls:
                print(f"  {n}", file=sys.stderr)
            if a.fill:
                print("refusing to fill: no surface was touched", file=sys.stderr)
            elif a.render:
                print("rendered placeholders only", file=sys.stderr)
            return 1
        if a.fill:
            done = fill(root, data)
            print("filled: " + (", ".join(done) if done else "no slots found"))
        elif a.check:
            print("data file complete: no nulls")
        return 0
    except BenchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
