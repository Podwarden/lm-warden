"""Gates for scripts/bench-figures.py: the A/B data file, the slot convention, ingest.

No number here is a measurement: fixtures are synthetic and only prove the plumbing.
"""

import copy
import importlib.util
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "bench-figures.py"
DATA = REPO / "assets" / "bench" / "dp-affinity-ab.json"


@pytest.fixture(scope="module")
def bf():
    spec = importlib.util.spec_from_file_location("bench_figures", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _step(c, sessions, p50, p95, busy, hits, qs, tok, sticky=None, spilled=None):
    return {
        "concurrency": c,
        "sessions_per_h": sessions,
        "ttfb_p50_s": p50,
        "ttfb_p95_s": p95,
        "replicas_busy": busy,
        "output_tok_per_s": tok,
        "prefix_hits_by_replica": hits,
        "prefix_queries_by_replica": qs,
        "sticky_delta": sticky,
        "spilled_delta": spilled,
    }


def _run(path: Path, on: bool):
    path.mkdir(parents=True)
    steps = [
        _step(
            4,
            100.0,
            1.0,
            2.0,
            2.0,
            [10, 10, 0, 0],
            [100, 100, 0, 0],
            10.0,
            30 if on else None,
            10 if on else None,
        ),
        _step(
            8,
            150.0 if on else 120.0,
            1.5,
            3.5,
            3.5,
            [30, 30, 30, 30],
            [60, 60, 60, 60],
            20.0,
            90 if on else None,
            10 if on else None,
        ),
    ]
    summ = {
        "label": "x",
        "config": {"step_seconds": 90.0, "turns": 4, "system_tokens": 3500, "turn_tokens": 300},
        "steps": steps,
        "replicas_end": [
            {"rank": i, "prefix_cache_hits": 50.0 * (i + 1), "prefix_cache_queries": 100.0}
            for i in range(4)
        ],
        "dp": {"data_parallel_size": 4, "affinity_enabled": on, "spill_threshold": 6},
    }
    (path / "summary.json").write_text(json.dumps(summ))
    (path / "requests.jsonl").write_text(
        "\n".join(
            json.dumps({"t": t, "session": s, "in": n, "error": None})
            for t, s, n in [(1, 0, 6000), (2, 0, 6500), (1, 1, 6200)]
        )
        + "\n"
    )


_KEEP = {"box.gpus", "box.layout", "box.vllm"}


def _blank(obj, prefix=""):
    """The data file as it was before the A/B: only fixed facts and before_7gpu survive."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            if (
                k == "before_7gpu"
                or path in _KEEP
                or (prefix.startswith("router_smoke.") and k in ("route", "status", "reason"))
            ):
                out[k] = v
            else:
                out[k] = _blank(v, path)
        return out
    if isinstance(obj, list):
        return [_blank(v, prefix) for v in obj]
    return None


def pending_data() -> dict:
    d = _blank(json.loads(DATA.read_text()))
    d["ramp"] = d["ramp"][:1]
    return d


@pytest.fixture
def repo(tmp_path, bf):
    """A scratch repo root: the shipped (null) data file plus one surface with slots."""
    (tmp_path / "assets" / "bench").mkdir(parents=True)
    (tmp_path / "assets" / "bench" / "dp-affinity-ab.json").write_text(json.dumps(pending_data()))
    (tmp_path / "README.md").write_text(
        "p95 {{SLOT:peak.off.ttfb_p95_s}} s to {{SLOT:peak.on.ttfb_p95_s}} s at {{SLOT:peak.workers}} workers; "
        "{{SLOT:box.served_name}} {{SLOT:ramp[0].workers}} {{SLOT:peak.on.sticky_pct}}\n"
    )
    return tmp_path


def _full_args(tmp_path):
    _run(tmp_path / "off", False)
    _run(tmp_path / "on", True)
    (tmp_path / "smoke.json").write_text(
        json.dumps({"cases": {"haiku_local": {"requests": 5, "route": "local", "ttfb_p50_s": 0.3}}})
    )
    sets = {
        "box.model_hf": "org/model",
        "box.served_name": "small-model",
        "box.quant": "awq",
        "box.lm_warden": "v1",
        "box.max_num_seqs": 64,
        "box.max_model_len": 8192,
        "router_smoke.opus_passthrough.requests": 3,
        "router_smoke.opus_passthrough.ttfb_p50_s": 0.5,
        "router_smoke.fallback_forced.requests": 2,
        "router_smoke.fallback_forced.reason": "target_not_loaded",
        "router_smoke.refused_no_relay.requests": 1,
    }
    args = [
        "--off",
        str(tmp_path / "off"),
        "--on",
        str(tmp_path / "on"),
        "--smoke",
        str(tmp_path / "smoke.json"),
    ]
    for k, v in sets.items():
        args += ["--set", f"{k}={json.dumps(v)}"]
    return args


# ---- the shipped data file -------------------------------------------------------


def test_data_file_before_7gpu_is_the_issue_286_table():
    d = json.loads(DATA.read_text())
    rows = [
        (
            r["workers"],
            r["tasks_per_h"],
            r["ttfb_p50_s"],
            r["ttfb_p95_s"],
            r["gpus_busy_avg"],
            r["gpu_util_avg_pct"],
        )
        for r in d["before_7gpu"]["rows"]
    ]
    assert rows == [
        (16, 158, 12, 39, 2.1, 28),
        (20, 179, 12, 35, 2.3, 30),
        (24, 200, 15, 44, 2.8, 37),
        (30, 197, 20, 50, 3.3, 42),
        (44, 189, 26, 78, 3.3, 42),
    ]
    assert d["before_7gpu"]["single_tp4_tasks_per_h"] == 93
    assert d["before_7gpu"]["tp4_prefix_hit_pct"] == 82


def test_only_before_7gpu_carries_numbers_before_the_ab_runs(bf):
    d = pending_data()
    nulls = bf.find_nulls(d)
    assert nulls
    assert not [n for n in nulls if n.startswith("before_7gpu")]
    # the measured block is entirely null (no invented numbers)
    for arm in ("off", "on"):
        assert all(v is None for v in d["ramp"][0][arm].values())


def test_data_file_has_no_internal_names():
    text = DATA.read_text().lower()
    # The denylist entries are assembled so this public file never spells an
    # internal host name itself.
    for bad in ("lmw" + "48", "lmwarden" + ".com", "ollama", "co-authored", "noreply", "10.10."):
        assert bad not in text


# ---- gates -----------------------------------------------------------------------


def test_check_lists_nulls_and_fails(bf, repo, capsys):
    rc = bf.main(["--root", str(repo), "--check"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "ramp[0].on.tasks_per_h" in err and "box.model_hf" in err


def test_fill_refuses_on_nulls_and_touches_nothing(bf, repo):
    before = (repo / "README.md").read_text()
    assert bf.main(["--root", str(repo), "--fill"]) == 1
    assert (repo / "README.md").read_text() == before


def test_render_with_nulls_writes_pending_placeholders_only(bf, repo):
    assert bf.main(["--root", str(repo), "--render"]) == 1
    svg = (repo / "assets/bench/c1-ttfb-light.svg").read_text()
    assert "measurement pending" in svg
    ET.fromstring(svg)
    assert not (repo / "assets/bench/t1-ramp.md").exists()
    # the mechanism drawing and the issue's table need no measurement
    ET.fromstring((repo / "assets/diagrams/replica-affinity-dark.svg").read_text())
    assert "| 16 | 158 |" in (repo / "assets/bench/t2-before-7gpu.md").read_text()


def test_unknown_slot_is_an_error(bf, repo):
    d = json.loads((repo / "assets/bench/dp-affinity-ab.json").read_text())
    with pytest.raises(bf.BenchError):
        bf.resolve_slot(d, "box.nope")
    with pytest.raises(bf.BenchError, match="null"):
        bf.resolve_slot(d, "box.model_hf")


# ---- ingest + end to end ---------------------------------------------------------


def test_ingest_maps_harness_fields(bf, tmp_path):
    args = _full_args(tmp_path)
    base = json.loads(DATA.read_text())
    sets = [args[i + 1] for i, a in enumerate(args) if a == "--set"]
    d = bf.ingest(base, tmp_path / "off", tmp_path / "on", tmp_path / "smoke.json", sets)
    assert bf.find_nulls(d) == []
    r0, r1 = d["ramp"]
    assert r0["workers"] == 4 and r1["workers"] == 8
    assert r0["off"]["prefix_hit_pct"] == 10.0  # 20 / 200, sum over replicas, not a mean of rates
    assert r1["on"]["prefix_hit_pct"] == 50.0
    assert r0["on"]["sticky_pct"] == 75.0 and r0["on"]["spilled_pct"] == 25.0
    assert r0["on"]["tokens_per_min"] == 600
    # lifetime per-replica counters are not ingested: the arms share one engine run
    assert "per_replica_hit_pct" not in d
    assert d["spill_threshold"] == 6
    assert d["workload"]["step_minutes"] == 1.5
    assert d["workload"]["first_prompt_tokens_range"] == [6000, 6200]
    assert d["router_smoke"]["haiku_local"]["ttfb_p50_s"] == 0.3


def test_ingest_rejects_mismatched_steps_swapped_arms_and_missing_counters(bf, tmp_path):
    base = json.loads(DATA.read_text())
    _run(tmp_path / "off", False)
    _run(tmp_path / "on", True)
    with pytest.raises(bf.BenchError, match="swapped"):
        bf.ingest(base, tmp_path / "on", tmp_path / "off")
    s = json.loads((tmp_path / "on/summary.json").read_text())
    s["steps"] = s["steps"][:1]
    (tmp_path / "on/summary.json").write_text(json.dumps(s))
    with pytest.raises(bf.BenchError, match="steps differ"):
        bf.ingest(base, tmp_path / "off", tmp_path / "on")
    s = json.loads((tmp_path / "off/summary.json").read_text())
    del s["steps"][0]["prefix_hits_by_replica"]
    (tmp_path / "off/summary.json").write_text(json.dumps(s))
    s2 = json.loads((tmp_path / "on/summary.json").read_text())
    s2["steps"] = json.loads((tmp_path / "off/summary.json").read_text())["steps"]
    (tmp_path / "on/summary.json").write_text(json.dumps(s2))
    with pytest.raises(bf.BenchError, match="raw prefix"):
        bf.ingest(base, tmp_path / "off", tmp_path / "on")


def test_set_rejects_unknown_fields_and_wrong_smoke_route(bf, tmp_path):
    base = json.loads(DATA.read_text())
    with pytest.raises(bf.BenchError):
        bf.ingest(base, sets=["box.invented=1"])
    (tmp_path / "s.json").write_text(
        json.dumps({"cases": {"haiku_local": {"route": "passthrough"}}})
    )
    with pytest.raises(bf.BenchError, match="route"):
        bf.ingest(base, smoke=tmp_path / "s.json")


def test_fill_one_command_ingests_renders_and_resolves_every_slot(bf, repo, tmp_path):
    rc = bf.main(["--root", str(repo), "--fill", *_full_args(tmp_path)])
    assert rc == 0
    text = (repo / "README.md").read_text()
    assert "{{SLOT:" not in text
    # peak = step with the highest `on` tasks/h (workers 8); its p95 off/on are 3.5 / 3.5
    assert text.startswith("p95 3.5 s to 3.5 s at 8 workers; small-model 4 ")
    assert text.rstrip().endswith("90")  # sticky_pct at the peak step: 90 / (90 + 10)
    data = json.loads((repo / "assets/bench/dp-affinity-ab.json").read_text())
    assert data["captured"] and re.fullmatch(r"\d{4}-\d{2}-\d{2}", data["captured"])
    for rel in ("c1-ttfb", "c2-busy", "c3-hits", "c4-tasks"):
        for theme in ("light", "dark"):
            svg = (repo / f"assets/bench/{rel}-{theme}.svg").read_text()
            assert "pending" not in svg
            ET.fromstring(svg)
            assert "var(--" not in svg  # GitHub files carry literal colours
    landing = (repo / "app/landing/figures/c1.svg").read_text()
    ET.fromstring(landing)
    assert "var(--ink)" in landing and "var(--ink-2)" in landing
    assert "4 → " not in landing
    t1 = (repo / "assets/bench/t1-ramp.md").read_text()
    assert "| 4 | 100" in t1 and "→" in t1
    assert 'class="n"' in (repo / "app/landing/figures/t1-ramp.html").read_text()


def test_filled_chart_uses_only_the_inks_and_the_two_validated_series(bf, tmp_path):
    """Text and rules in the page's inks; the two series in the palette-validated pair
    (never the label-tape amber #F2A516). Narrow cuts use the same colours."""
    base = json.loads(DATA.read_text())
    args = _full_args(tmp_path)
    sets = [args[i + 1] for i, a in enumerate(args) if a == "--set"]
    d = bf.ingest(base, tmp_path / "off", tmp_path / "on", tmp_path / "smoke.json", sets)
    allowed = {"#1D1913", "#565049", "#CBC7BF", "#EFECE7", "#A86400", "#2A78D6"}
    for fn in (bf.chart_c1, bf.chart_c2, bf.chart_c3, bf.chart_c4):
        for lay in (bf.WIDE, bf.NARROW):
            svg = fn(bf.LIGHT, d, lay)
            ET.fromstring(svg)
            colors = set(re.findall(r"#[0-9A-Fa-f]{6}", svg))
            assert colors <= allowed, colors
            assert "#A86400" in colors and "#2A78D6" in colors
    tokens = bf.chart_c1(bf.TOKENS, d, bf.NARROW)
    assert "var(--series-on)" in tokens and "var(--series-off)" in tokens


def test_ttfb_chart_is_log_scaled_and_narrow_cut_is_phone_sized(bf):
    d = json.loads(DATA.read_text())
    if bf.find_nulls(d):
        pytest.skip("A/B not run yet")
    wide, narrow = bf.chart_c1(bf.LIGHT, d), bf.chart_c1(bf.LIGHT, d, bf.NARROW)
    assert "log scale" in wide and "log scale" in narrow
    assert 'viewBox="0 0 340 ' in narrow
    assert 'viewBox="0 0 340 ' in bf.diagram_d2(bf.LIGHT, bf.NARROW)


def test_slot_gate_flips_when_the_data_file_is_complete(bf):
    """While the A/B has not run, `{{SLOT:` may appear in the surfaces. The moment
    assets/bench/dp-affinity-ab.json has no nulls, none may survive in a shipped surface."""
    d = json.loads(DATA.read_text())
    if bf.find_nulls(d):
        pytest.skip("A/B not run yet: data file has nulls, slots are allowed in the surfaces")
    assert bf.surfaces_with_slots(REPO) == []


def test_fill_all_or_nothing_on_unknown_slot(bf, repo, tmp_path):
    data = copy.deepcopy(json.loads(DATA.read_text()))
    data["ramp"] = [
        {"workers": 4, "off": {"tasks_per_h": 1}, "on": {"tasks_per_h": 2, "ttfb_p95_s": 3}}
    ]
    (repo / "README.md").write_text("{{SLOT:peak.workers}} {{SLOT:bogus.path}}\n")
    with pytest.raises(bf.BenchError):
        bf.fill(repo, data)
    assert (repo / "README.md").read_text() == "{{SLOT:peak.workers}} {{SLOT:bogus.path}}\n"
