"""Live proof of strict priority admission + FIFO-within-tier queueing.

Talks to a real vllm-warden deployment over HTTP. Gated by the `live`
pytest marker and the VW_LIVE_* env contract -- see conftest.py and
README.md. NEVER collected by the default `pytest` / `pytest tests/unit`
run (pyproject.toml addopts carries `-m "not live"`).

Run it:

    set -a; . /path/outside/repo/vw-live.env; set +a
    pytest -m live -s tests/live/test_priority_queueing.py

VW_LIVE_DRY_RUN=1 exercises isolation + the token lifecycle and exits
before any load -- see README.md "Rehearsal".
"""

from __future__ import annotations

import asyncio
import time

import pytest

from tests.live import _analysis as analysis
from tests.live._env import LiveConfig, redact_base_url
from tests.live._load import CalibrationResult, ProbeToken, RoundResult, calibrate, run_round
from tests.live._openai import OpenAIClient
from tests.live._report import RoundReport, RunReport

pytestmark = pytest.mark.live


def _summarize_round(
    rr: RoundResult,
    *,
    probes: list[ProbeToken],
    send_order: dict[str, int],
    cfg: LiveConfig,
    filler0_duration_s: float | None,
) -> RoundReport:
    records: list[analysis.ProbeRecord] = []
    for p in probes:
        timing = rr.probe_timings.get(p.label)
        row = rr.finished_rows.get(p.label)
        client_ttft = None
        client_done = None
        if timing is not None:
            if timing.t_first_frame is not None:
                client_ttft = timing.t_first_frame - timing.t_send
            if timing.t_done is not None:
                client_done = timing.t_done - timing.t_send
        records.append(
            analysis.ProbeRecord(
                label=p.label,
                priority=p.priority,
                send_order=send_order[p.label],
                t_send=rr.t_send.get(p.label, 0.0),
                t_burst_end=rr.t_burst_end or 0.0,
                started_iso=row.get("started_iso") if row else None,
                queued_s=row.get("queued_s") if row else None,
                finish_reason=(row.get("finish_reason") if row else None)
                or (timing.finish_reason if timing else None),
                orphan=bool(row.get("orphan")) if row else False,
                status=timing.status if timing else None,
                client_ttft_s=client_ttft,
                client_done_s=client_done,
            )
        )

    verdicts: dict[str, list[str]] = {}
    if rr.tainted:
        verdicts["tainted"] = [rr.taint_reason or "tainted"]
        ordered = records
    else:
        adm = analysis.admission_order(records)
        ordered = adm
        verdicts["A3_saturated"] = analysis.check_a3_saturated(records)
        verdicts["B_strict_priority"] = analysis.check_b_strict_priority(adm)
        verdicts["B_prime_client_order"] = analysis.check_b_prime_client_order(adm)
        verdicts["C_fifo_within_tier"] = analysis.check_c_fifo_within_tier(adm)
        verdicts["D1_queued_increasing"] = analysis.check_d1_queued_increasing_across_tiers(adm)
        d2_ok, spread = analysis.check_d2_spread(adm, min_spread_s=cfg.min_queue_spread_s)
        verdicts["D2_spread"] = [] if d2_ok else [f"spread {spread:.3f}s < {cfg.min_queue_spread_s}s"]
        verdicts["D3_first_admitted"] = analysis.check_d3_first_admitted(
            adm, hold_s=cfg.hold_s, filler0_duration_s=filler0_duration_s
        )
        verdicts["D4_basic"] = analysis.check_d4_basic(records)

    rows = []
    for idx, r in enumerate(ordered):
        rows.append(
            {
                "label": r.label,
                "priority": r.priority,
                "send": r.send_order,
                "admit": idx if r.started_iso else -1,
                "started_iso": r.started_iso or "",
                "queued_s": r.queued_s if r.queued_s is not None else "",
                "client_ttft_s": round(r.client_ttft_s, 3) if r.client_ttft_s is not None else "",
                "duration_s": round(r.client_done_s, 3) if r.client_done_s is not None else "",
                "finish": r.finish_reason or "",
            }
        )

    filler_order = [
        i for i, _ in sorted(rr.filler_timings.items(), key=lambda kv: kv[1].t_done or 0.0)
    ]

    return RoundReport(
        index=rr.index,
        tainted=rr.tainted,
        taint_reason=rr.taint_reason,
        setup_failure=rr.setup_failure,
        saturation_snapshot=rr.saturation_snapshot,
        burst_snapshot=rr.burst_snapshot,
        filler_completion_order=filler_order,
        rows=rows,
        verdicts=verdicts,
        errors=rr.errors,
    )


def _aggregate_report(report: RunReport) -> None:
    by_priority: dict[int, dict[str, list[float]]] = {}
    ttft_pairs: list[tuple[float, float]] = []
    done_pairs: list[tuple[float, float]] = []
    for r in report.rounds:
        if r.tainted:
            continue
        for idx, row in enumerate(r.rows):
            prio = row["priority"]
            qs = row["queued_s"]
            if isinstance(qs, int | float):
                bucket = by_priority.setdefault(prio, {"queued_s": [], "ranks": []})
                bucket["queued_s"].append(float(qs))
                bucket["ranks"].append(float(idx))
            ttft = row["client_ttft_s"]
            done = row["duration_s"]
            if isinstance(ttft, int | float):
                ttft_pairs.append((float(prio), float(ttft)))
            if isinstance(done, int | float):
                done_pairs.append((float(prio), float(done)))

    for prio, data in by_priority.items():
        pct = analysis.percentiles(data["queued_s"])
        report.aggregate_by_priority[prio] = {
            "n": len(data["queued_s"]),
            **pct,
            "mean_rank": sum(data["ranks"]) / len(data["ranks"]) if data["ranks"] else 0.0,
        }

    if len(ttft_pairs) >= 2:
        report.spearman["ttft"] = analysis.spearman(
            [p for p, _ in ttft_pairs], [v for _, v in ttft_pairs]
        )
    if len(done_pairs) >= 2:
        report.spearman["done"] = analysis.spearman(
            [p for p, _ in done_pairs], [v for _, v in done_pairs]
        )


#: loop_scope="session" -- MUST match the loop_scope on every async fixture
#: in conftest.py (admin/model_info/test_tokens/isolation). Without it this
#: test gets its own function-scoped event loop while those session-scoped
#: fixtures run on a persistent session loop; the first time the test body
#: reuses the admin fixture's already-connected httpx client from its own,
#: different loop, httpcore raises "is bound to a different event loop"
#: (see the comment above the `admin` fixture for the full trace).
@pytest.mark.asyncio(loop_scope="session")
async def test_priority_and_queueing(
    live_cfg: LiveConfig, admin, model_info: dict, test_tokens: dict, isolation: dict
) -> None:
    created = test_tokens["created"]
    filler = ProbeToken(
        label="filler",
        priority=live_cfg.filler_priority,
        token_id=created["filler"]["id"],
        plaintext=created["filler"]["plaintext"],
    )
    probes = [
        ProbeToken(
            label=label,
            priority=prio,
            token_id=created[label]["id"],
            plaintext=created[label]["plaintext"],
        )
        for label, prio in zip(live_cfg.probe_labels, live_cfg.priorities, strict=False)
    ]
    send_order = {label: i for i, label in enumerate(live_cfg.probe_labels)}

    oai = OpenAIClient(live_cfg.base_url, verify=live_cfg.httpx_verify)
    report = RunReport(
        run_id=test_tokens["run_id"],
        base_url_redacted=redact_base_url(live_cfg.base_url),
        model=live_cfg.model,
        backend=model_info.get("backend"),
        max_inflight=live_cfg.max_inflight,
        calibration={},
        paused_count=isolation["paused_count"],
        drain_seconds=isolation["drain_seconds"],
        foreign_inflight_at_pause=isolation["foreign_inflight_at_pause"],
    )
    hard_failures: list[str] = []

    try:
        async with asyncio.timeout(live_cfg.timeout_s):
            if live_cfg.dry_run:
                print(
                    "[tests/live] VW_LIVE_DRY_RUN=1: isolation + token lifecycle only, "
                    "exiting before load."
                )
                report.total_load_seconds = 0.0
                # Minor: a dry run proves isolation + token lifecycle, not
                # the priority/queueing claim this test exists to make --
                # SKIP, not PASS, so it can never be mistaken for a real
                # green result. finally below still writes the report.
                pytest.skip("VW_LIVE_DRY_RUN=1: isolation + token lifecycle rehearsed, no load run")

            calib: CalibrationResult = await calibrate(oai, cfg=live_cfg, filler_token=filler.plaintext)
            report.calibration = {
                "decode_rate_tps": round(calib.decode_rate_tps, 2),
                "base_tokens": calib.base_tokens,
                "step_tokens": calib.step_tokens,
            }
            print(f"[tests/live] calibration: {report.calibration}")

            round_index = 0
            load_start = time.monotonic()
            # I5: stop pausing everyone for up to VW_LIVE_MAX_LOAD_S just to
            # accumulate more of the same failure -- 2 consecutive
            # setup_failure rounds (our harness/environment, NOT foreign
            # traffic) abort the load loop early. The round(s) already
            # recorded are hard failures regardless; this only shortens how
            # long real users' tokens stay paused once it's clear the
            # result won't change.
            consecutive_setup_failures = 0

            while True:
                elapsed = time.monotonic() - load_start
                if round_index >= live_cfg.min_rounds and elapsed >= live_cfg.min_load_s:
                    break
                if elapsed >= live_cfg.max_load_s:
                    break

                rr = await run_round(
                    index=round_index,
                    cfg=live_cfg,
                    admin=admin,
                    oai=oai,
                    calib=calib,
                    model_row_id=model_info.get("id"),
                    filler=filler,
                    probes=probes,
                )
                filler0 = rr.filler_timings.get(0)
                filler0_duration_s = (
                    filler0.t_done - filler0.t_send
                    if filler0 is not None and filler0.t_done is not None
                    else None
                )
                round_report = _summarize_round(
                    rr,
                    probes=probes,
                    send_order=send_order,
                    cfg=live_cfg,
                    filler0_duration_s=filler0_duration_s,
                )
                report.rounds.append(round_report)
                print(
                    f"[tests/live] round {round_index}: tainted={rr.tainted} "
                    f"setup_failure={rr.setup_failure} ({rr.taint_reason or 'clean'})"
                )
                if rr.setup_failure:
                    # I5: a hard failure, never silently excluded.
                    hard_failures.append(f"round {round_index} setup failure: {rr.taint_reason}")
                    consecutive_setup_failures += 1
                elif not rr.tainted:
                    for check, violations in round_report.verdicts.items():
                        if violations:
                            hard_failures.append(f"round {round_index} {check}: {violations}")
                    consecutive_setup_failures = 0
                else:
                    # Foreign-traffic-only taint: excluded, reported, not a
                    # hard failure, and doesn't count toward the abort streak.
                    consecutive_setup_failures = 0
                round_index += 1

                if consecutive_setup_failures >= 2:
                    hard_failures.append(
                        "aborting load early: 2 consecutive setup_failure rounds"
                    )
                    break

            report.total_load_seconds = time.monotonic() - load_start
            report.clean_rounds = sum(1 for r in report.rounds if not r.tainted)
            report.tainted_rounds = sum(1 for r in report.rounds if r.tainted)

            _aggregate_report(report)

            if live_cfg.assert_ttft:
                for key in ("ttft", "done"):
                    rho = report.spearman.get(key)
                    if rho is None or rho > -0.8:
                        hard_failures.append(f"E_ttft_correlation({key}): rho={rho}, required <= -0.8")

            if report.clean_rounds < live_cfg.min_clean_rounds:
                hard_failures.append(
                    f"only {report.clean_rounds} clean round(s), need >= {live_cfg.min_clean_rounds}"
                )
    finally:
        await oai.aclose()
        json_path, md_path = report.write(live_cfg.report_dir)
        print(f"[tests/live] report written: {json_path} / {md_path}")
        print(report.to_markdown())

    assert not hard_failures, "\n".join(hard_failures)
