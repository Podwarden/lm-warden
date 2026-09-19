"""Report writers: text (printed with -s), JSON, Markdown.

Only token ids and our own generated names ever appear -- never plaintext
keys, never credentials, never another operator's token names (counts
only), never a hostname (the base_url is redacted to scheme://<redacted>
before it ever reaches a RunReport).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RoundReport:
    index: int
    tainted: bool
    taint_reason: str | None
    #: True iff `tainted` is a hard-failure (harness/scheduler) cause rather
    #: than mere foreign-traffic contamination -- see RoundResult in _load.py.
    setup_failure: bool
    saturation_snapshot: dict | None
    burst_snapshot: dict | None
    filler_completion_order: list[int]
    #: one dict per probe: label, priority, send#, admit#, started_iso,
    #: queued_s, client_ttft_s, duration_s, finish -- ordered by admission.
    rows: list[dict]
    #: check name -> list of violation strings (empty list = PASS)
    verdicts: dict[str, list[str]]
    errors: list[str] = field(default_factory=list)


@dataclass
class RunReport:
    run_id: str
    base_url_redacted: str
    model: str
    backend: str | None
    max_inflight: int
    calibration: dict
    paused_count: int
    drain_seconds: float
    foreign_inflight_at_pause: int
    rounds: list[RoundReport] = field(default_factory=list)
    aggregate_by_priority: dict[int, dict] = field(default_factory=dict)
    spearman: dict[str, float | None] = field(default_factory=dict)
    total_load_seconds: float = 0.0
    clean_rounds: int = 0
    tainted_rounds: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "base_url": self.base_url_redacted,
            "model": self.model,
            "backend": self.backend,
            "max_inflight": self.max_inflight,
            "calibration": self.calibration,
            "paused_count": self.paused_count,
            "drain_seconds": self.drain_seconds,
            "foreign_inflight_at_pause": self.foreign_inflight_at_pause,
            "rounds": [asdict(r) for r in self.rounds],
            "aggregate_by_priority": self.aggregate_by_priority,
            "spearman": self.spearman,
            "total_load_seconds": self.total_load_seconds,
            "clean_rounds": self.clean_rounds,
            "tainted_rounds": self.tainted_rounds,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str)

    def to_markdown(self) -> str:
        lines = [
            f"# Live priority/queueing run {self.run_id}",
            "",
            f"- model: `{self.model}` backend: `{self.backend}`",
            f"- max_inflight: {self.max_inflight}",
            f"- calibration: {self.calibration}",
            f"- paused tokens: {self.paused_count}, drain: {self.drain_seconds:.1f}s, "
            f"foreign in-flight at pause: {self.foreign_inflight_at_pause}",
            f"- rounds: {len(self.rounds)} run / {self.clean_rounds} clean / "
            f"{self.tainted_rounds} tainted",
            f"- total load seconds: {self.total_load_seconds:.1f}",
            "",
            "## Per-round verdicts",
            "",
        ]
        for r in self.rounds:
            title = f"### Round {r.index}"
            if r.setup_failure:
                title += f" (SETUP FAILURE / HARD FAIL: {r.taint_reason or 'unspecified'})"
            elif r.tainted:
                title += f" (TAINTED / excluded: {r.taint_reason or 'unspecified'})"
            lines.append(title)
            lines.append("")
            lines.append(
                "| label | priority | send# | admit# | started_iso | queued_s | "
                "client_ttft_s | duration_s | finish |"
            )
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for row in r.rows:
                lines.append(
                    "| {label} | {priority} | {send} | {admit} | {started_iso} | "
                    "{queued_s} | {client_ttft_s} | {duration_s} | {finish} |".format(**row)
                )
            lines.append("")
            for check, violations in r.verdicts.items():
                verdict = "PASS" if not violations else "FAIL"
                line = f"- **{check}: {verdict}**"
                if violations:
                    line += " -- " + "; ".join(violations)
                lines.append(line)
            if r.errors:
                lines.append(f"- errors: {'; '.join(r.errors)}")
            if r.saturation_snapshot is not None:
                lines.append(f"- saturation_snapshot: {r.saturation_snapshot}")
            if r.burst_snapshot is not None:
                lines.append(f"- burst_snapshot: {r.burst_snapshot}")
            if r.filler_completion_order:
                lines.append(f"- filler completion order: {r.filler_completion_order}")
            lines.append("")
        lines.append("## Aggregate per priority")
        lines.append("")
        lines.append("| priority | n | queued_s p50 | queued_s p95 | queued_s max | mean admit rank |")
        lines.append("|---|---|---|---|---|---|")
        for prio, agg in sorted(self.aggregate_by_priority.items()):
            lines.append(
                f"| {prio} | {agg['n']} | {agg['p50']:.3f} | {agg['p95']:.3f} | "
                f"{agg['max']:.3f} | {agg['mean_rank']:.2f} |"
            )
        lines.append("")
        lines.append(f"Spearman (priority vs client TTFT): {self.spearman.get('ttft')}")
        lines.append(f"Spearman (priority vs client done): {self.spearman.get('done')}")
        return "\n".join(lines)

    def write(self, report_dir: str) -> tuple[str, str]:
        os.makedirs(report_dir, exist_ok=True)
        json_path = os.path.join(report_dir, f"{self.run_id}.json")
        md_path = os.path.join(report_dir, f"{self.run_id}.md")
        with open(json_path, "w") as f:
            f.write(self.to_json())
        with open(md_path, "w") as f:
            f.write(self.to_markdown())
        return json_path, md_path
