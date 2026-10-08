"""The llama.cpp fixtures are real captured output, and they say where from.

Sub-project C writes its argv, its log grammar and its metric parser against
these files rather than against anyone's memory of what llama-server prints.
That only holds if the files stay honest, so this module asserts the four things
that would make them dishonest: a missing capture, a fixture that no longer
parses, a README that no longer records which build produced them, and a log
polluted with terminal escapes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

FIX = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "llamacpp"

REQUIRED = (
    "README.md",
    "version.txt",
    "help.txt",
    "metrics.txt",
    "metrics_second_scrape.txt",
    "health_ready.json",
    "health_loading.json",
    "models.json",
    "props.json",
    "log_startup_ok.txt",
    "log_oom.txt",
    "log_unknown_arch.txt",
    "log_ctx_capped.txt",
    "log_wrong_shard.txt",
)


@pytest.mark.parametrize("name", REQUIRED)
def test_fixture_present_and_non_empty(name):
    p = FIX / name
    assert p.is_file(), f"missing capture {name}"
    assert p.stat().st_size > 0


def test_readme_records_the_build_tag():
    text = (FIX / "README.md").read_text()
    assert re.search(r"\bb\d{5,6}\b", text), (
        "README must record the exact llama.cpp build tag (bNNNNN) the captures "
        "came from; Task 13 pins the same tag in the Dockerfile"
    )


def test_health_ready_is_the_documented_body():
    assert json.loads((FIX / "health_ready.json").read_text()) == {"status": "ok"}


def test_health_loading_is_a_503_error_envelope():
    body = json.loads((FIX / "health_loading.json").read_text())
    assert body["error"]["code"] == 503
    assert body["error"]["type"] == "unavailable_error"


def test_metrics_use_the_llamacpp_prefix():
    text = (FIX / "metrics.txt").read_text()
    assert "llamacpp:" in text
    assert "vllm:" not in text


def test_logs_carry_no_ansi_escapes():
    """Property 2 from the capture checklist. The supervisor writes the engine
    log to a plain file; escape codes would pollute the log viewer and break the
    diagnosis regexes. If this fails, the fix is a documented llama.cpp logging
    flag added to the argv in Task 7 -- never a wrapper program (spec §2)."""
    for p in sorted(FIX.glob("log_*.txt")):
        assert "\x1b[" not in p.read_text(), f"{p.name} contains ANSI escapes"


def test_counters_are_cumulative_and_only_the_gauges_reset():
    """Property 1, pinned as a test rather than left as a README note.

    Task 10 derives throughput from the COUNTERS because llama.cpp resets the
    two ``*_tokens_seconds`` gauges' bucket on every scrape
    (server-context.cpp:4657-4659) -- two scrapers would otherwise halve each
    other's numbers. If a counter ever starts resetting too, that rate math is
    wrong and this is where it surfaces.
    """
    from app.stats.prometheus import Metrics, parse_prometheus

    first = Metrics(parse_prometheus((FIX / "metrics.txt").read_text()))
    second = Metrics(parse_prometheus((FIX / "metrics_second_scrape.txt").read_text()))

    for counter in (
        "llamacpp:prompt_tokens_total",
        "llamacpp:prompt_tokens_cached_total",
        "llamacpp:prompt_seconds_total",
        "llamacpp:tokens_predicted_total",
        "llamacpp:tokens_predicted_seconds_total",
        "llamacpp:n_decode_total",
        "llamacpp:n_tokens_max",
    ):
        a, b = first.value(counter), second.value(counter)
        assert a is not None, f"{counter} absent from the first scrape"
        assert b == a, f"{counter} moved between scrapes: {a} -> {b}"

    for gauge in (
        "llamacpp:prompt_tokens_seconds",
        "llamacpp:predicted_tokens_seconds",
    ):
        assert first.value(gauge) > 0, f"{gauge} was already 0 on the first scrape"
        assert second.value(gauge) == 0.0, f"{gauge} did not reset on rescrape"


def test_the_documented_failure_causes_are_actually_in_the_logs():
    """The four failure fixtures each have to *contain* the thing Task 9 keys
    on, or the grammar written against them is untested."""
    cases = {
        "log_unknown_arch.txt": "unknown model architecture: 'notarealarch'",
        "log_ctx_capped.txt": "exceeds the training context of the model",
        "log_wrong_shard.txt": "model must be loaded with the first split",
        "log_oom.txt": "failed to allocate buffer for kv cache",
    }
    for name, needle in cases.items():
        assert needle in (FIX / name).read_text(), f"{name} no longer contains {needle!r}"


def test_log_tails_fit_the_diagnoser_window():
    """app/models/routes_api.py:_read_engine_log_tail(max_lines=200) is what
    feeds the diagnoser, so a fixture longer than that would be testing text the
    product never sees."""
    for p in sorted(FIX.glob("log_*.txt")):
        assert len(p.read_text().splitlines()) <= 200, f"{p.name} exceeds the 200-line tail"
