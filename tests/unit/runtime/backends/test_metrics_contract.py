"""One reading shape, two dialects.

EngineReading is the seam between "which metric names does this engine use" (the
backend's problem) and "what does the dashboard show" (the stats layer's
problem). The contract has exactly one rule, and the whole degrade-don't-lie
property rests on it: every field is a number or None, and None means THIS
ENGINE DOES NOT REPORT IT -- never zero, never "idle".

Task 10 appends the llamacpp row to DIALECTS. This module is written as a table
from the start so that adding a backend is adding one line.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from app.runtime.backends.llamacpp import LlamaCppBackend
from app.runtime.backends.metrics import EngineReading
from app.runtime.backends.vllm import VllmBackend

_FIX = Path(__file__).resolve().parents[2] / "stats" / "fixtures"
VLLM_METRICS = (_FIX / "vllm_metrics_0_25_1.txt").read_text()
LLAMACPP_METRICS = (
    Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "llamacpp" / "metrics.txt"
).read_text()

# (backend, sample exposition body). Adding a backend is adding a line -- which
# is the whole point of writing this as a table.
DIALECTS = [
    pytest.param(VllmBackend(), VLLM_METRICS, id="vllm"),
    pytest.param(LlamaCppBackend(), LLAMACPP_METRICS, id="llamacpp"),
]


def test_engine_reading_is_frozen():
    r = EngineReading()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.requests_running = 1


def test_engine_reading_defaults_to_all_none():
    """A backend that fills in nothing still produces a legal reading, and every
    panel renders 'not reported'. That is what makes adding a backend additive
    rather than a schema negotiation."""
    r = EngineReading()
    for f in dataclasses.fields(EngineReading):
        assert getattr(r, f.name) is None


@pytest.mark.parametrize("backend,body", DIALECTS)
def test_parse_metrics_returns_a_reading(backend, body):
    assert isinstance(backend.parse_metrics(body), EngineReading)


@pytest.mark.parametrize("backend,body", DIALECTS)
def test_every_field_is_a_number_or_none(backend, body):
    r = backend.parse_metrics(body)
    for f in dataclasses.fields(EngineReading):
        v = getattr(r, f.name)
        if f.name.endswith("_hist"):
            assert v is None or (isinstance(v, tuple) and len(v) == 3)
        else:
            assert v is None or isinstance(v, int | float)


@pytest.mark.parametrize("backend,body", DIALECTS)
def test_empty_body_is_none_not_a_blank_reading(backend, body):
    """A failed or empty scrape is 'no data', which the stats layer already
    renders as scrape_error. A blank EngineReading would instead be rendered as
    a healthy engine reporting nothing -- a different, and wrong, story."""
    assert backend.parse_metrics("") is None
    assert backend.parse_metrics("   \n") is None


@pytest.mark.parametrize("backend,body", DIALECTS)
def test_garbage_body_yields_an_all_none_reading_and_does_not_raise(backend, body):
    r = backend.parse_metrics("<html>502 Bad Gateway</html>")
    assert r is not None
    assert r.requests_running is None


def test_vllm_reading_matches_the_fixture():
    r = VllmBackend().parse_metrics(VLLM_METRICS)
    assert r.requests_running == 3.0
    assert r.requests_waiting == 2.0
    assert r.waiting_capacity == 2.0
    assert r.kv_cache_usage_perc == 0.5
    assert r.prompt_tokens_total == 1000000.0
    assert r.generation_tokens_total == 500000.0
    assert r.preemptions_total == 12.0
