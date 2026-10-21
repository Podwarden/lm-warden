"""The stress probe posts the engine's own model id, not the served name.

mlx_lm.server treats the request's ``model`` as a Hugging Face repo or path, so
the user-facing served name 404s there (same bug as the warm-up probe, 7d4d673a).
"""

from __future__ import annotations

import httpx
import pytest

from app.stress.gates import Reference
from app.stress.probes import build_suite
from tests.unit.stress.test_runner import FakeControl, db, engine, make_runner  # noqa: F401


class _Spy:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def stream(self, method, url, json=None, **_kw):
        self.payloads.append(json)
        raise httpx.ConnectError("spy")


async def _model_sent(control, db) -> str:  # noqa: F811
    run = make_runner(control, db)
    spy = _Spy()
    run._client = spy
    probe = build_suite(run_id="r", seed=1, approx_tokens=64)[0]
    await run._execute(probe, ref=Reference(output_tokens=0, checks_length=False), deadline_s=2.0)
    return spy.payloads[0]["model"]


@pytest.mark.parametrize(
    ("engine_id", "expect"),
    [("/cache/mlx/Qwen-4bit", "/cache/mlx/Qwen-4bit"), (None, "fake-model")],
    ids=["mlx-like", "no-engine-id"],
)
async def test_probe_uses_engine_model_id(engine, db, engine_id, expect):  # noqa: F811
    control = FakeControl(engine)
    control.engine_model_id = lambda: engine_id
    assert await _model_sent(control, db) == expect


async def test_probe_tolerates_control_without_accessor(engine, db):  # noqa: F811
    assert await _model_sent(FakeControl(engine), db) == "fake-model"
