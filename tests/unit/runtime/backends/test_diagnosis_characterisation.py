import pytest

from app.runtime.backends.vllm.diagnostics import diagnose_engine_log
from tests.unit.runtime.backends.corpus import LOG_CASES, load_diagnosis_golden

GOLDEN = load_diagnosis_golden()


@pytest.mark.parametrize("name,text", LOG_CASES, ids=[n for n, _ in LOG_CASES])
def test_diagnosis_matches_golden(name, text):
    d = diagnose_engine_log(text)
    actual = (
        None
        if d is None
        else {"message": d.message, "recommended_max_model_len": d.recommended_max_model_len}
    )
    assert actual == GOLDEN[name]


def test_config_echo_never_diagnoses_trust_remote_code():
    """The 2026-08-18 regression, pinned as its own named assertion.

    A TP-worker hang was reported as 'This model requires trust_remote_code'
    for 1h43m because the bare substring matched vLLM's own config echo. This
    test states the invariant in prose so a future reader knows why the regex
    in diagnostics.py is anchored on advice-shaped sentences only.
    """
    assert diagnose_engine_log("EngineArgs(model='org/m', trust_remote_code=False)") is None
    assert diagnose_engine_log("EngineArgs(model='org/m', trust_remote_code=True)") is None
