from app.setup.state_machine import STEPS, next_step


def test_steps_in_order():
    assert STEPS == ["welcome", "gpus", "hf_token", "admin", "done"]


def test_next_step_advances():
    assert next_step("welcome") == "gpus"
    assert next_step("gpus") == "hf_token"
    assert next_step("hf_token") == "admin"
    assert next_step("admin") == "done"


def test_next_step_done_stays_done():
    assert next_step("done") == "done"


def test_next_step_unknown_raises():
    import pytest

    with pytest.raises(ValueError):
        next_step("garbage")
