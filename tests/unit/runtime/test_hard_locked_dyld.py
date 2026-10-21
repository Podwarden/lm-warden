"""The three DYLD_* keys are the macOS spelling of LD_PRELOAD and must be
hard-locked exactly the way LD_PRELOAD already is (Metal design §5 security
addendum)."""

import pytest


def _model(extra_env: dict) -> object:
    class M:
        id = "test"
        gpu_indices = [0, 1]
        tensor_parallel_size = 2

    M.extra_env = extra_env
    return M()


@pytest.mark.parametrize(
    "key",
    ["DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH"],
)
def test_dyld_keys_are_hard_locked(key):
    """One leg per named key: all three sit in the global hard-locked set."""
    from app.runtime.backends.vllm.env import HARD_LOCKED_ENV_KEYS

    assert key in HARD_LOCKED_ENV_KEYS


def test_ld_preload_still_hard_locked():
    """Control: the original lock is unchanged, not just renamed."""
    from app.runtime.backends.vllm.env import HARD_LOCKED_ENV_KEYS

    assert "LD_PRELOAD" in HARD_LOCKED_ENV_KEYS


def test_bound_unnamed_dyld_key_is_not_locked():
    """Bound: only the three named keys are locked. DYLD_PRINT_LIBRARIES is a
    real DYLD_ variable (a debug trace toggle) that the addendum does not name,
    so it must stay out of the set -- a wider lock would be an overreach."""
    from app.runtime.backends.vllm.env import HARD_LOCKED_ENV_KEYS

    assert "DYLD_PRINT_LIBRARIES" not in HARD_LOCKED_ENV_KEYS


@pytest.mark.parametrize(
    "key",
    ["DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH"],
)
def test_setting_a_dyld_key_is_refused_at_write(key):
    """The extra-env write path refuses each DYLD key loudly, exactly as it
    refuses LD_PRELOAD today."""
    from app.runtime.backends.vllm.env import build_subprocess_env

    model = _model({key: "/tmp/evil"})
    with pytest.raises(ValueError, match="hard-locked"):
        build_subprocess_env(model, hf_token="t", hf_cache_dir="/c")


@pytest.mark.parametrize(
    "key",
    ["DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH"],
)
def test_dyld_key_is_refused_regardless_of_backend_allowlist(key):
    """The hard lock is checked FIRST and is global: even a backend allowlist
    that would accept DYLD_* prefixes cannot unlock one."""
    from app.runtime.backends.vllm.env import filter_extra_env

    with pytest.raises(ValueError, match="hard-locked"):
        filter_extra_env(
            {key: "/tmp/evil"},
            prefixes=("DYLD_",),
            exact=frozenset(),
        )
