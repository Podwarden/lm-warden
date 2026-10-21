"""LMW_PLATFORM_PROFILE selects the registry's backend set and default.

Step 2 of the DEFAULT_BACKEND migration (Metal design s10, OVERSEER decision,
planner iter 22): the registry answers "which backends, and what is the
default" from a deployment PROFILE, not from ``sys.platform``. Resolving by
platform would make the whole unit suite see a different registry on a
developer's Mac, on the Mac merge gate, and in CI. The Swift launcher sets
``LMW_PLATFORM_PROFILE=apple`` (a P7 follow-up); everything that does not opt
in stays ``linux`` and is byte-identical to today.

Every NEW name in this file is imported INSIDE the test functions, so the
file still collects on a tree that does not have them yet (a module-level
import of a missing name makes the whole file an ImportError, which is red
for the wrong reason).
"""

import os
import re
import subprocess
import sys

import pytest

from app.runtime.backends import registry


def _sys_platform_reads(module):
    """Occurrences of ``sys.platform`` in the module's source.

    Re-scan per assertion: the module is imported once and shared, so a test
    that monkeypatches its source would otherwise be defeated by the first
    scan having cached the original.
    """
    import inspect

    return len(re.findall(r"sys\.platform", inspect.getsource(module)))


# ---------------------------------------------------------------------------
# app/platform_profile.py
# ---------------------------------------------------------------------------


def test_resolve_profile_unset_defaults_to_linux():
    from app.platform_profile import resolve_profile

    assert resolve_profile({}) == "linux"


def test_resolve_profile_empty_string_defaults_to_linux():
    """An exported-but-empty variable is UNSET, not a profile name."""
    from app.platform_profile import resolve_profile

    assert resolve_profile({"LMW_PLATFORM_PROFILE": ""}) == "linux"


def test_resolve_profile_apple_is_apple():
    from app.platform_profile import resolve_profile

    assert resolve_profile({"LMW_PLATFORM_PROFILE": "apple"}) == "apple"


def test_resolve_profile_linux_is_linux():
    from app.platform_profile import resolve_profile

    assert resolve_profile({"LMW_PLATFORM_PROFILE": "linux"}) == "linux"


@pytest.mark.parametrize(
    "value",
    ["Apple", "APPLE", "mac", "macos", "linux ", " linux", "linux\n"],
)
def test_resolve_profile_unknown_values_raise_naming_the_accepted(value):
    """EXACT match, case-sensitive: no title-casing, no trimming, no aliases.
    The arms are one per way a value can miss the boundary -- wrong case,
    an alias of a platform, and whitespace (leading/trailing/newline) that a
    ``.strip()`` would quietly accept."""
    from app.platform_profile import resolve_profile

    with pytest.raises(ValueError) as e:
        resolve_profile({"LMW_PLATFORM_PROFILE": value})
    # The message names BOTH accepted values: an operator who typo'd the
    # profile gets the fix in the same sentence as the refusal.
    assert "linux" in str(e.value)
    assert "apple" in str(e.value)


def test_profile_constant_is_linux_when_the_variable_is_unset(monkeypatch):
    """``PROFILE = resolve_profile(os.environ)`` at import; CI and every test
    run without the variable, so the constant is the linux answer."""
    monkeypatch.delenv("LMW_PLATFORM_PROFILE", raising=False)
    import app.platform_profile as pp

    assert pp.PROFILE == "linux"


# ---------------------------------------------------------------------------
# registry.backends_for / registry.default_for
# ---------------------------------------------------------------------------


def test_backends_for_linux_lists_both_sorted():
    from app.runtime.backends.llamacpp import LlamaCppBackend
    from app.runtime.backends.vllm import VllmBackend

    backends = registry.backends_for("linux")
    assert sorted(backends) == ["llamacpp", "vllm"]
    assert isinstance(backends["vllm"], VllmBackend)
    assert isinstance(backends["llamacpp"], LlamaCppBackend)


def test_backends_for_apple_has_no_vllm():
    """The red-first assertion: apple serves llama.cpp and MLX, no vllm.
    (Red on the base tree as AttributeError: registry has no backends_for.)"""
    from app.runtime.backends.llamacpp import LlamaCppBackend

    backends = registry.backends_for("apple")
    assert "vllm" not in backends
    assert sorted(backends) == ["llamacpp", "mlx"]
    assert isinstance(backends["llamacpp"], LlamaCppBackend)


def test_backends_for_is_pure_not_the_module_dict():
    """A mutation of the answer must not leak into the live registry or into a
    second call: the profile is a question, not the one global. The live
    dict is identity-checked, not re-derived -- backends_for answers fresh,
    so re-deriving the linux set here would compare two different dicts."""
    apple = registry.backends_for("apple")
    sentinel = object()
    apple["vllm"] = sentinel
    assert "vllm" in registry._BACKENDS
    assert registry._BACKENDS["vllm"] is not sentinel
    assert sorted(registry.backends_for("apple")) == ["llamacpp", "mlx"]


def test_backends_for_unknown_profile_raises():
    """The registry refuses a profile it has never heard of, the way get()
    refuses a backend name: loud, at the seam, naming what was asked."""
    with pytest.raises(ValueError):
        registry.backends_for("windows")


def test_default_for_linux_is_vllm():
    assert registry.default_for("linux") == "vllm"


def test_default_for_apple_is_mlx():
    """MLX registers in the apple profile and is its default (OVERSEER
    DEFAULT_BACKEND decision, planner iter 22)."""
    assert registry.default_for("apple") == "mlx"


def test_default_for_unknown_profile_raises():
    with pytest.raises(ValueError):
        registry.default_for("windows")


@pytest.mark.parametrize("profile", ["linux", "apple"])
def test_default_is_a_member_of_its_own_backend_set(profile):
    """A default that is not in the set would make get(None) raise on every
    legacy NULL row of that profile. The migration's whole point is that
    NULL stays servable on both profiles."""
    assert registry.default_for(profile) in registry.backends_for(profile)


# ---------------------------------------------------------------------------
# module level: the live registry answers from LMW_PLATFORM_PROFILE
# ---------------------------------------------------------------------------


def test_module_level_linux_defaults_are_unchanged():
    """With the variable unset, the module behaves exactly as today: both
    backends, sorted, default vllm, and get()/is_known() keep their
    contracts (get(None) -> the default; unknown -> UnknownBackendError)."""
    assert registry.DEFAULT_BACKEND == "vllm"
    assert registry.available() == ("llamacpp", "vllm")
    assert registry.get(None).capabilities.name == "vllm"
    assert registry.get("").capabilities.name == "vllm"
    assert registry.is_known(None)
    assert not registry.is_known("sglang")
    with pytest.raises(registry.UnknownBackendError):
        registry.get("sglang")


def test_module_backend_dict_comes_from_the_profile():
    """_BACKENDS and DEFAULT_BACKEND are the PROFILE's answers: same keys,
    same default, and -- the point of the migration -- built from the
    profile resolved at import, not hard-coded. A second import under a
    different profile would get a different registry from the same source."""
    import app.platform_profile as pp

    assert registry.available() == tuple(sorted(registry.backends_for(pp.PROFILE)))
    assert registry.DEFAULT_BACKEND == registry.default_for(pp.PROFILE)


def test_no_sys_platform_in_platform_profile_or_registry():
    """The decision that was rejected: resolving by sys.platform. Neither
    module may read the platform; the profile env var is the only input."""
    import app.platform_profile as pp

    assert _sys_platform_reads(pp) == 0
    assert _sys_platform_reads(registry) == 0


# ---------------------------------------------------------------------------
# the live registry under each profile: a fresh interpreter per profile,
# because PROFILE / _BACKENDS / DEFAULT_BACKEND are bound once at import
# ---------------------------------------------------------------------------

_LIVE_PROBE = (
    "from app.runtime.backends import registry as r\n"
    "print(r.DEFAULT_BACKEND, ','.join(r.available()), r.get(None).capabilities.name)\n"
    "try:\n"
    "    r.get('vllm')\n"
    "    print('vllm-served')\n"
    "except r.UnknownBackendError:\n"
    "    print('vllm-unknown')\n"
)


def _run_probe(profile):
    env = {k: v for k, v in os.environ.items() if k != "LMW_PLATFORM_PROFILE"}
    if profile is not None:
        env["LMW_PLATFORM_PROFILE"] = profile
    return subprocess.run(
        [sys.executable, "-c", _LIVE_PROBE],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        (None, "vllm llamacpp,vllm vllm\nvllm-served\n"),
        ("", "vllm llamacpp,vllm vllm\nvllm-served\n"),
        ("linux", "vllm llamacpp,vllm vllm\nvllm-served\n"),
        ("apple", "mlx llamacpp,mlx mlx\nvllm-unknown\n"),
    ],
)
def test_live_registry_follows_the_environment(profile, expected):
    """The module-level binding, not only the pure helpers: on apple the live
    registry serves llama.cpp and MLX, get(None) resolves to MLX and get('vllm')
    raises UnknownBackendError; unset/''/linux are the unchanged linux set."""
    proc = _run_probe(profile)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == expected


def test_live_registry_refuses_an_unknown_profile_at_import():
    """A typo fails the import (and so startup) loudly, naming the accepted
    values -- never a silent fallback to the linux set."""
    proc = _run_probe("Apple")
    assert proc.returncode != 0
    assert "ValueError" in proc.stderr
    assert "LMW_PLATFORM_PROFILE='Apple'" in proc.stderr
    assert "linux, apple" in proc.stderr
