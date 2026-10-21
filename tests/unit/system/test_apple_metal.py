"""P6 (Metal design s7) — the device ceiling probe:
``MTLDevice.recommendedMaxWorkingSetSize`` read through pyobjc's ``Metal``.

Same contract as every Apple probe: never raise; None when Metal is not
there (Linux, no pyobjc, no device) or the reading is not a positive int.
The pure tests inject a fake ``Metal`` module, so they run on Linux.
"""

import platform
import subprocess
import sys
import types

import pytest


@pytest.fixture
def metal(monkeypatch):
    """The probe module with its process-wide device cache cleared."""
    from app.system.apple import metal as mod

    mod._reset_cache()
    yield mod
    mod._reset_cache()


def _fake_metal(monkeypatch, create):
    fake = types.ModuleType("Metal")
    fake.MTLCreateSystemDefaultDevice = create
    monkeypatch.setitem(sys.modules, "Metal", fake)
    return fake


class _Device:
    def __init__(self, value):
        self._value = value

    def recommendedMaxWorkingSetSize(self):  # noqa: N802 - Objective-C selector name
        if isinstance(self._value, BaseException):
            raise self._value
        return self._value


def _require_pyobjc_metal():
    """Skip a live test, with the reason, when pyobjc's Metal bridge is not
    installed (the app then falls back to hw.memsize, which the pure tests
    pin). Once ``import Metal`` works the live test runs and a None or
    out-of-range ceiling is a FAILURE, not a skip."""
    try:
        import Metal  # noqa: F401
    except ImportError as error:
        pytest.skip(
            "pyobjc-framework-Metal not importable "
            f"({error}); install requirements-macos.txt to run the live probe"
        )


def test_ceiling_read_from_default_device(monkeypatch, metal):
    _fake_metal(monkeypatch, lambda: _Device(12713115648))
    assert metal.device_ceiling() == 12713115648


def test_no_pyobjc_is_none(monkeypatch, metal):
    # sys.modules[name] = None makes ``import Metal`` raise ImportError.
    monkeypatch.setitem(sys.modules, "Metal", None)
    assert metal.device_ceiling() is None


def test_no_device_is_none(monkeypatch, metal):
    _fake_metal(monkeypatch, lambda: None)
    assert metal.device_ceiling() is None


def test_create_raising_is_none(monkeypatch, metal):
    def boom():
        raise RuntimeError("no GPU")

    _fake_metal(monkeypatch, boom)
    assert metal.device_ceiling() is None


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "12713115648", None, 2**63])
def test_bad_reading_is_none(monkeypatch, metal, value):
    _fake_metal(monkeypatch, lambda: _Device(value))
    assert metal.device_ceiling() is None


def test_property_raising_is_none(monkeypatch, metal):
    _fake_metal(monkeypatch, lambda: _Device(RuntimeError("selector gone")))
    assert metal.device_ceiling() is None


def test_device_created_once_and_property_reread(monkeypatch, metal):
    created = []
    device = _Device(10 * 2**30)

    def create():
        created.append(1)
        return device

    _fake_metal(monkeypatch, create)
    assert metal.device_ceiling() == 10 * 2**30
    # The ceiling is re-read each call (it may follow a wired-limit change);
    # the device is created once.
    device._value = 11 * 2**30
    assert metal.device_ceiling() == 11 * 2**30
    assert created == [1]


def test_missing_metal_is_cached_too(monkeypatch, metal):
    monkeypatch.setitem(sys.modules, "Metal", None)
    assert metal.device_ceiling() is None
    # A later import would succeed, but the cached "no Metal" answer stands:
    # the probe does not retry an import on every GPU poll.
    _fake_metal(monkeypatch, lambda: _Device(5))
    assert metal.device_ceiling() is None


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="needs a live Apple Silicon host",
)
@pytest.mark.macos
def test_darwin_live_device_ceiling(metal):
    _require_pyobjc_metal()
    ceiling = metal.device_ceiling()
    memsize = int(
        subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=10
        ).stdout
    )
    print(f"recommendedMaxWorkingSetSize={ceiling} hw.memsize={memsize}")
    assert isinstance(ceiling, int), "pyobjc Metal must read recommendedMaxWorkingSetSize"
    assert 0 < ceiling <= memsize


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="needs a live Apple Silicon host",
)
@pytest.mark.macos
async def test_darwin_live_budget_uses_real_ceiling(metal, monkeypatch):
    """The live Mac budget (GPU table total / shell usable_bytes) is the
    device ceiling, not the hw.memsize stand-in, and admission's weights
    budget is the device budget itself (no OS reserve on the ceiling)."""
    _require_pyobjc_metal()
    from app.models.capacity import unified_weights_budget
    from app.system import gpu

    cap = await gpu._apple_capacity()
    print(f"live capacity {cap!r} device_budget={cap.device_budget}")
    ceiling = metal.device_ceiling()
    assert cap.device_ceiling == ceiling
    assert cap.memsize and 0 < cap.device_budget <= cap.memsize
    assert cap.device_budget <= ceiling < cap.memsize
    assert await gpu._apple_budget_mib() == cap.device_budget // 2**20
    # A readable Metal ceiling already leaves the OS its share: no reserve.
    assert cap.has_device_ceiling
    assert cap.weights_budget() == cap.device_budget
    assert cap.weights_budget() == unified_weights_budget(
        cap.device_budget, 0, cap.reserve_basis, 0, 0, os_reserve=False
    )
