"""P6 (Metal design s8) — ``proc_pid_rusage`` -> ``ri_phys_footprint``.

``phys_footprint``/``footprints`` never raise and degrade every unreadable
footprint to None: None means "not reported", never 0. A fake ``libproc``
injected by the tests stands in for the real ctypes handle, so the whole
contract is exercised on Linux without a live process.

The fake writes through the ctypes pointer exactly as ``libproc`` would and
reads the field offset from ``proc``'s own struct definition — the same
layout the production code pins, never a private copy of it. A by-value
call, a dropped ``byref`` or a wrong offset turns the success leg red
instead of passing on shared layout."""

import os
import sys

import pytest


class _FakeLibProc:
    """A stand-in for a ctypes libproc handle.

    ``proc_pid_rusage`` must receive a ctypes pointer (``ctypes.byref`` or a
    ``POINTER``); the fake writes ``rusage`` into the buffer's
    ``ri_phys_footprint`` at the offset ``proc``'s struct reports — through
    the pointer, as ``libproc`` would. ``rc`` is the return code (0 =
    success, nonzero = "not reported"), ``raise_exc`` an exception to raise
    instead of calling. ``calls`` records every ``proc_pid_rusage``
    invocation so a test can assert the call never happened."""

    def __init__(self, rc=0, rusage=12345, raise_exc=None):
        self._rc = rc
        self._rusage = rusage
        self._raise = raise_exc
        self.calls = []

    def proc_pid_rusage(self, pid, flavor, buf):
        self.calls.append((pid, flavor))
        if self._raise is not None:
            raise self._raise
        import app.system.apple.proc as proc

        if self._rusage is not None:
            self._write_footprint(buf, self._rusage, proc._RusageInfoV2)
        return self._rc

    @staticmethod
    def _write_footprint(buf, value, rusage):
        """Write ``value`` at the struct's ``ri_phys_footprint`` offset.

        Raises (and the success leg goes red) when ``buf`` is not a ctypes
        pointer — a by-value call — or when the layout is not what ``proc``
        pins."""
        import ctypes

        if not isinstance(buf, type(ctypes.byref(rusage()))):
            raise TypeError(f"proc_pid_rusage got a by-value buffer, not a pointer: {buf!r}")
        addr = ctypes.c_void_p(
            ctypes.cast(buf, ctypes.c_void_p).value + rusage.ri_phys_footprint.offset
        )
        src = (ctypes.c_uint8 * 8)(*[b for b in value.to_bytes(8, "little")])
        ctypes.memmove(ctypes.cast(addr, ctypes.POINTER(ctypes.c_uint8)), src, 8)


def test_layout_matches_sdk_rusage_info_v2():
    # _RusageInfoV2 must mirror the SDK's struct rusage_info_v2
    # (sys/resource.h) exactly: ri_uuid (16 bytes) then 18 unsigned long
    # long fields, so the struct is 16 + 18 * 8 == 160 bytes and
    # ri_phys_footprint (the 8th field, index 8) sits at 16 + 7 * 8 == 72.
    # A wrong field list, an assumed alignment or a missing uuid all change
    # one of these two numbers.
    import ctypes

    import app.system.apple.proc as proc

    assert ctypes.sizeof(proc._RusageInfoV2) == 160
    assert proc._RusageInfoV2.ri_phys_footprint.offset == 72


def test_layout_field_count_and_uuid():
    import ctypes

    import app.system.apple.proc as proc

    r = proc._RusageInfoV2
    inst = r()
    # 1 uuid field + 18 uint64 fields, nothing else.
    assert len(r._fields_) == 19
    assert r._fields_[0] == ("ri_uuid", ctypes.c_uint8 * 16)
    # The footprint sits 7 uint64 fields after the uuid: 16 + 7 * 8 == 72.
    assert r._fields_[8][0] == "ri_phys_footprint"
    assert (
        r.ri_phys_footprint.offset
        == ctypes.addressof(inst.ri_uuid) - ctypes.addressof(inst) + 16 + 7 * 8
    )


def test_fake_success_reads_phys_footprint():
    # (a) a fake libproc that fills ri_phys_footprint returns it, byte for
    # byte, read back from the pointer the production code handed in.
    from app.system.apple.proc import phys_footprint

    fake = _FakeLibProc(rc=0, rusage=12345)
    assert phys_footprint(1, libproc=fake) == 12345
    assert fake.calls == [(1, 2)]


def test_fake_success_zero_rc_and_zero_footprint_boundaries():
    # rc exactly 0 is the success bound and a footprint of 0 is a real
    # reading (0 bytes), not the "not reported" sentinel.
    from app.system.apple.proc import phys_footprint

    fake = _FakeLibProc(rc=0, rusage=0)
    assert phys_footprint(1, libproc=fake) == 0


def test_fake_nonzero_rc_returns_none():
    # (b) rc just past the success bound (1) means "not reported" -> None,
    # and the filled buffer is never read back.
    from app.system.apple.proc import phys_footprint

    fake = _FakeLibProc(rc=1, rusage=99999)
    assert phys_footprint(1, libproc=fake) is None


def test_fake_raising_returns_none():
    # (c) any exception from the call degrades to None, never raises.
    from app.system.apple.proc import phys_footprint

    fake = _FakeLibProc(rc=0, rusage=12345, raise_exc=OSError("no such process"))
    assert phys_footprint(1, libproc=fake) is None


def test_linux_no_injection_returns_none(monkeypatch):
    # (d) off darwin with no libproc injected, the result is None without any
    # ctypes work (this whole lane runs on Linux).
    from app.system.apple import proc

    monkeypatch.setattr(proc.sys, "platform", "linux")
    assert proc.phys_footprint(1) is None


@pytest.mark.skipif(sys.platform != "darwin", reason="needs a live darwin libproc")
@pytest.mark.macos
def test_darwin_live_pid_footprint_is_plausible():
    # On darwin, a real (this) pid must read back a real footprint: an int
    # between 1 MiB and 64 GiB. A 0 reading is a broken offset, and a dead
    # pid cannot catch one — it only proves the None path.
    from app.system.apple.proc import phys_footprint

    value = phys_footprint(os.getpid())
    assert isinstance(value, int)
    assert 2**20 < value < 64 * 2**30


def test_footprints_mixes_int_and_none():
    # (e) one pid succeeds, the other gets a nonzero rc: the dict mixes an
    # int and a None and is keyed by the pid passed in.
    from app.system.apple.proc import phys_footprint

    result = {
        1: phys_footprint(1, libproc=_FakeLibProc(rc=0, rusage=5000)),
        2: phys_footprint(2, libproc=_FakeLibProc(rc=1, rusage=777)),
    }
    assert result == {1: 5000, 2: None}


def test_footprints_helper_maps_every_pid_in_order(monkeypatch):
    # footprints(pids) maps each pid through phys_footprint, in input order,
    # keyed by it; per-pid injection is proven above via phys_footprint.
    from app.system.apple import proc

    seen = []

    def fake_pf(pid, *, libproc=None):
        seen.append(pid)
        return {3: 42, 4: None, 5: 42}[pid]

    monkeypatch.setattr(proc, "phys_footprint", fake_pf)
    assert proc.footprints([3, 4, 5]) == {3: 42, 4: None, 5: 42}
    assert seen == [3, 4, 5]


def test_pid_zero_returns_none_without_calling():
    # (f) pid 0 is the bound: the libproc is never invoked.
    from app.system.apple.proc import phys_footprint

    fake = _FakeLibProc(rc=0, rusage=12345)
    assert phys_footprint(0, libproc=fake) is None
    assert fake.calls == []


def test_negative_pid_returns_none_without_calling():
    # (f) a negative pid is just past the bound: also never invoked.
    from app.system.apple.proc import phys_footprint

    fake = _FakeLibProc(rc=0, rusage=12345)
    assert phys_footprint(-1, libproc=fake) is None
    assert fake.calls == []
