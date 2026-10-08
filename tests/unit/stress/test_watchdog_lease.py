"""The lease's integration with the watchdog (design §5.5).

The lease is only worth anything at the point `wants_restart` is consulted,
because that is the single predicate BOTH recovery paths use --
`restore_after_warden_restart` (watchdog.py:666) and `restart_crashed_models`
(watchdog.py:707).

v1 of the design guarded `check_once` instead. That is the path that does NOT
fire on a crash: it only probes rows whose status is 'loaded' (watchdog.py:761),
and a crash flips the row to 'failed' first. So the guard would have sat on the
wrong function while the runner and the sweep both called `_restart` -- which
force-unloads unconditionally (watchdog.py:546) -- and whichever lost the race
would have destroyed the other's freshly loaded engine.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.runtime.watchdog import wants_restart
from app.stress.lease import LeaseRegistry


@dataclass
class _Row:
    id: str
    status: str
    prior_status: str | None


def test_a_crashed_model_is_normally_restartable():
    assert wants_restart(_Row("m1", "failed", "loaded")) is True


def test_a_leased_model_is_left_to_its_stress_run():
    """The run detects the crash itself and recovers on its own schedule.

    It is already probing far more tightly than the watchdog's
    interval 30s x threshold 3 ~= 90s detection latency.
    """
    leases = LeaseRegistry(now=lambda: 1000.0)
    leases.take("m1", run_id="r1")
    assert wants_restart(_Row("m1", "failed", "loaded"), leases=leases) is False


def test_an_unleased_model_is_still_restarted_during_someone_else_s_run():
    """A neighbour keeps its watchdog protection.

    Our run may disturb it -- that is precisely when it most needs the
    watchdog, and claiming its lease would remove that protection at the worst
    moment.
    """
    leases = LeaseRegistry(now=lambda: 1000.0)
    leases.take("m1", run_id="r1")
    assert wants_restart(_Row("m2", "failed", "loaded"), leases=leases) is True


def test_an_expired_lease_returns_the_model_to_the_watchdog():
    """The failure mode the heartbeat exists for.

    If the runner is SIGKILLed its `finally` never runs, so the lease must
    lapse on its own or the model stays unsupervised indefinitely.
    """
    t = [1000.0]
    leases = LeaseRegistry(now=lambda: t[0], ttl_s=60.0)
    leases.take("m1", run_id="r1")
    t[0] = 2000.0
    assert wants_restart(_Row("m1", "failed", "loaded"), leases=leases) is True


def test_no_registry_means_the_old_behaviour_exactly():
    """Every existing caller passes nothing and must be unaffected."""
    assert wants_restart(_Row("m1", "failed", "loaded")) is True
    assert wants_restart(_Row("m1", "failed", None)) is False
    assert wants_restart(_Row("m1", "loaded", None)) is False


def test_the_lease_does_not_override_the_other_conditions():
    """A row that was never serving is still not restartable, leased or not.

    The lease suppresses recovery; it does not become a reason to attempt one.
    """
    leases = LeaseRegistry(now=lambda: 1000.0)
    leases.take("m1", run_id="r1")
    assert wants_restart(_Row("m1", "failed", "pulling"), leases=leases) is False
    assert wants_restart(_Row("m1", "pulled", None), leases=leases) is False


def test_check_once_is_the_path_that_actually_fires_for_a_stress_run():
    """Found in review, 2026-09-04. The lease guarded the wrong doors.

    `wants_restart` gates `restore_after_warden_restart` and
    `restart_crashed_models`, and the MR that added the lease claimed those
    were "both recovery paths". They are not the path a stress run provokes:
    `check_once` probes rows whose status is still 'loaded' and calls
    `_restart` directly, and a wedged engine under a stress run is exactly
    that -- nothing has set it 'failed', so `wants_restart` is never consulted
    and its lease check never applies.

    This asserts the guard exists in that function, which is a weak test and
    deliberately better than none: exercising the real loop needs a supervisor,
    a live port and a health probe. It fails if the guard is removed.
    """
    import inspect

    from app.runtime import watchdog

    src = inspect.getsource(watchdog.check_once)
    assert "stress_leases" in src, "check_once no longer reads the lease registry"
    assert "is_held" in src, "check_once no longer skips leased models"
