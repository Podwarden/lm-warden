"""The stress lease (design §5.5).

A stress run deliberately crashes an engine. The watchdog's RestartBudget
allows 3 restarts per 3600s and, on exhaustion, leaves the model dead until a
human intervenes -- so three deliberate crashes would take a production model
down for good. The lease makes the watchdog stand down for one model while its
run owns recovery.

Two things v1 of the design got wrong, both fixed here:

* It guarded `check_once`. But `check_once` only probes rows with
  status == 'loaded' (watchdog.py:761), and a real crash flips the row to
  'failed' FIRST -- so recovery actually runs through `restart_crashed_models`
  (watchdog.py:680), which v1 left unguarded. Both would then call `_restart`,
  which force-unloads unconditionally, and whichever lost the race would
  destroy the other's freshly loaded engine.
* It proposed an expiry of "2x expected runtime" -- a SIX HOUR window on
  thorough, during which a dead runner leaves a production model unsupervised.
  `finally` does not run on SIGKILL.
"""

from __future__ import annotations

from app.stress.lease import LeaseRegistry


def test_a_model_without_a_lease_is_not_held():
    reg = LeaseRegistry(now=lambda: 1000.0)
    assert reg.is_held("m1") is False


def test_taking_a_lease_holds_the_model():
    reg = LeaseRegistry(now=lambda: 1000.0)
    reg.take("m1", run_id="r1")
    assert reg.is_held("m1") is True


def test_a_lease_expires_without_a_heartbeat():
    """Bounded exposure is the whole point.

    If the runner dies without clearing the lease -- SIGKILL, container
    restart, an exception inside the finally itself -- the model must return to
    the watchdog's care quickly, not in six hours.
    """
    t = [1000.0]
    reg = LeaseRegistry(now=lambda: t[0], ttl_s=60.0)
    reg.take("m1", run_id="r1")
    t[0] = 1059.0
    assert reg.is_held("m1") is True
    t[0] = 1061.0
    assert reg.is_held("m1") is False


def test_a_heartbeat_extends_the_lease():
    t = [1000.0]
    reg = LeaseRegistry(now=lambda: t[0], ttl_s=60.0)
    reg.take("m1", run_id="r1")
    t[0] = 1050.0
    reg.heartbeat("m1")
    t[0] = 1100.0
    assert reg.is_held("m1") is True


def test_a_heartbeat_for_an_unheld_model_does_nothing():
    """It must not resurrect a lease the runner already released."""
    reg = LeaseRegistry(now=lambda: 1000.0)
    reg.heartbeat("m1")
    assert reg.is_held("m1") is False


def test_releasing_clears_the_lease_immediately():
    reg = LeaseRegistry(now=lambda: 1000.0)
    reg.take("m1", run_id="r1")
    reg.release("m1")
    assert reg.is_held("m1") is False


def test_releasing_is_idempotent():
    """It is called from a `finally`, which can run after an earlier release."""
    reg = LeaseRegistry(now=lambda: 1000.0)
    reg.take("m1", run_id="r1")
    reg.release("m1")
    reg.release("m1")
    assert reg.is_held("m1") is False


def test_a_lease_covers_only_its_own_model():
    """A neighbour has no lease and stays under the watchdog's care.

    We crash our own engine; recovering theirs is not our job, and claiming
    their lease would remove their protection at the exact moment our run is
    most likely to disturb them.
    """
    reg = LeaseRegistry(now=lambda: 1000.0)
    reg.take("m1", run_id="r1")
    assert reg.is_held("m2") is False


def test_boot_clears_every_lease():
    """A run is in-process and dies with the warden.

    Whatever it held is meaningless afterwards, and the expiry alone would
    leave a model unsupervised for a further TTL after every restart.
    """
    reg = LeaseRegistry(now=lambda: 1000.0)
    reg.take("m1", run_id="r1")
    reg.take("m2", run_id="r2")
    assert reg.clear_all() == 2
    assert reg.is_held("m1") is False
    assert reg.is_held("m2") is False


def test_the_run_id_is_recoverable_for_diagnostics():
    reg = LeaseRegistry(now=lambda: 1000.0)
    reg.take("m1", run_id="r1")
    assert reg.holder("m1") == "r1"
    assert reg.holder("m2") is None


def test_an_expired_lease_reports_no_holder():
    t = [1000.0]
    reg = LeaseRegistry(now=lambda: t[0], ttl_s=60.0)
    reg.take("m1", run_id="r1")
    t[0] = 2000.0
    assert reg.holder("m1") is None


def test_one_run_per_gpu_set_not_per_model():
    """Two runs sharing a card each attribute the other's pressure to itself.

    The lease is per-model because recovery is per-model, but admission has to
    be per-GPU or both measurements are wrong and neither knows it.
    """
    reg = LeaseRegistry(now=lambda: 1000.0)
    reg.take("m1", run_id="r1", gpu_uuids=("GPU-a", "GPU-b"))
    assert reg.gpus_busy(("GPU-b",)) is True
    assert reg.gpus_busy(("GPU-c",)) is False


def test_releasing_frees_the_gpus():
    reg = LeaseRegistry(now=lambda: 1000.0)
    reg.take("m1", run_id="r1", gpu_uuids=("GPU-a",))
    reg.release("m1")
    assert reg.gpus_busy(("GPU-a",)) is False


def test_an_expired_lease_frees_its_gpus():
    t = [1000.0]
    reg = LeaseRegistry(now=lambda: t[0], ttl_s=60.0)
    reg.take("m1", run_id="r1", gpu_uuids=("GPU-a",))
    t[0] = 2000.0
    assert reg.gpus_busy(("GPU-a",)) is False
