"""The stress lease (design §5.5).

A stress run deliberately crashes an engine. ``RestartBudget``
(``app/runtime/watchdog.py:209``) allows 3 restarts per 3600s and, on
exhaustion, writes ``status='failed'`` and leaves the model dead until a human
intervenes. Three deliberate crashes reach that. The watchdog also needs
``interval 30s × threshold 3`` ≈ 90s merely to notice each death, which the
runner beats by an order of magnitude because it is already probing.

So for the duration of a run the watchdog stands down for that one model and
the runner owns recovery.

**Check this in ``wants_restart()``**, which both watchdog paths consult — not
in ``check_once``. ``check_once`` only probes rows whose status is ``loaded``
(``watchdog.py:761``), but a real crash flips the row to ``failed`` first, so
recovery actually runs through ``restart_crashed_models``
(``watchdog.py:680``). Guarding only the first leaves the path that actually
fires unguarded, and then both it and the runner call ``_restart`` — which
force-unloads unconditionally (``watchdog.py:546``) — so whichever loses the
race destroys the other's freshly loaded engine.

**The lease is a heartbeat, not a long expiry.** A TTL sized to the run (the
v1 proposal was 2× expected runtime, i.e. six hours on ``thorough``) leaves a
production model unsupervised for that whole window if the runner dies — and
``finally`` does not run on SIGKILL, on a container restart, or when an
exception is raised inside the ``finally`` itself. Renewing every ~15s against
a 60s TTL is the same amount of code and bounds the exposure to a minute.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

#: The runner renews well inside this. Sized so a dead runner returns the model
#: to the watchdog within a minute rather than within a run length.
DEFAULT_TTL_S = 60.0


@dataclass
class _Lease:
    run_id: str
    expires_at: float
    gpu_uuids: tuple[str, ...]


class LeaseRegistry:
    """In-memory, because a run is in-process and dies with the warden.

    Persisting a lease would only create rows that outlive the thing they
    describe. Boot clears the registry by construction; ``clear_all`` exists so
    that is explicit rather than incidental.
    """

    def __init__(
        self,
        *,
        now: Callable[[], float] = time.monotonic,
        ttl_s: float = DEFAULT_TTL_S,
    ) -> None:
        self._now = now
        self._ttl_s = ttl_s
        self._leases: dict[str, _Lease] = {}

    def take(self, model_id: str, *, run_id: str, gpu_uuids: Iterable[str] = ()) -> None:
        self._leases[model_id] = _Lease(
            run_id=run_id,
            expires_at=self._now() + self._ttl_s,
            gpu_uuids=tuple(gpu_uuids),
        )

    def heartbeat(self, model_id: str) -> None:
        """Extend an existing lease. Never creates one.

        Creating on heartbeat would let a stale timer resurrect a lease the
        runner has already released, silently disarming the watchdog for a
        model nobody is testing.
        """
        lease = self._leases.get(model_id)
        if lease is not None:
            lease.expires_at = self._now() + self._ttl_s

    def release(self, model_id: str) -> None:
        """Idempotent: it is called from a ``finally`` that may run twice."""
        self._leases.pop(model_id, None)

    def is_held(self, model_id: str) -> bool:
        """The predicate ``wants_restart()`` consults."""
        return self._live(model_id) is not None

    def holder(self, model_id: str) -> str | None:
        lease = self._live(model_id)
        return lease.run_id if lease else None

    def gpus_busy(self, gpu_uuids: Iterable[str]) -> bool:
        """Admission is per-GPU-set, even though recovery is per-model.

        Two runs sharing a card each attribute the other's memory and
        scheduling pressure to their own limit, and neither can tell. The lease
        stays per-model because that is the unit recovery acts on; this is the
        separate question of whether a new run may start at all.
        """
        wanted = set(gpu_uuids)
        for model_id in list(self._leases):
            lease = self._live(model_id)
            if lease and wanted & set(lease.gpu_uuids):
                return True
        return False

    def clear_all(self) -> int:
        n = len(self._leases)
        self._leases.clear()
        return n

    def _live(self, model_id: str) -> _Lease | None:
        lease = self._leases.get(model_id)
        if lease is None:
            return None
        if lease.expires_at <= self._now():
            # Expiry is evaluated lazily on read, so a dead runner needs no
            # reaper task to release what it was holding.
            del self._leases[model_id]
            return None
        return lease
