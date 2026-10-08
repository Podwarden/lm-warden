"""Cross-package DB constants.

Lifted out of ``app/models/routes_api.py`` so other subsystems (e.g.
``app/cache/routes_api.py``) can guard the same state-transition rules
without copying the tuple — see vllm-warden#114.
"""

# Statuses where a ``models`` row is actively owned by the runtime
# subsystem (vLLM subprocess running or transitioning). Mutating the
# row OR its on-disk HF cache from outside the supervisor while it's
# in one of these states will produce a noisy failure mid-load /
# mid-unload. Same set guarded by ``delete_model`` in
# ``app/models/routes_api.py`` and the cache routes in
# ``app/cache/routes_api.py``.
ACTIVE_STATUSES: tuple[str, ...] = (
    "loaded",
    "loading",
    "unloading",
    "pulling",
)
