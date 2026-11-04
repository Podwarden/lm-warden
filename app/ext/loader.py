"""Entry-point loader for edition extensions (group ``lm_warden.extensions``).

An entry point names a ``register(reg: Registry) -> None`` callable. A failing
extension is recorded in ``reg.load_errors`` and logged, never fatal: the app
still boots as whatever the other extensions make it.
"""

import logging
from importlib.metadata import EntryPoint, entry_points

from app.ext import EDITIONS, Registry

GROUP = "lm_warden.extensions"
log = logging.getLogger(__name__)


def _entry_points() -> list[EntryPoint]:
    return list(entry_points(group=GROUP))


def load_extensions(reg: Registry) -> None:
    """Run each extension against a scratch registry and merge it into ``reg``
    only if ``register`` returned: a failing extension leaves no router,
    feature or edition bump behind."""
    for ep in sorted(_entry_points(), key=lambda e: e.name):
        scratch = Registry()
        try:
            register = ep.load()
            register(scratch)
            if scratch.edition not in EDITIONS:  # pragma: no cover - set_edition guards
                raise ValueError(f"unknown edition {scratch.edition!r}")
        except Exception as exc:
            reg.load_errors.append(f"{ep.name}: {exc!r}")
            log.error("extension %s failed to load: %r", ep.name, exc, exc_info=True)
            continue
        reg.set_edition(scratch.edition)
        reg.features |= scratch.features
        reg.routers.extend(scratch.routers)
        reg.public_routes.extend(scratch.public_routes)
        reg.settings_keys.extend(scratch.settings_keys)
        reg.home_providers.extend(scratch.home_providers)
        reg.load_errors.extend(scratch.load_errors)
