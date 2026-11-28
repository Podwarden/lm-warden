"""Whether Prompt Lens is on, and turning it on or off while the app runs.

Precedence (owner request 2026-11-22):

1. the runtime setting ``prompt_lens_enabled`` (Settings page, the ``settings``
   KV), when it has been set;
2. otherwise the env: ``VW_PROMPT_LENS_ENABLED`` (app/config.py);
3. otherwise on (default flipped to on 2026-11-26, #346).

The runtime value lives on ``app.state.prompt_lens_runtime`` (True, False, or
absent/None for "not set"): read once at startup (``load``) and replaced by the
settings PATCH (``apply``), so the proxy's per-request check is two attribute
reads and no database. The env value is read from ``app.state.settings`` each
time.

Turning it off (``apply(app, False)`` while it was on) stops new captures at
once (every tap checks ``enabled``, and the hub stops accepting events, so a
request already running captures nothing more), drops what memory holds (the
replay ring and key windows, the whole-section store, the image store) and ends
every open stream: the hub puts its stop mark on each one's queue. A key
holder's stream also re-checks ``enabled`` on its own timer. Kept captures on
disk (per-key ``keep_captures``) stay; they are deleted only when that per-key
switch is turned off.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Literal

from app.db.database import open_db
from app.db.repos.settings import SettingsRepo

log = logging.getLogger("vllm_warden.prompt_lens")

#: The runtime settings key (app/settings/routes_api.py).
SETTING_KEY = "prompt_lens_enabled"

_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}

Source = Literal["setting", "env", "default"]


def parse_stored(raw: str | None) -> bool | None:
    """A stored KV value as a bool; None for absent or unreadable."""
    if raw is None:
        return None
    v = raw.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return None


def save_lock(app: Any) -> asyncio.Lock:
    """Serialises saves of the switch (app/settings/routes_api.py)."""
    lock = getattr(app.state, "prompt_lens_save_lock", None)
    if not isinstance(lock, asyncio.Lock):
        lock = asyncio.Lock()
        app.state.prompt_lens_save_lock = lock
    return lock


def _runtime(app: Any) -> bool | None:
    rt = getattr(app.state, "prompt_lens_runtime", None)
    return rt if isinstance(rt, bool) else None


def _env(app: Any) -> bool:
    return bool(getattr(app.state.settings, "prompt_lens_enabled", True))


def configured(app: Any) -> bool:
    """The switch alone (setting, else env, else on), hub or not."""
    rt = _runtime(app)
    return rt if rt is not None else _env(app)


def enabled(app: Any) -> bool:
    """True when Prompt Lens is on: the switch says so and the hub exists."""
    return getattr(app.state, "prompt_lens_hub", None) is not None and configured(app)


def status(app: Any) -> dict[str, Any]:
    """What the Settings page shows: the value in effect and where it comes from."""
    rt = _runtime(app)
    env_var = getattr(app.state.settings, "prompt_lens_env_var", None)
    env_var = env_var if isinstance(env_var, str) else None
    source: Source = "setting" if rt is not None else ("env" if env_var else "default")
    return {
        "enabled": enabled(app),
        "source": source,
        "setting": rt,
        "env_var": env_var,
        "env_value": _env(app) if env_var else None,
    }


def apply(app: Any, value: bool | None) -> None:
    """Set the runtime value (None: not set, the env decides) and act on the
    change: off drops memory and ends streams; on lets the hub accept again."""
    app.state.prompt_lens_runtime = value
    if enabled(app):
        app.state.prompt_lens_hub.accepting = True
    else:
        _drop(app)


def _drop(app: Any) -> None:
    hub = getattr(app.state, "prompt_lens_hub", None)
    if hub is not None:
        hub.drop_all()
    for name in ("prompt_lens_sections", "prompt_lens_media"):
        store = getattr(app.state, name, None)
        if store is not None:
            store.clear()


async def load(app: Any) -> None:
    """Read the runtime value from the settings KV (startup)."""
    async with open_db(app.state.settings.db_path) as db:
        raw = await SettingsRepo(db).get(SETTING_KEY)
    app.state.prompt_lens_runtime = parse_stored(raw)
    st = status(app)
    if st["source"] == "setting" and st["env_var"]:
        log.info(
            "Prompt Lens is %s from the runtime setting (overrides %s)", _word(st), st["env_var"]
        )


async def record_change(
    db: Any, *, actor: str, via: str, before: dict[str, Any], after: dict[str, Any]
) -> None:
    """One ``prompt_lens_switch_audit`` row for a save of the switch. Commits."""
    await db.execute(
        "INSERT INTO prompt_lens_switch_audit(ts, actor, via, old_enabled, new_enabled,"
        " old_source, new_source, env_var) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            time.time(),
            actor,
            via,
            int(bool(before["enabled"])),
            int(bool(after["enabled"])),
            before["source"],
            after["source"],
            after.get("env_var"),
        ),
    )
    await db.commit()


async def last_change(db: Any) -> dict[str, Any] | None:
    """The newest ``prompt_lens_switch_audit`` row, or None."""
    cur = await db.execute(
        "SELECT ts, actor, via, old_enabled, new_enabled, old_source, new_source"
        " FROM prompt_lens_switch_audit ORDER BY id DESC LIMIT 1"
    )
    row = await cur.fetchone()
    if row is None:
        return None
    ts, actor, via, old_e, new_e, old_s, new_s = row
    return {
        "ts": float(ts),
        "actor": actor,
        "via": via,
        "old_enabled": bool(old_e),
        "new_enabled": bool(new_e),
        "old_source": old_s,
        "new_source": new_s,
    }


def _word(st: dict[str, Any]) -> str:
    return "on" if st["enabled"] else "off"
