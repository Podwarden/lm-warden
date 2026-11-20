"""Who may set up the first admin of a fresh install.

conventions: lifecycle.bootstrap-admin@1 lifecycle.retire-bootstrap@1

Until 2026-11 the first visitor to a fresh install's /ui/setup created the
admin. On a public address that is whoever finds it first. Now there are two
ways, and neither is open to a stranger:

* **Unattended** -- ``LM_BOOTSTRAP_ADMIN_USERNAME`` and
  ``LM_BOOTSTRAP_ADMIN_PASSWORD`` create the admin at first boot and close the
  wizard. A half-set pair, a blank username or a password the wizard would
  refuse stops the boot and names the problem; it never falls back to the
  wizard. Once an admin exists they create nothing, and the log asks for them
  to be removed. The admin they made stays *temporary* (a log warning at every
  boot, ``temporary_admin`` on its sign-in answer) until a person sets a new
  password on it.
* **Wizard, behind a one-time setup code.** Every ``POST /api/setup/*`` needs
  ``X-Setup-Code``. The code is ``LM_SETUP_CODE`` when set (install.sh and the
  Hub generate one), else one generated here, kept in ``<data dir>/setup-code``
  (0600, so it survives a restart) and printed in the log at boot. It dies
  with the wizard: the file is removed when the admin is created.

The macOS app is the one exception: a server it launched (``VW_SHELL_TOKEN``
is set) skips the code for a direct loopback peer (no proxy header). The
person at the Mac owns it, the app binds 127.0.0.1 unless told to share on
the LAN, and it offers no way to set an env var. A LAN peer, or anything
through Caddy/Tailscale (which add X-Forwarded-For), still needs the code.

New names use the ``LM_`` prefix (conventions ops.env-naming: LM Warden is
moving from ``VW_``).
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import bcrypt
from fastapi import HTTPException, Request

from app.auth.origin import browser_origin, settings_origin_allowed
from app.auth.passwords import new_password_problem
from app.auth.throttle import LoginThrottle, throttle_client, too_many_attempts
from app.db.database import open_db
from app.db.repos.settings import SettingsRepo
from app.db.repos.setup import SetupRepo
from app.system.shell_auth import is_forwarded, is_loopback_client

logger = logging.getLogger(__name__)

USERNAME_ENV = "LM_BOOTSTRAP_ADMIN_USERNAME"
PASSWORD_ENV = "LM_BOOTSTRAP_ADMIN_PASSWORD"
SETUP_CODE_ENV = "LM_SETUP_CODE"
SETUP_CODE_HEADER = "X-Setup-Code"
SETUP_CODE_FILE = "setup-code"
#: Settings key naming the admin that still holds the configured password.
TEMPORARY_ADMIN_KEY = "bootstrap_admin_temporary"
MIN_SETUP_CODE_CHARS = 16
#: Crockford base32: no I, L, O or U to misread. 16 characters = 80 bits.
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


class BootstrapMisconfigured(RuntimeError):
    """The first-run configuration is half-set or too weak: refuse to boot."""


@dataclass(frozen=True)
class SetupLock:
    """What guards the wizard. ``code`` None: the wizard is closed (an admin
    exists), so the setup routes' own step checks refuse everything."""

    code: str | None
    source: str | None  # "env" | "file" | None


def bootstrap_admin_from_env(env: Mapping[str, str]) -> tuple[str, str] | None:
    """``(username, password)`` from LM_BOOTSTRAP_ADMIN_*, None when neither is
    set; BootstrapMisconfigured for anything in between."""
    username = env.get(USERNAME_ENV, "")
    password = env.get(PASSWORD_ENV, "")
    if not username and not password:
        return None
    if not password:
        raise BootstrapMisconfigured(
            f"{USERNAME_ENV} is set but {PASSWORD_ENV} is not: set both, or neither."
        )
    if not username:
        raise BootstrapMisconfigured(
            f"{PASSWORD_ENV} is set but {USERNAME_ENV} is not: set both, or neither."
        )
    if not username.strip():
        raise BootstrapMisconfigured(f"{USERNAME_ENV} is blank.")
    problem = new_password_problem(password)
    if problem is not None:
        raise BootstrapMisconfigured(f"{PASSWORD_ENV}: {problem}.")
    return username.strip(), password


def setup_code_from_env(env: Mapping[str, str]) -> str | None:
    raw = env.get(SETUP_CODE_ENV, "").strip()
    if not raw:
        return None
    if len(_normalize(raw)) < MIN_SETUP_CODE_CHARS:
        raise BootstrapMisconfigured(
            f"{SETUP_CODE_ENV} must have at least {MIN_SETUP_CODE_CHARS} characters "
            "(install.sh generates one: openssl rand -hex 16)."
        )
    return raw


def _normalize(code: str) -> str:
    """People retype codes: case, spaces and dashes do not matter."""
    return "".join(ch for ch in code.upper() if ch not in " -\t\r\n")


def generate_setup_code() -> str:
    raw = "".join(secrets.choice(_ALPHABET) for _ in range(16))
    return "-".join(raw[i : i + 4] for i in range(0, 16, 4))


def code_path(data_dir: Path) -> Path:
    return data_dir / SETUP_CODE_FILE


def _read_or_create_code_file(path: Path) -> tuple[str, bool]:
    """``(code, kept_in_file)``. A data dir or code file this process cannot
    read or write must not stop the boot: the code then lives in memory only
    (until the next restart) and the log is the one place it is shown."""
    try:
        existing = path.read_text().strip()
    except FileNotFoundError:
        existing = ""
    except OSError:
        logger.warning("could not read %s; using a setup code kept in memory", path, exc_info=True)
        return generate_setup_code(), False
    if len(_normalize(existing)) >= MIN_SETUP_CODE_CHARS:
        try:
            os.chmod(path, 0o600)
        except OSError:
            logger.warning("could not make %s private (0600)", path, exc_info=True)
        return existing, True
    code = generate_setup_code()
    try:
        path.unlink(missing_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(code + "\n")
    except OSError:
        logger.warning("could not write %s; using a setup code kept in memory", path, exc_info=True)
        return code, False
    return code, True


def discard_code_file(data_dir: Path) -> None:
    try:
        code_path(data_dir).unlink(missing_ok=True)
    except OSError:
        logger.warning("could not remove %s", code_path(data_dir), exc_info=True)


def _hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


async def _detected_gpu_indices() -> list[int]:
    from app.system import gpu as gpu_mod

    try:
        return sorted(g.index for g in await gpu_mod.query_gpus())
    except Exception:  # noqa: BLE001 -- no GPU tool is "no GPUs", not a failed boot
        logger.warning(
            "bootstrap admin: could not list GPUs; allow them in Settings", exc_info=True
        )
        return []


async def prepare_first_run(db_path: Path, data_dir: Path, env: Mapping[str, str]) -> SetupLock:
    """Run at boot, after migrations. Validates the first-run variables (raises
    BootstrapMisconfigured), creates the env admin when there is none, and
    returns what locks the wizard."""
    creds = bootstrap_admin_from_env(env)
    env_code = setup_code_from_env(env)

    async with open_db(db_path) as db:
        cur = await db.execute("SELECT COUNT(*) FROM users")
        row = await cur.fetchone()
        users = int(row[0]) if row else 0
        temporary = await SettingsRepo(db).get(TEMPORARY_ADMIN_KEY)

        if users and creds is not None:
            logger.warning(
                "%s / %s are set, but an admin already exists, so they were ignored. "
                "Remove them from the environment: they still hold a password.",
                USERNAME_ENV,
                PASSWORD_ENV,
            )
        if users:
            if temporary:
                logger.warning(
                    "The admin %r still has the password from %s. Sign in and set a new "
                    "password (Settings) so that one, which also sits in your compose or "
                    ".env file, stops working.",
                    temporary,
                    PASSWORD_ENV,
                )
            discard_code_file(data_dir)
            return SetupLock(code=None, source=None)

        if creds is not None:
            username, password = creds
            allowed = await _detected_gpu_indices()
            await db.execute("BEGIN IMMEDIATE")
            try:
                await db.execute(
                    "INSERT INTO users(username, password_hash) VALUES (?, ?)",
                    (username, _hash(password)),
                )
                draft = (await SetupRepo(db).get()).draft
                draft.setdefault("allowed_gpu_indices", allowed)
                await db.execute(
                    "UPDATE setup_state SET step = 'done', draft = ?, updated_at = datetime('now') "
                    "WHERE id = 1",
                    (json.dumps(draft),),
                )
                await db.execute(
                    "INSERT INTO settings(key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (TEMPORARY_ADMIN_KEY, username),
                )
                await db.commit()
            except BaseException:
                await db.rollback()
                raise
            logger.warning(
                "bootstrap admin %r created from %s / %s; the setup wizard is closed. "
                "Sign in and set a new password, then remove both variables.",
                username,
                USERNAME_ENV,
                PASSWORD_ENV,
            )
            discard_code_file(data_dir)
            return SetupLock(code=None, source=None)

    if env_code is not None:
        discard_code_file(data_dir)
        logger.warning(
            "First-run setup is locked: the setup wizard asks for the code in %s.", SETUP_CODE_ENV
        )
        return SetupLock(code=env_code, source="env")

    path = code_path(data_dir)
    code, in_file = _read_or_create_code_file(path)
    # Printed so `docker compose logs api` shows it; it works once, and only
    # until the admin exists (the file is removed then).
    where = f"also in {path}" if in_file else "kept in memory only: it changes on restart"
    logger.warning(
        "First-run setup is locked. Setup code: %s  (%s; the setup wizard asks "
        "for it once. Set %s to choose your own.)",
        code,
        where,
        SETUP_CODE_ENV,
    )
    return SetupLock(code=code, source="file")


SETUP_CODE_THROTTLED_MESSAGE = "too many wrong setup codes; retry later"
#: The throttle's "username": every wrong code counts against one subject.
_THROTTLE_SUBJECT = "setup-code"


def _code_throttle(request: Request) -> LoginThrottle:
    state = request.app.state
    throttle = getattr(state, "setup_code_throttle", None)
    if throttle is None:
        throttle = state.setup_code_throttle = LoginThrottle()
    return throttle


def _lock(request: Request) -> SetupLock:
    lock = getattr(request.app.state, "setup_lock", None)
    # Fail closed: a missing lock (the lifespan never ran prepare_first_run)
    # refuses rather than opening the wizard.
    return lock if isinstance(lock, SetupLock) else SetupLock(code="", source=None)


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _loopback_host(request: Request) -> bool:
    """The Host header names loopback literally (any port). A DNS-rebound
    page reaches 127.0.0.1 too, but its Host is still its own name."""
    host = request.headers.get("host", "").strip().lower()
    if host.startswith("["):
        name = host[1:].split("]", 1)[0]
    else:
        name = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    return name in _LOOPBACK_HOSTS


def code_exempt(request: Request) -> bool:
    """The macOS app's own window: a shell-launched server, a direct loopback
    peer, no proxy header, a loopback Host, and no foreign browser origin.

    The last two matter: any web page the Mac user opens can fire a
    no-cors POST at http://127.0.0.1:<port> (cookie-less, so the CSRF layer
    stands aside), and a DNS-rebinding page arrives from loopback as well.
    """
    if getattr(request.app.state, "shell_token", None) is None:
        return False
    if not is_loopback_client(request) or is_forwarded(request) or not _loopback_host(request):
        return False
    origin = browser_origin(request)
    return origin is None or settings_origin_allowed(request, origin)


def code_required(request: Request) -> bool:
    return _lock(request).code is not None and not code_exempt(request)


async def require_setup_code(request: Request) -> None:
    """Dependency on every POST /api/setup/*. Reads the header from the
    request (not a declared parameter), so the OpenAPI contract is unchanged."""
    lock = _lock(request)
    if lock.code is None or code_exempt(request):
        return
    sent = request.headers.get(SETUP_CODE_HEADER, "")
    if not sent.strip():
        raise HTTPException(403, "setup code required")
    # Wrong codes go through a login throttle of their own (app/auth/throttle.py):
    # each failure is answered late, and a public address that keeps failing is
    # locked out (429), the right code included, until the lock expires.
    throttle = _code_throttle(request)
    client = throttle_client(request)
    wait = throttle.retry_after(client)
    if wait > 0:
        raise too_many_attempts(wait, SETUP_CODE_THROTTLED_MESSAGE)
    if not lock.code or not hmac.compare_digest(
        _normalize(sent).encode(), _normalize(lock.code).encode()
    ):
        await throttle.failed(client, _THROTTLE_SUBJECT)
        raise HTTPException(403, "setup code invalid")
    throttle.success(client, _THROTTLE_SUBJECT)


def close_setup(request: Request) -> None:
    """The admin exists: the code dies with the wizard."""
    request.app.state.setup_lock = SetupLock(code=None, source=None)
    discard_code_file(request.app.state.settings.data_dir)


async def retire_temporary_admin(db_path: Path, username: str | None) -> None:
    """A person set a new password on the configured admin: it is a normal
    admin from now on (lifecycle.retire-bootstrap)."""
    async with open_db(db_path) as db:
        repo = SettingsRepo(db)
        temporary = await repo.get(TEMPORARY_ADMIN_KEY)
        if not temporary:
            return
        await db.execute("DELETE FROM settings WHERE key = ?", (TEMPORARY_ADMIN_KEY,))
        await db.commit()
    logger.warning(
        "bootstrap admin %r retired: its password was changed by %r; the password from %s "
        "no longer works.",
        temporary,
        username,
        PASSWORD_ENV,
    )


async def is_temporary_admin(db_path: Path) -> bool:
    """LM Warden has one admin: the flag names it (for the log), and a rename
    does not take the flag off -- only a new password does."""
    async with open_db(db_path) as db:
        return bool(await SettingsRepo(db).get(TEMPORARY_ADMIN_KEY))
