"""Shared pytest fixtures.

S5 (#104): the fixture ordering here intentionally seeds the DB BEFORE the
FastAPI TestClient's lifespan opens an aiosqlite connection. The original
shape (test calls ``client.get("/healthz")`` then ``_seed_done(...)``) raced
with the lifespan's WAL-mode aiosqlite connection: the sync sqlite3.connect
in ``_seed_done`` opens its own connection on a DB file that the async
connection already holds a WAL writer on. The fix is to ensure migrations
run, the lifespan releases its DB connection back to the pool, and seeding
happens via the SAME aiosqlite path the app uses — no more sync/async mix.

S2 (#55): a second, related flake — nondeterministic 401 "invalid
credentials" on the first POST /api/auth/login — is mitigated at the
SQLite level by ``seed_admin_user`` (matching PRAGMAs + explicit
``BEGIN IMMEDIATE`` + ``wal_checkpoint(FULL)``). ``jwt_login`` adds a
bounded retry-on-401 as defence-in-depth: with #104's checkpoint fix the
retry rarely fires, but it covers any residual aiosqlite read-after-write
window AND any future transient 401 not caused by seeding (route-handler
scheduling, token-cache warm-up, etc.) without papering over real bugs
(non-401 responses surface immediately).
"""

import asyncio
import json
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

# Hermetic Hugging Face: ``TokenizerCache.get`` calls
# ``AutoTokenizer.from_pretrained(hf_repo)`` for whatever repo a test seeded
# (``Qwen/Qwen3.5-9B`` in the proxy tests), which reaches huggingface.co and,
# on a developer machine, also picks up whatever happens to be in
# ``~/.cache/huggingface``. Measured on tests/unit/proxy/test_proxy_route.py:
# 13.1 s online vs 7.6 s offline, and on a cold CI container it is a real
# download per run. Offline mode + an empty HF_HOME makes the load fail
# immediately, which is the code path the cache already handles by falling
# back to a character estimate (see app/proxy/tokenizers.py::count). Tests
# that need a real tokenizer stub ``app.state.tokenizers``.
#
# Both variables are read by ``huggingface_hub.constants`` at import time, so
# they must be set here, before anything imports transformers, not in a
# fixture. ``setdefault`` so an operator can still opt back in explicitly.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault(
    "HF_HOME", str(Path(tempfile.gettempdir()) / "vllm-warden-tests-hf-home")
)
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)

import bcrypt  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# ---------------------------------------------------------------------------
# Suite-wide speed fixtures (perf/test-suite-speed)
# ---------------------------------------------------------------------------
#
# Two fixed costs dominated the unit suite (measured 2026-09-03, 2035 tests,
# 367.6 s serial on an M-series Mac before / 60.6 s after):
#
#   * bcrypt at the production cost factor. ``bcrypt.gensalt()`` defaults to
#     12 rounds (~370 ms/op on an M-series Mac, more on the CI runners). The
#     suite performs ~640 hashpw+checkpw calls per run — every
#     ``seed_admin_user`` hashes and every ``jwt_login`` verifies — which was
#     ~240 s of those 367.6 s. Nothing in the tree asserts on
#     the cost factor; the hash format, the verify path and the production
#     code are all unchanged. ``_fast_bcrypt`` lowers the DEFAULT rounds to
#     bcrypt's minimum (4) for the duration of each test by patching the
#     module attribute every call site resolves at call time. Production
#     defaults are untouched.
#
#   * Running all 28 schema migrations against a fresh SQLite file inside
#     every ``client`` lifespan (28 ``BEGIN IMMEDIATE``/``COMMIT`` pairs, each
#     an fsync). ``migrated_db_template`` runs them ONCE per session through
#     the exact same ``open_db`` + ``apply_migrations`` path and the ``client``
#     fixture ``shutil.copyfile``s the result into the test's data dir. The
#     lifespan still calls ``apply_migrations`` on it, sees every file already
#     recorded in ``schema_migrations`` and applies nothing. A test that
#     genuinely needs to observe migrations being applied by the lifespan
#     opts out with ``@pytest.mark.fresh_db``.

_FAST_BCRYPT_ROUNDS = 4
_REAL_GENSALT = bcrypt.gensalt
# Same content as app.auth.routes._DUMMY_HASH, at the test cost factor, so the
# unknown-user login path is as cheap as the known-user one during tests.
_FAST_DUMMY_HASH = bcrypt.hashpw(
    b"timing-equalizer", _REAL_GENSALT(_FAST_BCRYPT_ROUNDS)
).decode()


def _fast_gensalt(rounds: int = _FAST_BCRYPT_ROUNDS, prefix: bytes = b"2b") -> bytes:
    return _REAL_GENSALT(rounds, prefix)


@pytest.fixture(autouse=True)
def _fast_bcrypt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop bcrypt's default cost to the minimum for the duration of a test."""
    monkeypatch.setattr(bcrypt, "gensalt", _fast_gensalt)
    # The timing-equaliser hash is computed at import time at the production
    # cost, so an unknown-user login would still pay ~370 ms per checkpw.
    from app.auth import routes as auth_routes

    monkeypatch.setattr(auth_routes, "_DUMMY_HASH", _FAST_DUMMY_HASH)


@pytest.fixture(scope="session")
def migrated_db_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A SQLite file with every migration in app/db/sql applied, built once.

    Built through the production ``open_db`` + ``apply_migrations`` path so
    the PRAGMAs, journal mode and ``schema_migrations`` bookkeeping are
    byte-for-byte what the lifespan would have produced. The WAL is
    checkpointed and truncated before the connection closes so the main file
    is self-contained and safe to ``copyfile``.
    """
    from app.db.database import open_db
    from app.db.migrations import apply_migrations

    path = tmp_path_factory.mktemp("db-template") / "vllm-warden.db"

    async def build() -> None:
        async with open_db(path) as db:
            await apply_migrations(db)
            await db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    asyncio.run(build())
    return path


@pytest.fixture
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("VW_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VW_HF_CACHE_DIR", str(tmp_path / "hf-cache"))
    monkeypatch.setenv("VW_COOKIE_SECRET", "test-secret-32-bytes-min-padding!")
    monkeypatch.setenv("VW_CONTAINER_GPU_COUNT", "4")
    # chat2 attachment uploads run a real `shutil.disk_usage(data_dir)` free-
    # space check against `chat_quota_free_floor_bytes` (default 5 GiB). On a
    # disk-constrained runner every upload in the suite would 413 with
    # `quota_exceeded` — a failure that has nothing to do with the test. Pin
    # the floor to 0 here; the floor itself is covered by an explicit test
    # that raises it and monkeypatches `shutil.disk_usage`
    # (tests/unit/chat2/test_attachments_routes.py).
    monkeypatch.setenv("VW_CHAT_QUOTA_FREE_FLOOR_BYTES", "0")
    # The engine watchdog's FIRST tick runs the moment the lifespan starts
    # (interval 30 s after that). Proxy tests seed a 'loaded' model, register
    # a fake port on the supervisor and patch ``httpx.AsyncClient.send``
    # process-wide; the watchdog's health probe of that fake port then lands
    # inside the patch and is recorded as the test's "forwarded request"
    # (``calls[0]["stream"] is False``, empty body). The race window was
    # hidden by ~750 ms of production-cost bcrypt per test and became
    # deterministic once that went away. A unit-test app must not run an
    # autonomous prober against ports the test itself registered as fakes:
    # disable the probe/restart loop here. Restore-on-boot still runs (it is
    # outside the ``enabled`` check) and the watchdog itself is unit-tested
    # directly in tests/unit/test_engine_watchdog.py.
    monkeypatch.setenv("VW_WATCHDOG_ENABLED", "0")
    return tmp_path


@pytest.fixture
def client(
    request: pytest.FixtureRequest, tmp_data_dir: Path, migrated_db_template: Path
) -> TestClient:
    from app.main import build_app

    if request.node.get_closest_marker("fresh_db") is None:
        # Pre-migrated copy; the lifespan's apply_migrations finds nothing to
        # do. See the module docstring block above.
        shutil.copyfile(migrated_db_template, tmp_data_dir / "vllm-warden.db")
    app = build_app()
    with TestClient(app) as c:
        yield c


def csrf_header(client: TestClient) -> dict[str, str]:
    """Mint a CSRF token via /api/csrf and return a header dict."""
    r = client.get("/api/csrf")
    return {"X-CSRF-Token": r.json()["csrf"]}


# ---------------------------------------------------------------------------
# Seeding helpers (#104 + #55)
# ---------------------------------------------------------------------------
#
# Previously each test file rolled its own `_seed_done()` that used a SYNC
# ``sqlite3.connect`` to write into the same DB file the async TestClient
# lifespan had opened in WAL mode. SQLite supports concurrent readers across
# the two driver families but the failure mode under contention (rare but
# real in CI's shared runner) is that the sync write commits to a journal
# the async readers don't pick up until they reopen — manifesting as 401
# "invalid credentials" because the admin row appears to be missing.
#
# ``seed_admin_user`` below performs the same INSERT but routes it through
# the same connection-open path the app uses, so PRAGMAs and journal mode
# match exactly. The previous _seed_done helper in test files is updated
# to delegate here.


def seed_admin_user(
    db_path: Path,
    username: str = "admin",
    password: str = "hunter2",
    allowed_gpu_indices: list[int] | None = None,
) -> None:
    """Seed an admin user and mark setup as done.

    Uses sqlite3 (sync) but opens with the SAME WAL/foreign_keys PRAGMAs
    the async ``open_db`` uses, so the journal state stays consistent for
    any subsequent aiosqlite connection. Must be called AFTER the client
    fixture's lifespan finishes startup (the lifespan runs migrations that
    create the ``users`` and ``setup_state`` tables).
    """
    pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode()
    indices = [0, 1, 2, 3] if allowed_gpu_indices is None else allowed_gpu_indices
    with sqlite3.connect(db_path, isolation_level=None) as db:
        # Match the async open_db PRAGMAs so we share the same journal mode
        # rather than letting sqlite3's default rollback-journal mode fight
        # the aiosqlite WAL writer.
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA journal_mode = WAL")
        # Wrap the seeds in an explicit transaction so the WAL frame lands
        # in one commit boundary (avoids partial-visibility races with the
        # async reader picking up half a write).
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "INSERT INTO users(username, password_hash) VALUES (?, ?)",
            (username, pw),
        )
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": indices}),),
        )
        db.execute("COMMIT")
        # Checkpoint the WAL so the seed becomes visible to readers using a
        # bare ``sqlite3.connect`` without WAL pragmas (defensive — every
        # in-tree reader is WAL-aware, but it's cheap insurance).
        db.execute("PRAGMA wal_checkpoint(FULL)")


# ---------------------------------------------------------------------------
# Auth login helper (#55)
# ---------------------------------------------------------------------------
#
# ``jwt_login`` is the defence-in-depth complement to ``seed_admin_user``:
# with #104's WAL checkpoint the seed is observable to any new connection
# the moment ``seed_admin_user`` returns, but we still retry-on-401 with
# bounded exponential backoff to cover:
#
#   * any residual aiosqlite read-after-write window on slow CI runners
#     (the original #55 symptom);
#   * any FUTURE transient 401 not caused by seeding (route-handler async
#     scheduling, token-cache warm-up, snapshot-isolation hiccups under
#     contention) — we want one canonical login helper, not 19 bespoke
#     copies that each have to be patched if a new race surfaces.
#
# Any non-401 response (422 malformed body, 500 server error, etc.)
# surfaces immediately so we don't paper over real bugs.

def seed_admin_token(
    db_path: Path,
    *,
    name: str = "automation",
    created_by: str = "admin",
    scope: str = "admin",
    plaintext: str | None = None,
    expires_at: str | None = None,
    revoked_at: str | None = None,
    rotated_at: str | None = None,
    paused_at: str | None = None,
) -> tuple[str, str]:
    """Insert an api_tokens row directly and return ``(token_id, plaintext)``.

    For states the API cannot reach in one call (expired, paused, a vw_ secret
    on an admin row, …). Timestamps are SQLite UTC strings -- build them with
    ``app.db.repos.tokens.sqlite_utc_in``. Same PRAGMAs, transaction and WAL
    checkpoint as ``seed_admin_user``, for the same reason (#55/#104).
    """
    import secrets

    from app.auth.bearer import generate_admin_token
    from app.db.repos.tokens import hash_token

    secret = plaintext or generate_admin_token()
    token_id = secrets.token_hex(16)
    with sqlite3.connect(db_path, isolation_level=None) as db:
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope, expires_at, "
            " revoked_at, rotated_at, paused_at, created_by) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (token_id, name, secret[:8], hash_token(secret), scope, expires_at,
             revoked_at, rotated_at, paused_at, created_by),
        )
        db.execute("COMMIT")
        db.execute("PRAGMA wal_checkpoint(FULL)")
    return token_id, secret


LOGIN_RETRY_MAX = 5
LOGIN_RETRY_BACKOFF_S = (0.01, 0.02, 0.05, 0.1, 0.2)


def jwt_login(
    client: TestClient,
    *,
    username: str = "admin",
    password: str = "hunter2",
) -> dict[str, str]:
    """POST /api/auth/login and return a ready-to-merge Authorization header.

    Retries up to ``LOGIN_RETRY_MAX`` times on 401 with exponential
    backoff. With #104's ``seed_admin_user`` fix the retry rarely fires;
    it remains as defence in depth for future transient 401 conditions
    (see module docstring). Any non-401 failure surfaces immediately.
    """
    last_status = -1
    last_text = ""
    for attempt in range(LOGIN_RETRY_MAX):
        r = client.post(
            "/api/auth/login", json={"username": username, "password": password}
        )
        if r.status_code == 200:
            return {"Authorization": f"Bearer {r.json()['access_token']}"}
        last_status = r.status_code
        last_text = r.text
        if r.status_code != 401:
            break
        backoff = LOGIN_RETRY_BACKOFF_S[min(attempt, len(LOGIN_RETRY_BACKOFF_S) - 1)]
        time.sleep(backoff)
    raise AssertionError(
        f"jwt_login: POST /api/auth/login returned {last_status} after "
        f"{LOGIN_RETRY_MAX} attempts (last response: {last_text!r})"
    )


# ---------------------------------------------------------------------------
# Admin-token helpers (spec 2026-09-19): shared, never copied into test files
# ---------------------------------------------------------------------------


def bearer(secret: str) -> dict[str, str]:
    """``Authorization: Bearer <secret>`` -- an admin (vwa_) or inference (vw_)
    token, or a session JWT."""
    return {"Authorization": f"Bearer {secret}"}


@pytest.fixture
def flush_audit(client: TestClient):
    """Call it to write every queued admin-audit row NOW; returns the count.

    The audit middleware enqueues and a background flusher batches (#258), so
    a row is not in the table the instant a request returns. Rather than
    sleeping past the flush interval, a test that reads ``admin_audit``
    directly asks for the drain: ``flush_audit()`` runs the writer's
    ``flush()`` on the app's own event loop through the TestClient's portal,
    so the awaitable the app uses is the one the test awaits -- no second
    loop, no sync write racing the async one.

    An async test drives the app on its own loop (``live_app``) and awaits
    ``live_app.state.admin_audit.flush()`` directly instead.
    """
    def _flush() -> int:
        portal = getattr(client, "portal", None)
        assert portal is not None, "flush_audit needs the client's lifespan running"
        return int(portal.call(client.app.state.admin_audit.flush))

    return _flush


@pytest.fixture
def seeded_db(client: TestClient, tmp_data_dir: Path) -> Path:
    """The ``client`` app's database with the admin user seeded (setup done),
    ready for ``jwt_login`` and ``seed_admin_token``."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    seed_admin_user(db_path)
    return db_path


@pytest.fixture
async def live_app(tmp_data_dir: Path, migrated_db_template: Path):
    """The real app with its lifespan running on the test's own event loop,
    for tests that drive it over raw ASGI (tests/asgi_stream.py) -- the only
    way to open an endless stream -- or call its dependencies directly. The
    admin user is seeded."""
    from app.main import build_app

    shutil.copyfile(migrated_db_template, tmp_data_dir / "vllm-warden.db")
    app = build_app()
    async with app.router.lifespan_context(app):
        seed_admin_user(tmp_data_dir / "vllm-warden.db")
        yield app
