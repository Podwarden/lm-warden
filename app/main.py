import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.auth.csrf import csrf_check, ensure_csrf_id
from app.config import load_settings
from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo
from app.db.repos.runtime import RuntimeRepo
from app.db.repos.stress_runs import StressRunRepo
from app.runtime.backends.vllm.env import warn_if_shm_undersized
from app.runtime.boot_reconcile import reconcile_stranded_models
from app.runtime.port_alloc import PortAllocator
from app.runtime.supervisor import Supervisor


def _build_engine_driver(settings):
    """Pick the engine driver from settings (#160). Default is the
    in-container subprocess driver. The docker driver is opt-in and, until
    image-channel resolution lands in P2 (#161), requires an explicit
    VLLM_ENGINE_IMAGE so it never guesses an image."""
    from pathlib import Path

    if settings.engine_driver == "docker":
        import os

        import docker

        from app.runtime.engine.docker_socket import DockerSocketDriver

        image = os.environ.get("VLLM_ENGINE_IMAGE")
        if not image:
            raise RuntimeError(
                "VW_ENGINE_DRIVER=docker requires VLLM_ENGINE_IMAGE to be set "
                "(the engine container image). Channel-based image resolution "
                "lands in P2/#161; until then the image must be explicit."
            )
        # Pass the same per-model logs dir the subprocess driver uses so the
        # docker driver mirrors the engine container's stdout+stderr into
        # <data_dir>/logs/<model_id>.log — the only place routes_logs.py reads
        # from. Without it the UI Live-logs panel is stale/empty under the
        # docker driver. (#177 follow-up)
        return DockerSocketDriver(
            client=docker.from_env(),
            image=image,
            log_dir=str(Path(settings.data_dir) / "logs"),
        )
    from app.runtime.engine.local_subprocess import LocalSubprocessDriver

    return LocalSubprocessDriver(
        log_dir=str(Path(settings.data_dir) / "logs"),
        log_max_bytes=settings.engine_log_max_bytes,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.hf_cache_dir.mkdir(parents=True, exist_ok=True)
    app.state.settings = settings

    from app.auth.jwt_secret import load_jwt_secret

    app.state.jwt_secret = load_jwt_secret(settings.db_path)

    from app.auth.sse_tickets import TicketStore

    app.state.sse_tickets = TicketStore(
        secret=app.state.jwt_secret,
        ttl_seconds=settings.sse_ticket_ttl_seconds,
    )

    from app.auth.stream_registry import StreamRegistry

    app.state.stream_registry = StreamRegistry()

    async with open_db(settings.db_path) as db:
        await apply_migrations(db)

    # #210 — under the local-subprocess driver the vLLM engine inherits this
    # process's /dev/shm. On Kubernetes that is 64 MiB unless the pod spec asks
    # for more, which is far too little for tensor-parallel workers and shows
    # up as an uncatchable SIGBUS minutes into serving rather than as a startup
    # failure. Say so in the log while the operator is still reading it.
    warn_if_shm_undersized(settings.engine_driver)

    app.state.supervisor = Supervisor(
        app.state.settings,
        driver=_build_engine_driver(app.state.settings),
    )
    app.state.port_allocator = PortAllocator(start=10000, end=10999)

    # #236 — a row left in a transient status ('loading'/'unloading'/'pulling')
    # by a warden that died mid-operation describes an in-process object that
    # no longer exists, and BOTH routes out of those statuses 409. Demote them
    # to something an operator can act on before anything else touches the
    # table.
    #
    # ORDER MATTERS TWICE:
    #   * after the supervisor exists, because "is this load real?" is
    #     answered by the supervisor holding a handle — at boot it never does,
    #     nothing has been spawned yet, but keying on the handle is what makes
    #     the function safe to reuse outside boot;
    #   * before mark_runtime_dead_on_startup, which would otherwise have
    #     already swept every transient row into 'failed' and left this with
    #     nothing to see.
    # Rows that were genuinely SERVING ('loaded', or a watchdog restore in
    # flight) are deliberately left to the call below, which records
    # prior_status so the watchdog restores them.
    await reconcile_stranded_models(settings, app.state.supervisor)

    async with open_db(settings.db_path) as db:
        await ModelRepo(db).mark_runtime_dead_on_startup()
        await RuntimeRepo(db).clear_all()
        # A stress run is an in-process asyncio task, so it dies with the
        # warden. A row left 'running' would block the cooldown forever AND
        # carry a search bracket that was never confirmed -- publishing its
        # lower bound would understate the limit at full confidence.
        #
        # Deliberately AFTER reconcile_stranded_models (#236): a run that died
        # mid-load leaves both a stress row and a model row stranded, and the
        # model is the one an operator will look at first.
        interrupted = await StressRunRepo(db).mark_interrupted_on_startup()
        if interrupted:
            logging.getLogger(__name__).warning(
                "stress: marked %d run(s) interrupted -- the warden restarted "
                "while they were in progress",
                interrupted,
            )

    # The watchdog stands down for a model under a live lease
    # (``watchdog.wants_restart`` reads ``app_state.stress_leases``), so this
    # must exist before ``run_watchdog_forever`` starts below. Cleared
    # explicitly rather than relying on the registry being fresh: leases are
    # in-memory precisely because nothing that outlives this process may hold
    # one, and boot is where that is asserted.
    from app.stress.lease import LeaseRegistry

    app.state.stress_leases = LeaseRegistry()
    app.state.stress_leases.clear_all()

    from app.proxy.tokenizers import TokenizerCache

    app.state.tokenizers = TokenizerCache()

    # S5 (#104) — STRICT priority scheduler. An in-process singleton because
    # the warden runs a single uvicorn worker per pod; if we ever scale
    # workers, swap for a Redis-backed implementation (see
    # app/proxy/scheduler.py module docstring).
    from app.proxy.scheduler import PriorityScheduler

    app.state.scheduler = PriorityScheduler()

    # S8 (#117) — chat playground singletons. ``playground_store`` caches
    # the `vw-playground` bearer plaintext server-side (browser never sees
    # it). ``chat_active_requests`` is a counter the Playwright suite polls
    # to verify abort-cleanup. Both are process-local for the same reason
    # the scheduler is — single uvicorn worker.
    from app.chat.active_requests import ActiveRequestCounter
    from app.chat.playground_store import PlaygroundStore

    app.state.playground_store = PlaygroundStore()
    app.state.chat_active_requests = ActiveRequestCounter()

    # God mode (live prompt/output viewer) — in-memory broadcast hub. Always
    # instantiated (cheap); the proxy taps only publish when
    # settings.godmode_enabled is true, so when off the hot path never touches
    # it. Bounds are read once here from settings.
    from app.proxy.godmode import GodModeHub

    app.state.godmode_hub = GodModeHub(
        ring_events=settings.godmode_ring_events,
        ring_chars=settings.godmode_ring_chars,
    )

    # God-mode media store (inline images) — out-of-band blob store so image
    # payloads never inflate the event ring. Always instantiated (cheap);
    # only written when settings.godmode_enabled is true.
    from app.proxy.godmode import GodModeMediaStore

    app.state.godmode_media = GodModeMediaStore(
        store_chars=settings.godmode_media_store_chars,
        max_item_chars=settings.godmode_max_image_chars,
    )

    # Live request registry (feature/live-stats-dashboard, Plane B). In-process,
    # single-worker, lock-light — tracks every in-flight /v1 request with token
    # name + client IP + context tokens for GET /api/stats/requests. dev-2 hooks
    # register/deregister into app/proxy/routes.py::_forward (fail-open).
    from app.proxy.request_registry import RequestRegistry

    app.state.request_registry = RequestRegistry()

    # Requests that have COMPLETED. The registry holds only in-flight ones and
    # drops each in the streaming `finally`, taking its duration, TTFT and
    # finish reason with it. They go to SQLite (request_history) through a
    # queue: the proxy enqueues from `_deregister` and never touches the DB
    # on the slot-release path; the writer task below drains in batches. This
    # replaced an in-memory ring that kept 15 minutes and died with the
    # process -- see app/stats/request_history.py for why.
    from app.stats.request_history import RequestHistoryStore

    request_history = RequestHistoryStore(settings.db_path)
    app.state.request_history = request_history

    from app.runtime.stats_pruner import run_pruner_forever
    from app.runtime.stats_sampler import run_sampler_forever
    from app.runtime.watchdog import run_watchdog_forever

    sampler_task = asyncio.create_task(run_sampler_forever(settings))
    pruner_task = asyncio.create_task(run_pruner_forever(settings))
    watchdog_task = asyncio.create_task(run_watchdog_forever(settings, app.state))
    history_task = asyncio.create_task(request_history.run_forever())

    # Chat2 (2026-08-23) — orphan/TTL/LRU collector for attachments (spec §3).
    # Same "log and keep going" shape as the other background loops above.
    from app.chat2.gc import run_gc_forever

    gc_task = asyncio.create_task(run_gc_forever(settings))
    app.state.chat2_gc_task = gc_task

    try:
        yield
    finally:
        sampler_task.cancel()
        pruner_task.cancel()
        watchdog_task.cancel()
        gc_task.cancel()
        # The history writer flushes what is queued on cancellation, so a
        # request that finished a moment before shutdown is not lost.
        history_task.cancel()
        await asyncio.gather(
            sampler_task,
            pruner_task,
            watchdog_task,
            gc_task,
            history_task,
            return_exceptions=True,
        )

        # Chat2 detached turns — cancel still-running turn runners FIRST so
        # their CancelledError path persists an 'aborted' tail (shielded, so
        # this really waits for the persist) and releases the chat locks
        # while the loop is still alive. Runs before the _BACKGROUND grace
        # below because the runners live in that same set.
        live_turns = getattr(app.state, "chat2_live_turns", None)
        if live_turns is not None:
            await live_turns.shutdown()

        # Chat2 (2026-08-23) review finding — the turn endpoint's fire-and-forget
        # persistence tasks (app.chat2.routes_turn._BACKGROUND) are kept alive by
        # a strong ref specifically so they survive their originating request's
        # cancellation (see that module's docstring). Nothing else awaits them,
        # so without this the app can shut down mid-persist and drop an
        # assistant row/ledger entry. Give them a short grace period to finish
        # instead of hanging shutdown on them forever. Note `asyncio.wait_for`
        # DOES cancel what it is waiting on when the timeout fires: a task
        # still running after 5s is cancelled, not left alone — the warning
        # below is the record that a persist may have been cut short.
        from app.chat2.routes_turn import _BACKGROUND

        if _BACKGROUND:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*_BACKGROUND, return_exceptions=True),
                    timeout=5.0,
                )
            except TimeoutError:
                logging.getLogger(__name__).warning(
                    "chat2: %d background turn-persistence task(s) still running "
                    "at shutdown after 5s grace period",
                    len(_BACKGROUND),
                )


def build_app() -> FastAPI:
    app = FastAPI(title="LLM Warden", lifespan=lifespan)

    from app.setup import routes_api as setup_routes_api

    app.include_router(setup_routes_api.router)

    from app.models import routes_api as models_routes_api

    app.include_router(models_routes_api.router)

    from app.models import routes_logs as models_routes_logs

    app.include_router(models_routes_logs.router)

    # Model stress test (docs/superpowers/specs/2026-09-03-model-stress-test-design.md).
    # Shares the /api/models prefix: POST /{id}/stress starts a run,
    # GET /{id}/capabilities returns the measured record. Operator JWT only --
    # a run deliberately crashes the engine, so it is not reachable with a /v1
    # API token (see app/stress/routes_api.py::require_operator).
    from app.stress import routes_api as stress_routes_api

    app.include_router(stress_routes_api.router)

    # #177: engine-version dropdown — GET /api/templates/engine-versions.
    # Backs the try-stack vLLM-version field with the published
    # vllm/vllm-openai semver tags (6h family-keyed cache over Docker Hub).
    from app.templates import routes_api as templates_routes_api

    app.include_router(templates_routes_api.router)

    from app.auth.routes import router as auth_router

    app.include_router(auth_router)

    from app.tokens import routes_api as tokens_routes_api

    app.include_router(tokens_routes_api.router)

    from app.stats import routes_api as stats_routes_api

    app.include_router(stats_routes_api.router)

    # Live realtime stats dashboard (feature/live-stats-dashboard). Two
    # independent planes behind a fixed API contract (docs/live-stats-spec.md):
    #   Plane A — engine /metrics scraper SSE:  GET /api/stats/live
    #   Plane B — live per-request registry:     GET /api/stats/requests
    # Registered here up front so the two backend slices never collide in
    # main.py. Both surface on the NEW /ui/stats/live page; /stats is untouched.
    from app.stats import live_engine as stats_live_engine
    from app.stats import live_requests as stats_live_requests

    app.include_router(stats_live_engine.router)
    app.include_router(stats_live_requests.router)

    # HF cache management — vllm-warden#114. Lives next to stats because
    # the UI surfaces it as a section on /stats; the routes are
    # JWT-gated like every other /api/*.
    from app.cache import routes_api as cache_routes_api

    app.include_router(cache_routes_api.router)

    from app.settings import routes_api as settings_routes_api

    app.include_router(settings_routes_api.router)
    app.include_router(settings_routes_api.model_settings_router)

    from app.proxy import routes as proxy_routes

    app.include_router(proxy_routes.router)

    from app.proxy import routes_godmode as proxy_routes_godmode

    app.include_router(proxy_routes_godmode.router)

    from app.system import routes_version as system_routes_version

    app.include_router(system_routes_version.router)

    from app.system import routes_gpus as system_routes_gpus

    app.include_router(system_routes_gpus.router)

    # #177: active engine-driver capability (can it swap the engine image to a
    # pinned vLLM version?). Consumed by the Try-stack panel to disable +
    # explain the version selector under the in-container subprocess driver.
    from app.system import routes_engine as system_routes_engine

    app.include_router(system_routes_engine.router)

    # #148: static system inventory (CPU/RAM/GPU/OS/Docker) consumed by
    # the /stats System Configuration panel. 60s in-process cache lives
    # on app.state.system_info_cache (lazy-init on first request).
    from app.system import routes_info as system_routes_info

    app.include_router(system_routes_info.router)

    from app.header import routes_api as header_routes_api

    app.include_router(header_routes_api.router)

    # S4: built-in tuning presets ("Apply preset" dropdown on /settings).
    # Read-only — FE applies preset.settings via the existing PATCH
    # /api/models/{id}/settings endpoint, no new write path.
    from app.presets import routes_api as presets_routes_api

    app.include_router(presets_routes_api.router)

    # S8 (#117): /chat playground — JWT-authed SSE proxy + admin
    # active-requests diagnostic. Mounts /api/chat/* and
    # /api/admin/active-requests; routes are in app/chat/routes_api.py.
    from app.chat import routes_api as chat_routes_api

    app.include_router(chat_routes_api.router)

    # #155 — Unified-port architecture: public landing page at /_landing
    # served behind Caddy's `handle /` rewrite. Route is intentionally
    # NOT JWT-gated (the whole point is that an anonymous browser hitting
    # the unified-port root sees a useful page). Opt-out via the
    # `landing_page_enabled` runtime setting → route returns 404.
    from app.landing import routes as landing_routes

    app.include_router(landing_routes.router)

    # Chat2 (2026-08-23) — T5 image attachments (upload/serve/delete under
    # /api/chat2/attachments) + T13 chats CRUD/fork/defaults/models/budget
    # under /api/chat2/chats, /defaults, /models, /budget, /_whoami + T14 the
    # streaming turn (POST /api/chat2/chats/{id}/turns).
    from app.chat2 import routes_attachments as chat2_routes_attachments
    from app.chat2 import routes_chats as chat2_routes_chats
    from app.chat2 import routes_turn as chat2_routes_turn
    from app.chat2.body_limit import Chat2BodyLimitMiddleware
    from app.chat2.budget import AlwaysAllow
    from app.chat2.live import LiveTurns
    from app.chat2.locks import TurnLocks

    # In-process singletons — single uvicorn worker, same rationale as
    # the scheduler above. TurnLocks enforces one turn in flight
    # per chat; AlwaysAllow is the warden's no-op budget policy (the Hub
    # wires its rolling-window budget checker in here instead). LiveTurns is
    # the detached-turn registry (feat/chat2-detached-turns): a turn's SSE
    # frames live here so a disconnected client can re-attach and the runner
    # survives navigation/reload.
    app.state.chat2_turn_locks = TurnLocks()
    app.state.chat2_budget = AlwaysAllow()
    app.state.chat2_live_turns = LiveTurns()

    app.include_router(chat2_routes_attachments.router)
    app.include_router(chat2_routes_chats.router)
    app.include_router(chat2_routes_turn.router)

    # Middleware registration order matters: in Starlette the LAST-added middleware
    # (whether via @app.middleware("http") or app.add_middleware() — the decorator
    # is sugar for the latter) is the OUTERMOST wrapper (first to run on every
    # request).
    #
    # Desired request-path order:
    #   Chat2BodyLimitMiddleware (outermost — reject an oversized declared
    #                             Content-Length on POST /api/chat2/attachments
    #                             before anything else, including CSRF, runs)
    #   ensure_csrf_id           (populates request.state.csrf_id / csrf_token)
    #   csrf_check               (validates X-CSRF-Token after csrf_id is set)
    #
    # Therefore: csrf_check is added first (→ innermost), ensure_csrf_id next,
    # Chat2BodyLimitMiddleware last (→ outermost).

    @app.middleware("http")
    async def _csrf_check(request: Request, call_next):
        return await csrf_check(request, call_next)

    @app.middleware("http")
    async def _ensure_csrf_id(request: Request, call_next):
        return await ensure_csrf_id(request, call_next)

    # Chat2 (2026-08-23) — T5 fix round 2: FastAPI resolves route dependencies
    # (the UploadFile/Form parameters on POST /api/chat2/attachments) by
    # awaiting Request.form(), which spools the ENTIRE multipart body before
    # the route function body ever runs — an in-route Content-Length check
    # alone is too late to stop a huge declared upload from being read onto
    # disk/into memory. This raw-ASGI middleware runs ahead of FastAPI's
    # routing entirely, so it can reject on the header alone.
    app.add_middleware(Chat2BodyLimitMiddleware)

    @app.get("/api/csrf")
    async def get_csrf_token(request: Request) -> dict:
        return {"csrf": request.state.csrf_token}

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        # Always 200 while the control plane is up: this is the container's
        # liveness probe, and failing it over lost GPU telemetry would get a
        # still-serving engine restarted. The GPU state rides along for
        # operators and the Hub to read (#255) -- the last-known result of
        # the GPU probe, from memory, never a fresh nvidia-smi call.
        from app.system.gpu import gpu_probe_health

        return {"ok": True, "gpu": gpu_probe_health().as_dict()}

    return app


app = build_app()
