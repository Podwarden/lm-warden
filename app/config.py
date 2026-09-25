import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from app.utils.client_ip import DEFAULT_TRUSTED_PROXIES, parse_trusted_proxies

log = logging.getLogger("vllm_warden.config")

_TRUTHY = {"1", "true", "yes", "on"}

# Recognised VW_RUNAWAY_MODE values; anything else falls back to the off-safe
# default so a typo can never silently arm enforcement on shared prod.
_RUNAWAY_MODES = {"off", "log", "enforce"}


def _truthy(raw: str) -> bool:
    return raw.strip().lower() in _TRUTHY


_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


def parse_canonical_origin(raw: str) -> str:
    """Normalise VW_LANDING_CANONICAL_URL to `scheme://host[:port]`, or "".

    The value is echoed into the public landing page (`<link rel=canonical>`,
    Open Graph URLs, robots.txt, sitemap.xml), so it is held to the shape of
    an http(s) ORIGIN and nothing else: no path beyond "/", no query, no
    fragment, no userinfo, a plain DNS name (or IPv4) and a valid port.
    Anything else returns "" -- the page then stays noindex, exactly as if
    the variable were unset. A trailing slash is tolerated and dropped.
    """
    value = raw.strip()
    if not value:
        return ""
    try:
        parts = urlsplit(value)
        port = parts.port  # raises ValueError on a non-numeric/out-of-range port
    except ValueError:
        return ""
    host = (parts.hostname or "").lower()
    if (
        parts.scheme.lower() not in ("http", "https")
        or not host
        or not _HOST_RE.match(host)
        or parts.username is not None
        or parts.password is not None
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
        or "@" in parts.netloc
    ):
        return ""
    netloc = host if port is None else f"{host}:{port}"
    return f"{parts.scheme.lower()}://{netloc}"


def _parse_trusted_proxies(raw: str) -> tuple[str, ...]:
    """VW_TRUSTED_PROXIES: comma-separated CIDRs. Blank = the built-in
    default (loopback + private ranges); "none" = trust no peer. A malformed
    entry raises, so a typo fails start-up instead of changing who is
    trusted."""
    if not raw.strip():
        return DEFAULT_TRUSTED_PROXIES
    if raw.strip().lower() == "none":
        return ()
    specs = tuple(p.strip() for p in raw.split(",") if p.strip())
    try:
        parse_trusted_proxies(specs)
    except ValueError as exc:
        raise ValueError(f"VW_TRUSTED_PROXIES: {exc}") from exc
    return specs


def _parse_origins(raw: str) -> tuple[str, ...]:
    out = []
    for part in raw.split(","):
        cleaned = part.strip().rstrip("/")
        if cleaned:
            out.append(cleaned)
    return tuple(out)


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    hf_cache_dir: Path
    cookie_secret: str
    container_gpu_count: int
    bind_host: str = "0.0.0.0"
    bind_port: int = 8080
    load_timeout_s: float = 600.0
    # Max time the warmup verification probe waits for a successful
    # POST /v1/completions before marking the load failed. Closes the
    # window between /health 200 and actual serving readiness (e.g.
    # Qwen3-VL's _warmup_mm_processor). Configurable via env override.
    warmup_probe_timeout_s: float = 600.0
    # Server-side wall-clock backstop for a single proxied request (streaming
    # or not). 0.0 = disabled (default; behaviour-identical to pre-reaper).
    # When > 0, a streamed generation that has run longer than this many
    # seconds is torn down: the upstream vLLM socket is closed so the engine
    # aborts the generation and frees its KV blocks, and the scheduler slot is
    # released. This is the guaranteed reaper for the "client abandoned the
    # request but the TCP connection stayed transport-alive" case, where no
    # http.disconnect ever reaches uvicorn and Starlette never cancels the
    # body iterator — so nothing else can reclaim the slot. Set to a real
    # value (e.g. 600) in the deployment env to bound worst-case slot pinning.
    request_max_wall_s: float = 0.0
    session_access_ttl_minutes: int = 15
    session_refresh_ttl_days: int = 7
    allowed_origins: tuple[str, ...] = ("http://localhost:3000",)
    trust_proxy_origin: bool = False
    # Peers (CIDRs) whose X-Forwarded-For is believed when a security decision
    # is keyed on the client's address -- today the login and bearer throttles
    # (app/auth/throttle.py, app/utils/client_ip.py::trusted_client_ip).
    # VW_TRUSTED_PROXIES; blank = loopback + private ranges, "none" = trust no
    # peer (key on the socket address only).
    trusted_proxies: tuple[str, ...] = DEFAULT_TRUSTED_PROXIES
    # When no origin is configured, accept the origin this request was actually
    # addressed to (see app/auth/origin.py). Defaults False so the documented
    # Settings-level guarantee below still holds: constructing
    # Settings(allowed_origins=()) blocks every origin. Only load_settings
    # turns this on, and only when the operator configured nothing.
    derive_origin_from_request: bool = False
    sse_ticket_ttl_seconds: int = 60
    # #160 — which engine driver Supervisor uses to run the vLLM engine.
    # "local" = in-container subprocess (default, behaviour-identical to
    # pre-#160); "docker" = sibling container via the host docker socket
    # (requires VLLM_ENGINE_IMAGE; image-channel resolution lands in P2/#161).
    engine_driver: str = "local"
    # God mode (live prompt/output viewer) — OFF by default. When false the
    # proxy hot path is byte-identical to pre-godmode: no hub call happens at
    # any tap. All bounds are read once at startup into GodModeHub.
    godmode_enabled: bool = False
    godmode_ring_events: int = 2000
    godmode_ring_chars: int = 4_000_000
    godmode_max_prompt_chars: int = 16000
    # Display-only capture window: when a prompt exceeds godmode_max_prompt_chars
    # the tap keeps the head PLUS this many chars from the tail (joined by an
    # elision marker), so the newest turn — which lives at the tail — survives
    # even under a giant repeated system prompt. Never affects the forwarded
    # request. 0 = pure head-slice (pre-#tail behaviour).
    godmode_prompt_tail_chars: int = 4000
    # God-mode media store (spec 2026-08-03 — inline images). Bounds for the
    # out-of-band image blob store; only consulted when godmode_enabled.
    godmode_media_store_chars: int = 64_000_000
    godmode_max_image_chars: int = 14_000_000
    godmode_max_images_per_req: int = 16
    # Content logging (diagnostic) — see app/proxy/content_log.py. Disabled by
    # default; when off the proxy forward path is byte-identical to a build
    # without it. Hard-scoped to an explicit token allowlist so this can run on
    # a shared server without ever logging all traffic: a request is logged
    # only when content_log_enabled is True AND its token id is in
    # content_log_tokens.
    #
    # Two independent caps, and NEITHER has a value meaning "unlimited":
    #   content_log_max_chars bounds one RECORD (prompt and completion
    #     separately). A negative value is clamped to 0 by load_settings and
    #     again in content_log._cap -- it is not an escape hatch.
    #   content_log_max_bytes bounds the FILE. On reaching it the sink STOPS
    #     writing and warns once; it does not rotate and does not delete. The
    #     Hub template pins vllm-warden-data at 10 GiB shared with the SQLite
    #     database, so 512 MiB leaves the database an order of magnitude of
    #     headroom while still holding roughly 6,000 records at the 80 KB an
    #     80,000-char record costs. <= 0 switches the cap off, matching
    #     engine_log_max_bytes above.
    #
    # content_log_path defaults to <data_dir>/logs/content.jsonl -- see
    # load_settings. It MUST follow data_dir: pinned to a literal /data it
    # escaped a moved VW_DATA_DIR onto the container's writable layer, filling
    # the node's disk instead of the PVC and vanishing on restart.
    content_log_enabled: bool = False
    content_log_tokens: frozenset[str] = frozenset()
    content_log_path: Path = Path("/data/logs/content.jsonl")
    content_log_max_chars: int = 40000
    content_log_max_bytes: int = 512 * 1024 * 1024
    # Runaway-generation detector (app/proxy/runaway.py) — see
    # docs/superpowers/specs/2026-07-21-runaway-detector-design.md. Off by
    # default: when runaway_mode == "off" the proxy forward path is
    # byte-identical to a build without the detector — no detector is
    # constructed and NO upstream stream-forcing happens. "log" runs the
    # detector and records a would-trip incident via the content-logger sink
    # without interrupting; "enforce" tears the generation down on a trip. The
    # three thresholds carry conservative defaults so a confident pathology
    # signal is required before any teardown.
    runaway_mode: str = "off"
    runaway_think_budget: int = 24000
    runaway_repeat_max: int = 6
    runaway_hard_max: int = 96000

    # --- Engine watchdog (2026-08-17) --------------------------------------
    # _watch_exit waits on the `vllm serve` WRAPPER, but vLLM v1 runs EngineCore
    # and the TP workers underneath it. Four times on 2026-08-17 EngineCore died
    # while the wrapper stayed alive, so the wait never returned, the row kept
    # status='loaded', and the proxy forwarded to a dead engine indefinitely.
    # The watchdog probes the engine's OWN /health instead.
    watchdog_enabled: bool = True
    watchdog_interval_s: float = 30.0
    watchdog_failure_threshold: int = 3      # ~90s before declaring death
    watchdog_max_restarts: int = 3           # per window, then leave it failed
    watchdog_restart_window_s: float = 3600.0
    watchdog_crash_keep: int = 20            # crash dirs retained per model
    # Bytes of engine log copied into each crash dir. Was the WHOLE log, which
    # made /data occupancy `watchdog_crash_keep x current log size` -- a moving
    # target that grows with uptime, not a ceiling. Measured 2026-08-18: 20
    # crash dirs x a 6.8 MB log = 131 MB, all of the volume's usage. The log is
    # append-only across loads and never rotated, so a longer-lived engine
    # simply makes every copy bigger. A tail is what anyone reads anyway -- the
    # crash is at the END of the file.
    watchdog_crash_log_bytes: int = 4 * 1024 * 1024
    # Rotate the live engine log once it passes this, keeping one .1 alongside.
    # /data also holds the SQLite DB, and filling it takes the database down
    # with it -- the 2026-06-15 ENOSPC incident.
    engine_log_max_bytes: int = 32 * 1024 * 1024

    # --- Chat2 attachments (2026-08-23) ------------------------------------
    chat_quota_user_bytes: int = 2 * 1024**3
    chat_quota_free_floor_bytes: int = 5 * 1024**3
    chat_attachment_ttl_days: int = 90

    # --- Per-request history (app/stats/request_history.py) ----------------
    # One row per completed /v1 request, in SQLite. Pruned by age AND by row
    # count, because either alone fails: a quiet deployment would keep a
    # trickle forever under a count cap, and a load test at 10 req/s writes
    # ~860k rows a day under an age cap. 30 days is four times the widest
    # window the stats page offers (7d), so the 7d view is always served in
    # full and a last-N latency basis still has depth when traffic is thin.
    # At the observed thousands of requests a day that is tens of MB on the
    # 10 GiB data volume; the row cap bounds the pathological case at ~40 MB
    # whatever the request rate.
    request_history_retention_days: int = 30
    request_history_max_rows: int = 200_000

    # --- Public landing page (app/landing) ---------------------------------
    # The public origin this instance's landing page may be indexed under,
    # e.g. "https://lmwarden.com". Empty (the default, and every normal
    # self-hosted install) keeps the page noindex, robots.txt disallow-all
    # and no sitemap. Validated by parse_canonical_origin: an invalid value
    # is ignored with a warning rather than echoed into the page.
    landing_canonical_url: str = ""

    @property
    def db_path(self) -> Path:
        return self.data_dir / "vllm-warden.db"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def crashes_dir(self) -> Path:
        return self.data_dir / "crashes"

    @property
    def hf_token_path(self) -> Path:
        return self.data_dir / "hf-token"


def load_settings() -> Settings:
    data_dir = Path(os.environ.get("VW_DATA_DIR", "/data"))
    # HF model cache lives on its own PVC in production so the data PVC can
    # stay small. Compose mounts the cache at /root/.cache/huggingface to
    # match the HF library default; VW_HF_CACHE_DIR overrides that path.
    hf_cache_dir = Path(
        os.environ.get("VW_HF_CACHE_DIR", "/root/.cache/huggingface")
    )
    secret = os.environ.get("VW_COOKIE_SECRET")
    if not secret or len(secret) < 32:
        raise RuntimeError("VW_COOKIE_SECRET must be set and >=32 chars")
    gpu_count = int(os.environ.get("VW_CONTAINER_GPU_COUNT", "0"))
    # `None` when unset, distinct from a set-but-blank value. Both mean "not
    # configured" here, but reading it with a default would make an unset var
    # look configured and silently disable derivation.
    raw_origins = os.environ.get("VW_FRONTEND_ORIGIN") or ""
    # An unset or explicitly-empty VW_FRONTEND_ORIGIN means "nobody told us our
    # public URL" — the normal state behind a reverse proxy, since nothing in
    # the deploy flow knows it at container-build time.
    #
    # The localhost fallback STAYS, because local development is genuinely
    # cross-origin: the frontend dev server runs on :3000 while this API is on
    # :8080, so the browser's Origin never equals the host it addressed and
    # derivation alone would reject every dev request.
    #
    # What the fallback could not do is serve a proxied install, where
    # localhost is never the browser's origin — so every request carrying a
    # real Origin was 403'd and every page reload logged the operator out. The
    # old comment claimed this fallback existed so a blank value would not
    # "silently lock admins out of refresh"; it did precisely that.
    #
    # So derivation is added ALONGSIDE it rather than replacing it: an
    # unconfigured deployment accepts both the dev origin and whatever host it
    # is actually served on. Configuring VW_FRONTEND_ORIGIN is a deliberate act
    # — that allowlist is then honoured exactly, with derivation off.
    #
    # The Settings-level fail-closed guarantee is unchanged: constructing
    # Settings(allowed_origins=()) directly — with derive_origin_from_request
    # and trust_proxy_origin both False — still blocks every origin.
    configured = _parse_origins(raw_origins)
    derive_origin_from_request = not configured
    allowed_origins = configured or ("http://localhost:3000",)
    trust_proxy_origin = _truthy(os.environ.get("VW_TRUST_PROXY_ORIGIN", ""))
    trusted_proxies = _parse_trusted_proxies(os.environ.get("VW_TRUSTED_PROXIES", ""))
    warmup_probe_timeout_s = float(
        os.environ.get("VW_WARMUP_PROBE_TIMEOUT_S", "600.0")
    )
    engine_driver = os.environ.get("VW_ENGINE_DRIVER", "local")
    watchdog_enabled = _truthy(os.environ.get("VW_WATCHDOG_ENABLED", "1"))
    watchdog_interval_s = float(os.environ.get("VW_WATCHDOG_INTERVAL_S", "30.0"))
    watchdog_failure_threshold = int(
        os.environ.get("VW_WATCHDOG_FAILURE_THRESHOLD", "3")
    )
    watchdog_max_restarts = int(os.environ.get("VW_WATCHDOG_MAX_RESTARTS", "3"))
    watchdog_restart_window_s = float(
        os.environ.get("VW_WATCHDOG_RESTART_WINDOW_S", "3600.0")
    )
    watchdog_crash_keep = int(os.environ.get("VW_WATCHDOG_CRASH_KEEP", "20"))
    watchdog_crash_log_bytes = int(
        os.environ.get("VW_WATCHDOG_CRASH_LOG_BYTES", str(4 * 1024 * 1024))
    )
    engine_log_max_bytes = int(
        os.environ.get("VW_ENGINE_LOG_MAX_BYTES", str(32 * 1024 * 1024))
    )
    godmode_enabled = _truthy(os.environ.get("VW_GODMODE_ENABLED", ""))
    godmode_ring_events = int(os.environ.get("VW_GODMODE_RING_EVENTS", "2000"))
    godmode_ring_chars = int(os.environ.get("VW_GODMODE_RING_CHARS", "4000000"))
    godmode_max_prompt_chars = int(
        os.environ.get("VW_GODMODE_MAX_PROMPT_CHARS", "16000")
    )
    godmode_prompt_tail_chars = int(
        os.environ.get("VW_GODMODE_PROMPT_TAIL_CHARS", "4000")
    )
    godmode_media_store_chars = int(
        os.environ.get("VW_GODMODE_MEDIA_STORE_CHARS", "64000000")
    )
    godmode_max_image_chars = int(
        os.environ.get("VW_GODMODE_MAX_IMAGE_CHARS", "14000000")
    )
    godmode_max_images_per_req = int(
        os.environ.get("VW_GODMODE_MAX_IMAGES_PER_REQ", "16")
    )
    request_max_wall_s = float(os.environ.get("VW_REQUEST_MAX_WALL_S", "0.0"))
    content_log_enabled = _truthy(os.environ.get("VW_CONTENT_LOG_ENABLED", ""))
    # Comma-separated token ids; blanks dropped. Empty/unset => log nothing
    # (the gate in content_log.should_log also requires the flag above).
    content_log_tokens = frozenset(
        t.strip()
        for t in os.environ.get("VW_CONTENT_LOG_TOKENS", "").split(",")
        if t.strip()
    )
    # Defaults UNDER data_dir rather than to a literal /data/logs. VW_DATA_DIR
    # is a knob the PodWarden Hub template exposes and interpolates; with the
    # literal, moving it left the content log writing to /data/logs -- no
    # longer the mounted volume, so records landed on the container's writable
    # layer, filled the NODE's disk instead of the PVC, and vanished on
    # restart. Silent on all three counts. An explicit VW_CONTENT_LOG_PATH
    # still wins: pointing it at another volume is a legitimate choice.
    # Blank counts as unset -- the Hub compose renders "${VW_CONTENT_LOG_PATH:-}"
    # and an empty value would otherwise become Path("") == Path("."), the
    # process CWD.
    _content_log_path_raw = os.environ.get("VW_CONTENT_LOG_PATH", "").strip()
    content_log_path = (
        Path(_content_log_path_raw)
        if _content_log_path_raw
        else data_dir / "logs" / "content.jsonl"
    )
    # A character count has no negative reading. `-1` used to disable the cap
    # entirely -- an undocumented escape hatch, and the value an operator
    # naturally reaches for because VW_REQUEST_MAX_WALL_S=0 does mean "no cap".
    # Clamped to 0 (capture nothing) and said out loud: fail-safe beats
    # unbounded records on the volume that also holds the database.
    content_log_max_chars = int(os.environ.get("VW_CONTENT_LOG_MAX_CHARS", "40000"))
    if content_log_max_chars < 0:
        log.warning(
            "VW_CONTENT_LOG_MAX_CHARS=%d is negative; there is no value that "
            "means unlimited. Clamped to 0 (records keep their metadata and "
            "capture no prompt or completion text). Raise the number to "
            "capture more.",
            content_log_max_chars,
        )
        content_log_max_chars = 0
    # Bounds the FILE, not a record. Fail-safe: the sink stops writing and
    # warns once on reaching this -- no rotation, no deletion. Same shape and
    # same <= 0 = off convention as VW_ENGINE_LOG_MAX_BYTES.
    content_log_max_bytes = int(
        os.environ.get("VW_CONTENT_LOG_MAX_BYTES", str(512 * 1024 * 1024))
    )
    # Unknown / typo'd modes fall back to "off" so the forward path stays
    # byte-identical rather than silently arming enforcement on shared prod.
    runaway_mode = os.environ.get("VW_RUNAWAY_MODE", "off").strip().lower()
    if runaway_mode not in _RUNAWAY_MODES:
        runaway_mode = "off"
    runaway_think_budget = int(os.environ.get("VW_RUNAWAY_THINK_BUDGET", "24000"))
    runaway_repeat_max = int(os.environ.get("VW_RUNAWAY_REPEAT_MAX", "6"))
    runaway_hard_max = int(os.environ.get("VW_RUNAWAY_HARD_MAX", "96000"))
    chat_quota_user_bytes = int(os.environ.get("VW_CHAT_QUOTA_USER_BYTES", 2 * 1024**3))
    chat_quota_free_floor_bytes = int(
        os.environ.get("VW_CHAT_QUOTA_FREE_FLOOR_BYTES", 5 * 1024**3)
    )
    chat_attachment_ttl_days = int(os.environ.get("VW_CHAT_ATTACHMENT_TTL_DAYS", 90))
    request_history_retention_days = int(
        os.environ.get("VW_REQUEST_HISTORY_RETENTION_DAYS", "30")
    )
    request_history_max_rows = int(
        os.environ.get("VW_REQUEST_HISTORY_MAX_ROWS", "200000")
    )
    _canonical_raw = os.environ.get("VW_LANDING_CANONICAL_URL", "")
    landing_canonical_url = parse_canonical_origin(_canonical_raw)
    if _canonical_raw.strip() and not landing_canonical_url:
        log.warning(
            "VW_LANDING_CANONICAL_URL=%r is not an http(s) origin such as "
            "https://llm.example.com (no path, query or credentials); ignored, "
            "so the landing page stays noindex.",
            _canonical_raw,
        )
    return Settings(
        data_dir=data_dir,
        hf_cache_dir=hf_cache_dir,
        cookie_secret=secret,
        container_gpu_count=gpu_count,
        allowed_origins=allowed_origins,
        trust_proxy_origin=trust_proxy_origin,
        trusted_proxies=trusted_proxies,
        derive_origin_from_request=derive_origin_from_request,
        warmup_probe_timeout_s=warmup_probe_timeout_s,
        engine_driver=engine_driver,
        watchdog_enabled=watchdog_enabled,
        watchdog_interval_s=watchdog_interval_s,
        watchdog_failure_threshold=watchdog_failure_threshold,
        watchdog_max_restarts=watchdog_max_restarts,
        watchdog_restart_window_s=watchdog_restart_window_s,
        watchdog_crash_keep=watchdog_crash_keep,
        watchdog_crash_log_bytes=watchdog_crash_log_bytes,
        engine_log_max_bytes=engine_log_max_bytes,
        godmode_enabled=godmode_enabled,
        godmode_ring_events=godmode_ring_events,
        godmode_ring_chars=godmode_ring_chars,
        godmode_max_prompt_chars=godmode_max_prompt_chars,
        godmode_prompt_tail_chars=godmode_prompt_tail_chars,
        godmode_media_store_chars=godmode_media_store_chars,
        godmode_max_image_chars=godmode_max_image_chars,
        godmode_max_images_per_req=godmode_max_images_per_req,
        request_max_wall_s=request_max_wall_s,
        content_log_enabled=content_log_enabled,
        content_log_tokens=content_log_tokens,
        content_log_path=content_log_path,
        content_log_max_chars=content_log_max_chars,
        content_log_max_bytes=content_log_max_bytes,
        runaway_mode=runaway_mode,
        runaway_think_budget=runaway_think_budget,
        runaway_repeat_max=runaway_repeat_max,
        runaway_hard_max=runaway_hard_max,
        chat_quota_user_bytes=chat_quota_user_bytes,
        chat_quota_free_floor_bytes=chat_quota_free_floor_bytes,
        chat_attachment_ttl_days=chat_attachment_ttl_days,
        request_history_retention_days=request_history_retention_days,
        request_history_max_rows=request_history_max_rows,
        landing_canonical_url=landing_canonical_url,
    )
