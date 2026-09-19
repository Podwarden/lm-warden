"""VW_LIVE_* environment contract for tests/live.

Parsed once per session by the `live_cfg` fixture. Nothing here is hardcoded
-- every value the test talks to comes from an env var, sourced by the
operator from a private env file outside the repo (`set -a; . vw-live.env;
set +a`). `LiveConfig.__repr__` redacts the admin username and password;
no other code path in this package ever prints them.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

#: Required env vars. Any missing -> the whole tests/live directory skips.
#: VW_LIVE_ACK_PAUSE_ALL is checked separately (it must equal "1", not just
#: be present) but reported alongside these by `missing_vars()`.
REQUIRED_VARS = (
    "VW_LIVE_BASE_URL",
    "VW_LIVE_ADMIN_USER",
    "VW_LIVE_ADMIN_PASSWORD",
    "VW_LIVE_MODEL",
)

_DEFAULT_PRIORITIES = "0,0,2,3,4,4,6,7,9,9"
_PROBE_LABELS = ("p0a", "p0b", "p2", "p3", "p4a", "p4b", "p6", "p7", "p9a", "p9b")


def _int_env(env: dict, name: str, default: int) -> int:
    v = env.get(name)
    return int(v) if v not in (None, "") else default


def _float_env(env: dict, name: str, default: float) -> float:
    v = env.get(name)
    return float(v) if v not in (None, "") else default


def _bool_env(env: dict, name: str, default: bool) -> bool:
    v = env.get(name)
    if v in (None, ""):
        return default
    return v == "1"


def redact_base_url(base_url: str) -> str:
    """scheme://<redacted> -- never let a hostname reach a report file."""
    parts = urlsplit(base_url)
    return f"{parts.scheme}://<redacted>"


@dataclass
class LiveConfig:
    base_url: str
    admin_user: str
    admin_password: str
    model: str
    ack_pause_all: bool

    max_inflight: int = 16
    priorities: tuple[int, ...] = ()
    filler_priority: int = 1
    filler_prompt_tokens: int = 3000
    probe_prompt_tokens: int = 2000
    hold_s: float = 8.0
    step_s: float = 0.75
    filler_max_tokens: int = 512
    probe_max_tokens: int = 64
    burst_gap_ms: int = 150
    min_rounds: int = 4
    min_load_s: float = 300.0
    max_load_s: float = 540.0
    min_clean_rounds: int = 3
    timeout_s: float = 900.0
    drain_timeout_s: float = 180.0
    saturation_timeout_s: float = 60.0
    round_timeout_s: float = 180.0
    min_queue_spread_s: float = 1.0
    assert_ttft: bool = False
    state_dir: str = ""
    report_dir: str = ""
    token_prefix: str = "vwlive"
    sweep_leftovers: bool = False
    verify_tls: bool = True
    ca_bundle: str | None = None
    #: Rehearsal-only switch (not in the plan's original table -- added
    #: during implementation, see tests/live/README.md "Rehearsal"). When
    #: set, the `model_info` fixture skips resolving a real model (the local
    #: dev stack has no inference engine) and the test exits after isolation
    #: + token lifecycle, before any load. Exercises: login, CSRF, token
    #: create/delete, pause-every-other-token, state file, restore.
    dry_run: bool = False

    def __repr__(self) -> str:  # never leak credentials
        return (
            f"LiveConfig(base_url={self.base_url!r}, admin_user=<redacted>, "
            f"admin_password=<redacted>, model={self.model!r}, "
            f"max_inflight={self.max_inflight}, dry_run={self.dry_run})"
        )

    @property
    def probe_labels(self) -> tuple[str, ...]:
        return _PROBE_LABELS[: len(self.priorities)]

    @property
    def httpx_verify(self) -> bool | str:
        if self.ca_bundle:
            return self.ca_bundle
        return self.verify_tls


def missing_vars(env: dict | None = None) -> list[str]:
    """Names of required vars that are absent, plus the safety interlock if
    it isn't exactly "1". Empty list means the contract is satisfiable."""
    import os

    e = env if env is not None else os.environ
    missing = [v for v in REQUIRED_VARS if not e.get(v)]
    if e.get("VW_LIVE_ACK_PAUSE_ALL") != "1":
        missing.append("VW_LIVE_ACK_PAUSE_ALL=1 (safety interlock)")
    return missing


def load_config(env: dict | None = None) -> LiveConfig:
    """Build a LiveConfig from the environment.

    Call only after `missing_vars()` is empty -- this does not itself
    validate presence, only parses/defaults.
    """
    import os

    e = env if env is not None else os.environ

    priorities_raw = e.get("VW_LIVE_PRIORITIES", _DEFAULT_PRIORITIES)
    priorities = tuple(int(x) for x in priorities_raw.split(",") if x.strip())

    # I4: NOT tempfile.gettempdir() -- on Linux that's frequently a tmpfs
    # (wiped on reboot) and on any platform it can be cleaned by an OS
    # janitor between "the process got killed" and "an operator runs
    # restore.py". The one artifact standing between a crash and a real
    # user's token staying paused forever must survive both.
    default_dir = os.path.join(os.path.expanduser("~/.local/state"), "vw-live")
    ca_bundle = e.get("VW_LIVE_CA_BUNDLE") or None

    return LiveConfig(
        base_url=e["VW_LIVE_BASE_URL"].rstrip("/"),
        admin_user=e["VW_LIVE_ADMIN_USER"],
        admin_password=e["VW_LIVE_ADMIN_PASSWORD"],
        model=e["VW_LIVE_MODEL"],
        ack_pause_all=e.get("VW_LIVE_ACK_PAUSE_ALL") == "1",
        max_inflight=_int_env(e, "VW_LIVE_MAX_INFLIGHT", 16),
        priorities=priorities,
        filler_priority=_int_env(e, "VW_LIVE_FILLER_PRIORITY", 1),
        filler_prompt_tokens=_int_env(e, "VW_LIVE_FILLER_PROMPT_TOKENS", 3000),
        probe_prompt_tokens=_int_env(e, "VW_LIVE_PROBE_PROMPT_TOKENS", 2000),
        hold_s=_float_env(e, "VW_LIVE_HOLD_S", 8.0),
        step_s=_float_env(e, "VW_LIVE_STEP_S", 0.75),
        filler_max_tokens=_int_env(e, "VW_LIVE_FILLER_MAX_TOKENS", 512),
        probe_max_tokens=_int_env(e, "VW_LIVE_PROBE_MAX_TOKENS", 64),
        burst_gap_ms=_int_env(e, "VW_LIVE_BURST_GAP_MS", 150),
        min_rounds=_int_env(e, "VW_LIVE_MIN_ROUNDS", 4),
        min_load_s=_float_env(e, "VW_LIVE_MIN_LOAD_S", 300.0),
        max_load_s=_float_env(e, "VW_LIVE_MAX_LOAD_S", 540.0),
        min_clean_rounds=_int_env(e, "VW_LIVE_MIN_CLEAN_ROUNDS", 3),
        timeout_s=_float_env(e, "VW_LIVE_TIMEOUT_S", 900.0),
        drain_timeout_s=_float_env(e, "VW_LIVE_DRAIN_TIMEOUT_S", 180.0),
        saturation_timeout_s=_float_env(e, "VW_LIVE_SATURATION_TIMEOUT_S", 60.0),
        round_timeout_s=_float_env(e, "VW_LIVE_ROUND_TIMEOUT_S", 180.0),
        min_queue_spread_s=_float_env(e, "VW_LIVE_MIN_QUEUE_SPREAD_S", 1.0),
        assert_ttft=_bool_env(e, "VW_LIVE_ASSERT_TTFT", False),
        state_dir=e.get("VW_LIVE_STATE_DIR") or default_dir,
        report_dir=e.get("VW_LIVE_REPORT_DIR") or default_dir,
        token_prefix=e.get("VW_LIVE_TOKEN_PREFIX", "vwlive"),
        sweep_leftovers=_bool_env(e, "VW_LIVE_SWEEP_LEFTOVERS", False),
        verify_tls=_bool_env(e, "VW_LIVE_VERIFY_TLS", True),
        ca_bundle=ca_bundle,
        dry_run=_bool_env(e, "VW_LIVE_DRY_RUN", False),
    )
