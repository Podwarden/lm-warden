"""Pure router logic (#287): pattern validation, matching, settings parsing.

No I/O here. ``RuleSpec`` / ``RouterConfig`` / ``RuleSet`` are the immutable
snapshot the data plane reads; ``app.router.state.RouterState`` builds and
caches them.
"""

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# The header Claude Code carries the LM Warden key in (ANTHROPIC_CUSTOM_HEADERS).
HEADER_NAME = "X-LMWarden-Key"
PATTERN_MAX = 128

_PATTERN_RE = re.compile(r"^[A-Za-z0-9._:*?-]+$")
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1"})
DEFAULT_UPSTREAM_URL = "https://api.anthropic.com"

#: Claude model ids a rule pattern is tried against to name an example model
#: (the Connect page's verify request, GET /api/connect/clients). The same
#: list as ``KNOWN_CLAUDE_IDS`` in frontend/src/lib/router.ts; a test pins
#: the two together.
KNOWN_CLAUDE_IDS: tuple[str, ...] = (
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
    "claude-3-5-haiku-latest",
    "claude-3-5-haiku-20241022",
    "claude-sonnet-4-5",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-latest",
    "claude-opus-4-1",
    "claude-opus-4-1-20250805",
    "claude-opus-4-20250514",
)


def validate_pattern(pattern: str) -> str:
    """Return ``pattern`` unchanged, or raise ``ValueError("pattern: ...")``.

    1..128 chars of ``[A-Za-z0-9._:*?-]`` with at least one non-wildcard
    character (a bare ``*`` would swallow the warden's own served names).
    """
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("pattern: must not be empty")
    if len(pattern) > PATTERN_MAX:
        raise ValueError(f"pattern: at most {PATTERN_MAX} characters")
    if not _PATTERN_RE.match(pattern):
        raise ValueError("pattern: only letters, digits and . _ : - with the wildcards * and ?")
    if not pattern.strip("*?"):
        raise ValueError("pattern: must contain at least one non-wildcard character")
    return pattern


@dataclass(frozen=True)
class RuleSpec:
    id: str
    position: int
    pattern: str
    target_model_id: str
    enabled: bool
    fallback: bool
    strip_thinking: bool
    min_max_tokens: int


def match_rule(model: object, rules: Sequence[RuleSpec]) -> RuleSpec | None:
    """First enabled rule (in the given order) whose glob matches ``model``.

    Case-sensitive. A non-string model never matches.
    """
    if not isinstance(model, str):
        return None
    for rule in rules:
        if rule.enabled and fnmatchcase(model, rule.pattern):
            return rule
    return None


def validate_upstream_url(url: str) -> str:
    """Return ``url`` unchanged, or raise ``ValueError``.

    ``https://`` only (``http://`` is accepted for localhost / 127.0.0.1, for
    tests and local stubs); an optional path prefix, no userinfo, query or
    fragment.
    """
    if not isinstance(url, str) or not url or url != url.strip():
        raise ValueError("upstream_url: must be an https:// URL")
    try:
        parts = urlsplit(url)
        host = parts.hostname
        parts.port  # noqa: B018 - raises ValueError on a bad port
    except ValueError as exc:
        raise ValueError("upstream_url: not a valid URL") from exc
    if not host:
        raise ValueError("upstream_url: host is required")
    if parts.scheme == "https":
        pass
    elif parts.scheme == "http" and host in _LOCAL_HOSTS:
        pass
    else:
        raise ValueError("upstream_url: must be https:// (http:// only for localhost)")
    if parts.username is not None or parts.password is not None:
        raise ValueError("upstream_url: credentials in the URL are not allowed")
    if parts.query or parts.fragment or "?" in url or "#" in url:
        raise ValueError("upstream_url: no query or fragment")
    return url


@dataclass(frozen=True)
class RouterConfig:
    enabled: bool = False
    upstream_url: str = DEFAULT_UPSTREAM_URL
    passthrough_unmatched: bool = True
    local_header_timeout_s: int = 60
    local_nonstream_timeout_s: int = 120
    breaker_threshold: int = 3
    breaker_open_s: int = 60
    max_body_mb: int = 32


# settings-table key -> RouterConfig field
ROUTER_KEYS: dict[str, str] = {
    "router_enabled": "enabled",
    "router_upstream_url": "upstream_url",
    "router_passthrough_unmatched": "passthrough_unmatched",
    "router_local_header_timeout_s": "local_header_timeout_s",
    "router_local_nonstream_timeout_s": "local_nonstream_timeout_s",
    "router_breaker_threshold": "breaker_threshold",
    "router_breaker_open_s": "breaker_open_s",
    "router_max_body_mb": "max_body_mb",
}

# field -> inclusive (lo, hi) for the integer settings
INT_RANGES: dict[str, tuple[int, int]] = {
    "local_header_timeout_s": (1, 600),
    "local_nonstream_timeout_s": (1, 3600),
    "breaker_threshold": (1, 100),
    "breaker_open_s": (1, 3600),
    "max_body_mb": (1, 256),
}
_BOOL_FIELDS = frozenset({"enabled", "passthrough_unmatched"})

_warned: set[str] = set()


def _warn_once(key: str, value: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning("router: ignoring unusable setting %s (using the default)", key)


def _parse_bool(value: str) -> bool | None:
    v = value.strip().lower()
    if v in ("true", "1", "yes", "on"):
        return True
    if v in ("false", "0", "no", "off"):
        return False
    return None


def config_from_kv(kv: Mapping[str, str]) -> RouterConfig:
    """Build a ``RouterConfig`` from the settings kv rows.

    Fail-open: a missing or unusable value yields the default for THAT key
    only (logged once), never an exception.
    """
    values: dict[str, object] = {}
    for key, fieldname in ROUTER_KEYS.items():
        raw = kv.get(key)
        if raw is None:
            continue
        parsed: object | None
        if fieldname in _BOOL_FIELDS:
            parsed = _parse_bool(raw)
        elif fieldname == "upstream_url":
            try:
                parsed = validate_upstream_url(raw.strip())
            except ValueError:
                parsed = None
        else:
            lo, hi = INT_RANGES[fieldname]
            try:
                n = int(raw.strip())
            except ValueError:
                n = lo - 1
            parsed = n if lo <= n <= hi else None
        if parsed is None:
            _warn_once(key, raw)
            continue
        values[fieldname] = parsed
    return RouterConfig(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class RuleSet:
    """Settings + ordered rules, as loaded at ``version``.

    ``served_names`` maps each rule's target model id to its current served
    name (``None`` when the row no longer exists), joined at load for display.
    """

    config: RouterConfig
    rules: tuple[RuleSpec, ...]
    served_names: Mapping[str, str | None]
    version: int
