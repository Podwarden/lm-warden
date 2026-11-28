"""Public price lists: where Prompt Lens prices come from (#347).

Two sources, both public, read without credentials:

* **OpenRouter** (primary): ``GET https://openrouter.ai/api/v1/models``. A
  documented public API, under 1 MB, one schema for every vendor, per-token
  ``pricing.prompt`` / ``completion`` / ``input_cache_read`` / ``input_cache_write``
  (strings, currency per token, USD). Its prices are the vendors' list prices.
* **LiteLLM** (fallback): ``model_prices_and_context_window.json`` in
  github.com/BerriAI/litellm (MIT; about 5 MB, community maintained). Keyed by
  the vendors' own API model names, with cache read and write prices. Read only
  when OpenRouter fails.

Only these two HTTPS hosts are ever contacted (``ALLOWED_HOSTS``), without
redirects, credentials or cookies, with timeouts and a size cap. Nothing about
the install is sent. This module does no database work: ``price_store`` does.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)
ALLOWED_HOSTS = frozenset({"openrouter.ai", "raw.githubusercontent.com"})
MAX_BYTES = 16 * 1024 * 1024
TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
#: A source that yields fewer rows than this is treated as broken, not as a price cut.
MIN_ROWS = 20

OPENROUTER_HOST = "openrouter.ai"
#: Vendor name in an OpenRouter id (``anthropic/claude-...``) -> the vendor's API host.
VENDOR_HOSTS = {
    "anthropic": "api.anthropic.com",
    "openai": "api.openai.com",
    "google": "generativelanguage.googleapis.com",
    "x-ai": "api.x.ai",
}
#: LiteLLM ``litellm_provider`` -> (API host, key prefix to strip).
LITELLM_HOSTS = {
    "anthropic": ("api.anthropic.com", ""),
    "openai": ("api.openai.com", ""),
    "gemini": ("generativelanguage.googleapis.com", "gemini/"),
    "xai": ("api.x.ai", "xai/"),
    "openrouter": (OPENROUTER_HOST, "openrouter/"),
}
#: Names for the providers this module knows; any other host shows its own name.
HOST_LABELS = {
    "api.anthropic.com": "Anthropic",
    "api.openai.com": "OpenAI",
    "generativelanguage.googleapis.com": "Google Gemini",
    "api.x.ai": "xAI",
    OPENROUTER_HOST: "OpenRouter",
}
_PER_M_MICROS = 1e12  # currency per token -> micro-units per million tokens
_DATE_SUFFIX = re.compile(r"-(?:\d{8}|\d{4}-\d{2}-\d{2})$")


@dataclass(frozen=True)
class PriceRow:
    host: str
    model_key: str
    model: str
    input_per_m_micros: int
    output_per_m_micros: int | None
    cache_read_per_m_micros: int | None
    cache_write_per_m_micros: int | None
    source: str
    fetched_at: str
    currency: str = "USD"


def model_key(name: str) -> str:
    """A model name folded for matching: lower case, no trailing date, dots as
    dashes (``claude-sonnet-4.5`` and ``claude-sonnet-4-5-20250929`` meet)."""
    k = name.strip().lower()
    k = _DATE_SUFFIX.sub("", k)
    return k.replace(".", "-")


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _micros(v: Any) -> int | None:
    """A per-token price (number or numeric string) as micro-units per million
    tokens; None when absent, negative or not a number."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f < 0 or f > 1.0:  # NaN, negative (OpenRouter's -1 = variable), absurd
        return None
    return round(f * _PER_M_MICROS)


def _row(host: str, name: str, p: dict[str, Any], source: str, at: str) -> PriceRow | None:
    inp = _micros(p.get("input"))
    if inp is None:
        return None
    return PriceRow(
        host,
        model_key(name),
        name,
        inp,
        _micros(p.get("output")),
        _micros(p.get("read")),
        _micros(p.get("write")),
        source,
        at,
    )


def parse_openrouter(data: Any, at: str | None = None) -> list[PriceRow]:
    """Rows from OpenRouter's ``/api/v1/models`` answer: every model under
    ``openrouter.ai`` (full id), and the four vendors' own models also under the
    vendor's API host (id without the vendor prefix; variants such as ``:free``
    only under OpenRouter)."""
    at = at or now_iso()
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list):
        raise ValueError("OpenRouter: no data list")
    rows: dict[tuple[str, str], PriceRow] = {}
    for m in models:
        if not isinstance(m, dict) or not isinstance(m.get("id"), str):
            continue
        pr = m.get("pricing")
        if not isinstance(pr, dict):
            continue
        p = {
            "input": pr.get("prompt"),
            "output": pr.get("completion"),
            "read": pr.get("input_cache_read"),
            "write": pr.get("input_cache_write"),
        }
        mid = m["id"]
        r = _row(OPENROUTER_HOST, mid, p, "openrouter", at)
        if r is None:
            continue
        rows[(r.host, r.model_key)] = r
        vendor, _, rest = mid.partition("/")
        host = VENDOR_HOSTS.get(vendor)
        if host and rest and ":" not in rest:
            v = _row(host, rest, p, "openrouter", at)
            if v is not None:
                rows.setdefault((v.host, v.model_key), v)
    return list(rows.values())


def parse_litellm(data: Any, at: str | None = None) -> list[PriceRow]:
    """Rows from LiteLLM's price file for the hosts in ``LITELLM_HOSTS``; entries
    for other providers, and for models that are not chat or completion, are skipped."""
    at = at or now_iso()
    if not isinstance(data, dict):
        raise ValueError("LiteLLM: not an object")
    rows: dict[tuple[str, str], PriceRow] = {}
    for key, v in data.items():
        if not isinstance(key, str) or not isinstance(v, dict):
            continue
        target = LITELLM_HOSTS.get(str(v.get("litellm_provider")))
        if target is None or v.get("mode") not in (None, "chat", "completion", "responses"):
            continue
        host, prefix = target
        name = key[len(prefix) :] if prefix and key.startswith(prefix) else key
        if "/" in name and host != OPENROUTER_HOST:
            continue  # a routed variant (bedrock-style or regional path), not the vendor's own id
        r = _row(
            host,
            name,
            {
                "input": v.get("input_cost_per_token"),
                "output": v.get("output_cost_per_token"),
                "read": v.get("cache_read_input_token_cost"),
                "write": v.get("cache_creation_input_token_cost"),
            },
            "litellm",
            at,
        )
        if r is not None:
            rows[(r.host, r.model_key)] = r
    return list(rows.values())


async def fetch_json(url: str, *, transport: httpx.AsyncBaseTransport | None = None) -> Any:
    """GET ``url`` (one of the two allow-listed https hosts) and parse its JSON.
    No redirects, no environment proxy, no credentials; connect and read
    timeouts; the body is cut off at ``MAX_BYTES``."""
    parts = httpx.URL(url)
    if parts.scheme != "https" or parts.host not in ALLOWED_HOSTS:
        raise ValueError(f"price fetch: host not allowed: {parts.host}")
    async with (
        httpx.AsyncClient(
            timeout=TIMEOUT,
            follow_redirects=False,
            trust_env=False,
            headers={"User-Agent": "lm-warden-price-refresh", "Accept": "application/json"},
            transport=transport,
        ) as client,
        client.stream("GET", url) as resp,
    ):
        if resp.status_code != 200:
            raise RuntimeError(f"price fetch: HTTP {resp.status_code} from {parts.host}")
        size = 0
        chunks: list[bytes] = []
        async for chunk in resp.aiter_bytes():
            size += len(chunk)
            if size > MAX_BYTES:
                raise RuntimeError(f"price fetch: {parts.host} answer over {MAX_BYTES} bytes")
            chunks.append(chunk)
    return json.loads(b"".join(chunks))


async def fetch_rows(
    *, transport: httpx.AsyncBaseTransport | None = None
) -> tuple[list[PriceRow], str, str | None]:
    """The rows of the newest answer: OpenRouter, else LiteLLM. Returns
    ``(rows, source, note)``; ``note`` names why the primary was skipped. Raises
    when neither source gives a usable list."""
    note: str | None = None
    try:
        rows = parse_openrouter(await fetch_json(OPENROUTER_URL, transport=transport))
        if len(rows) >= MIN_ROWS:
            return rows, "openrouter", None
        note = f"openrouter: only {len(rows)} priced models"
    except Exception as exc:  # noqa: BLE001 -- any failure falls back
        note = f"openrouter: {type(exc).__name__}: {str(exc)[:120]}"
    try:
        rows = parse_litellm(await fetch_json(LITELLM_URL, transport=transport))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"{note}; litellm: {type(exc).__name__}: {str(exc)[:120]}") from exc
    if len(rows) < MIN_ROWS:
        raise RuntimeError(f"{note}; litellm: only {len(rows)} priced models")
    return rows, "litellm", note
