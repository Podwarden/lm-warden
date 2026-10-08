"""Resolve a channel → the published vLLM versions that map to a real image.

The try-stack panel (#177) lets an operator pick a ``(channel, vllm_version)``
combo. The version field was free-text; this module backs it with the actual
``vllm/vllm-openai`` semver tags published on Docker Hub so the dropdown only
offers versions that will resolve to an image (``app/runtime/backends/vllm/images.py``).

Three concerns live here, deliberately separated so the network layer is the
only thing tests need to stub:

  - ``resolve_family``: ``channel`` → image family (reuses the resolver's
    ``_RESOLVABLE`` map). Non-resolvable channels return ``None``.
  - ``filter_sort_dedupe_tags``: pure tag-name → version-string transform
    (keep ``^v\\d+\\.\\d+\\.\\d+$``, strip ``v``, semver-desc, dedupe).
  - ``fetch_docker_hub_tags`` + ``EngineVersionsCache``: the bounded,
    paginated Docker Hub query and a 6h in-process TTL cache keyed by image
    family. Docker Hub rate-limits anonymous tag queries hard, so we never
    hit it per request, and serve stale-on-error rather than 500 the page.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable

import httpx

from app.runtime.backends.vllm.images import _RESOLVABLE

logger = logging.getLogger(__name__)

# Match an exact ``vX.Y.Z`` Docker Hub tag — no pre-release / nightly / latest.
# The leading ``v`` is the upstream convention (vllm/vllm-openai:v0.20.0); we
# strip it for the wire shape since the resolver re-adds it.
_SEMVER_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")

# Docker Hub paginates anonymous tag listings; cap the pages we follow so a
# family with a long tag history can't make this unbounded. 5 pages * 100 =
# 500 most-recently-updated tags, which comfortably covers every published
# vllm/vllm-openai semver release.
_MAX_PAGES = 5
_PAGE_SIZE = 100
_DOCKER_HUB_TIMEOUT_S = 10.0

# 6h: the longest end of the issue's "~1-6h" range. Published vLLM releases
# are infrequent (weeks apart) and Docker Hub's anonymous rate limit is the
# real constraint, so a long TTL is strictly better here.
DEFAULT_TTL_SECONDS = 6 * 60 * 60.0


def resolve_family(channel: str) -> str | None:
    """Return the image family for ``channel``, or ``None`` if not resolvable.

    Mirrors ``resolver._RESOLVABLE`` (today every CUDA channel →
    ``vllm/vllm-openai``). rocm/cpu/xpu/unknown channels return ``None`` — they
    already 400 on try-stack, so the dropdown simply offers nothing for them.
    """
    return _RESOLVABLE.get(channel)


def filter_sort_dedupe_tags(tag_names: list[str]) -> list[str]:
    """Keep ``vX.Y.Z`` tags, strip the ``v``, sort semver-desc, dedupe.

    Pure function — the network layer hands it raw tag names and it returns
    the wire-ready version list (newest first). Non-matching tags (``latest``,
    ``nightly``, ``v0.1.2rc1``) are dropped.
    """
    seen: set[tuple[int, int, int]] = set()
    parsed: list[tuple[int, int, int]] = []
    for name in tag_names:
        m = _SEMVER_TAG_RE.match(name)
        if m is None:
            continue
        key = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if key in seen:
            continue
        seen.add(key)
        parsed.append(key)
    parsed.sort(reverse=True)
    return [f"{a}.{b}.{c}" for a, b, c in parsed]


async def fetch_docker_hub_tags(
    family: str, *, client: httpx.AsyncClient | None = None
) -> list[str]:
    """Fetch + normalise the published semver tags for ``family``.

    Pages the Docker Hub tags API (newest-updated first) up to ``_MAX_PAGES``,
    collecting raw tag names, then runs them through
    :func:`filter_sort_dedupe_tags`. Raises ``httpx.HTTPError`` on a transport
    or status failure — the cache layer turns that into stale-or-empty.

    ``client`` is injectable so tests can supply a transport; in production a
    short-lived client is created with a bounded timeout.
    """
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=_DOCKER_HUB_TIMEOUT_S)
    raw_names: list[str] = []
    try:
        url: str | None = (
            f"https://hub.docker.com/v2/repositories/{family}/tags"
            f"?page_size={_PAGE_SIZE}&ordering=last_updated"
        )
        pages = 0
        while url and pages < _MAX_PAGES:
            resp = await client.get(url)
            resp.raise_for_status()
            payload = resp.json()
            for entry in payload.get("results") or []:
                name = entry.get("name")
                if isinstance(name, str):
                    raw_names.append(name)
            # Docker Hub returns an absolute ``next`` URL (or null on last page).
            nxt = payload.get("next")
            url = nxt if isinstance(nxt, str) and nxt else None
            pages += 1
    finally:
        if owns_client:
            await client.aclose()
    return filter_sort_dedupe_tags(raw_names)


class EngineVersionsCache:
    """6h in-process TTL cache of published versions, keyed by image family.

    Same single-``asyncio.Lock`` shape as ``_DiscoveryCache`` in
    ``app/models/routes_api.py``: concurrent requests inside the TTL window
    collapse to one Docker Hub fetch per family. On a fetch error we serve the
    last good value if we have one (stale-on-error), else an empty list — we
    never propagate the third-party failure as a 500. The error is returned
    alongside the versions so the route can reflect it in a field.
    """

    def __init__(
        self,
        *,
        ttl: float = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        fetcher: Callable[[str], Awaitable[list[str]]] | None = None,
    ) -> None:
        self._ttl = ttl
        self._clock = clock
        self._lock = asyncio.Lock()
        # family -> (fetched_at, versions)
        self._cache: dict[str, tuple[float, list[str]]] = {}
        self._fetcher = fetcher or fetch_docker_hub_tags

    async def get(self, family: str) -> tuple[list[str], str | None]:
        """Return ``(versions, error)`` for ``family``.

        ``error`` is ``None`` on a cache hit or a fresh successful fetch, and a
        short sanitised string when the fetch failed (versions then come from
        stale cache, or are empty if nothing was cached yet).
        """
        async with self._lock:
            now = self._clock()
            entry = self._cache.get(family)
            if entry is not None and (now - entry[0]) < self._ttl:
                return entry[1], None
            try:
                versions = await self._fetcher(family)
            except Exception:  # noqa: BLE001 — third-party hiccup must not 500
                logger.warning(
                    "Docker Hub tag fetch failed for %s; serving %s",
                    family,
                    "stale cache" if entry is not None else "empty list",
                    exc_info=True,
                )
                if entry is not None:
                    return entry[1], "docker_hub_unavailable"
                return [], "docker_hub_unavailable"
            self._cache[family] = (now, versions)
            return versions, None
