"""Unit tests for the #177 engine-versions dropdown backing logic.

Three layers, all network-free:
  - resolve_family: channel → image family (resolvable vs not)
  - filter_sort_dedupe_tags: pure tag-name normaliser
  - EngineVersionsCache: TTL hit/miss + stale-on-error, fetcher injected
  - fetch_docker_hub_tags: pagination + filtering, HTTP stubbed via pytest-httpx
"""

from __future__ import annotations

import httpx
import pytest

from app.runtime.backends.vllm.versions import (
    EngineVersionsCache,
    fetch_docker_hub_tags,
    filter_sort_dedupe_tags,
    resolve_family,
)

# ---- resolve_family -------------------------------------------------------


def test_resolve_family_cuda_channels_map_to_upstream():
    for ch in ("cuda-stable", "cuda-edge", "cuda-legacy"):
        assert resolve_family(ch) == "vllm/vllm-openai"


def test_resolve_family_non_resolvable_channels_return_none():
    for ch in ("rocm", "cpu", "xpu", "totally-made-up"):
        assert resolve_family(ch) is None


# ---- filter_sort_dedupe_tags ----------------------------------------------


def test_filter_keeps_only_exact_semver_strips_v_sorts_desc():
    raw = ["v0.20.0", "latest", "v0.9.0", "nightly", "v0.21.0", "v0.20.0rc1"]
    assert filter_sort_dedupe_tags(raw) == ["0.21.0", "0.20.0", "0.9.0"]


def test_filter_dedupes_repeated_tags():
    raw = ["v0.20.0", "v0.20.0", "v0.19.0"]
    assert filter_sort_dedupe_tags(raw) == ["0.20.0", "0.19.0"]


def test_filter_sorts_by_semver_not_lexicographically():
    # "0.9.0" must sort below "0.10.0" — string sort would invert these.
    raw = ["v0.9.0", "v0.10.0", "v0.2.0"]
    assert filter_sort_dedupe_tags(raw) == ["0.10.0", "0.9.0", "0.2.0"]


def test_filter_empty_input_returns_empty():
    assert filter_sort_dedupe_tags([]) == []


# ---- EngineVersionsCache --------------------------------------------------


@pytest.mark.asyncio
async def test_cache_miss_then_hit_collapses_to_one_fetch():
    calls = {"n": 0}

    async def fetcher(family: str) -> list[str]:
        calls["n"] += 1
        return ["0.21.0", "0.20.0"]

    clock = {"t": 1000.0}
    cache = EngineVersionsCache(ttl=100.0, clock=lambda: clock["t"], fetcher=fetcher)

    v1, e1 = await cache.get("vllm/vllm-openai")
    assert v1 == ["0.21.0", "0.20.0"]
    assert e1 is None
    # Within TTL → served from cache, no second fetch.
    clock["t"] = 1050.0
    v2, e2 = await cache.get("vllm/vllm-openai")
    assert v2 == ["0.21.0", "0.20.0"]
    assert e2 is None
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_cache_refetches_after_ttl_expiry():
    calls = {"n": 0}

    async def fetcher(family: str) -> list[str]:
        calls["n"] += 1
        return [f"0.{calls['n']}.0"]

    clock = {"t": 0.0}
    cache = EngineVersionsCache(ttl=100.0, clock=lambda: clock["t"], fetcher=fetcher)

    await cache.get("fam")
    clock["t"] = 200.0  # past TTL
    v2, _ = await cache.get("fam")
    assert calls["n"] == 2
    assert v2 == ["0.2.0"]


@pytest.mark.asyncio
async def test_cache_serves_stale_on_fetch_error():
    state = {"fail": False}

    async def fetcher(family: str) -> list[str]:
        if state["fail"]:
            raise httpx.ConnectError("docker hub down")
        return ["0.21.0"]

    clock = {"t": 0.0}
    cache = EngineVersionsCache(ttl=100.0, clock=lambda: clock["t"], fetcher=fetcher)

    v1, e1 = await cache.get("fam")
    assert v1 == ["0.21.0"] and e1 is None
    # TTL expires, next fetch fails → stale value served + error reflected.
    state["fail"] = True
    clock["t"] = 200.0
    v2, e2 = await cache.get("fam")
    assert v2 == ["0.21.0"]
    assert e2 == "docker_hub_unavailable"


@pytest.mark.asyncio
async def test_cache_returns_empty_with_error_when_no_prior_value():
    async def fetcher(family: str) -> list[str]:
        raise httpx.ConnectError("docker hub down")

    cache = EngineVersionsCache(ttl=100.0, clock=lambda: 0.0, fetcher=fetcher)
    versions, error = await cache.get("fam")
    assert versions == []
    assert error == "docker_hub_unavailable"


# ---- fetch_docker_hub_tags (HTTP stubbed) ---------------------------------


@pytest.mark.asyncio
async def test_fetch_single_page(httpx_mock):
    httpx_mock.add_response(
        url="https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags"
        "?page_size=100&ordering=last_updated",
        json={
            "results": [
                {"name": "v0.21.0"},
                {"name": "latest"},
                {"name": "v0.20.0"},
            ],
            "next": None,
        },
    )
    async with httpx.AsyncClient() as client:
        out = await fetch_docker_hub_tags("vllm/vllm-openai", client=client)
    assert out == ["0.21.0", "0.20.0"]


@pytest.mark.asyncio
async def test_fetch_follows_pagination(httpx_mock):
    base = "https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags"
    httpx_mock.add_response(
        url=f"{base}?page_size=100&ordering=last_updated",
        json={"results": [{"name": "v0.21.0"}], "next": f"{base}?page=2"},
    )
    httpx_mock.add_response(
        url=f"{base}?page=2",
        json={"results": [{"name": "v0.20.0"}], "next": None},
    )
    async with httpx.AsyncClient() as client:
        out = await fetch_docker_hub_tags("vllm/vllm-openai", client=client)
    assert out == ["0.21.0", "0.20.0"]


@pytest.mark.asyncio
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_fetch_caps_page_count(httpx_mock):
    # Every page advertises a `next`; the fetcher must stop after _MAX_PAGES (5)
    # rather than looping forever. Pages 6/7 are mocked but must NOT be hit.
    base = "https://hub.docker.com/v2/repositories/fam/tags"
    httpx_mock.add_response(
        url=f"{base}?page_size=100&ordering=last_updated",
        json={"results": [{"name": "v0.9.0"}], "next": f"{base}?page=2"},
    )
    for p in range(2, 8):
        httpx_mock.add_response(
            url=f"{base}?page={p}",
            json={"results": [{"name": f"v0.{p}.0"}], "next": f"{base}?page={p + 1}"},
        )
    async with httpx.AsyncClient() as client:
        out = await fetch_docker_hub_tags("fam", client=client)
    # 5 pages fetched: v0.9.0 (p1) + v0.2.0..v0.5.0 (p2-5). p6+ never requested.
    assert out == ["0.9.0", "0.5.0", "0.4.0", "0.3.0", "0.2.0"]
