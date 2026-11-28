"""A tokenizer that will not load must not be re-fetched on every request.

Reported from a live install: with a model whose tokenizer cannot be loaded
(GGUF-only with no tokenizer_repo, a gated repo with no token, a tokenizer that
needs remote code, or no route to huggingface.co), every /v1 request called
``AutoTokenizer.from_pretrained`` again -- several HTTP round trips to the Hub
on the hot path -- because ``TokenizerCache`` remembered successes only. The
``_estimating`` set silenced the warning but not the load.

The cache now remembers a failed load for a while and estimates in the
meantime; it retries after the window, and an unload (``evict``) clears the
memory at once, so fixing the model's tokenizer_repo still takes effect on the
next load.
"""

from __future__ import annotations

import pytest

from app.proxy import tokenizers as tk
from app.proxy.tokenizers import TokenizerCache

pytestmark = pytest.mark.asyncio


class _Counter:
    def __init__(self, fail: bool = True) -> None:
        self.calls: list[str] = []
        self.fail = fail

    def from_pretrained(self, repo, **_kw):
        self.calls.append(repo)
        if self.fail:
            raise OSError("does not appear to have a file named tokenizer.json")
        return type("Tok", (), {"encode": staticmethod(lambda s: s.split())})()


@pytest.fixture
def hub(monkeypatch):
    c = _Counter()
    monkeypatch.setattr(tk, "AutoTokenizer", c)
    return c


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(tk, "_now", lambda: now[0])
    return now


async def test_a_failed_load_is_attempted_once_across_many_requests(hub, clock):
    cache = TokenizerCache()
    for _ in range(20):
        assert await cache.count("org/gguf-only", "a b c d") > 0
    assert hub.calls == ["org/gguf-only"]


async def test_the_failure_is_remembered_per_tried_repo(hub, clock):
    cache = TokenizerCache()
    await cache.count("org/a", "x")
    await cache.count("org/b", "x")
    await cache.count("org/a", "x")
    await cache.count("org/b", "x")
    assert hub.calls == ["org/a", "org/b"]


async def test_the_load_is_retried_after_the_window(hub, clock):
    cache = TokenizerCache()
    await cache.count("org/gguf-only", "x")
    clock[0] += tk.FAILED_LOAD_RETRY_S - 1
    await cache.count("org/gguf-only", "x")
    assert len(hub.calls) == 1
    clock[0] += 2
    await cache.count("org/gguf-only", "x")
    assert len(hub.calls) == 2


async def test_a_retry_that_succeeds_gives_exact_counts_and_clears_estimating(hub, clock):
    cache = TokenizerCache()
    await cache.count("org/flaky", "a b c d e f g h")
    assert cache.estimating() == frozenset({"org/flaky"})
    hub.fail = False
    clock[0] += tk.FAILED_LOAD_RETRY_S + 1
    assert await cache.count("org/flaky", "a b c") == 3
    assert cache.estimating() == frozenset()


async def test_evict_forgets_the_failure_so_a_fixed_row_loads_at_once(hub, clock):
    """Unload is how an operator applies a corrected tokenizer_repo; it must
    not have to wait out the retry window."""
    cache = TokenizerCache()
    await cache.count("org/gguf-only", "x", fallback_repo="org/wrong-tokenizer")
    await cache.evict("org/gguf-only")
    hub.fail = False
    assert await cache.count("org/gguf-only", "a b", fallback_repo="org/right-tokenizer") == 2
    assert hub.calls == ["org/wrong-tokenizer", "org/right-tokenizer"]
