"""#264 — the proxy tokenizer must never execute a model repo's Python.

``models.trust_remote_code`` was never emitted to any engine (pinned by
tests/unit/runtime/backends/test_launch_characterisation.py::
test_no_case_sets_trust_remote_code). Its ONLY consumer was this cache, which
passed it to ``AutoTokenizer.from_pretrained`` -- so the flag ran the target
repository's Python *inside the warden process*: the process that holds the
SQLite DB and VW_JWT_SECRET, on an unauthenticated-to-the-column ``/v1``
data-plane request, for token accounting only, under every driver.

Counting is accounting, not inference. It already fails open to a character
estimate (test_token_accounting_fallback.py), so the cost of never trusting
remote code is at worst an estimate for a repo whose tokenizer needs custom
code -- and an estimate is what this cache already produces for a GGUF-only
repo today. Code execution in the warden process is not a price worth paying
for accounting precision.
"""

from __future__ import annotations

import inspect

import pytest

from app.proxy.tokenizers import TokenizerCache

pytestmark = pytest.mark.asyncio


class _Recorder:
    """Records every ``from_pretrained`` kwarg the cache ever passes."""

    calls: list[tuple[tuple, dict]] = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.calls.append((args, kwargs))
        return type("_Tok", (), {"encode": staticmethod(lambda s: list(s))})()


@pytest.fixture
def recorder(monkeypatch):
    rec = type("R", (_Recorder,), {"calls": []})
    monkeypatch.setattr("app.proxy.tokenizers.AutoTokenizer", rec)
    return rec


async def test_the_tokenizer_is_constructed_with_trust_remote_code_false(recorder):
    """Explicitly False, not merely absent: this is the security assertion, and
    it must not rest on a transformers default we do not control."""
    cache = TokenizerCache()
    await cache.get("org/needs-custom-code")
    assert recorder.calls, "the cache did not construct a tokenizer at all"
    for _args, kwargs in recorder.calls:
        assert kwargs.get("trust_remote_code") is False


async def test_counting_never_trusts_remote_code(recorder):
    cache = TokenizerCache()
    await cache.count("org/needs-custom-code", "a b c", fallback_repo=None)
    await cache.count("org/gguf-only", "a b c", fallback_repo="org/safetensors")
    assert recorder.calls
    assert all(k.get("trust_remote_code") is False for _a, k in recorder.calls)


async def test_the_cache_takes_no_trust_remote_code_argument():
    """The column cannot reach the tokenizer if the parameter does not exist.

    A caller that still passes it -- app/proxy/routes.py did, three times --
    fails loudly at the call site rather than quietly re-enabling execution.
    """
    for fn in (TokenizerCache.get, TokenizerCache.count):
        assert "trust_remote_code" not in inspect.signature(fn).parameters


async def test_the_cache_key_is_the_repo_alone(recorder):
    """With the flag gone there is one entry per repo, not two. A stale
    ``(repo, bool)`` key would leave a second, unreachable slot behind."""
    cache = TokenizerCache()
    a = await cache.get("org/m")
    b = await cache.get("org/m")
    assert a is b
    assert cache.size() == 1
    assert list(cache._cache) == ["org/m"]


async def test_a_repo_whose_tokenizer_needs_custom_code_still_counts(monkeypatch):
    """The whole blast radius, in one test: transformers refuses a custom-code
    tokenizer when the flag is off, and counting must degrade to an estimate
    rather than 500 a request the engine could have served."""

    class _RefusesWithoutTheFlag:
        @staticmethod
        def from_pretrained(repo, *, trust_remote_code=False, **k):
            if not trust_remote_code:
                raise ValueError(
                    f"Loading {repo} requires you to execute the configuration "
                    "file in that repo on your local machine."
                )
            raise AssertionError("the cache must never ask for remote code")

    monkeypatch.setattr("app.proxy.tokenizers.AutoTokenizer", _RefusesWithoutTheFlag)
    cache = TokenizerCache()
    n = await cache.count("org/custom-code-tokenizer", "hello world")
    assert n > 0
    # And the operator is told, through the same surface a GGUF-only repo uses.
    assert cache.estimating() == frozenset({"org/custom-code-tokenizer"})
