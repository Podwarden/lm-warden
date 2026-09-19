"""S7 (#124) — code-review finding #5 — tokenizer cache must flush on unload.

The proxy's ``TokenizerCache`` (app/proxy/tokenizers.py) holds one
fully-loaded ``AutoTokenizer`` per (hf_repo, trust_remote_code) tuple.
Before #124 the cache had no eviction — a long-lived warden that
load/unloads a rotating set of models accumulated one entry per
ever-seen repo, eventually OOMing the process.

This module covers the in-process counter-unload contract:

  * ``evict(hf_repo)`` drops both trust_remote_code variants for the repo
  * ``evict`` on an unseen repo is a no-op and returns 0
  * After eviction, a subsequent ``count`` re-fetches via
    ``AutoTokenizer.from_pretrained`` rather than returning a stale entry
  * Eviction is keyed by hf_repo only — evicting "modelA" does NOT drop
    cached entries for "modelB" (no over-eviction)
  * The unload HTTP route wires the cache eviction in: a successful
    ``POST /api/models/{id}/unload`` results in ``evict`` being called
    against the model's hf_repo (end-to-end smoke via TestClient)
"""

import sqlite3
from unittest.mock import MagicMock, patch

from tests.conftest import csrf_header, jwt_login, seed_admin_user


async def test_evict_drops_both_trust_variants():
    from app.proxy.tokenizers import TokenizerCache
    fake = MagicMock()
    fake.encode = lambda s: list(s.encode())
    with patch(
        "app.proxy.tokenizers.AutoTokenizer.from_pretrained",
        return_value=fake,
    ):
        cache = TokenizerCache()
        await cache.get("Qwen/Qwen3.5-9B", trust_remote_code=False)
        await cache.get("Qwen/Qwen3.5-9B", trust_remote_code=True)
        # Sanity: both variants share the hf_repo but have distinct cache keys.
        assert cache.size() == 2
        dropped = await cache.evict("Qwen/Qwen3.5-9B")
    assert dropped == 2
    assert cache.size() == 0


async def test_evict_unseen_repo_is_noop():
    from app.proxy.tokenizers import TokenizerCache
    cache = TokenizerCache()
    dropped = await cache.evict("never/seen")
    assert dropped == 0
    assert cache.size() == 0


async def test_after_eviction_get_refetches():
    """The whole point of eviction is to free the entry — a subsequent
    ``count`` must hit ``AutoTokenizer.from_pretrained`` again, not return
    a stale dict entry. The load counter pins this contract."""
    from app.proxy.tokenizers import TokenizerCache
    fake = MagicMock()
    fake.encode = lambda s: list(s.encode())
    with patch(
        "app.proxy.tokenizers.AutoTokenizer.from_pretrained",
        return_value=fake,
    ) as load:
        cache = TokenizerCache()
        await cache.get("Qwen/Qwen3.5-9B", trust_remote_code=False)
        assert load.call_count == 1
        await cache.evict("Qwen/Qwen3.5-9B")
        # After eviction, the next get() must re-fetch.
        await cache.get("Qwen/Qwen3.5-9B", trust_remote_code=False)
        assert load.call_count == 2


async def test_evict_does_not_overshoot_to_other_repos():
    """Evicting model A must not drop model B's cached tokenizer — the
    cache key includes hf_repo and only the matching repo is removed."""
    from app.proxy.tokenizers import TokenizerCache
    fake = MagicMock()
    fake.encode = lambda s: list(s.encode())
    with patch(
        "app.proxy.tokenizers.AutoTokenizer.from_pretrained",
        return_value=fake,
    ):
        cache = TokenizerCache()
        await cache.get("repo/A", trust_remote_code=False)
        await cache.get("repo/B", trust_remote_code=False)
        assert cache.size() == 2
        dropped = await cache.evict("repo/A")
    assert dropped == 1
    assert cache.size() == 1
    # repo/B must still be cached.
    assert ("repo/B", False) in cache._cache


def test_unload_route_evicts_tokenizer_cache(tmp_data_dir, client):
    """End-to-end: a successful unload via POST /api/models/{id}/unload
    triggers a TokenizerCache.evict for the model's hf_repo. The
    supervisor's unload is patched to a no-op so we test the route
    plumbing in isolation (the supervisor itself has its own coverage).
    """
    client.get("/healthz")  # boot lifespan
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    # Seed a model row in 'loaded' status so the unload route doesn't 409.
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-evict', 'm-evict', 'Qwen/Qwen3.5-9B', 'main', "
            "'[0]', 1, NULL, NULL, 0.9, 0, '[]', 'loaded')"
        )
        db.commit()
    # Pre-warm the cache so we can assert it shrank after unload.
    cache = client.app.state.tokenizers
    fake = MagicMock()
    fake.encode = lambda s: list(s.encode())
    with patch(
        "app.proxy.tokenizers.AutoTokenizer.from_pretrained",
        return_value=fake,
    ):
        import asyncio
        # asyncio.run, not get_event_loop().run_until_complete: the latter
        # raises "There is no current event loop in thread 'MainThread'" when
        # nothing earlier in this worker happened to set one, which made this
        # test fail intermittently under `pytest -n auto` and deterministically
        # when tests/unit/stats ran on its own. asyncio.run owns its loop.
        asyncio.run(cache.get("Qwen/Qwen3.5-9B", trust_remote_code=False))
    assert cache.size() == 1

    auth = jwt_login(client)
    csrf = csrf_header(client)
    # Patch the supervisor's unload to a no-op — we only care that the
    # route reaches the eviction tail after unload.unload() returns OK.
    async def _noop_unload(model_id, *, force=False):
        return None
    with patch.object(client.app.state.supervisor, "unload", _noop_unload):
        r = client.post(
            "/api/models/m-evict/unload",
            headers={**auth, **csrf},
        )
    assert r.status_code == 202, r.text
    # #166 — teardown (and the tokenizer-cache eviction tail) now runs in a
    # background task, so the route returns 'unloading' and the cache shrinks
    # asynchronously; poll for it rather than asserting inline.
    assert r.json() == {"status": "unloading"}
    import time
    for _ in range(60):
        if cache.size() == 0:
            break
        time.sleep(0.05)
    # The hf_repo entry is gone.
    assert cache.size() == 0
