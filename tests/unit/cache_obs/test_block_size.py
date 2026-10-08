# tests/unit/cache_obs/test_block_size.py
"""The engine's KV block size per engine run (#301): learned off the request
path, read on it as a dictionary lookup."""

from types import SimpleNamespace

import pytest

from app.cache_obs.block_size import BlockSizes, learn_once

PROD = (
    'vllm:cache_config_info{block_size="784",mamba_block_size="16",'
    'mamba_cache_mode="align",num_gpu_blocks="382",engine="0"} 1.0\n'
)


class _Sup:
    def __init__(self, ports, gen=1):
        self._ports = ports
        self._gen = gen

    def get_port(self, mid):
        return self._ports.get(mid)

    def get_host(self, mid):
        return "127.0.0.1"


def _model(mid="qwen", backend="vllm", status="loaded"):
    return SimpleNamespace(id=mid, backend=backend, status=status)


def _state(sup):
    return SimpleNamespace(supervisor=sup, engine_block_sizes=BlockSizes())


def _epoch(state):
    return f"e:{state.supervisor._gen}"


@pytest.fixture(autouse=True)
def _fake_epoch(monkeypatch):
    import app.proxy.routes_dp as routes_dp

    monkeypatch.setattr(routes_dp, "engine_epoch", lambda state, m: _epoch(state))


def test_store_is_scoped_to_one_engine_run():
    s = BlockSizes()
    assert s.get("qwen", "v:1") is None and not s.known("qwen", "v:1")
    s.set("qwen", "v:1", 784)
    assert s.get("qwen", "v:1") == 784 and s.known("qwen", "v:1")
    # a reload is another engine run: the old block size is not inherited
    assert s.get("qwen", "v:2") is None and not s.known("qwen", "v:2")


async def test_learns_vllm_block_size_once_per_engine_run():
    state = _state(_Sup({"qwen": 10000}))
    calls = []

    async def fetch(host, port, path):
        calls.append((host, port, path))
        return PROD

    await learn_once(state, [_model()], fetch)
    assert state.engine_block_sizes.get("qwen", _epoch(state)) == 784
    await learn_once(state, [_model()], fetch)
    assert calls == [("127.0.0.1", 10000, "/metrics")]  # known: no second scrape
    state.supervisor._gen = 2  # reloaded
    await learn_once(state, [_model()], fetch)
    assert len(calls) == 2


async def test_failed_scrape_stays_unknown_and_retries():
    state = _state(_Sup({"qwen": 10000}))
    bodies = [None, PROD]

    async def fetch(host, port, path):
        return bodies.pop(0)

    await learn_once(state, [_model()], fetch)
    assert not state.engine_block_sizes.known("qwen", _epoch(state))
    await learn_once(state, [_model()], fetch)
    assert state.engine_block_sizes.get("qwen", _epoch(state)) == 784


async def test_vllm_without_cache_config_info_is_known_none():
    state = _state(_Sup({"qwen": 10000}))

    async def fetch(host, port, path):
        return 'vllm:num_requests_running{model_name="m"} 0\n'

    await learn_once(state, [_model()], fetch)
    assert state.engine_block_sizes.known("qwen", _epoch(state))
    assert state.engine_block_sizes.get("qwen", _epoch(state)) is None


async def test_llamacpp_is_never_scraped_and_unloaded_models_skipped():
    state = _state(_Sup({"llama": 10001, "qwen": 10000}))

    async def fetch(host, port, path):
        raise AssertionError("no scrape expected")

    await learn_once(
        state, [_model("llama", backend="llamacpp"), _model("qwen", status="unloaded")], fetch
    )
    assert state.engine_block_sizes.get("llama", _epoch(state)) is None
    assert not state.engine_block_sizes.known("qwen", _epoch(state))


async def test_fetch_exception_is_contained():
    state = _state(_Sup({"qwen": 10000}))

    async def fetch(host, port, path):
        raise RuntimeError("boom")

    await learn_once(state, [_model()], fetch)
    assert not state.engine_block_sizes.known("qwen", _epoch(state))
