import asyncio
import sys

import httpx

from tests.conftest import wait_until_async


async def _health_ok(client: httpx.AsyncClient) -> bool:
    try:
        r = await client.get("http://127.0.0.1:18001/health")
        return r.status_code == 200
    except Exception:
        return False


async def test_fake_vllm_health_and_completions():
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "tests.fakes.fake_vllm",
        "--port",
        "18001",
        "--served-model-name",
        "fake-model",
    )
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            await wait_until_async(
                lambda: _health_ok(c), what="fake_vllm to become healthy", child=proc
            )

            r = await c.get("http://127.0.0.1:18001/v1/models")
            assert r.status_code == 200
            assert r.json()["data"][0]["id"] == "fake-model"

            r = await c.post(
                "http://127.0.0.1:18001/v1/chat/completions",
                json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert r.status_code == 200
            data = r.json()
            assert data["choices"][0]["message"]["content"]
            assert data["usage"]["prompt_tokens"] >= 1
    finally:
        if proc.returncode is None:
            proc.terminate()
        await proc.wait()
