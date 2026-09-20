"""End-to-end test: POST /api/auth/logout cancels any active SSE streams
owned by the same user.

This stitches together the three moving parts that earlier unit tests cover
in isolation:

  * `app.auth.routes.logout` calls `app.state.stream_registry.cancel_user(...)`.
  * The SSE handler in `app.models.routes_logs.stream_logs` registers
    `asyncio.current_task()` on first iteration and unregisters in `finally`.
  * `StreamRegistry.cancel_user` calls `Task.cancel()` on every active task.

Run everything inside one event loop via `httpx.AsyncClient` +
`ASGITransport(app)` so that:

  * The streaming generator's `asyncio.current_task()` is a real Task on the
    same loop that the logout request runs on — `.cancel()` is well-defined
    (no cross-loop hazard like the one we hit in Task 1.7 with the sync
    TestClient).
  * Both requests share `app.state`, so the registry state is observable.
"""
import asyncio

import httpx
import pytest
from httpx import ASGITransport

from tests.conftest import seed_admin_user


# #55 fix — route through the shared barrier-aware helper. The
# downstream POST /api/auth/login is async (httpx.AsyncClient) and
# can't reuse the sync ``jwt_login`` helper; the read-back barrier in
# ``seed_admin_user`` is what closes the race for this caller.
def _seed_done(db_path):
    seed_admin_user(db_path, allowed_gpu_indices=[0])


@pytest.mark.integration
async def test_logout_cancels_active_sse_stream(tmp_path, monkeypatch):
    monkeypatch.setenv("VW_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VW_HF_CACHE_DIR", str(tmp_path / "hf-cache"))
    monkeypatch.setenv("VW_COOKIE_SECRET", "test-secret-32-bytes-min-padding!")
    monkeypatch.setenv("VW_CONTAINER_GPU_COUNT", "4")

    from app.main import build_app
    app = build_app()

    # Drive the lifespan manually so app.state.{sse_tickets,stream_registry,...}
    # are populated and DB migrations have run before we seed the admin row.
    lifespan_cm = app.router.lifespan_context(app)
    await lifespan_cm.__aenter__()
    try:
        _seed_done(tmp_path / "vllm-warden.db")

        # The SSE handler 404s without an existing log file. Create a non-empty
        # one so the generator enters the tail loop (where it blocks on sleeps)
        # rather than completing before cancellation can hit it.
        log_path = tmp_path / "logs" / "demo.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("seed-line\n")

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", timeout=10.0
        ) as client:
            # 1. JWT login
            r = await client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "hunter2"},
            )
            assert r.status_code == 200, r.text
            access = r.json()["access_token"]
            auth = {"Authorization": f"Bearer {access}"}

            # 2. Mint a single-use SSE ticket bound to the stream path.
            stream_path = "/api/models/demo/logs/stream"
            r = await client.post(
                "/api/auth/sse-ticket", json={"path": stream_path}, headers=auth,
            )
            assert r.status_code == 200, r.text
            ticket = r.json()["ticket"]

            registry = app.state.stream_registry

            stream_exc: list[BaseException] = []

            async def consume_stream() -> None:
                """Open the SSE stream and pull at least one chunk so the
                generator's first iteration runs (which is where it registers
                itself). Then keep iterating until the server closes the
                response — that happens when logout cancels our task."""
                try:
                    async with client.stream(
                        "GET", f"{stream_path}?ticket={ticket}",
                    ) as resp:
                        assert resp.status_code == 200, (
                            f"stream did not open: {resp.status_code} {await resp.aread()!r}"
                        )
                        async for _chunk in resp.aiter_bytes():
                            # We don't care about the bytes; we just need the
                            # generator to have started so registration has
                            # happened.
                            pass
                except Exception as exc:
                    stream_exc.append(exc)
                    raise

            stream_task = asyncio.create_task(consume_stream())

            # 3. Wait for the generator's first iteration to register the task.
            #    ASGITransport pumps the body iterator lazily; registration
            #    happens once it pulls the first chunk. Poll a short deadline.
            async def wait_registered(deadline: float) -> bool:
                loop = asyncio.get_running_loop()
                while loop.time() < deadline:
                    if registry.count("session:admin") >= 1:
                        return True
                    await asyncio.sleep(0.05)
                return False

            loop = asyncio.get_running_loop()
            # 5s for registration to defend against ASGI lazy-body-pump priming.
            registered = await wait_registered(loop.time() + 5.0)
            assert registered, (
                "stream did not register within 5s — generator may not have started "
                "iterating; check ASGI body-pump or generator pre-yield work"
            )

            # 4. Logout. Must include matching Origin header (origin_check_dep).
            r = await client.post(
                "/api/auth/logout",
                headers={**auth, "Origin": "http://localhost:3000"},
            )
            assert r.status_code == 204, r.text

            # 5. Within 2 seconds, registry should be empty for this user.
            async def wait_unregistered(deadline: float) -> bool:
                loop = asyncio.get_running_loop()
                while loop.time() < deadline:
                    if registry.count("session:admin") == 0:
                        return True
                    await asyncio.sleep(0.05)
                return False

            # 2s for the post-logout cancellation contract.
            unregistered = await wait_unregistered(loop.time() + 2.0)
            assert unregistered, (
                "stream still registered 2s after logout — cancellation did not "
                "propagate; this is a real bug in registry or handler"
            )

            # 6. Streaming task should finish promptly. The handler's finally
            #    block unregisters and the StreamingResponse closes the body,
            #    so the client-side `async for` ends without an exception
            #    (Starlette swallows the CancelledError on the gen). Either
            #    completion or a CancelledError/HTTPError is acceptable —
            #    what's NOT acceptable is hanging.
            try:
                await asyncio.wait_for(stream_task, timeout=2.0)
            except TimeoutError:
                stream_task.cancel()
                pytest.fail(
                    "client-side stream task did not finish within 2s of logout"
                )
            except (asyncio.CancelledError, httpx.HTTPError):
                # Acceptable terminal states.
                pass

            # Sanity: if the stream task did raise, surface it for visibility
            # but don't fail — the cancel contract is "stream ends", not
            # "stream ends gracefully".
            if stream_exc:
                assert isinstance(
                    stream_exc[0], asyncio.CancelledError | httpx.HTTPError
                ), f"unexpected stream exception type: {type(stream_exc[0])!r}"
    finally:
        await lifespan_cm.__aexit__(None, None, None)
