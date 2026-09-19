"""Admin API client: login, CSRF, auto re-login, token CRUD, stats reads.

One cookie-carrying httpx.AsyncClient per session (app/auth/csrf.py mints
`vw_csrf_id` on the login response; the same jar + header pair is valid for
every later mutating call under /api/tokens). Never logs the login body, the
JWT, or a created token's plaintext -- callers must not print `plaintext`
either.
"""

from __future__ import annotations

import time
from typing import Any

import httpx


class AdminError(RuntimeError):
    pass


class AdminSession:
    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        *,
        verify: bool | str = True,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._username = username
        self._password = password
        # I2: a bounded read timeout, not `read=None` -- this is the admin
        # (control-plane) client, which only ever does short JSON round
        # trips; an unbounded read here can hang the isolation drain/restore
        # loop indefinitely on a wedged connection. The data-plane SSE
        # client (_openai.py) legitimately needs `read=None` since a stream
        # can run for minutes -- that one is deliberately different.
        #
        # `transport` is test-only (tests/unit/live/test_live_admin.py uses
        # httpx.MockTransport to exercise list_tokens()'s pagination without
        # a real server); production code never passes it.
        self._client = httpx.AsyncClient(
            base_url=base_url, verify=verify, timeout=httpx.Timeout(30), transport=transport
        )
        self._access_token: str | None = None
        self._expires_at: float = 0.0
        self._csrf: str | None = None

    async def aclose(self) -> None:
        await self._client.aclose()

    async def login(self) -> None:
        resp = await self._client.post(
            "/api/auth/login", json={"username": self._username, "password": self._password}
        )
        if resp.status_code != 200:
            raise AdminError(f"login failed: HTTP {resp.status_code}")
        data = resp.json()
        self._access_token = data["access_token"]
        # 60s safety margin before the real expiry -- a run longer than the
        # 15-minute access TTL must re-login before it actually expires.
        self._expires_at = time.monotonic() + max(0, data["expires_in"] - 60)
        await self._refresh_csrf()

    async def _refresh_csrf(self) -> None:
        resp = await self._client.get("/api/csrf", headers=self._auth_header())
        resp.raise_for_status()
        self._csrf = resp.json()["csrf"]

    def _auth_header(self) -> dict[str, str]:
        if self._access_token is None:
            raise AdminError("not logged in")
        return {"Authorization": f"Bearer {self._access_token}"}

    async def _ensure_fresh(self) -> None:
        if self._access_token is None or time.monotonic() >= self._expires_at:
            await self.login()

    async def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        await self._ensure_fresh()
        headers = dict(kwargs.pop("headers", None) or {})
        headers.update(self._auth_header())
        if method.upper() not in ("GET", "HEAD", "OPTIONS") and self._csrf:
            headers["X-CSRF-Token"] = self._csrf
        resp = await self._client.request(method, path, headers=headers, **kwargs)
        if resp.status_code == 401:
            await self.login()
            headers.update(self._auth_header())
            resp = await self._client.request(method, path, headers=headers, **kwargs)
        if resp.status_code == 403:
            detail = ""
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                pass
            if "csrf" in detail.lower():
                await self._refresh_csrf()
                headers["X-CSRF-Token"] = self._csrf or ""
                resp = await self._client.request(method, path, headers=headers, **kwargs)
        return resp

    # ---- token CRUD ---------------------------------------------------

    async def create_token(self, *, name: str, priority: int, expires_in_days: int = 1) -> dict:
        resp = await self.request(
            "POST",
            "/api/tokens",
            json={"name": name, "priority": priority, "expires_in_days": expires_in_days},
        )
        if resp.status_code != 201:
            raise AdminError(f"create_token({name!r}) failed: HTTP {resp.status_code}")
        return resp.json()

    async def delete_token(self, token_id: str) -> bool:
        resp = await self.request("DELETE", f"/api/tokens/{token_id}")
        if resp.status_code in (204, 404):
            return True
        raise AdminError(f"delete_token({token_id}) failed: HTTP {resp.status_code}")

    async def set_paused(self, token_id: str, paused: bool) -> dict | None:
        """PATCH {"paused": ...}. None on 404 (gone). A dict with
        `{"conflict": True}` on 409 (expired/revoked -- only true for pause,
        never for unpause). Otherwise the token body."""
        resp = await self.request("PATCH", f"/api/tokens/{token_id}", json={"paused": paused})
        if resp.status_code == 404:
            return None
        if resp.status_code == 409:
            return {"conflict": True}
        resp.raise_for_status()
        return resp.json()

    async def list_tokens(self) -> list[dict]:
        """All tokens, paged.

        I-b: GET /api/tokens is becoming paged (default limit 50, response
        `{items, total, limit, offset, ...}`). We ask for a large page
        (500) and keep going while `offset + len(items) < total`. The
        CURRENT (pre-paging) production shape, `{"items": [...]}` with no
        `total`, is treated as already-complete -- one call, no pagination
        fields to page against. Without this, `select_foreign_tokens` (and
        the sweep paths in conftest.py / restore.py) would silently miss
        any token past the first page.
        """
        all_items: list[dict] = []
        offset = 0
        while True:
            resp = await self.request("GET", "/api/tokens", params={"limit": 500, "offset": offset})
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", [])
            all_items.extend(items)
            total = data.get("total")
            if total is None:
                break  # current, unpaged shape -- one call is the whole list
            offset += len(items)
            if not items or offset >= total:
                break
        return all_items

    async def get_token(self, token_id: str) -> dict | None:
        resp = await self.request("GET", f"/api/tokens/{token_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    # ---- models / stats -------------------------------------------------

    async def list_models(self) -> list[dict]:
        resp = await self.request("GET", "/api/models")
        resp.raise_for_status()
        return resp.json()["models"]

    async def live_requests(self) -> dict:
        resp = await self.request("GET", "/api/stats/requests")
        resp.raise_for_status()
        return resp.json()

    async def finished(self, *, model_row_id: str | None, limit: int = 200) -> dict:
        params: dict[str, Any] = {"limit": limit}
        if model_row_id is not None:
            params["models"] = model_row_id
        resp = await self.request("GET", "/api/stats/live/finished", params=params)
        resp.raise_for_status()
        return resp.json()
