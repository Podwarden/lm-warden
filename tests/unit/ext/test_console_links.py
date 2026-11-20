"""Extension console links (spec 2026-11-20): a generic list of plain links an
extension adds to the console menu. Core registers none."""

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.ext import ConsoleLink, Registry
from tests.conftest import FakeEntryPoint, jwt_login, seed_admin_user


def test_core_registers_no_links(client: TestClient, tmp_data_dir: Path) -> None:
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    body = client.get("/api/edition", headers=jwt_login(client)).json()
    assert body["console_links"] == []


def test_an_extension_link_is_listed(
    ext_registry: Any, client_factory: Any, tmp_data_dir: Path
) -> None:
    def register(reg: Registry) -> None:
        reg.console_links.append(ConsoleLink(label="Extra", href="/_x/page"))

    ext_registry(FakeEntryPoint("x", register))
    with client_factory() as c:
        seed_admin_user(tmp_data_dir / "vllm-warden.db")
        body = c.get("/api/edition", headers=jwt_login(c)).json()
    assert body["console_links"] == [{"label": "Extra", "href": "/_x/page"}]
