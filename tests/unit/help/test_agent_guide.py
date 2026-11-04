"""/agent-guide.md is core, not website: it answers whatever the home-page mode."""

from pathlib import Path

from fastapi.testclient import TestClient

from tests.unit.home.test_modes import _put

REPO = Path(__file__).resolve().parents[3]
PKG = REPO / "app" / "help"


def test_agent_guide_served_when_site_off(client: TestClient, tmp_data_dir: Path) -> None:
    _put(tmp_data_dir, "home_page_mode", "off")
    r = client.get("/agent-guide.md")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/markdown")
    assert r.headers["x-robots-tag"] == "noindex"
    assert r.text == (PKG / "agent-guide.md").read_text(encoding="utf-8")
    assert "{{" not in r.text


def test_agent_guide_head(client: TestClient) -> None:
    assert client.head("/agent-guide.md").status_code == 200


def test_agent_guide_does_not_mint_csrf_cookie(client: TestClient) -> None:
    assert "set-cookie" not in client.get("/agent-guide.md").headers


def test_guide_ships_inside_the_package() -> None:
    """The macOS bundle ships only `app`; the guide must sit next to the route."""
    from app.help import routes

    assert routes.GUIDE_PATH == PKG / "agent-guide.md"
    assert routes.GUIDE_PATH.is_file()


def test_documents_copy_is_identical() -> None:
    assert (REPO / "documents" / "AGENT-GUIDE.md").read_bytes() == (
        PKG / "agent-guide.md"
    ).read_bytes()


def test_coding_agents_doc_names_every_catalogue_client() -> None:
    from app.connect.catalog import CLIENTS

    text = (REPO / "documents" / "CODING-AGENTS.md").read_text(encoding="utf-8")
    for c in CLIENTS:
        assert f"`{c.id}`" in text, c.id


# The "What works" wording per catalogue support level (the Connect page's own
# pills, frontend/src/lib/connect.ts SUPPORT, with "Full" spelled out).
_SUPPORT_WORDING = {
    "full": "Full support",
    "local_only": "Local models only",
    "documented": "Documented, not run",
    "unsupported": "Not supported",
}
_PROTOCOL_WORDING = {"anthropic": "Anthropic Messages", "openai": "OpenAI "}


def test_coding_agents_table_matches_the_catalogue() -> None:
    """Each client's row states the catalogue's protocol and support level,
    not only its id: a client the catalogue marks documented (not run) must
    not read as tested, and vice versa."""
    from app.connect.catalog import CLIENTS

    text = (REPO / "documents" / "CODING-AGENTS.md").read_text(encoding="utf-8")
    rows = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[1].startswith("`") and cells[1].endswith("`"):
            rows[cells[1].strip("`")] = cells
    assert set(rows) == {c.id for c in CLIENTS}
    for c in CLIENTS:
        _name, _id, protocol, works = rows[c.id]
        assert protocol.startswith(_PROTOCOL_WORDING[c.protocol]), (c.id, protocol)
        # Responses only where the catalogue says the client speaks only it.
        responses_only = "speaks only the Responses API" in c.capability
        assert (protocol == "OpenAI Responses") == responses_only, (c.id, protocol)
        assert works.startswith(_SUPPORT_WORDING[c.support]), (c.id, works)
        if "Local models only" in c.capability:
            assert "local models only" in works.lower(), (c.id, works)
