"""Agent-tuning page and the agent guide (spec 2026-10-07 §5)."""

import re
from pathlib import Path

from fastapi.testclient import TestClient


def test_agent_guide_served_as_markdown(client: TestClient) -> None:
    r = client.get("/agent-guide.md")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/markdown")
    assert r.headers["x-robots-tag"] == "noindex"
    assert r.text.startswith("# Driving LM Warden from a coding agent")
    assert "{{" not in r.text
    assert "set-cookie" not in r.headers


def test_every_endpoint_named_exists_in_the_api(client: TestClient) -> None:
    """The page and the guide name only endpoints the API really has, with
    the method it really answers."""
    schema = client.app.openapi()["paths"]
    text = client.get("/agent-tuning").text + client.get("/agent-guide.md").text
    named = set(re.findall(r"\b(GET|POST|PATCH|DELETE) (/api/[a-z0-9_/{}-]+)", text))
    assert len(named) >= 8
    for method, path in named:
        assert path in schema, path
        assert method.lower() in schema[path], (method, path)


def test_prompts_have_copy_buttons_and_point_at_the_guide(client: TestClient) -> None:
    body = client.get("/agent-tuning").text
    prompts = re.findall(r'<pre id="(prompt-\d)"[^>]*>(.*?)</pre>', body, re.S)
    assert len(prompts) == 4
    for pid, text in prompts:
        assert f'data-copy="{pid}"' in body
        assert "/agent-guide.md" in text


def test_safety_notes_present(client: TestClient) -> None:
    body = client.get("/agent-tuning").text
    assert "full admin" in body and "crashes an engine" in body and "Revoke" in body


def test_llms_txt_links_the_guide(client: TestClient) -> None:
    assert "/agent-guide.md" in client.get("/llms.txt").text


def test_disabled_site_hides_the_guide(client: TestClient, tmp_data_dir: Path) -> None:
    from tests.unit.landing.test_routes import _disable

    _disable(tmp_data_dir)
    assert client.get("/agent-guide.md").status_code == 404
