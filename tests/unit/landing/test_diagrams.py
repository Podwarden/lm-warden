"""Flow diagrams on the coding-agent pages (owner request 2026-10-07): a
generic one at the top of /coding-agents and on Home, and each agent's own
version at the top of its page, generated from the Connect catalogue."""

import re

from fastapi.testclient import TestClient

from app.connect.catalog import CLIENTS
from app.landing.diagrams import AGENTS, GENERIC

ENDPOINT = {"anthropic": "/v1/messages", "openai": "/v1/chat/completions"}


def _main(body: str) -> str:
    return re.search(r"<main\b.*?</main>", body, re.S).group(0)


def test_every_catalogue_agent_but_claude_code_has_a_diagram() -> None:
    """Claude Code has the full router drawing instead."""
    assert set(AGENTS) == {c.id for c in CLIENTS} - {"claude-code"}


def test_hub_opens_with_the_generic_diagram(client: TestClient) -> None:
    main = _main(client.get("/coding-agents").text)
    fig = main.index('aria-labelledby="flow-generic-title"')
    assert fig < main.index('<table class="clients"')
    assert fig < main.index("You don’t write the config by hand")


def test_home_shows_the_generic_diagram(client: TestClient) -> None:
    assert 'aria-labelledby="flow-generic-title"' in client.get("/_landing").text


def test_each_agent_page_opens_with_its_own_diagram(client: TestClient) -> None:
    for c in CLIENTS:
        if c.id == "claude-code":
            continue
        main = _main(client.get(f"/coding-agents/{c.id}").text)
        title = f'aria-labelledby="flow-{c.id}-title"'
        assert title in main, c.id
        assert main.index(title) < main.index('<dl class="facts">'), c.id
        svg = main[main.index(title) - 400 : main.index("</svg>", main.index(title))]
        endpoint = "/v1/responses" if c.id == "codex-cli" else ENDPOINT[c.protocol]
        assert endpoint in svg, c.id
        for mono, _note in AGENTS[c.id].settings:
            assert mono in svg, (c.id, mono)


def test_anthropic_branch_only_where_it_exists(client: TestClient) -> None:
    for c in CLIENTS:
        if c.id == "claude-code":
            continue
        main = _main(client.get(f"/coding-agents/{c.id}").text)
        assert ("api.anthropic.com" in main) == (c.protocol == "anthropic"), c.id
    assert GENERIC.relay  # the generic drawing names it, marked Claude Code only


def test_claude_code_page_opens_with_the_router_drawing(client: TestClient) -> None:
    main = _main(client.get("/coding-agents/claude-code").text)
    assert main.index('<svg class="flow"') < main.index('<dl class="facts">')
    assert main.index('aria-labelledby="router-title"') < main.index("<h2>Setup</h2>")


def test_every_diagram_has_a_phone_cut_and_unique_ids(client: TestClient) -> None:
    for path in ["/coding-agents", "/_landing"] + [f"/coding-agents/{a}" for a in AGENTS]:
        body = client.get(path).text
        assert body.count('class="v-wide"') == body.count('class="v-narrow"'), path
        ids = re.findall(r'\bid="([^"]+)"', body)
        assert len(ids) == len(set(ids)), path
