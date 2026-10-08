"""The client catalogue behind GET /api/connect/clients (connect-clients plan,
Task 1): shape, template language, the Claude Code golden fixture shared with
vitest, and the server mirror of the UI's glob over known Claude ids."""

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from app.connect.catalog import (
    CLIENTS,
    KEY_PLACEHOLDER,
    MODEL_PLACEHOLDER,
    PLACEHOLDERS,
    ClientDef,
    ModeDef,
    example_model_for,
    render,
    render_file,
)
from app.router.rules import KNOWN_CLAUDE_IDS

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = json.loads((ROOT / "tests/fixtures/connect/claude-code.golden.json").read_text())
_PH = re.compile(r"\{\{([a-z_]+)\}\}")
FULL_VARS = {
    "origin": "https://warden.example",
    "key": "vw_test",
    "model": "qwen",
    "header": "X-LMWarden-Key",
    "context": "65536",
}


def _client(cid: str) -> ClientDef:
    return next(c for c in CLIENTS if c.id == cid)


def _mode(cid: str, mid: str) -> ModeDef:
    return next(m for m in _client(cid).modes if m.id == mid)


def _file(cid: str, mid: str, fid: str) -> Any:
    return next(f for f in _mode(cid, mid).files if f.id == fid)


def _golden_vars() -> dict[str, str]:
    return {
        "origin": GOLDEN["origin"],
        "header": GOLDEN["header"],
        "key": GOLDEN["key"],
        "model": GOLDEN["model"],
        "context": "65536",
    }


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in _strings(k) + _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def test_the_placeholder_language_is_exactly_five_names() -> None:
    assert set(PLACEHOLDERS) == {"origin", "key", "model", "header", "context"}
    assert KEY_PLACEHOLDER == "vw_YOUR_KEY"
    assert MODEL_PLACEHOLDER == "your-served-model-name"


def test_ids_are_unique_slugs_in_display_order() -> None:
    ids = [c.id for c in CLIENTS]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[a-z0-9-]+", i) for i in ids)
    assert ids[0] == "claude-code"
    assert {
        "claude-code",
        "anthropic-sdk",
        "codex-cli",
        "opencode",
        "aider",
        "grok-cli",
        "openai-sdk",
        "continue",
        "cline",
        "cursor",
    } == set(ids)


@pytest.mark.parametrize("client", CLIENTS, ids=lambda c: c.id)
def test_every_client_is_complete(client: ClientDef) -> None:
    assert client.name and client.summary and client.capability
    assert client.group in ("anthropic", "openai", "sdk", "editors")
    assert client.protocol in ("anthropic", "openai")
    assert client.support in ("full", "local_only", "documented", "unsupported")
    assert client.docs.url.startswith("https://")
    date.fromisoformat(client.docs.accessed)
    # Task 4 (live verification) filled these from real runs.
    assert client.verified is not None
    if client.support == "unsupported":
        assert client.modes == ()
    else:
        assert client.modes
    for mode in client.modes:
        assert mode.files, f"{client.id}/{mode.id} has no file"
        assert len({f.id for f in mode.files}) == len(mode.files)
        assert mode.verify is not None
        assert mode.verify.method == "POST"
        assert mode.verify.path in ("/v1/messages", "/v1/chat/completions", "/v1/responses")
        if client.id == "codex-cli":
            expected = "openai_responses"
        else:
            expected = "anthropic_message" if client.protocol == "anthropic" else "openai_chat"
        assert mode.verify.expect == expected
        assert mode.verify.covers
    for req in client.requirements:
        assert req.id and req.text
        if req.modes is not None:
            assert set(req.modes) <= {m.id for m in client.modes}


@pytest.mark.parametrize("client", CLIENTS, ids=lambda c: c.id)
def test_templates_use_only_the_five_placeholders(client: ClientDef) -> None:
    for mode in client.modes:
        for f in mode.files:
            found = _PH.findall(f.template)
            assert set(found) <= set(PLACEHOLDERS), f"{client.id}/{f.id}: {found}"
            assert f.template.count("{{") == len(found), f"{client.id}/{f.id}: stray {{{{"
            assert render(f.template, FULL_VARS).count("{{") == 0
        verify = mode.verify
        assert verify is not None
        for s in _strings(dict(verify.headers)) + _strings(verify.body):
            assert set(_PH.findall(s)) <= set(PLACEHOLDERS) | {"verify_model"}


@pytest.mark.parametrize("client", CLIENTS, ids=lambda c: c.id)
def test_no_template_carries_a_real_key(client: ClientDef) -> None:
    for mode in client.modes:
        for f in mode.files:
            assert not re.search(r"vw[a]?_[A-Za-z0-9]", f.template), f"{client.id}/{f.id}"


def test_render_is_a_plain_string_replace() -> None:
    assert render("{{origin}}/v1 {{key}} {{origin}}", FULL_VARS) == (
        "https://warden.example/v1 vw_test https://warden.example"
    )
    assert render("{{unknown}} {{model}}", {"model": "m"}) == "{{unknown}} m"


def test_openai_clients_point_at_v1_and_anthropic_clients_do_not() -> None:
    for client in CLIENTS:
        for mode in client.modes:
            for f in mode.files:
                if "{{origin}}" not in f.template:
                    continue
                if client.protocol == "openai":
                    assert "{{origin}}/v1" in f.template, f"{client.id}/{f.id}"
                else:
                    assert "{{origin}}/v1\n" not in f.template
                    assert '"{{origin}}/v1"' not in f.template


def test_claude_code_router_files_equal_the_golden_fixture() -> None:
    v = _golden_vars()
    assert render(_file("claude-code", "router", "shell").template, v) == GOLDEN["env"]
    assert render(_file("claude-code", "router", "settings").template, v) == GOLDEN["settings_json"]


def test_claude_code_local_shell_extends_the_golden_local_only_block() -> None:
    out = render(_file("claude-code", "local", "shell").template, _golden_vars())
    assert out.startswith(GOLDEN["local_only"] + "\n")
    tail = out[len(GOLDEN["local_only"]) + 1 :].splitlines()
    assert tail == [
        "export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1",
        "export CLAUDE_CODE_MAX_CONTEXT_TOKENS=65536",
    ]


def test_claude_code_local_settings_is_valid_json_with_the_same_keys() -> None:
    out = json.loads(render(_file("claude-code", "local", "settings").template, _golden_vars()))
    assert out["env"]["ANTHROPIC_AUTH_TOKEN"] == "vw_YOUR_KEY"
    assert out["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "qwen"
    assert out["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "65536"


def test_claude_code_router_verify_uses_the_header_and_no_authorization() -> None:
    verify = _mode("claude-code", "router").verify
    assert verify is not None
    assert verify.headers["{{header}}"] == "{{key}}"
    assert not any(k.lower() in ("authorization", "x-api-key") for k in verify.headers)
    assert verify.body["model"] == "{{verify_model}}"
    assert verify.not_covered
    local = _mode("claude-code", "local").verify
    assert local is not None
    assert local.headers["x-api-key"] == "{{key}}"
    assert local.body["model"] == "{{model}}"


def test_claude_code_needs_49k_context_and_a_relay_key_in_router_mode() -> None:
    reqs = {r.id: r for r in _client("claude-code").requirements}
    assert reqs["min_context"].min_context_tokens == 49152
    assert not reqs["min_context"].soft
    assert reqs["recommended_context"].min_context_tokens == 131072
    assert reqs["recommended_context"].soft
    assert reqs["relay_key"].relay_key and reqs["relay_key"].modes == ("router",)
    assert reqs["router_on"].router_on and reqs["router_on"].modes == ("router",)
    assert "2.1.227" in reqs["client_version"].text
    assert _mode("claude-code", "router").needs_relay_key
    assert not _mode("claude-code", "local").needs_relay_key


def test_codex_is_local_only_over_responses() -> None:
    import tomllib

    codex = _client("codex-cli")
    assert codex.support == "local_only"
    assert "Responses" in codex.capability and "stateless" in codex.capability
    assert "web_search" in codex.capability
    (mode,) = codex.modes
    assert not mode.needs_relay_key
    config = next(f for f in mode.files if f.id == "config")
    assert config.path == "~/.codex/config.toml" and config.language == "toml"
    parsed = tomllib.loads(render(config.template, FULL_VARS))
    provider = parsed["model_providers"]["lmwarden"]
    assert provider["wire_api"] == "responses"
    assert provider["base_url"] == f"{FULL_VARS['origin']}/v1"
    assert provider["env_key"] == "LMWARDEN_KEY"
    assert parsed["model_provider"] == "lmwarden"
    assert parsed["model"] == FULL_VARS["model"]
    assert parsed["model_context_window"] == int(FULL_VARS["context"])
    assert "model_supports_reasoning_summaries" not in parsed  # ignored by Codex 0.160
    # the key never lands in a committed file: it is exported
    assert "{{key}}" not in config.template
    shell = next(f for f in mode.files if f.id == "shell")
    assert "export LMWARDEN_KEY={{key}}" in shell.template and "codex" in shell.template
    v = mode.verify
    assert v is not None and v.path == "/v1/responses" and v.expect == "openai_responses"
    assert v.headers["Authorization"] == "Bearer {{key}}"
    assert v.body == {
        "model": "{{model}}",
        "input": "Reply with OK.",
        "max_output_tokens": 16,
        "stream": False,
    }


def test_codex_config_equals_the_golden_fixture() -> None:
    golden = json.loads((ROOT / "tests/fixtures/connect/codex-cli.golden.json").read_text())
    got = render(_file("codex-cli", "default", "config").template, golden["vars"])
    assert got == golden["config_toml"]


def test_continue_has_a_tools_and_a_no_tools_variant() -> None:
    files = _mode("continue", "default").files
    variants = {f.requires_tools: f for f in files}
    assert set(variants) == {True, False}
    assert "capabilities: [tool_use]" in variants[True].template
    assert "tool_use" not in variants[False].template
    assert all("useResponsesApi: false" in f.template for f in files)


def test_editors_without_a_file_use_fields() -> None:
    for cid in ("cline", "cursor"):
        assert _client(cid).support == "documented"
        assert [f.language for f in _mode(cid, "default").files] == ["fields"]
    assert any(r.id == "reachable" for r in _client("cursor").requirements)


def test_render_file_notes_a_guessed_context_in_files_with_comments() -> None:
    aider_shell = _file("aider", "default", "shell")
    grok = _file("grok-cli", "default", "config")
    v = {**FULL_VARS, "context": "32768"}
    assert render_file(grok, v, context_known=True) == render(grok.template, v)
    guessed = render_file(grok, v, context_known=False)
    assert "context_window = 32768" in guessed
    assert re.search(r"^# .*32768.*guess", guessed, re.M)
    # No {{context}} in the file: nothing to note.
    assert render_file(aider_shell, v, context_known=False) == render(aider_shell.template, v)
    # JSON takes no comments.
    meta = _file("aider", "default", "metadata")
    assert json.loads(render_file(meta, v, context_known=False))


# -- example_model_for: server mirror of the UI's globMatches --------------


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("claude-haiku*", "claude-haiku-4-5"),
        ("claude-sonnet*", "claude-sonnet-4-5"),
        ("claude-?-5-haiku*", "claude-3-5-haiku-latest"),
        ("claude-opus-4-1", "claude-opus-4-1"),
        ("my-exact-name", "my-exact-name"),
        ("gpt-*", None),
        ("Claude-haiku*", None),  # case-sensitive, like match_rule
        ("claude-haiku-4-?", "claude-haiku-4-5"),
    ],
)
def test_example_model_for(pattern: str, expected: str | None) -> None:
    assert example_model_for(pattern) == expected


def test_known_claude_ids_match_the_frontend_list() -> None:
    src = (ROOT / "frontend/src/lib/router.ts").read_text()
    block = re.search(r"KNOWN_CLAUDE_IDS[^=]*=\s*\[(.*?)\]", src, re.S)
    assert block is not None
    assert tuple(re.findall(r'"([^"]+)"', block.group(1))) == KNOWN_CLAUDE_IDS


# -- Task 4: live-verification records and the fixes the runs exposed -------

_EXPECTED_VERIFIED = {
    "claude-code": "run",
    "anthropic-sdk": "run",
    "opencode": "run",
    "aider": "run",
    "grok-cli": "run",
    "openai-sdk": "run",
    "continue": "documented",
    "cline": "documented",
    "cursor": "documented",
    "codex-cli": "run",
}


@pytest.mark.parametrize("client", CLIENTS, ids=lambda c: c.id)
def test_verified_records_say_what_was_run_and_when(client: ClientDef) -> None:
    v = client.verified
    assert v is not None
    assert v.status == _EXPECTED_VERIFIED[client.id]
    date.fromisoformat(v.date)
    if v.status in ("run", "failed"):
        # A run names the client version it ran.
        assert v.client_version
    if v.status == "failed":
        assert v.note
    if v.status == "documented":
        assert v.client_version is None
    # The GUI editors and the unsupported client never claim a run.
    if client.support in ("documented", "unsupported"):
        assert v.status != "run"


def test_claude_code_record_says_both_modes_run_live() -> None:
    v = _client("claude-code").verified
    assert v is not None and v.note
    assert v.status == "run" and v.date == "2026-10-04"
    assert v.client_version == "Claude Code 2.1.289"
    assert "local-only and router modes run live" in v.note.lower()
    assert "400" not in v.note and "pending" not in v.note


def test_claude_code_local_mode_tips_the_unknown_model_warning() -> None:
    reqs = [r for r in _client("claude-code").requirements if r.id == "unknown_model_window"]
    assert len(reqs) == 1
    r = reqs[0]
    assert r.soft and r.modes == ("local",)
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" in r.text and "200k" in r.text


def test_codex_record_says_what_ran_live() -> None:
    v = _client("codex-cli").verified
    assert v is not None and v.note
    assert v.status == "run" and v.date == "2026-10-05"
    assert v.client_version == "codex-cli 0.160.0"
    assert "workspace-write" in v.note and "cached tokens" in v.note
    assert 'model_reasoning_effort = "none"' in v.note


def test_grok_config_sends_session_titles_to_the_warden_model() -> None:
    import tomllib

    # Grok Build asks for a session title with grok-4.6 on the BYOK base URL
    # unless [models] session_summary names the BYOK model (live run, Task 4).
    grok = _file("grok-cli", "default", "config")
    parsed = tomllib.loads(render(grok.template, FULL_VARS))
    assert parsed["models"] == {"default": "lmwarden", "session_summary": "lmwarden"}
    assert parsed["model"]["lmwarden"]["model"] == "qwen"


def test_grok_community_mode_says_tool_calls_fail() -> None:
    mode = _mode("grok-cli", "community")
    assert "tool call" in mode.description.lower()
    assert "bun" in mode.description.lower()
