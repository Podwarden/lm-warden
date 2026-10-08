"""The client catalogue behind ``GET /api/connect/clients``.

One entry per harness an operator may point at this warden: what protocol it
speaks, what it honestly gets here (``support``), the files to paste and a
``verify`` request definition the UI (and a CLI user) can send to prove the
snippet works. Plan: docs/superpowers/plans/2026-10-04-connect-clients.md
(§2 contract, §4 per-tool specifications, with the doc URLs and access dates
that went into ``docs``).

Templates speak a five-placeholder language -- ``{{origin}}``, ``{{key}}``,
``{{model}}``, ``{{header}}``, ``{{context}}`` -- and rendering is a plain
string replace (``render``), mirrored by ``renderTemplate`` in
frontend/src/lib/connect.ts. A verify body may also use ``{{verify_model}}``.
No template holds a key: the server only ever renders ``KEY_PLACEHOLDER``.

The Claude Code templates are the same text as ``envSnippet`` /
``settingsJsonSnippet`` / ``localOnlySnippet`` in frontend/src/lib/router.ts;
tests/fixtures/connect/claude-code.golden.json pins both sides.

``verified`` records, per client, what the live-verification task (plan §11)
ran against lmwarden.com, with which client version and when; GUI editors are
``documented`` (config checked against the docs, the request they send run
with curl) and nothing is ``run`` that was not run.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Literal

from app.router.rules import KNOWN_CLAUDE_IDS

PLACEHOLDERS: tuple[str, ...] = ("origin", "key", "model", "header", "context")
KEY_PLACEHOLDER = "vw_YOUR_KEY"
MODEL_PLACEHOLDER = "your-served-model-name"
#: ``{{context}}`` when the selected model's window is unknown (or no model).
DEFAULT_CONTEXT = 32768
DOCS_ACCESSED = "2026-10-04"
VERIFIED_ON = "2026-10-04"

Group = Literal["anthropic", "openai", "sdk", "editors"]
Protocol = Literal["anthropic", "openai"]
Support = Literal["full", "local_only", "documented", "unsupported"]
Language = Literal["bash", "json", "toml", "yaml", "python", "typescript", "fields"]
Expect = Literal["anthropic_message", "openai_chat", "openai_responses"]
VerifiedStatus = Literal["run", "documented", "failed"]

_COMMENT_PREFIX: dict[str, str] = {
    "bash": "#",
    "toml": "#",
    "yaml": "#",
    "python": "#",
    "typescript": "//",
}


@dataclass(frozen=True)
class Requirement:
    id: str
    text: str
    min_context_tokens: int | None = None
    soft: bool = False
    needs_tools: bool = False
    router_on: bool = False
    relay_key: bool = False
    #: Mode ids this applies to; None = every mode.
    modes: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Docs:
    url: str
    accessed: str = DOCS_ACCESSED


@dataclass(frozen=True)
class Verified:
    status: VerifiedStatus
    date: str
    client_version: str | None
    note: str | None


@dataclass(frozen=True)
class FileDef:
    id: str
    label: str
    path: str | None
    language: Language
    template: str
    #: None: always shown. True/False: shown only when the picked model's
    #: ``supports_tools`` matches (Continue's two config variants).
    requires_tools: bool | None = None


@dataclass(frozen=True)
class VerifyDef:
    path: str
    headers: Mapping[str, str]
    body: Mapping[str, Any]
    expect: Expect
    covers: str
    not_covered: str | None = None
    method: str = "POST"


@dataclass(frozen=True)
class ModeDef:
    id: str
    title: str
    description: str
    needs_relay_key: bool
    files: tuple[FileDef, ...]
    verify: VerifyDef | None


@dataclass(frozen=True)
class ClientDef:
    id: str
    name: str
    group: Group
    protocol: Protocol
    support: Support
    summary: str
    capability: str
    docs: Docs
    requirements: tuple[Requirement, ...] = ()
    modes: tuple[ModeDef, ...] = ()
    verified: Verified | None = field(default=None)


# -- rendering --------------------------------------------------------------


def render(template: str, variables: Mapping[str, str]) -> str:
    """Replace each ``{{name}}`` with ``variables[name]``; anything else stays."""
    out = template
    for name, value in variables.items():
        out = out.replace("{{" + name + "}}", value)
    return out


def render_file(f: FileDef, variables: Mapping[str, str], *, context_known: bool) -> str:
    """``render`` plus, when the context window is a guess and the file both
    uses ``{{context}}`` and has a comment syntax, a trailing comment saying so."""
    out = render(f.template, variables)
    prefix = _COMMENT_PREFIX.get(f.language)
    if not context_known and prefix and "{{context}}" in f.template:
        out += (
            f"\n{prefix} The model's context window is unknown: "
            f"{variables.get('context', DEFAULT_CONTEXT)} is a guess. "
            "Set it to the model's real window."
        )
    return out


def example_model_for(pattern: str) -> str | None:
    """A Claude model id a rule ``pattern`` matches: the first known id it
    matches (case-sensitive, like ``match_rule``), else the pattern itself when
    it has no wildcard, else None."""
    for cid in KNOWN_CLAUDE_IDS:
        if fnmatchcase(cid, pattern):
            return cid
    if "*" in pattern or "?" in pattern:
        return None
    return pattern


# -- shared pieces ----------------------------------------------------------

_PROMPT = "Reply with the single word OK."
_MESSAGES = [{"role": "user", "content": _PROMPT}]


def _anthropic_verify(
    auth: Mapping[str, str], model: str, covers: str, not_covered: str | None = None
) -> VerifyDef:
    return VerifyDef(
        path="/v1/messages",
        headers={**auth, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        body={"model": model, "max_tokens": 16, "messages": _MESSAGES},
        expect="anthropic_message",
        covers=covers,
        not_covered=not_covered,
    )


def _openai_verify(covers: str, not_covered: str | None = None) -> VerifyDef:
    return VerifyDef(
        path="/v1/chat/completions",
        headers={"Authorization": "Bearer {{key}}", "content-type": "application/json"},
        body={"model": "{{model}}", "max_tokens": 16, "stream": False, "messages": _MESSAGES},
        expect="openai_chat",
        covers=covers,
        not_covered=not_covered,
    )


def _responses_verify(covers: str, not_covered: str | None = None) -> VerifyDef:
    return VerifyDef(
        path="/v1/responses",
        headers={"Authorization": "Bearer {{key}}", "content-type": "application/json"},
        body={
            "model": "{{model}}",
            "input": "Reply with OK.",
            "max_output_tokens": 16,
            "stream": False,
        },
        expect="openai_responses",
        covers=covers,
        not_covered=not_covered,
    )


_ROUTER_VERIFY = _anthropic_verify(
    {"{{header}}": "{{key}}"},
    "{{verify_model}}",
    covers="A rule-matched Claude model served locally, with the key in {{header}}.",
    not_covered=(
        "Pass-through to Anthropic needs your own Claude login, which this test does not "
        "send. Run the client and watch Router > Activity."
    ),
)
_LOCAL_ANTHROPIC_VERIFY = _anthropic_verify(
    {"x-api-key": "{{key}}"},
    "{{model}}",
    covers="The picked model answering an Anthropic Messages request, with the key in x-api-key.",
)
_CHAT_VERIFY = _openai_verify(
    covers="The picked model answering a Chat Completions request, with the key as a Bearer token."
)
_CHAT_VERIFY_RAW = _openai_verify(
    covers=(
        "The request this tool sends (Chat Completions, key as a Bearer token), "
        "sent from this page."
    ),
    not_covered="The tool itself: it needs its own settings screen, so it was not run here.",
)

_ROUTER_CAPABILITY = (
    "The Claude models your rules name run here; every other model and /v1 path goes to "
    "Anthropic on your own login."
)
_ROUTER_REQS = (
    Requirement(
        id="router_on",
        text="Routing must be on, with at least one enabled rule.",
        router_on=True,
        modes=("router",),
    ),
    Requirement(
        id="relay_key",
        text="The key must be allowed to relay to Anthropic (May relay).",
        relay_key=True,
        modes=("router",),
    ),
)
_TOOLS = Requirement(
    id="tools", text="Needs a model with tool calling (agentic edits).", needs_tools=True
)
_SOFT_32K = Requirement(
    id="min_context",
    text="Works best with at least 32,768 tokens of context.",
    min_context_tokens=32768,
    soft=True,
)


# -- Claude Code (plan §4.1) ------------------------------------------------

_CC_ROUTER_SHELL = (
    "export ANTHROPIC_BASE_URL={{origin}}\n"
    'export ANTHROPIC_CUSTOM_HEADERS="{{header}}: {{key}}"\n'
    "claude"
)
_CC_ROUTER_SETTINGS = (
    "{\n"
    '  "env": {\n'
    '    "ANTHROPIC_BASE_URL": "{{origin}}",\n'
    '    "ANTHROPIC_CUSTOM_HEADERS": "{{header}}: {{key}}"\n'
    "  }\n"
    "}"
)
_CC_LOCAL_SHELL = (
    "export ANTHROPIC_BASE_URL={{origin}}\n"
    "export ANTHROPIC_AUTH_TOKEN={{key}}\n"
    "export ANTHROPIC_MODEL={{model}}\n"
    "export ANTHROPIC_DEFAULT_OPUS_MODEL={{model}}\n"
    "export ANTHROPIC_DEFAULT_SONNET_MODEL={{model}}\n"
    "export ANTHROPIC_DEFAULT_HAIKU_MODEL={{model}}\n"
    "export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1\n"
    "export CLAUDE_CODE_MAX_CONTEXT_TOKENS={{context}}"
)
_CC_LOCAL_SETTINGS = (
    "{\n"
    '  "env": {\n'
    '    "ANTHROPIC_BASE_URL": "{{origin}}",\n'
    '    "ANTHROPIC_AUTH_TOKEN": "{{key}}",\n'
    '    "ANTHROPIC_MODEL": "{{model}}",\n'
    '    "ANTHROPIC_DEFAULT_OPUS_MODEL": "{{model}}",\n'
    '    "ANTHROPIC_DEFAULT_SONNET_MODEL": "{{model}}",\n'
    '    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "{{model}}",\n'
    '    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",\n'
    '    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "{{context}}"\n'
    "  }\n"
    "}"
)

CLAUDE_CODE = ClientDef(
    id="claude-code",
    name="Claude Code",
    group="anthropic",
    protocol="anthropic",
    support="full",
    summary="Keep your Claude login and run the models your rules name here, or run it fully local.",
    capability=(
        f"{_ROUTER_CAPABILITY} Never put the warden key in ANTHROPIC_API_KEY or "
        "ANTHROPIC_AUTH_TOKEN in router mode: that would replace your login. Local only runs "
        "everything on one served model, including the small background tasks, and nothing "
        "goes to Anthropic."
    ),
    docs=Docs("https://code.claude.com/docs/en/llm-gateway"),
    requirements=(
        Requirement(
            id="client_version",
            text="Claude Code 2.1.227 or newer (ANTHROPIC_CUSTOM_HEADERS).",
            modes=("router",),
        ),
        *_ROUTER_REQS,
        Requirement(
            id="min_context",
            text=(
                "Needs at least 49,152 tokens of context: the first turn alone is about "
                "38,000 tokens before any file is read."
            ),
            min_context_tokens=49152,
        ),
        Requirement(
            id="recommended_context",
            text="131,072 tokens or more for real sessions; less works for short ones.",
            min_context_tokens=131072,
            soft=True,
        ),
        Requirement(
            id="unknown_model_window",
            text=(
                "Claude Code may warn that the model name is not in its catalogue. That is "
                "informational: it assumes a 200k window and auto-compacts accordingly. Set "
                "CLAUDE_CODE_MAX_CONTEXT_TOKENS to the model's real context window."
            ),
            modes=("local",),
            soft=True,
        ),
    ),
    modes=(
        ModeDef(
            id="router",
            title="Router: keep your Claude login",
            description=(
                "Your Claude login keeps flowing to Anthropic; the warden key travels in its "
                "own header, and the models your rules name are answered here."
            ),
            needs_relay_key=True,
            files=(
                FileDef("shell", "Shell", None, "bash", _CC_ROUTER_SHELL),
                FileDef(
                    "settings",
                    "settings.json",
                    "~/.claude/settings.json",
                    "json",
                    _CC_ROUTER_SETTINGS,
                ),
            ),
            verify=_ROUTER_VERIFY,
        ),
        ModeDef(
            id="local",
            title="Local only (no Anthropic account)",
            description=(
                "Every model slot is the picked served model and the warden key is the auth "
                "token. Nothing goes to Anthropic."
            ),
            needs_relay_key=False,
            files=(
                FileDef("shell", "Shell", None, "bash", _CC_LOCAL_SHELL),
                FileDef(
                    "settings",
                    "settings.json",
                    "~/.claude/settings.json",
                    "json",
                    _CC_LOCAL_SETTINGS,
                ),
            ),
            verify=_LOCAL_ANTHROPIC_VERIFY,
        ),
    ),
    verified=Verified(
        status="run",
        date=VERIFIED_ON,
        client_version="Claude Code 2.1.289",
        note=(
            "Local-only and router modes run live. Router: claude-haiku-4-5 was answered "
            "here, including a file read, and the default model went to Anthropic on the "
            "Claude login. Local only: a normal reply came back from a served model."
        ),
    ),
)


# -- Anthropic SDKs (plan §4.2) ---------------------------------------------

_CLAUDE_EXAMPLE = "claude-haiku-4-5"

_ASDK_ROUTER_PY = (
    "import anthropic\n"
    "\n"
    "# api_key is your own Anthropic key (ANTHROPIC_API_KEY), used for pass-through.\n"
    "client = anthropic.Anthropic(\n"
    '    base_url="{{origin}}",\n'
    '    default_headers={"{{header}}": "{{key}}"},\n'
    ")\n"
    "message = client.messages.create(\n"
    f'    model="{_CLAUDE_EXAMPLE}",  # a model your rules serve here; others go to Anthropic\n'
    "    max_tokens=256,\n"
    f'    messages=[{{"role": "user", "content": "{_PROMPT}"}}],\n'
    ")\n"
    "print(message.content[0].text)"
)
_ASDK_ROUTER_TS = (
    'import Anthropic from "@anthropic-ai/sdk";\n'
    "\n"
    "// apiKey is your own Anthropic key (ANTHROPIC_API_KEY), used for pass-through.\n"
    "const client = new Anthropic({\n"
    '  baseURL: "{{origin}}",\n'
    '  defaultHeaders: { "{{header}}": "{{key}}" },\n'
    "});\n"
    "const message = await client.messages.create({\n"
    f'  model: "{_CLAUDE_EXAMPLE}", // a model your rules serve here; others go to Anthropic\n'
    "  max_tokens: 256,\n"
    f'  messages: [{{ role: "user", content: "{_PROMPT}" }}],\n'
    "});\n"
    "console.log(message.content);"
)
_ASDK_ROUTER_CURL = (
    "curl {{origin}}/v1/messages \\\n"
    '  -H "{{header}}: {{key}}" \\\n'
    '  -H "x-api-key: $ANTHROPIC_API_KEY" \\\n'
    '  -H "anthropic-version: 2023-06-01" \\\n'
    '  -H "content-type: application/json" \\\n'
    f'  -d \'{{"model": "{_CLAUDE_EXAMPLE}", "max_tokens": 256, '
    f'"messages": [{{"role": "user", "content": "{_PROMPT}"}}]}}\''
)
_ASDK_LOCAL_PY = (
    "import anthropic\n"
    "\n"
    'client = anthropic.Anthropic(base_url="{{origin}}", api_key="{{key}}")\n'
    "message = client.messages.create(\n"
    '    model="{{model}}",\n'
    "    max_tokens=256,\n"
    f'    messages=[{{"role": "user", "content": "{_PROMPT}"}}],\n'
    ")\n"
    "print(message.content[0].text)"
)
_ASDK_LOCAL_TS = (
    'import Anthropic from "@anthropic-ai/sdk";\n'
    "\n"
    'const client = new Anthropic({ baseURL: "{{origin}}", apiKey: "{{key}}" });\n'
    "const message = await client.messages.create({\n"
    '  model: "{{model}}",\n'
    "  max_tokens: 256,\n"
    f'  messages: [{{ role: "user", content: "{_PROMPT}" }}],\n'
    "});\n"
    "console.log(message.content);"
)
_ASDK_LOCAL_CURL = (
    "curl {{origin}}/v1/messages \\\n"
    '  -H "x-api-key: {{key}}" \\\n'
    '  -H "anthropic-version: 2023-06-01" \\\n'
    '  -H "content-type: application/json" \\\n'
    '  -d \'{"model": "{{model}}", "max_tokens": 256, '
    f'"messages": [{{"role": "user", "content": "{_PROMPT}"}}]}}\''
)

ANTHROPIC_SDK = ClientDef(
    id="anthropic-sdk",
    name="Anthropic SDK (Python, TypeScript, curl)",
    group="sdk",
    protocol="anthropic",
    support="full",
    summary="Your Anthropic API key stays in api_key; the warden key rides in its own header.",
    capability=(
        f"{_ROUTER_CAPABILITY} Local only needs no Anthropic account: the warden key is the "
        "api_key and the model is a served name."
    ),
    docs=Docs("https://platform.claude.com/docs/en/api/sdks/python"),
    requirements=_ROUTER_REQS,
    modes=(
        ModeDef(
            id="router",
            title="Router: keep your Anthropic key",
            description=(
                "base_url is this warden; your Anthropic API key stays in api_key and the "
                "warden key goes in default headers."
            ),
            needs_relay_key=True,
            files=(
                FileDef("python", "Python", None, "python", _ASDK_ROUTER_PY),
                FileDef("typescript", "TypeScript", None, "typescript", _ASDK_ROUTER_TS),
                FileDef("curl", "curl", None, "bash", _ASDK_ROUTER_CURL),
            ),
            verify=_ROUTER_VERIFY,
        ),
        ModeDef(
            id="local",
            title="Local only",
            description="The warden key is the api_key and the model is a served name.",
            needs_relay_key=False,
            files=(
                FileDef("python", "Python", None, "python", _ASDK_LOCAL_PY),
                FileDef("typescript", "TypeScript", None, "typescript", _ASDK_LOCAL_TS),
                FileDef("curl", "curl", None, "bash", _ASDK_LOCAL_CURL),
            ),
            verify=_LOCAL_ANTHROPIC_VERIFY,
        ),
    ),
    verified=Verified(
        status="run",
        date=VERIFIED_ON,
        client_version="anthropic 1.11.0 (Python), @anthropic-ai/sdk 0.131.0, curl",
        note=(
            "Python, TypeScript and curl in both modes; router mode's claude-haiku-4-5 was "
            "answered here."
        ),
    ),
)


# -- OpenCode (plan §4.4) ---------------------------------------------------

_OPENCODE_JSON = (
    "{\n"
    '  "$schema": "https://opencode.ai/config.json",\n'
    '  "provider": {\n'
    '    "lmwarden": {\n'
    '      "npm": "@ai-sdk/openai-compatible",\n'
    '      "name": "LM Warden",\n'
    '      "options": { "baseURL": "{{origin}}/v1", "apiKey": "{{key}}" },\n'
    '      "models": {\n'
    '        "{{model}}": { "name": "{{model}}", '
    '"limit": { "context": {{context}}, "output": 8192 } }\n'
    "      }\n"
    "    }\n"
    "  },\n"
    '  "model": "lmwarden/{{model}}"\n'
    "}"
)

OPENCODE = ClientDef(
    id="opencode",
    name="OpenCode",
    group="openai",
    protocol="openai",
    support="local_only",
    summary="LM Warden as an OpenAI-compatible provider for OpenCode, one served model.",
    capability=(
        "Local models only, over Chat Completions (@ai-sdk/openai-compatible; @ai-sdk/openai "
        "would use the Responses API, which this warden translates for Codex CLI only). The model id must be "
        "a served name from /v1/models. For a committed file, use {env:LMWARDEN_KEY} instead "
        "of the key. Router mode through OpenCode's Anthropic provider is not covered yet."
    ),
    docs=Docs("https://opencode.ai/docs/providers/"),
    requirements=(_TOOLS, _SOFT_32K),
    modes=(
        ModeDef(
            id="default",
            title="Local models",
            description="Add LM Warden as a provider in opencode.json and make it the default.",
            needs_relay_key=False,
            files=(
                FileDef("config", "opencode.json", "opencode.json", "json", _OPENCODE_JSON),
                FileDef("shell", "Shell", None, "bash", f'opencode run "{_PROMPT}"'),
            ),
            verify=_CHAT_VERIFY,
        ),
    ),
    verified=Verified(
        status="run",
        date=VERIFIED_ON,
        client_version="opencode 1.18.31",
        note="A one-word reply and a read-file tool call.",
    ),
)


# -- Aider (plan §4.5) ------------------------------------------------------

_AIDER_SHELL = (
    "export OPENAI_API_BASE={{origin}}/v1\n"
    "export OPENAI_API_KEY={{key}}\n"
    "aider --model openai/{{model}}"
)
_AIDER_META = (
    "{\n"
    '  "openai/{{model}}": {\n'
    '    "max_input_tokens": {{context}},\n'
    '    "max_output_tokens": 8192,\n'
    '    "max_tokens": 8192,\n'
    '    "input_cost_per_token": 0,\n'
    '    "output_cost_per_token": 0,\n'
    '    "litellm_provider": "openai",\n'
    '    "mode": "chat"\n'
    "  }\n"
    "}"
)

# One fresh id per launch becomes the session the warden shows and routes on.
# The header rides in model settings (aider reads extra_params.extra_headers).
_AIDER_SESSION = (
    "# Run this instead of `aider ...`: it writes a fresh session id per launch.\n"
    "SID=$(python3 -c 'import uuid; print(uuid.uuid4())')\n"
    "cat > .aider.model.settings.yml <<EOF\n"
    "- name: openai/{{model}}\n"
    "  extra_params:\n"
    "    extra_headers:\n"
    "      X-Session-Id: $SID\n"
    "EOF\n"
    "aider --model openai/{{model}}"
)

AIDER = ClientDef(
    id="aider",
    name="Aider",
    group="openai",
    protocol="openai",
    support="local_only",
    summary="Aider on a served model through its OpenAI-compatible provider.",
    capability=(
        "Local models only, over Chat Completions. Aider edits through text formats, so the "
        "model needs no tool calling. The metadata file only stops Aider warning about an "
        "unknown model and sets its context size."
    ),
    docs=Docs("https://aider.chat/docs/llms/openai-compat.html"),
    requirements=(_SOFT_32K,),
    modes=(
        ModeDef(
            id="default",
            title="Local models",
            description="Point Aider's OpenAI provider at this warden.",
            needs_relay_key=False,
            files=(
                FileDef("shell", "Shell", None, "bash", _AIDER_SHELL),
                FileDef(
                    "metadata",
                    ".aider.model.metadata.json",
                    ".aider.model.metadata.json",
                    "json",
                    _AIDER_META,
                ),
                FileDef(
                    "session", "Session id per launch (optional)", None, "bash", _AIDER_SESSION
                ),
            ),
            verify=_CHAT_VERIFY,
        ),
    ),
    verified=Verified(
        status="run",
        date=VERIFIED_ON,
        client_version="aider 0.86.2",
        note="A one-word reply and a question about a read-only file in the chat.",
    ),
)


# -- Grok CLI (plan §4.6) ---------------------------------------------------

_GROK_TOML = (
    "[model.lmwarden]\n"
    'model = "{{model}}"\n'
    'name = "LM Warden · {{model}}"\n'
    'base_url = "{{origin}}/v1"\n'
    'env_key = "LMWARDEN_KEY"\n'
    'api_backend = "chat_completions"\n'
    "context_window = {{context}}\n"
    "\n"
    "[models]\n"
    'default = "lmwarden"\n'
    'session_summary = "lmwarden"'
)
_GROK_COMMUNITY_SHELL = (
    "export GROK_BASE_URL={{origin}}/v1\n"
    "export GROK_API_KEY={{key}}\n"
    "export GROK_MODEL={{model}}\n"
    "npx -y grok-dev"
)

GROK_CLI = ClientDef(
    id="grok-cli",
    name="Grok CLI",
    group="openai",
    protocol="openai",
    support="local_only",
    summary="xAI's Grok Build CLI with LM Warden as a bring-your-own-key model.",
    capability=(
        "Local models only, over Chat Completions (api_backend = chat_completions). There is "
        "no pass-through to xAI, and a BYOK model needs no xAI login. session_summary keeps "
        "Grok's session-title requests on this model; without it Grok asks this warden for "
        "grok-4.6 and gets a 404. Grok adds its own sections to config.toml when it runs. "
        "The community grok-cli (npm grok-dev) is a different tool: chat works, tool calls "
        "do not."
    ),
    docs=Docs("https://docs.x.ai/build/settings/reference"),
    requirements=(_TOOLS, _SOFT_32K),
    modes=(
        ModeDef(
            id="default",
            title="Grok Build (official)",
            description=(
                "A model entry in ~/.grok/config.toml; the key comes from LMWARDEN_KEY. "
                "Install: curl -fsSL https://x.ai/cli/install.sh | bash, or "
                "npm i -g @xai-official/grok"
            ),
            needs_relay_key=False,
            files=(
                FileDef("config", "config.toml", "~/.grok/config.toml", "toml", _GROK_TOML),
                FileDef(
                    "shell", "Shell", None, "bash", "export LMWARDEN_KEY={{key}}\ngrok -m lmwarden"
                ),
            ),
            verify=_CHAT_VERIFY,
        ),
        ModeDef(
            id="community",
            title="Community grok-cli",
            description=(
                "The community grok-cli reads its base URL, key and model from the env and "
                "needs Bun installed. Chat works; tool calls fail, because its xAI SDK rejects "
                "tool-call chunks that arrive in parts, as vLLM streams them."
            ),
            needs_relay_key=False,
            files=(FileDef("shell", "Shell", None, "bash", _GROK_COMMUNITY_SHELL),),
            verify=_CHAT_VERIFY,
        ),
    ),
    verified=Verified(
        status="run",
        date=VERIFIED_ON,
        client_version="grok 1.0.46 (Grok Build), grok-dev 1.1.7",
        note=(
            "Grok Build: a one-word reply and a read_file tool call, with no xAI login. "
            "grok-dev: chat works; its tool calls fail with AI_TypeValidationError."
        ),
    ),
)


# -- OpenAI SDKs (plan §4.7) ------------------------------------------------

_OSDK_PY = (
    "from openai import OpenAI\n"
    "\n"
    'client = OpenAI(base_url="{{origin}}/v1", api_key="{{key}}")\n'
    "response = client.chat.completions.create(\n"
    '    model="{{model}}",\n'
    f'    messages=[{{"role": "user", "content": "{_PROMPT}"}}],\n'
    ")\n"
    "print(response.choices[0].message.content)"
)
_OSDK_TS = (
    'import OpenAI from "openai";\n'
    "\n"
    'const client = new OpenAI({ baseURL: "{{origin}}/v1", apiKey: "{{key}}" });\n'
    "const response = await client.chat.completions.create({\n"
    '  model: "{{model}}",\n'
    f'  messages: [{{ role: "user", content: "{_PROMPT}" }}],\n'
    "});\n"
    "console.log(response.choices[0].message.content);"
)
_OSDK_CURL = (
    "curl {{origin}}/v1/chat/completions \\\n"
    '  -H "Authorization: Bearer {{key}}" \\\n'
    '  -H "content-type: application/json" \\\n'
    '  -d \'{"model": "{{model}}", "stream": false, '
    f'"messages": [{{"role": "user", "content": "{_PROMPT}"}}]}}\''
)

OPENAI_SDK = ClientDef(
    id="openai-sdk",
    name="OpenAI SDK (Python, TypeScript, curl)",
    group="sdk",
    protocol="openai",
    support="local_only",
    summary="Any OpenAI SDK, or anything with an OpenAI-compatible base URL setting.",
    capability=(
        "Local models only. There is no pass-through to OpenAI: a request for gpt-* gets 404. "
        "Chat Completions and Completions are served; /v1/responses is a stateless translation "
        "to Chat Completions (see Codex CLI)."
    ),
    docs=Docs("https://github.com/openai/openai-python#readme"),
    modes=(
        ModeDef(
            id="default",
            title="Local models",
            description="base_url is this warden's /v1 and the warden key is the API key.",
            needs_relay_key=False,
            files=(
                FileDef("python", "Python", None, "python", _OSDK_PY),
                FileDef("typescript", "TypeScript", None, "typescript", _OSDK_TS),
                FileDef("curl", "curl", None, "bash", _OSDK_CURL),
            ),
            verify=_CHAT_VERIFY,
        ),
    ),
    verified=Verified(
        status="run",
        date=VERIFIED_ON,
        client_version="openai 3.24.0 (Python), openai 7.28.0 (TypeScript), curl",
        note="Python, TypeScript and curl, one chat completion each.",
    ),
)


# -- Continue (plan §4.8) ---------------------------------------------------

#: The three GUI editors: config doc-checked, the raw request run (plan §11.8).
_DOCUMENTED = Verified(
    status="documented",
    date=VERIFIED_ON,
    client_version=None,
    note=(
        "Config checked against the docs; the streaming Chat Completions request it sends, "
        "with a tool call, passed with curl."
    ),
)


def _continue_yaml(tools: bool) -> str:
    caps = "    capabilities: [tool_use]\n" if tools else ""
    return (
        "name: LM Warden\n"
        "version: 0.0.1\n"
        "schema: v1\n"
        "models:\n"
        "  - name: {{model}}\n"
        "    provider: openai\n"
        "    model: {{model}}\n"
        "    apiBase: {{origin}}/v1\n"
        "    apiKey: {{key}}\n"
        "    roles: [chat, edit, apply]\n"
        f"{caps}"
        "    useResponsesApi: false   # names like o*/gpt-5* would otherwise use /responses\n"
        "    defaultCompletionOptions:\n"
        "      contextLength: {{context}}\n"
        "      maxTokens: 8192"
    )


CONTINUE = ClientDef(
    id="continue",
    name="Continue",
    group="editors",
    protocol="openai",
    support="documented",
    summary="Continue's OpenAI provider pointed at this warden, one served model.",
    capability=(
        "Local models only, over Chat Completions. The config follows Continue's docs and the "
        "request it sends was tested from this page; the editor itself was not run. Router "
        "mode would need custom headers Continue does not document."
    ),
    docs=Docs("https://docs.continue.dev/reference"),
    requirements=(_SOFT_32K,),
    modes=(
        ModeDef(
            id="default",
            title="Local models",
            description="A model entry in ~/.continue/config.yaml.",
            needs_relay_key=False,
            files=(
                FileDef(
                    "config",
                    "config.yaml",
                    "~/.continue/config.yaml",
                    "yaml",
                    _continue_yaml(True),
                    requires_tools=True,
                ),
                FileDef(
                    "config-no-tools",
                    "config.yaml",
                    "~/.continue/config.yaml",
                    "yaml",
                    _continue_yaml(False),
                    requires_tools=False,
                ),
            ),
            verify=_CHAT_VERIFY_RAW,
        ),
    ),
    verified=_DOCUMENTED,
)


# -- Cline (plan §4.9) ------------------------------------------------------

CLINE = ClientDef(
    id="cline",
    name="Cline",
    group="editors",
    protocol="openai",
    support="documented",
    summary="Cline's OpenAI Compatible provider, filled in from its settings screen.",
    capability=(
        "Local models only, over streaming Chat Completions. Cline has no config file: enter "
        "these values under API Provider > OpenAI Compatible. Its Anthropic provider has a "
        "custom base URL but no custom header, so router mode is not available."
    ),
    docs=Docs("https://docs.cline.bot/provider-config/openai-compatible"),
    requirements=(_TOOLS, _SOFT_32K),
    modes=(
        ModeDef(
            id="default",
            title="Local models",
            description="Settings > API Provider > OpenAI Compatible.",
            needs_relay_key=False,
            files=(
                FileDef(
                    "fields",
                    "Settings",
                    None,
                    "fields",
                    "Base URL: {{origin}}/v1\n"
                    "API Key: {{key}}\n"
                    "Model ID: {{model}}\n"
                    "Context Window Size: {{context}}",
                ),
            ),
            verify=_CHAT_VERIFY_RAW,
        ),
    ),
    verified=_DOCUMENTED,
)


# -- Cursor (plan §4.10) ----------------------------------------------------

CURSOR = ClientDef(
    id="cursor",
    name="Cursor",
    group="editors",
    protocol="openai",
    support="documented",
    summary="Cursor's OpenAI base URL override, for chat with a served model.",
    capability=(
        "Requests are issued from Cursor's servers, so this warden's URL must be reachable "
        "from the internet; a LAN or localhost warden will not work. Custom keys only work "
        "with chat models: Tab completion keeps using Cursor's own models, and Agent with "
        "Cursor's models rejects custom keys. Cursor's current docs no longer describe the "
        "Override OpenAI Base URL setting; it is known from Cursor staff forum replies."
    ),
    docs=Docs("https://cursor.com/docs/settings/api-keys"),
    requirements=(
        Requirement(
            id="reachable",
            text="Cursor's servers call this URL, so it must be reachable from the internet.",
        ),
    ),
    modes=(
        ModeDef(
            id="default",
            title="Chat with a local model",
            description="Cursor Settings > Models, then add the model name as a custom model.",
            needs_relay_key=False,
            files=(
                FileDef(
                    "fields",
                    "Settings",
                    None,
                    "fields",
                    "OpenAI API Key: {{key}}\n"
                    "Override OpenAI Base URL: {{origin}}/v1\n"
                    "Custom model: {{model}}",
                ),
            ),
            verify=_CHAT_VERIFY_RAW,
        ),
    ),
    verified=_DOCUMENTED,
)


# -- Codex CLI (plan §4.3; Responses API plan 2026-10-05) ------------------

_CODEX_TOML = (
    'model = "{{model}}"\n'
    'model_provider = "lmwarden"\n'
    "# Codex assumes a 272k window for models it does not know; set the real one.\n"
    "model_context_window = {{context}}\n"
    "\n"
    "[model_providers.lmwarden]\n"
    'name = "LM Warden"\n'
    'base_url = "{{origin}}/v1"\n'
    'env_key = "LMWARDEN_KEY"\n'
    'wire_api = "responses"'
)

CODEX_CLI = ClientDef(
    id="codex-cli",
    name="Codex CLI",
    group="openai",
    protocol="openai",
    support="local_only",
    summary="LM Warden as a custom model provider for Codex CLI, one served model.",
    capability=(
        "Local models only. Codex speaks only the Responses API; the warden translates "
        "POST /v1/responses to Chat Completions for the served model, so the reply, tool calls "
        "and usage reach Codex as Responses events. It is stateless: Codex sends store:false "
        "and the whole conversation each turn, and previous_response_id is refused. The "
        "thinking is on by default (Codex asks for reasoning summaries and the model's reasoning "
        'is shown as a summary; to turn it off, set model_reasoning_effort = "none" in '
        "config.toml), hosted web search is dropped "
        '(set web_search = "disabled"), and router rules do not apply. Set '
        "model_context_window to the model's real window: Codex assumes 272k for names it "
        "does not know. The engine needs tool calling enabled (vLLM: --enable-auto-tool-choice "
        "and a --tool-call-parser). A request for a model that is not loaded gets 404 and "
        "Codex retries for several seconds before it gives up."
    ),
    docs=Docs("https://learn.chatgpt.com/docs/config-file/config-reference"),
    requirements=(_TOOLS, _SOFT_32K),
    modes=(
        ModeDef(
            id="default",
            title="Local models",
            description="Add LM Warden as a model provider in ~/.codex/config.toml.",
            needs_relay_key=False,
            files=(
                FileDef("config", "config.toml", "~/.codex/config.toml", "toml", _CODEX_TOML),
                FileDef(
                    "shell",
                    "Shell",
                    None,
                    "bash",
                    f'export LMWARDEN_KEY={{{{key}}}}\ncodex exec "{_PROMPT}"',
                ),
            ),
            verify=_responses_verify(
                covers=(
                    "The picked model answering a Responses API request, with the key as a "
                    "Bearer token."
                ),
                not_covered="Codex's own tool calls and streaming.",
            ),
        ),
    ),
    verified=Verified(
        status="run",
        date="2026-10-05",
        client_version="codex-cli 0.160.0",
        note=(
            "Two turns with a shell tool call, a workspace-write file edit, the session pinned "
            "by session-id and cached tokens measured on turn 2. Thinking on by default and "
            'off with model_reasoning_effort = "none", both run live.'
        ),
    ),
)


#: Display order: the page's tool list and the API's ``clients`` array.
CLIENTS: tuple[ClientDef, ...] = (
    CLAUDE_CODE,
    ANTHROPIC_SDK,
    OPENCODE,
    AIDER,
    GROK_CLI,
    OPENAI_SDK,
    CONTINUE,
    CLINE,
    CURSOR,
    CODEX_CLI,
)
