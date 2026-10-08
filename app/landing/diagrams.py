"""Request-flow diagrams for the coding-agent pages (owner request 2026-10-07).

One generic drawing (top of /coding-agents and Home) and one per agent (top
of /coding-agents/<id>), all drawn by the same two functions in the visual
language of the hand-drawn router figure (figures/d1-flow.svg): the `flow`
SVG classes in layout.html, a wide cut and a narrow phone cut swapped by
`.swap-1200`. Claude Code keeps the full router drawing instead.

The setting names come from the Connect catalogue's config templates
(app/connect/catalog.py); tests/unit/landing/test_diagrams.py checks every
name appears in its drawing and that the Anthropic branch is drawn only for
the Anthropic-protocol clients, where it exists.
"""

from __future__ import annotations

import html
from dataclasses import dataclass


@dataclass(frozen=True)
class Agent:
    name: str
    #: (setting, what it holds): two lines in the agent's box.
    settings: tuple[tuple[str, str], ...]
    #: Label on the arrow into the warden.
    endpoint: str
    #: What the warden's middle row says it speaks.
    protocol: str
    #: Draw the pass-through branch to api.anthropic.com.
    relay: bool = False
    relay_note: str = "rules send other Claude models here, on your own login"


_OPENAI = "OpenAI Chat Completions"

GENERIC = Agent(
    name="Your coding agent",
    settings=(("base URL", "the warden, at /v1"), ("API key", "vw_…, one key per app")),
    endpoint="POST /v1/…",
    protocol="the OpenAI and Anthropic APIs",
    relay=True,
    relay_note="Claude Code only: other Claude models go here, on your own login",
)

AGENTS: dict[str, Agent] = {
    "anthropic-sdk": Agent(
        "Anthropic SDK",
        (("base_url", "points at the warden"), ("X-LMWarden-Key", "vw_…, in default headers")),
        "POST /v1/messages",
        "Anthropic Messages",
        relay=True,
        relay_note="rules send other Claude models here, on your own API key",
    ),
    "opencode": Agent(
        "OpenCode",
        (("baseURL", "the warden, at /v1"), ("apiKey", "vw_…, the warden key")),
        "POST /v1/chat/completions",
        _OPENAI,
    ),
    "aider": Agent(
        "Aider",
        (("OPENAI_API_BASE", "the warden, at /v1"), ("OPENAI_API_KEY", "vw_…, the warden key")),
        "POST /v1/chat/completions",
        _OPENAI,
    ),
    "grok-cli": Agent(
        "Grok CLI",
        (("base_url", "the warden, at /v1"), ("env_key", "LMWARDEN_KEY holds vw_…")),
        "POST /v1/chat/completions",
        _OPENAI,
    ),
    "openai-sdk": Agent(
        "OpenAI SDK",
        (("base_url", "the warden, at /v1"), ("api_key", "vw_…, the warden key")),
        "POST /v1/chat/completions",
        _OPENAI,
    ),
    "continue": Agent(
        "Continue",
        (("apiBase", "the warden, at /v1"), ("apiKey", "vw_…, the warden key")),
        "POST /v1/chat/completions",
        _OPENAI,
    ),
    "cline": Agent(
        "Cline",
        (("Base URL", "the warden, at /v1"), ("API Key", "vw_…, the warden key")),
        "POST /v1/chat/completions",
        _OPENAI,
    ),
    "cursor": Agent(
        "Cursor",
        (("Override OpenAI Base URL", "the warden, at /v1"), ("OpenAI API Key", "vw_…")),
        "POST /v1/chat/completions",
        _OPENAI,
    ),
    "codex-cli": Agent(
        "Codex CLI",
        (("base_url", "the warden, at /v1"), ("env_key", "LMWARDEN_KEY holds vw_…")),
        "POST /v1/responses",
        "OpenAI Responses",
    ),
}

_e = html.escape


def _marker(uid: str) -> str:
    return (
        f'<defs><marker id="{uid}-head" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="9" markerHeight="9" orient="auto">'
        '<path d="M0 0 L10 5 L0 10 z" class="head"/></marker></defs>'
    )


def _desc(a: Agent) -> str:
    s = (
        f"{a.name} sends its requests to LM Warden with {a.settings[0][0]} set to the warden "
        f"and {a.settings[1][0]} holding a warden key. LM Warden checks the key, speaks "
        f"{a.protocol}, queues the request by key priority and hands it to the replica "
        "router, which keeps each session on the replica that already holds its cache, "
        "one model copy per GPU."
    )
    if a.relay:
        s += f" A separate path goes to api.anthropic.com: {a.relay_note}."
    return s


def flow_wide(a: Agent, uid: str) -> str:
    h = 400 if a.relay else 300
    (m1, n1), (m2, n2) = a.settings
    arrow = f'marker-end="url(#{uid}-head)"'
    verb, _, path = a.endpoint.partition(" ")
    out = [
        f'<svg class="flow" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1240 {h}" '
        f'width="1240" height="{h}" role="img" aria-labelledby="{uid}-title" aria-describedby="{uid}-desc">',
        f'<title id="{uid}-title">Where a request from {_e(a.name)} goes</title>',
        f'<desc id="{uid}-desc">{_e(_desc(a))}</desc>',
        _marker(uid),
        # the agent
        '<rect class="box" x="0.75" y="40.75" width="248.5" height="180" rx="3"/>',
        f'<text class="h" x="16" y="68">{_e(a.name)}</text>',
        f'<text class="m" x="16" y="104">{_e(m1)}</text>',
        f'<text class="s" x="16" y="124">{_e(n1)}</text>',
        f'<text class="m" x="16" y="164">{_e(m2)}</text>',
        f'<text class="s" x="16" y="184">{_e(n2)}</text>',
        f'<path class="edge" d="M250 130 H428" {arrow}/>',
        f'<text class="s" x="339" y="118" text-anchor="middle">{_e(verb)}</text>',
        f'<text class="s" x="339" y="152" text-anchor="middle">{_e(path)}</text>',
        # the warden
        '<rect class="box" x="430.75" y="40.75" width="328.5" height="200" rx="3"/>',
        '<text class="h" x="446" y="68">LM Warden</text>',
        '<line class="rule" x1="446" x2="744" y1="84" y2="84"/>',
        '<text class="t" x="446" y="108">checks the key: one per app</text>',
        '<line class="rule" x1="446" x2="744" y1="122" y2="122"/>',
        f'<text class="t" x="446" y="146">speaks {_e(a.protocol)}</text>',
        '<line class="rule" x1="446" x2="744" y1="160" y2="160"/>',
        '<text class="t" x="446" y="184">scheduler, key priority</text>',
        '<line class="rule" x1="446" x2="744" y1="198" y2="198"/>',
        '<text class="s" x="446" y="224">usage, latency and live stats per key</text>',
        f'<path class="edge" d="M760 140 H828" {arrow}/>',
        # the replica router
        '<rect class="box" x="830.75" y="70.75" width="208.5" height="140" rx="3"/>',
        '<text class="h" x="846" y="98">Replica router</text>',
        '<text class="s" x="846" y="128">a session goes back to</text>',
        '<text class="s" x="846" y="148">the replica that holds</text>',
        '<text class="s" x="846" y="168">its cache</text>',
        '<text class="s" x="846" y="196">one model copy per GPU</text>',
        # the GPUs
        '<text class="s" x="1239" y="28" text-anchor="end">your GPUs: vLLM or llama.cpp</text>',
        '<path class="edge" d="M1040 140 H1055 M1055 72 V208"/>',
    ]
    for i, y in enumerate((72, 140, 208)):
        out += [
            f'<path class="edge" d="M1055 {y} H1068" {arrow}/>',
            f'<rect class="box" x="1070.75" y="{y - 21.25}" width="168.5" height="42" rx="3"/>',
            f'<text class="t" x="1086" y="{y + 5}">replica {i}, GPU {i}</text>',
        ]
    out.append(
        '<text class="s" x="1239" y="254" text-anchor="end">'
        "a conversation stays on its replica</text>"
    )
    if a.relay:
        out += [
            f'<path class="edge" d="M595 242 V318" {arrow}/>',
            '<rect class="box" x="430.75" y="320.75" width="558.5" height="62" rx="3"/>',
            '<text class="h" x="446" y="346">api.anthropic.com</text>',
            f'<text class="s" x="446" y="369">{_e(a.relay_note)}</text>',
        ]
    out.append("</svg>")
    return "\n".join(out)


def flow_narrow(a: Agent, uid: str) -> str:
    h = 760 if a.relay else 640
    (m1, n1), (m2, n2) = a.settings
    arrow = f'marker-end="url(#{uid}-head)"'
    out = [
        f'<svg class="flow" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 340 {h}" '
        f'width="340" height="{h}" role="img" aria-labelledby="{uid}-title" aria-describedby="{uid}-desc">',
        f'<title id="{uid}-title">Where a request from {_e(a.name)} goes</title>',
        f'<desc id="{uid}-desc">{_e(_desc(a))} The same flow as the wide drawing, '
        "top to bottom.</desc>",
        _marker(uid),
        '<rect class="box" x="0.75" y="0.75" width="298.5" height="128" rx="3"/>',
        f'<text class="h" x="14" y="28">{_e(a.name)}</text>',
        f'<text class="m" x="14" y="56">{_e(m1)}</text>',
        f'<text class="s" x="14" y="74">{_e(n1)}</text>',
        f'<text class="m" x="14" y="100">{_e(m2)}</text>',
        f'<text class="s" x="14" y="118">{_e(n2)}</text>',
        f'<path class="edge" d="M150 130 V168" {arrow}/>',
        f'<text class="s" x="162" y="154">{_e(a.endpoint)}</text>',
        '<rect class="box" x="0.75" y="170.75" width="298.5" height="168" rx="3"/>',
        '<text class="h" x="14" y="198">LM Warden</text>',
        '<text class="t" x="14" y="228">checks the key: one per app</text>',
        f'<text class="t" x="14" y="256">speaks {_e(a.protocol)}</text>',
        '<text class="t" x="14" y="284">scheduler, key priority</text>',
        '<text class="s" x="14" y="312">usage, latency, live stats per key</text>',
        f'<path class="edge" d="M150 340 V378" {arrow}/>',
        '<rect class="box" x="0.75" y="380.75" width="298.5" height="96" rx="3"/>',
        '<text class="h" x="14" y="408">Replica router</text>',
        '<text class="s" x="14" y="434">a session goes back to the replica</text>',
        '<text class="s" x="14" y="454">that holds its cache</text>',
        '<path class="edge" d="M150 478 V496 M45 496 H255"/>',
    ]
    for i, x in enumerate((0, 105, 210)):
        out += [
            f'<path class="edge" d="M{x + 45} 496 V522" {arrow}/>',
            f'<rect class="box" x="{x + 0.75}" y="524.75" width="88.5" height="46" rx="3"/>',
            f'<text class="t" x="{x + 10}" y="547">replica {i}</text>',
            f'<text class="s" x="{x + 10}" y="563">GPU {i}</text>',
        ]
    out.append('<text class="s" x="0" y="598">your GPUs: vLLM or llama.cpp</text>')
    out.append('<text class="s" x="0" y="620">a conversation stays on its replica</text>')
    if a.relay:
        out += [
            f'<path class="edge" d="M300 254 H326 V694 H302" {arrow}/>',
            '<rect class="box" x="0.75" y="650.75" width="298.5" height="96" rx="3"/>',
            '<text class="h" x="14" y="678">api.anthropic.com</text>',
        ]
        # The note wraps onto two lines in the narrow cut.
        words = a.relay_note.split(" ")
        half = len(words) // 2
        out += [
            f'<text class="s" x="14" y="704">{_e(" ".join(words[:half]))}</text>',
            f'<text class="s" x="14" y="724">{_e(" ".join(words[half:]))}</text>',
        ]
    out.append("</svg>")
    return "\n".join(out)


def figure_html(a: Agent, uid: str, caption: str) -> str:
    """A figure in the page's diagram frame: wide cut, narrow phone cut, a
    caption and the same flow in words for screen readers."""
    return (
        '<figure class="diagram-fig flow-top">\n'
        '  <div class="diagram swap-1200">\n'
        f'    <div class="v-wide">\n{flow_wide(a, uid)}\n    </div>\n'
        f'    <div class="v-narrow">\n{flow_narrow(a, uid + "n")}\n    </div>\n'
        "  </div>\n"
        f"  <figcaption>{caption}</figcaption>\n"
        "</figure>\n"
        f'<p class="vh">In words: {_e(_desc(a))}</p>\n'
    )
