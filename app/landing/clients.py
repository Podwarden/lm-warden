"""Coding-agent pages generated from the Connect catalogue (spec 2026-10-07 §3).

The product's Connect page and the website read the same ClientDef, so a new
client, a changed snippet or a new verification date shows on both. Snippets
are rendered with placeholders, never a real key or origin."""

from __future__ import annotations

import html
import re

from app.connect.catalog import (
    CLIENTS,
    DEFAULT_CONTEXT,
    KEY_PLACEHOLDER,
    MODEL_PLACEHOLDER,
    ClientDef,
    FileDef,
    render_file,
)
from app.landing.diagrams import AGENTS, figure_html
from app.landing.pages import Page
from app.router.rules import HEADER_NAME

_VARS = {
    "origin": "https://your-warden",
    "key": KEY_PLACEHOLDER,
    "model": MODEL_PLACEHOLDER,
    "header": HEADER_NAME,
    "context": str(DEFAULT_CONTEXT),
}
_PROTOCOL = {"anthropic": "Anthropic Messages", "openai": "OpenAI Chat Completions"}
_SUPPORT = {
    "full": "Full support",
    "local_only": "Local models only",
    "documented": "Configured from its docs",
    "unsupported": "Not supported yet",
}
_VERIFIED = {
    "run": "Run against a live warden",
    "documented": "Checked against its docs",
    "failed": "Failed when run",
}


_APOS = re.compile(r"(?<=\w)'(?=\w)")
_DQUOTES = re.compile(r'"([^"]*)"')


def _e(text: str) -> str:
    """Catalogue prose for the website: escaped, with the site's curly
    apostrophes and quotes (the catalogue itself keeps straight ones, which
    the console renders as they are). Never used for snippets."""
    text = _DQUOTES.sub("\u201c\\1\u201d", _APOS.sub("\u2019", text))
    return html.escape(text, quote=False)


def _code(text: str) -> str:
    return html.escape(text, quote=False)


def short_name(c: ClientDef) -> str:
    return c.name.split(" (")[0]


def protocol(c: ClientDef) -> str:
    # Codex CLI speaks the Responses API; the catalogue files it under openai.
    if c.id == "codex-cli":
        return "OpenAI Responses"
    return _PROTOCOL[c.protocol]


def _description(c: ClientDef) -> str:
    text = f"{short_name(c)} with LM Warden: {c.summary}"
    if len(text) < 156:
        return text
    return text[:152].rsplit(" ", 1)[0] + "…"


def _page(c: ClientDef) -> Page:
    return Page(
        path=f"/coding-agents/{c.id}",
        fragment="",
        title=f"{short_name(c)} with a local LLM on your GPUs | LM Warden",
        description=_description(c),
        nav="agents",
        crumbs=(("Coding agents", "/coding-agents"), (short_name(c), f"/coding-agents/{c.id}")),
        jsonld=("article",),
        client_id=c.id,
    )


CLIENT_PAGES: dict[str, Page] = {c.id: _page(c) for c in CLIENTS}
_BY_ID: dict[str, ClientDef] = {c.id: c for c in CLIENTS}


def client_def(client_id: str) -> ClientDef:
    return _BY_ID[client_id]


def _verified_html(c: ClientDef) -> str:
    v = c.verified
    if v is None:
        return "Not yet run against a warden."
    parts = [f"{_VERIFIED[v.status]} on {_e(v.date)}"]
    if v.client_version:
        parts.append(f"with {_e(v.client_version)}")
    note = f" {_e(v.note)}" if v.note else ""
    return f"{' '.join(parts)}.{note}"


def _snippet(mode_id: str, f: FileDef) -> str:
    sid = f"snip-{mode_id}-{f.id}"
    where = f" (<code>{_code(f.path)}</code>)" if f.path else ""
    return (
        f'<figure class="snippet"><figcaption>{_e(f.label)}{where}</figcaption>'
        f'<div class="cmd"><pre id="{sid}" tabindex="0">'
        f"{_code(render_file(f, _VARS, context_known=False))}</pre>"
        f'<button class="btn" type="button" data-copy="{sid}"><span>Copy</span></button></div>'
        "</figure>"
    )


def client_body(c: ClientDef, router_section: str) -> str:
    """The page body for one client. Claude Code's page also carries the
    router section that used to sit under the single page's hero."""
    reqs = "".join(f"<li>{_e(r.text)}</li>" for r in c.requirements)
    modes = "".join(
        f"<h3>{_e(m.title)}</h3><p>{_e(m.description)}</p>"
        + "".join(_snippet(m.id, f) for f in m.files)
        for m in c.modes
    )
    requirements = f"<h2>Requirements</h2><ul>{reqs}</ul>" if reqs else ""
    head = f"""        <h1 id="client-title" class="as-h2">{_e(short_name(c))} on your own GPUs.</h1>
        <p class="lede">{_e(c.summary)}</p>
"""
    if c.id == "claude-code":
        # The router drawing is the page's opening argument, so its section
        # sits between the header and the facts.
        top = f"""    <section class="sec client" aria-labelledby="client-title">
      <div class="wrap">
{head}      </div>
    </section>
{router_section}    <section class="sec client" aria-label="Setup">
      <div class="wrap">
"""
    else:
        top = f"""    <section class="sec client" aria-labelledby="client-title">
      <div class="wrap">
{head}{figure_html(AGENTS[c.id], f"flow-{c.id}", _caption(c))}"""
    return (
        top
        + f"""        <dl class="facts">
          <dt>Protocol</dt><dd>{_e(protocol(c))}</dd>
          <dt>Support</dt><dd>{_e(_SUPPORT[c.support])}</dd>
          <dt>Verified</dt><dd>{_verified_html(c)}</dd>
        </dl>
        <p>{_e(c.capability)}</p>
        {requirements}
        <h2>Setup</h2>
        <p>In the console, <strong>Connect</strong> writes these files with your warden’s address, your key and the model you picked. Here they are with placeholders.</p>
        {modes}
        <p>Official documentation: <a href="{html.escape(c.docs.url)}" rel="noopener">{_code(c.docs.url)}</a> (read {_e(c.docs.accessed)}).</p>
        <p class="related">Related: <a href="/coding-agents">all coding agents</a> · <a href="/agent-tuning">let your agent pick the model</a> · <a href="/api-keys">API keys</a></p>
      </div>
    </section>
"""
    )


def _caption(c: ClientDef) -> str:
    return (
        f"Where a request from {_e(short_name(c))} goes: the two settings Connect writes, "
        "the warden in the middle, then one copy of the model per GPU."
    )


def clients_table_html() -> str:
    def run_date(c: ClientDef) -> str:
        return _e(c.verified.date) if c.verified and c.verified.status == "run" else "docs only"

    rows = "".join(
        f'<tr><th scope="row"><a href="/coding-agents/{c.id}">{_e(short_name(c))}</a></th>'
        f"<td>{_e(protocol(c))}</td><td>{_e(_SUPPORT[c.support])}</td><td>{run_date(c)}</td></tr>"
        for c in CLIENTS
    )
    return (
        '<div class="table-wrap"><table class="clients"><thead><tr><th scope="col">Client</th>'
        '<th scope="col">Protocol</th><th scope="col">Support</th>'
        '<th scope="col">Run live</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )
