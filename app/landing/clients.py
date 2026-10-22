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
#: What a client can do through the warden. The website does not set the
#: clients we ran apart from the ones configured from their docs (owner,
#: 2026-10-21), so the catalogue's "documented" support reads as what those
#: clients get: every one of them is an OpenAI-protocol client on local
#: models, with no pass-through. The verification record stays in the
#: catalogue, for the console's Connect page, and is not printed here.
_SUPPORT = {
    "full": "Full support",
    "local_only": "Local models only",
    "documented": "Local models only",
    "unsupported": "Not supported yet",
}
#: Catalogue prose that states a verification status ("the editor itself was
#: not run"); dropped from the website's copy for the same reason.
_STATUS_SENTENCE = re.compile(
    r"\s*[^.]*\b(?:was|were) (?:not run|tested from this page)[^.]*\.(?=\s|$)"
)


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


def capability(c: ClientDef) -> str:
    """The catalogue's capability text without any verification-status
    sentence (see _STATUS_SENTENCE)."""
    return _STATUS_SENTENCE.sub("", c.capability).strip()


#: Literal code in the catalogue's prose that the patterns below would not
#: find whole (it has spaces or quotes), longest first.
_CODE_LITERALS = (
    "curl -fsSL https://x.ai/cli/install.sh | bash",
    "npm i -g @xai-official/grok",
    'model_reasoning_effort = "none"',
    'web_search = "disabled"',
    "api_backend = chat_completions",
    "POST /v1/responses",
    "{env:LMWARDEN_KEY}",
    "store:false",
    "gpt-*",
    "grok-4.6",
    "grok-dev",
)
#: Code-like words in the catalogue's prose: env vars, snake_case names,
#: paths, npm packages, flags and file names.
_CODE_RE = re.compile(
    "|".join(
        [re.escape(lit) for lit in _CODE_LITERALS]
        + [
            r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b",
            r"\b[a-z]+(?:_[a-z]+)+\b",
            r"(?<![\w/])~?/[\w./*-]*[\w*]",
            r"@[\w-]+/[\w-]+",
            r"(?<![\w-])--[a-z][\w-]*",
            r"(?<![\w/])\.?[\w-]+(?:\.[\w-]+)*\.(?:json|toml|yaml)\b",
        ]
    )
)


def prose_html(text: str) -> str:
    """Catalogue prose as the website prints it: _e, with every code-like
    word (env vars, settings keys, paths, commands, flags, file names) in a
    <code> span, so the page shows it as code and a translation must keep
    it verbatim (the i18n check compares <code> spans)."""
    out, at = [], 0
    for m in _CODE_RE.finditer(text):
        out.append(_e(text[at : m.start()]))
        out.append(f"<code>{_code(m.group(0))}</code>")
        at = m.end()
    out.append(_e(text[at:]))
    return "".join(out)


#: File labels shared by many clients: one segment each.
_COMMON_LABELS = {"Shell": "common.file.shell", "Settings": "common.file.settings"}
_FILE_NAME_RE = re.compile(r"^\.?[\w.-]+\.\w+$")


def _label(c: ClientDef, mode_id: str, f: FileDef) -> str:
    """A snippet's label: a file name as code, a language or tool name as it
    is, anything else a segment."""
    if _FILE_NAME_RE.match(f.label):
        return f"<code>{_code(f.label)}</code>"
    if f.label in {"Python", "TypeScript", "curl"}:
        return _e(f.label)
    key = _COMMON_LABELS.get(f.label, f"clients.{c.id}.mode.{mode_id}.file.{f.id}")
    return f'<span data-t="{key}">{_e(f.label)}</span>'


def _snippet(c: ClientDef, mode_id: str, f: FileDef) -> str:
    sid = f"snip-{mode_id}-{f.id}"
    where = f" (<code>{_code(f.path)}</code>)" if f.path else ""
    return (
        f'<figure class="snippet"><figcaption>{_label(c, mode_id, f)}{where}</figcaption>'
        f'<div class="cmd"><pre id="{sid}" tabindex="0">'
        f"{_code(render_file(f, _VARS, context_known=False))}</pre>"
        f'<button class="btn" type="button" data-copy="{sid}"><span data-t="common.copy">Copy</span>'
        "</button></div>"
        "</figure>"
    )


#: The support levels' labels, one segment each (common.support.<key>).
_SUPPORT_KEY = {
    "Full support": "full",
    "Local models only": "local-only",
    "Not supported yet": "unsupported",
}


def _support(c: ClientDef) -> str:
    label = _SUPPORT[c.support]
    return f'<span data-t="common.support.{_SUPPORT_KEY[label]}">{_e(label)}</span>'


def client_body(c: ClientDef, router_section: str) -> str:
    """The page body for one client. Claude Code's page also carries the
    router section that used to sit under the single page's hero. Every
    sentence is an i18n segment: the catalogue's prose under
    clients.<id>.*, the page's own wording under common.*."""
    k = f"clients.{c.id}"
    reqs = "".join(f'<li data-t="{k}.req.{r.id}">{prose_html(r.text)}</li>' for r in c.requirements)
    modes = "".join(
        f'<h3 data-t="{k}.mode.{m.id}.title">{prose_html(m.title)}</h3>'
        f'<p data-t="{k}.mode.{m.id}.desc">{prose_html(m.description)}</p>'
        + "".join(_snippet(c, m.id, f) for f in m.files)
        for m in c.modes
    )
    requirements = (
        f'<h2 data-t="common.client.requirements">Requirements</h2><ul>{reqs}</ul>' if reqs else ""
    )
    head = f"""        <h1 id="client-title" class="as-h2" data-t="{k}.h1">{_e(short_name(c))} on your own GPUs.</h1>
        <p class="lede" data-t="{k}.summary">{prose_html(c.summary)}</p>
"""
    if c.id == "claude-code":
        # The router drawing is the page's opening argument, so its section
        # sits between the header and the facts.
        top = f"""    <section class="sec client" aria-labelledby="client-title">
      <div class="wrap">
{head}      </div>
    </section>
{router_section}    <section class="sec client" aria-label="Setup" data-t-aria-label="common.client.setup-section.aria-label">
      <div class="wrap">
"""
    else:
        top = f"""    <section class="sec client" aria-labelledby="client-title">
      <div class="wrap">
{head}{figure_html(AGENTS[c.id], f"flow-{c.id}", _caption(c), key=f"{k}.flow")}"""
    return (
        top
        + f"""        <dl class="facts">
          <dt data-t="common.client.protocol">Protocol</dt><dd>{_e(protocol(c))}</dd>
          <dt data-t="common.client.support">Support</dt><dd>{_support(c)}</dd>
        </dl>
        <p data-t="{k}.capability">{prose_html(capability(c))}</p>
        {requirements}
        <h2 data-t="common.client.setup">Setup</h2>
        <p data-t="common.client.setup-p">In the console, <strong>Connect</strong> writes these files with your warden’s address, your key and the model you picked. Here they are with placeholders.</p>
        {modes}
        <p data-t="common.client.docs">Official documentation: {{{{docs_link}}}} (read {{{{docs_date}}}}).</p>
        <p class="related" data-t="common.client.related">Related: <a href="/coding-agents">all coding agents</a> · <a href="/agent-tuning">let your agent pick the model</a> · <a href="/api-keys">API keys</a></p>
      </div>
    </section>
"""
    )


def docs_tokens(c: ClientDef) -> dict[str, str]:
    """{{docs_link}} and {{docs_date}} in the docs line (common.client.docs),
    filled after translation: one sentence for every client."""
    return {
        "docs_link": f'<a href="{html.escape(c.docs.url)}" rel="noopener">{_code(c.docs.url)}</a>',
        "docs_date": _e(c.docs.accessed),
    }


def _caption(c: ClientDef) -> str:
    return (
        f"Where a request from {_e(short_name(c))} goes: the two settings Connect writes, "
        "the warden in the middle, then one copy of the model per GPU."
    )


def clients_table_html() -> str:
    """The table of every client on /coding-agents: names and protocols as
    they are, headers and support levels as segments."""
    rows = "".join(
        f'<tr><th scope="row"><a href="/coding-agents/{c.id}">{_e(short_name(c))}</a></th>'
        f"<td>{_e(protocol(c))}</td><td>{_support(c)}</td></tr>"
        for c in CLIENTS
    )
    return (
        '<div class="table-wrap"><table class="clients"><thead><tr>'
        '<th scope="col" data-t="common.clients.client">Client</th>'
        '<th scope="col" data-t="common.client.protocol">Protocol</th>'
        '<th scope="col" data-t="common.client.support">Support</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )
