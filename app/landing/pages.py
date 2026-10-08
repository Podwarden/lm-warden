"""The website's page registry (spec 2026-10-07 §3/§4): one source for
routes, the menu, breadcrumbs, sitemap, llms.txt and tests. No imports from
`app`: app/auth/policy.py and app/auth/csrf.py import PUBLIC_PAGE_PATHS."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    path: str  # "/" or "/faq": no trailing slash
    fragment: str  # file under app/landing/pages/, "" when generated
    title: str
    description: str
    nav: str  # "home" | "agents" | "tuning" | "features" | "faq" | "legal"
    crumbs: tuple[tuple[str, str], ...] = ()  # (label, path) after Home
    jsonld: tuple[str, ...] = ()  # "app", "website", "faq", "howto", "article"
    tag_only: bool = False  # served only when a Google tag ID is set
    client_id: str | None = None


#: "LLM" stays in the home title and description for search; the product is
#: "LM Warden".
HOME = Page(
    path="/",
    fragment="home.html",
    title="LM Warden: self-hosted LLM gateway for your own GPUs",
    description=(
        "Serve LLMs from your own NVIDIA GPUs with vLLM and llama.cpp: one "
        "OpenAI-compatible /v1 endpoint, a key per app, and readings per card."
    ),
    nav="home",
    jsonld=("app", "website"),
)
AGENTS = Page(
    path="/coding-agents",
    fragment="coding-agents.html",
    title="Coding agents on your own GPUs | LM Warden",
    description=(
        "Run Claude Code, Codex CLI, OpenCode, Aider, Cline, Cursor and more against a "
        "model on your own NVIDIA GPUs, with the config written for you."
    ),
    nav="agents",
    crumbs=(("Coding agents", "/coding-agents"),),
    jsonld=("article",),
)
TUNING = Page(
    path="/agent-tuning",
    fragment="agent-tuning.html",
    title="Let your coding agent pick the model | LM Warden",
    description=(
        "Give Claude Code or Codex an admin token and it picks the model, engine and flags "
        "for your GPUs, benchmarking as it goes. Prompts included."
    ),
    nav="tuning",
    crumbs=(("Agent tuning", "/agent-tuning"),),
    jsonld=("howto",),
)
GPU = Page(
    path="/gpu-monitoring",
    fragment="gpu-monitoring.html",
    title="Per-card GPU monitoring for vLLM | LM Warden",
    description=(
        "Each NVIDIA card gets its own panel: temperature against its throttle point, power "
        "against its cap, the PCIe link, and every request in flight."
    ),
    nav="features",
    crumbs=(("Features", "/features"), ("GPU monitoring", "/gpu-monitoring")),
    jsonld=("article",),
)
CACHE = Page(
    path="/cache-aware-routing",
    fragment="cache-aware-routing.html",
    title="Cache-aware routing for vLLM replicas | LM Warden",
    description=(
        "Agent sessions go back to the replica that already holds their prompt in the prefix "
        "cache, so long conversations skip most of the prefill."
    ),
    nav="features",
    crumbs=(("Features", "/features"), ("Cache-aware routing", "/cache-aware-routing")),
    jsonld=("article",),
)
KEYS = Page(
    path="/api-keys",
    fragment="api-keys.html",
    title="API keys, priorities and usage per app | LM Warden",
    description=(
        "One key per app with its own priority, usage and queue times, behind "
        "OpenAI-compatible, Anthropic Messages and Responses endpoints."
    ),
    nav="features",
    crumbs=(("Features", "/features"), ("API keys", "/api-keys")),
    jsonld=("article",),
)
FEATURES = Page(
    path="/features",
    fragment="features.html",
    title="More features and lessons learned | LM Warden",
    description=(
        "God mode, admin tokens, the engine watchdog, and the two failures that shaped how "
        "LM Warden supervises vLLM."
    ),
    nav="features",
    crumbs=(("Features", "/features"),),
    jsonld=("article",),
)
FAQ = Page(
    path="/faq",
    fragment="faq.html",
    title="LM Warden FAQ: self-hosted LLM gateway questions",
    description=(
        "What people ask before installing LM Warden: hardware, engines, models, clients, "
        "security and how it compares to running vLLM directly."
    ),
    nav="faq",
    crumbs=(("FAQ", "/faq"),),
    jsonld=("faq",),
)
PRIVACY = Page(
    path="/privacy",
    fragment="privacy.html",
    title="Privacy and cookies | LM Warden",
    description=(
        "What this site measures with Google Analytics, when it asks first, and how to "
        "change your choice."
    ),
    nav="legal",
    crumbs=(("Privacy", "/privacy"),),
    tag_only=True,
)

#: The site's 404 page: rendered for unknown paths, never routed or listed.
NOT_FOUND = Page(
    path="/404",
    fragment="not-found.html",
    title="Page not found | LM Warden",
    description="This page does not exist on the LM Warden website.",
    nav="legal",
)

PAGES: tuple[Page, ...] = (
    HOME,
    AGENTS,
    TUNING,
    GPU,
    CACHE,
    KEYS,
    FEATURES,
    FAQ,
    PRIVACY,
)

#: Path template for generated client pages (app/landing/clients.py).
CLIENT_PATH = "/coding-agents/{client_id}"

#: Every route the website registers, for app/auth/policy.py (public) and
#: app/auth/csrf.py (no cookie): each page, its trailing-slash redirect, and
#: the client-page template.
PUBLIC_PAGE_PATHS: frozenset[str] = frozenset(
    {p.path for p in PAGES}
    | {p.path + "/" for p in PAGES if p.path != "/"}
    | {CLIENT_PATH, CLIENT_PATH + "/"}
)
