"""Extension registry (spec 2026-11-04, site split).

Core never imports an edition package. The Enterprise and website packages
plug in through Python entry points (group ``lm_warden.extensions``, see
``app/ext/loader.py``); each extension's ``register(reg)`` fills this registry
and ``build_app()`` reads it. With no extension installed the registry stays
empty and the app is the Community build.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from fastapi import APIRouter, Request
from fastapi.responses import Response

EDITIONS: tuple[str, ...] = ("community", "enterprise", "vendor")


@dataclass
class PublicRoute:
    """A route an extension serves without credentials."""

    method: str
    path: str
    prefix: bool = False
    mint_csrf: bool = False


@dataclass
class SettingKey:
    """A runtime-settings key contributed by an extension."""

    name: str
    restart_kind: str
    coerce: Callable[[Any], Any]
    #: Only a signed-in session may change it, never an admin token (core's
    #: SESSION_ONLY_RUNTIME_KEYS, extended per key).
    session_only: bool = False


@dataclass
class ConsoleLink:
    """An entry an extension adds to the console menu (spec 2026-11-20).

    ``href`` is a same-origin endpoint the console POSTs to as the signed-in
    admin; it answers ``{"url": "/..."}`` (and may set a cookie scoped to it), a page the
    extension serves itself, and the console navigates there. (A plain link
    cannot work: the console's access token lives in memory and the refresh
    cookie is scoped to /api/auth, so a navigation carries no credential.)
    Core draws the entry and nothing else."""

    label: str
    href: str


class HomeProvider(Protocol):
    mode: Literal["builtin", "custom"]
    priority: int

    async def home(self, request: Request) -> Response: ...

    async def not_found(self, request: Request) -> Response | None: ...

    async def robots_txt(self, request: Request) -> Response | None: ...


@dataclass
class Registry:
    edition: str = "community"
    features: set[str] = field(default_factory=set)
    routers: list[APIRouter] = field(default_factory=list)
    public_routes: list[PublicRoute] = field(default_factory=list)
    settings_keys: list[SettingKey] = field(default_factory=list)
    home_providers: list[HomeProvider] = field(default_factory=list)
    console_links: list[ConsoleLink] = field(default_factory=list)
    load_errors: list[str] = field(default_factory=list)

    def set_edition(self, name: str) -> None:
        """The highest of community < enterprise < vendor wins, whatever the
        load order. An unknown name is rejected (the loader records it)."""
        if name not in EDITIONS:
            raise ValueError(f"unknown edition {name!r}")
        if EDITIONS.index(name) > EDITIONS.index(self.edition):
            self.edition = name

    def add_feature(self, feature_id: str) -> None:
        self.features.add(feature_id)

    def has_feature(self, feature_id: str) -> bool:
        """The single feature check: every caller goes through here, so a
        licence can later intersect ``features`` in one place."""
        return feature_id in self.features


__all__ = [
    "ConsoleLink",
    "EDITIONS",
    "HomeProvider",
    "PublicRoute",
    "Registry",
    "SettingKey",
]
