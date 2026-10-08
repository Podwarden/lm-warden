"""``GET /api/connect/clients``: the Connect page's one read.

Returns the client catalogue (app/connect/catalog.py) with each file's
template and its rendering for this warden, plus the live facts every snippet
depends on: the origin clients should use, the router's header name and
enabled rules, and the registered models with their effective context window
and capability flags. ``require_jwt`` (a session or an admin token); a GET,
so no CSRF.

Never in the response: key plaintexts, key ids or names, request headers.
``{{key}}`` always renders as ``KEY_PLACEHOLDER``.
"""

from typing import Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from app.auth.deps import require_jwt
from app.auth.origin import _derived_origin
from app.connect.catalog import (
    CLIENTS,
    DEFAULT_CONTEXT,
    KEY_PLACEHOLDER,
    MODEL_PLACEHOLDER,
    ClientDef,
    Expect,
    Group,
    Language,
    ModeDef,
    Protocol,
    Support,
    VerifiedStatus,
    example_model_for,
    render_file,
)
from app.db.database import open_db
from app.db.repos.models import ModelRepo, ModelRow
from app.db.repos.router_rules import RouterRuleRepo
from app.db.repos.settings import SettingsRepo
from app.models.context_window import effective_context_window
from app.models.serialisation import _backend_of, _tristate
from app.router.rules import HEADER_NAME, ROUTER_KEYS, config_from_kv

router = APIRouter(prefix="/api/connect", tags=["connect"])


class ConnectRule(BaseModel):
    pattern: str
    target_served_name: str | None
    target_status: str | None
    fallback: bool
    #: First known Claude id the pattern matches, else the pattern when it has
    #: no wildcard, else null.
    example_model: str | None


class ConnectRouter(BaseModel):
    enabled: bool
    passthrough_unmatched: bool
    rules: list[ConnectRule]


class ConnectModel(BaseModel):
    id: str
    served_name: str
    status: str
    backend: str
    context_window: int | None
    supports_tools: bool | None
    supports_vision: bool | None
    supports_reasoning: bool | None


class ConnectRequirement(BaseModel):
    id: str
    text: str
    min_context_tokens: int | None
    soft: bool
    needs_tools: bool
    router_on: bool
    relay_key: bool
    #: Mode ids this requirement applies to; null means every mode.
    modes: list[str] | None


class ConnectDocs(BaseModel):
    url: str
    accessed: str


class ConnectVerified(BaseModel):
    status: VerifiedStatus
    date: str
    client_version: str | None
    note: str | None


class ConnectFile(BaseModel):
    id: str
    label: str
    path: str | None
    language: Language
    requires_tools: bool | None
    template: str
    rendered: str


class ConnectVerify(BaseModel):
    method: Literal["POST"]
    path: str
    headers: dict[str, str]
    body: dict[str, Any]
    expect: Expect
    covers: str
    not_covered: str | None


class ConnectMode(BaseModel):
    id: str
    title: str
    description: str
    needs_relay_key: bool
    files: list[ConnectFile]
    verify: ConnectVerify | None


class ConnectClient(BaseModel):
    id: str
    name: str
    group: Group
    protocol: Protocol
    support: Support
    summary: str
    capability: str
    requirements: list[ConnectRequirement]
    docs: ConnectDocs
    verified: ConnectVerified | None
    modes: list[ConnectMode]


class ConnectClientsOut(BaseModel):
    origin: str
    origin_source: Literal["public_url", "request"]
    header_name: str
    key_placeholder: str
    router: ConnectRouter
    models: list[ConnectModel]
    selected_model: str | None
    clients: list[ConnectClient]


def _model_out(request: Request, row: ModelRow) -> ConnectModel:
    state = request.app.state
    window = effective_context_window(state.settings, getattr(state, "supervisor", None), row)
    return ConnectModel(
        id=row.id,
        served_name=row.served_model_name,
        status=row.status,
        backend=_backend_of(row),
        context_window=window,
        supports_tools=_tristate(row.supports_tools),
        supports_vision=_tristate(row.supports_vision),
        supports_reasoning=_tristate(row.supports_reasoning),
    )


def _mode_out(mode: ModeDef, variables: dict[str, str], context_known: bool) -> ConnectMode:
    verify = mode.verify
    return ConnectMode(
        id=mode.id,
        title=mode.title,
        description=mode.description,
        needs_relay_key=mode.needs_relay_key,
        files=[
            ConnectFile(
                id=f.id,
                label=f.label,
                path=f.path,
                language=f.language,
                requires_tools=f.requires_tools,
                template=f.template,
                rendered=render_file(f, variables, context_known=context_known),
            )
            for f in mode.files
        ],
        verify=None
        if verify is None
        else ConnectVerify(
            method="POST",
            path=verify.path,
            headers=dict(verify.headers),
            body=dict(verify.body),
            expect=verify.expect,
            covers=verify.covers,
            not_covered=verify.not_covered,
        ),
    )


def _client_out(c: ClientDef, variables: dict[str, str], context_known: bool) -> ConnectClient:
    return ConnectClient(
        id=c.id,
        name=c.name,
        group=c.group,
        protocol=c.protocol,
        support=c.support,
        summary=c.summary,
        capability=c.capability,
        requirements=[
            ConnectRequirement(
                id=r.id,
                text=r.text,
                min_context_tokens=r.min_context_tokens,
                soft=r.soft,
                needs_tools=r.needs_tools,
                router_on=r.router_on,
                relay_key=r.relay_key,
                modes=list(r.modes) if r.modes is not None else None,
            )
            for r in c.requirements
        ],
        docs=ConnectDocs(url=c.docs.url, accessed=c.docs.accessed),
        verified=None
        if c.verified is None
        else ConnectVerified(
            status=c.verified.status,
            date=c.verified.date,
            client_version=c.verified.client_version,
            note=c.verified.note,
        ),
        modes=[_mode_out(m, variables, context_known) for m in c.modes],
    )


@router.get("/clients", response_model=ConnectClientsOut)
async def list_clients(
    request: Request,
    model: str | None = Query(
        default=None, description="A served model name to render the snippets for."
    ),
    _user: str = Depends(require_jwt),
) -> ConnectClientsOut:
    async with open_db(request.app.state.settings.db_path) as db:
        settings_repo = SettingsRepo(db)
        public_url = await settings_repo.get("public_url")
        kv = await settings_repo.get_many(list(ROUTER_KEYS))
        rows = await ModelRepo(db).list_all()
        rule_rows = await RouterRuleRepo(db).list_all()

    origin_source: Literal["public_url", "request"]
    if public_url:
        origin, origin_source = public_url.rstrip("/"), "public_url"
    else:
        origin = _derived_origin(request) or str(request.base_url).rstrip("/")
        origin_source = "request"

    rows.sort(key=lambda r: (r.status != "loaded", r.served_model_name))
    models = [_model_out(request, r) for r in rows]
    by_id = {r.id: r for r in rows}

    selected: ConnectModel | None = None
    if model is not None:
        selected = next((m for m in models if m.served_name == model), None)
    if selected is None:
        selected = next((m for m in models if m.status == "loaded"), None)

    cfg = config_from_kv(kv)
    rules: list[ConnectRule] = []
    for r in rule_rows:
        if not r.enabled:
            continue
        target = by_id.get(r.target_model_id)
        rules.append(
            ConnectRule(
                pattern=r.pattern,
                target_served_name=target.served_model_name if target else None,
                target_status=target.status if target else None,
                fallback=r.fallback,
                example_model=example_model_for(r.pattern),
            )
        )

    context = selected.context_window if selected is not None else None
    variables = {
        "origin": origin,
        "header": HEADER_NAME,
        "key": KEY_PLACEHOLDER,
        "model": selected.served_name if selected is not None else MODEL_PLACEHOLDER,
        "context": str(context if context else DEFAULT_CONTEXT),
    }
    context_known = bool(context)

    return ConnectClientsOut(
        origin=origin,
        origin_source=origin_source,
        header_name=HEADER_NAME,
        key_placeholder=KEY_PLACEHOLDER,
        router=ConnectRouter(
            enabled=cfg.enabled, passthrough_unmatched=cfg.passthrough_unmatched, rules=rules
        ),
        models=models,
        selected_model=selected.served_name if selected is not None else None,
        clients=[_client_out(c, variables, context_known) for c in CLIENTS],
    )
