"""Control API for the Claude Code model router (#287): settings, rules,
stats and decisions. Every route is ``require_jwt`` (session or admin token).

Every write ends with ``RouterState.invalidate()`` so the data plane reloads
the rule set on its next request.
"""

from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.auth.deps import require_jwt
from app.db.database import open_db
from app.db.repos.models import ModelRepo
from app.db.repos.router_rules import RouterRuleRepo, RouterRuleRow
from app.db.repos.settings import SettingsRepo
from app.router.rules import (
    HEADER_NAME,
    INT_RANGES,
    ROUTER_KEYS,
    RouterConfig,
    config_from_kv,
    validate_pattern,
    validate_upstream_url,
)

router = APIRouter(prefix="/api/router", tags=["router"])

MIN_MAX_TOKENS_CAP = 131072


def _ranged(name: str) -> Any:
    lo, hi = INT_RANGES[name]
    return Field(default=None, ge=lo, le=hi)


# -- settings ---------------------------------------------------------------


class RouterSettingsOut(BaseModel):
    enabled: bool
    upstream_url: str
    passthrough_unmatched: bool
    local_header_timeout_s: int
    local_nonstream_timeout_s: int
    breaker_threshold: int
    breaker_open_s: int
    max_body_mb: int
    header_name: str


def _reject_nulls(data: object) -> object:
    if isinstance(data, dict):
        for k, v in data.items():
            if v is None:
                raise ValueError(f"{k}: null is not allowed")
    return data


class RouterSettingsPatch(BaseModel):
    """Any subset of the eight writable keys; ``header_name`` is refused."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    upstream_url: str | None = None
    passthrough_unmatched: bool | None = None
    local_header_timeout_s: int | None = _ranged("local_header_timeout_s")
    local_nonstream_timeout_s: int | None = _ranged("local_nonstream_timeout_s")
    breaker_threshold: int | None = _ranged("breaker_threshold")
    breaker_open_s: int | None = _ranged("breaker_open_s")
    max_body_mb: int | None = _ranged("max_body_mb")

    @model_validator(mode="before")
    @classmethod
    def _no_nulls(cls, data: object) -> object:
        return _reject_nulls(data)

    @field_validator("upstream_url")
    @classmethod
    def _url(cls, v: str | None) -> str | None:
        return validate_upstream_url(v) if v is not None else v


def _settings_out(cfg: RouterConfig) -> RouterSettingsOut:
    return RouterSettingsOut(
        **{f: getattr(cfg, f) for f in ROUTER_KEYS.values()}, header_name=HEADER_NAME
    )


def _kv_value(v: object) -> str:
    return ("true" if v else "false") if isinstance(v, bool) else str(v)


@router.get("/settings", response_model=RouterSettingsOut)
async def get_settings(request: Request, _user: str = Depends(require_jwt)) -> RouterSettingsOut:
    async with open_db(request.app.state.settings.db_path) as db:
        kv = await SettingsRepo(db).get_many(list(ROUTER_KEYS))
    return _settings_out(config_from_kv(kv))


@router.patch("/settings", response_model=RouterSettingsOut)
async def patch_settings(
    body: RouterSettingsPatch, request: Request, _user: str = Depends(require_jwt)
) -> RouterSettingsOut:
    changes = body.model_dump(exclude_none=True)
    field_to_key = {f: k for k, f in ROUTER_KEYS.items()}
    async with open_db(request.app.state.settings.db_path) as db:
        repo = SettingsRepo(db)
        for f, v in changes.items():
            await repo.set(field_to_key[f], _kv_value(v))
        kv = await repo.get_many(list(ROUTER_KEYS))
    request.app.state.router.invalidate()
    return _settings_out(config_from_kv(kv))


# -- rules ------------------------------------------------------------------


class RuleIn(BaseModel):
    pattern: str
    target_model_id: str = Field(min_length=1)
    enabled: bool = True
    fallback: bool = True
    strip_thinking: bool = True
    min_max_tokens: int = Field(default=0, ge=0, le=MIN_MAX_TOKENS_CAP)

    @field_validator("pattern")
    @classmethod
    def _pattern(cls, v: str) -> str:
        return validate_pattern(v)


class RulePatch(BaseModel):
    pattern: str | None = None
    target_model_id: str | None = Field(default=None, min_length=1)
    enabled: bool | None = None
    fallback: bool | None = None
    strip_thinking: bool | None = None
    min_max_tokens: int | None = Field(default=None, ge=0, le=MIN_MAX_TOKENS_CAP)

    @model_validator(mode="before")
    @classmethod
    def _no_nulls(cls, data: object) -> object:
        return _reject_nulls(data)

    @field_validator("pattern")
    @classmethod
    def _pattern(cls, v: str | None) -> str | None:
        return validate_pattern(v) if v is not None else v


class RuleOut(BaseModel):
    id: str
    position: int
    pattern: str
    target_model_id: str
    target_served_name: str | None
    target_status: str | None
    enabled: bool
    fallback: bool
    strip_thinking: bool
    min_max_tokens: int
    created_at: str
    updated_at: str


class RulesOut(BaseModel):
    rules: list[RuleOut]


class RuleOrder(BaseModel):
    ids: list[str]


async def _rule_out(db: aiosqlite.Connection, r: RouterRuleRow) -> RuleOut:
    m = await ModelRepo(db).get(r.target_model_id)
    return RuleOut(
        id=r.id,
        position=r.position,
        pattern=r.pattern,
        target_model_id=r.target_model_id,
        target_served_name=m.served_model_name if m else None,
        target_status=m.status if m else None,
        enabled=r.enabled,
        fallback=r.fallback,
        strip_thinking=r.strip_thinking,
        min_max_tokens=r.min_max_tokens,
        created_at=r.created_at,
        updated_at=r.updated_at,
    )


async def _rules_out(db: aiosqlite.Connection) -> RulesOut:
    rows = await RouterRuleRepo(db).list_all()
    return RulesOut(rules=[await _rule_out(db, r) for r in rows])


async def _require_model(db: aiosqlite.Connection, model_id: str) -> None:
    if await ModelRepo(db).get(model_id) is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "target_model_id: unknown model")


@router.get("/rules", response_model=RulesOut)
async def list_rules(request: Request, _user: str = Depends(require_jwt)) -> RulesOut:
    async with open_db(request.app.state.settings.db_path) as db:
        return await _rules_out(db)


@router.post("/rules", response_model=RuleOut, status_code=status.HTTP_201_CREATED)
async def create_rule(body: RuleIn, request: Request, _user: str = Depends(require_jwt)) -> RuleOut:
    async with open_db(request.app.state.settings.db_path) as db:
        await _require_model(db, body.target_model_id)
        row = await RouterRuleRepo(db).create(
            body.pattern,
            body.target_model_id,
            body.enabled,
            body.fallback,
            body.strip_thinking,
            body.min_max_tokens,
        )
        out = await _rule_out(db, row)
    request.app.state.router.invalidate()
    return out


# Registered before ``/rules/{rule_id}`` readers; PUT does not collide with
# the PATCH/DELETE routes below, but keep the literal first for clarity.
@router.put("/rules/order", response_model=RulesOut)
async def reorder_rules(
    body: RuleOrder, request: Request, _user: str = Depends(require_jwt)
) -> RulesOut:
    async with open_db(request.app.state.settings.db_path) as db:
        try:
            await RouterRuleRepo(db).reorder(body.ids)
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
        out = await _rules_out(db)
    request.app.state.router.invalidate()
    return out


@router.patch("/rules/{rule_id}", response_model=RuleOut)
async def patch_rule(
    rule_id: str, body: RulePatch, request: Request, _user: str = Depends(require_jwt)
) -> RuleOut:
    fields: dict[str, Any] = body.model_dump(exclude_none=True)
    async with open_db(request.app.state.settings.db_path) as db:
        repo = RouterRuleRepo(db)
        if await repo.get(rule_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "rule not found")
        if "target_model_id" in fields:
            await _require_model(db, fields["target_model_id"])
        row = await repo.update(rule_id, **fields) if fields else await repo.get(rule_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "rule not found")
        out = await _rule_out(db, row)
    request.app.state.router.invalidate()
    return out


@router.delete("/rules/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_rule(
    rule_id: str, request: Request, _user: str = Depends(require_jwt)
) -> Response:
    async with open_db(request.app.state.settings.db_path) as db:
        deleted = await RouterRuleRepo(db).delete(rule_id)
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "rule not found")
    request.app.state.router.invalidate()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# -- stats / decisions ------------------------------------------------------


class Percentiles(BaseModel):
    p50: int | None
    p95: int | None


class RuleStats(BaseModel):
    rule_id: str
    pattern: str
    target_model_id: str
    target_served_name: str | None
    local: int
    fallback: int
    refused: int
    latency_ms: Percentiles
    ttfb_ms: Percentiles


class TargetStats(BaseModel):
    model_id: str
    served_name: str | None
    breaker: str
    consecutive_failures: int
    open_until: str | None
    last_reason: str | None
    failures: int


class PassthroughStats(BaseModel):
    requests: int
    errors: int
    latency_ms: Percentiles
    ttfb_ms: Percentiles
    input_tokens: int
    output_tokens: int


class RouterStatsOut(BaseModel):
    enabled: bool
    since: str | None
    totals: dict[str, int]
    by_reason: dict[str, int]
    error_reasons: dict[str, int]
    rules: list[RuleStats]
    targets: list[TargetStats]
    passthrough: PassthroughStats


class DecisionOut(BaseModel):
    ts: str
    path: str
    model_in: str | None
    model_out: str | None
    rule_id: str | None
    route: str
    reason: str | None
    status: int
    latency_ms: int
    ttfb_ms: int | None
    stream: bool
    token_name: str | None


class DecisionsOut(BaseModel):
    decisions: list[DecisionOut]


@router.get("/stats", response_model=RouterStatsOut)
async def get_stats(request: Request, _user: str = Depends(require_jwt)) -> dict[str, Any]:
    rs = await request.app.state.router.ruleset(request.app.state.settings.db_path)
    return request.app.state.router.snapshot(rs)  # type: ignore[no-any-return]


@router.post("/stats/reset", status_code=status.HTTP_204_NO_CONTENT)
async def reset_stats(request: Request, _user: str = Depends(require_jwt)) -> Response:
    request.app.state.router.reset()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/decisions", response_model=DecisionsOut)
async def get_decisions(
    request: Request,
    limit: int = Query(default=100, ge=1, le=200),
    _user: str = Depends(require_jwt),
) -> dict[str, Any]:
    return {"decisions": request.app.state.router.decisions(limit)}
