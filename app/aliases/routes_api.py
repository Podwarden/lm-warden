"""Control API for model aliases: a named, ordered list of target models plus
optional proxy-side settings (guard / thinking / params). Every route is
require_jwt (session or admin token).

Every write ends by invalidating the /v1 model cache (app/proxy/model_cache.py)
-- through ModelAliasRepo -- so the very next request sees the change.
"""

import secrets
from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.aliases.schemas import AliasesOut, AliasIn, AliasOut, AliasPatch, AliasTargetOut
from app.auth.deps import require_jwt
from app.db.database import open_db
from app.db.repos.model_aliases import AliasRow, ModelAliasRepo
from app.db.repos.models import ModelRepo

router = APIRouter(prefix="/api/aliases", tags=["aliases"])


async def _alias_target_out(db: aiosqlite.Connection, model_id: str) -> AliasTargetOut:
    m = await ModelRepo(db).get(model_id)
    return AliasTargetOut(
        id=model_id,
        served_model_name=m.served_model_name if m else None,
        status=m.status if m else None,
    )


async def _alias_out(db: aiosqlite.Connection, row: AliasRow) -> AliasOut:
    return AliasOut(
        id=row.id,
        name=row.name,
        targets=[await _alias_target_out(db, t) for t in row.targets],
        guard=row.guard,
        thinking_policy=row.thinking_policy,
        params=row.params,
        enabled=row.enabled,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


async def _require_targets(db: aiosqlite.Connection, targets: list[str]) -> None:
    repo = ModelRepo(db)
    missing = [t for t in targets if await repo.get(t) is None]
    if missing:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, f"targets: unknown model {missing[0]!r}"
        )


@router.get("", response_model=AliasesOut)
async def list_aliases(request: Request, _user: str = Depends(require_jwt)) -> AliasesOut:
    async with open_db(request.app.state.settings.db_path) as db:
        rows = await ModelAliasRepo(db).list()
        return AliasesOut(aliases=[await _alias_out(db, r) for r in rows])


@router.post("", response_model=AliasOut, status_code=status.HTTP_201_CREATED)
async def create_alias(
    body: AliasIn, request: Request, _user: str = Depends(require_jwt)
) -> AliasOut:
    async with open_db(request.app.state.settings.db_path) as db:
        repo = ModelAliasRepo(db)
        await _require_targets(db, body.targets)
        if await repo.get_by_name(body.name) is not None:
            raise HTTPException(
                status.HTTP_409_CONFLICT, f"alias name '{body.name}' is already in use"
            )
        row = await repo.create(
            AliasRow(
                id=secrets.token_hex(8),
                name=body.name,
                targets=list(body.targets),
                guard=body.guard,
                thinking_policy=body.thinking_policy,
                params={k: v.model_dump() for k, v in body.params.items()},
                enabled=body.enabled,
            )
        )
        return await _alias_out(db, row)


@router.get("/{alias_id}", response_model=AliasOut)
async def get_alias(alias_id: str, request: Request, _user: str = Depends(require_jwt)) -> AliasOut:
    async with open_db(request.app.state.settings.db_path) as db:
        row = await ModelAliasRepo(db).get(alias_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "alias not found")
        return await _alias_out(db, row)


@router.patch("/{alias_id}", response_model=AliasOut)
async def patch_alias(
    alias_id: str, body: AliasPatch, request: Request, _user: str = Depends(require_jwt)
) -> AliasOut:
    fields: dict[str, Any] = body.model_dump(exclude_none=True)
    async with open_db(request.app.state.settings.db_path) as db:
        repo = ModelAliasRepo(db)
        if await repo.get(alias_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "alias not found")
        if "targets" in fields:
            await _require_targets(db, fields["targets"])
        if "name" in fields:
            existing = await repo.get_by_name(fields["name"])
            if existing is not None and existing.id != alias_id:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    f"alias name '{fields['name']}' is already in use",
                )
        row = await repo.update(alias_id, fields) if fields else await repo.get(alias_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "alias not found")
        return await _alias_out(db, row)


@router.delete("/{alias_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_alias(
    alias_id: str, request: Request, _user: str = Depends(require_jwt)
) -> Response:
    async with open_db(request.app.state.settings.db_path) as db:
        deleted = await ModelAliasRepo(db).delete(alias_id)
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "alias not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
