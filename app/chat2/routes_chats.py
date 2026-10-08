from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, Response

from app.chat2 import catalog
from app.chat2.budget import BudgetDecision
from app.chat2.context import derive_context
from app.chat2.contract import CONTRACT_VERSION
from app.chat2.errors import api_error
from app.chat2.identity import current_user_id
from app.chat2.limits import ALLOWED_IMAGE_MIMES
from app.chat2.repo import DEFAULT_SETTINGS, Chat2Repo, ChatRow, MessageRow
from app.chat2.routes_attachments import _serialize as serialize_attachment
from app.chat2.schemas import ChatCreate, ChatPatch, DefaultsPut, ForkBody
from app.chat2.storage import file_path, unlink_quiet
from app.db.database import open_db

router = APIRouter(prefix="/api/chat2", tags=["chat2"])


def serialize_chat(
    row: ChatRow, *, model_loaded: bool, context: dict[str, Any] | None = None
) -> dict[str, Any]:
    d = {
        "id": row.id,
        "title": row.title,
        "title_source": row.title_source,
        "title_prev": row.title_prev,
        "model": row.model,
        # Whether `model` names something the turn route will actually serve
        # right now (#240). A chat pins its model by name and nothing constrains
        # that to the loaded set -- the model can be unloaded or replaced long
        # after the chat was created -- and the catalog (`GET /models`) lists
        # loaded models only, so without this the client could only infer the
        # mismatch from an absence, and a stale catalog turns that inference
        # into the wrong answer.
        "model_loaded": model_loaded,
        "settings": row.settings,
        "forked_from_chat_id": row.forked_from_chat_id,
        "forked_at_seq": row.forked_at_seq,
        "cost_micros_total": row.cost_micros_total,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "last_message_at": row.last_message_at,
    }
    if context is not None:
        d["context_full"] = context["full"]
    return d


def serialize_message(row: MessageRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "seq": row.seq,
        "role": row.role,
        "parts": row.parts,
        "model": row.model,
        "settings_snapshot": row.settings_snapshot,
        "usage": row.usage,
        "finish_reason": row.finish_reason,
        "error": row.error,
        "created_at": row.created_at,
    }


async def _model_loaded(settings: Any, chat: ChatRow) -> bool:  # noqa: ANN401
    """`get_model_info` answers None for anything but a loaded row (catalog.py),
    which is exactly the check the turn route 404s on."""
    return bool(chat.model) and await catalog.get_model_info(settings, chat.model) is not None


async def _describe(settings: Any, repo: Chat2Repo, chat: ChatRow) -> tuple[dict[str, Any], bool]:  # noqa: ANN401
    """The chat's context state and whether its model is servable -- one
    catalog lookup feeds both."""
    msgs = await repo.list_messages(chat.id)
    info = await catalog.get_model_info(settings, chat.model) if chat.model else None
    ctx = derive_context(msgs, chat.settings, info.context_window if info else None).as_event()
    return ctx, info is not None


async def _serialize_described(settings: Any, repo: Chat2Repo, chat: ChatRow) -> dict[str, Any]:  # noqa: ANN401
    ctx, loaded = await _describe(settings, repo, chat)
    return serialize_chat(chat, model_loaded=loaded, context=ctx)


@router.get("/_whoami")
async def whoami(user_id: int = Depends(current_user_id)) -> dict[str, Any]:
    return {"user_id": user_id, "contract": CONTRACT_VERSION}


@router.post("/chats", status_code=201)
async def create_chat(
    body: ChatCreate, request: Request, user_id: int = Depends(current_user_id)
) -> dict[str, Any]:
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        repo = Chat2Repo(db)
        defaults = await repo.get_defaults(user_id)
        model, base = defaults if defaults else (None, DEFAULT_SETTINGS)
        merged = body.settings.merge_into(base) if body.settings else dict(base)
        chat = await repo.create_chat(user_id, model=body.model or model, settings=merged)
        loaded = await _model_loaded(settings, chat)
    return serialize_chat(chat, model_loaded=loaded)


@router.get("/chats")
async def list_chats(request: Request, user_id: int = Depends(current_user_id)) -> dict[str, Any]:
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        repo = Chat2Repo(db)
        rows = await repo.list_chats(user_id)
        out = [await _serialize_described(settings, repo, r) for r in rows]
    return {"chats": out}


@router.get("/chats/{chat_id}")
async def get_chat(
    chat_id: str, request: Request, user_id: int = Depends(current_user_id)
) -> dict[str, Any]:
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        repo = Chat2Repo(db)
        chat = await repo.get_chat(user_id, chat_id)
        if chat is None:
            raise api_error(404, "not_found", "chat not found")
        msgs = await repo.list_messages(chat_id)
        atts = await repo.list_attachments(chat_id)
        ctx, loaded = await _describe(settings, repo, chat)
    secret = request.app.state.jwt_secret
    # Detached-turn reattach handle (app/chat2/live.py): non-null while a
    # runner is still streaming for this chat. Reported null the moment the
    # turn is done — even though the registry keeps the finished entry
    # replayable for its GC grace window — so the client only ever attaches
    # to answers that are genuinely still being written.
    live = request.app.state.chat2_live_turns.get(chat_id)
    return {
        "chat": serialize_chat(chat, model_loaded=loaded, context=ctx),
        "messages": [serialize_message(m) for m in msgs],
        "attachments": [serialize_attachment(secret, a) for a in atts],
        "context": ctx,
        "live_turn": (
            None
            if live is None or live.done
            else {"request_id": live.request_id, "message_id": live.message_id}
        ),
    }


@router.patch("/chats/{chat_id}")
async def patch_chat(
    chat_id: str, body: ChatPatch, request: Request, user_id: int = Depends(current_user_id)
) -> dict[str, Any]:
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        repo = Chat2Repo(db)
        chat = await repo.get_chat(user_id, chat_id)
        if chat is None:
            raise api_error(404, "not_found", "chat not found")
        merged = body.settings.merge_into(chat.settings) if body.settings else None
        upd = await repo.update_chat(
            user_id,
            chat_id,
            title=body.title,
            title_source="user" if body.title is not None else None,
            # A user rename retires the auto-title history: `title_prev` points
            # at a title the machine picked, and offering to undo back to it
            # after the user has said what they want is nonsense.
            clear_title_prev=body.title is not None,
            model=body.model,
            settings=merged,
        )
        assert upd is not None
        out = await _serialize_described(settings, repo, upd)
    return out


async def _unlink_orphans(settings: Any, repo: Chat2Repo, user_id: int, atts: list[Any]) -> None:  # noqa: ANN401
    for a in atts:
        if await repo.refcount(user_id, a.sha256) == 0:
            unlink_quiet(
                file_path(settings.data_dir, user_id, a.sha256, ALLOWED_IMAGE_MIMES[a.mime])
            )


@router.delete("/chats/{chat_id}", status_code=204)
async def delete_chat(
    chat_id: str, request: Request, user_id: int = Depends(current_user_id)
) -> Response:
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        repo = Chat2Repo(db)
        atts = await repo.list_attachments(chat_id)
        if not await repo.delete_chat(user_id, chat_id):
            raise api_error(404, "not_found", "chat not found")
        await _unlink_orphans(settings, repo, user_id, atts)
    return Response(status_code=204)


@router.delete("/chats")
async def delete_all(request: Request, user_id: int = Depends(current_user_id)) -> dict[str, int]:
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        repo = Chat2Repo(db)
        atts: list[Any] = []
        for c in await repo.list_chats(user_id):
            atts += await repo.list_attachments(c.id)
        n = await repo.delete_all_chats(user_id)
        await _unlink_orphans(settings, repo, user_id, atts)
    return {"deleted": n}


@router.post("/chats/{chat_id}/fork", status_code=201)
async def fork_chat(
    chat_id: str, body: ForkBody, request: Request, user_id: int = Depends(current_user_id)
) -> dict[str, Any]:
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        repo = Chat2Repo(db)
        new = await repo.fork(user_id, chat_id, at_seq=body.at_seq, edited_text=body.edited_text)
        if new is None:
            raise api_error(404, "not_found", "chat not found")
        loaded = await _model_loaded(settings, new)
    return serialize_chat(new, model_loaded=loaded)


@router.get("/defaults")
async def get_defaults(request: Request, user_id: int = Depends(current_user_id)) -> dict[str, Any]:
    async with open_db(request.app.state.settings.db_path) as db:
        d = await Chat2Repo(db).get_defaults(user_id)
    model, s = d if d else (None, DEFAULT_SETTINGS)
    return {"model": model, "settings": s}


@router.put("/defaults")
async def put_defaults(
    body: DefaultsPut, request: Request, user_id: int = Depends(current_user_id)
) -> dict[str, Any]:
    async with open_db(request.app.state.settings.db_path) as db:
        repo = Chat2Repo(db)
        d = await repo.get_defaults(user_id)
        base = d[1] if d else DEFAULT_SETTINGS
        merged = body.settings.merge_into(base)
        await repo.put_defaults(user_id, model=body.model, settings=merged)
    return {"model": body.model, "settings": merged}


@router.get("/models")
async def list_models(request: Request, _uid: int = Depends(current_user_id)) -> dict[str, Any]:
    return {"models": [m.as_dict() for m in await catalog.list_catalog(request.app.state.settings)]}


def serialize_budget(d: BudgetDecision) -> dict[str, Any] | None:
    if d.allowed and d.blocked_until is None:
        return None
    w = d.window
    return {
        "blocked_until": d.blocked_until,
        "window": None
        if w is None
        else {
            "used_micros": w.used_micros,
            "limit_micros": w.limit_micros,
            "resets_at": w.resets_at,
        },
    }


@router.get("/budget")
async def get_budget(request: Request, user_id: int = Depends(current_user_id)) -> dict[str, Any]:
    policy = request.app.state.chat2_budget
    decision = await policy.check(user_id, None)
    return {"budget": serialize_budget(decision)}
