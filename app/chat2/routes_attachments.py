from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from starlette.concurrency import run_in_threadpool

from app.chat2 import storage
from app.chat2.errors import api_error
from app.chat2.identity import current_user_id
from app.chat2.limits import (
    ALLOWED_IMAGE_MIMES,
    MAX_IMAGE_BYTES,
    MULTIPART_OVERHEAD_BYTES,
    SIGNED_URL_TTL_S,
)
from app.chat2.quota import QuotaExceeded, check_quota
from app.chat2.repo import Chat2Repo
from app.chat2.signing import attachment_url, verify_attachment
from app.db.database import open_db

router = APIRouter(prefix="/api/chat2/attachments", tags=["chat2"])


def _serialize(secret: str, row: Any) -> dict[str, Any]:
    return {
        "id": row.id,
        "chat_id": row.chat_id,
        "message_id": row.message_id,
        "kind": row.kind,
        "mime": row.mime,
        "size_bytes": row.size_bytes,
        "sha256": row.sha256,
        "width": row.width,
        "height": row.height,
        "evicted": row.evicted_at is not None,
        "url": None
        if row.evicted_at
        else attachment_url(secret, row.user_id, row.id, SIGNED_URL_TTL_S),
    }


def _reject_oversized_content_length(request: Request) -> None:
    """Reject a declared Content-Length over MAX_IMAGE_BYTES + overhead.

    Defence-in-depth only, NOT the fast path: by the time this route
    function runs, FastAPI has already awaited ``Request.form()`` to
    resolve the ``file``/``chat_id`` parameters, so the full request body
    has already been read (and, for a large upload, already spooled to
    memory/disk) regardless of what this check decides. The actual
    fast-path rejection -- which never reads the body at all -- is
    ``Chat2BodyLimitMiddleware`` (app/chat2/body_limit.py), registered
    ahead of routing in ``app.main.build_app``. This check is kept as a
    second line of defense in case that middleware is ever removed,
    bypassed, or misconfigured.
    """
    raw_header = request.headers.get("content-length")
    if raw_header is None:
        return
    try:
        declared = int(raw_header)
    except ValueError:
        return
    if declared > MAX_IMAGE_BYTES + MULTIPART_OVERHEAD_BYTES:
        # Same envelope Chat2BodyLimitMiddleware writes by hand -- keep them identical.
        raise api_error(413, "too_large", f"request body larger than {MAX_IMAGE_BYTES} bytes")


@router.post("", status_code=201)
async def upload(
    request: Request,
    file: UploadFile = File(...),
    chat_id: str = Form(...),
    user_id: int = Depends(current_user_id),
) -> dict[str, Any]:
    settings = request.app.state.settings
    _reject_oversized_content_length(request)
    raw = await file.read(MAX_IMAGE_BYTES + 1)
    if len(raw) > MAX_IMAGE_BYTES:
        raise api_error(413, "too_large", f"raw image larger than {MAX_IMAGE_BYTES} bytes")
    try:
        # CPU-bound Pillow decode/re-encode; offload so it doesn't block the
        # single-worker event loop for other in-flight requests.
        stored, data = await run_in_threadpool(storage.reencode_image, raw)
    except storage.ImageTooLarge as exc:
        # `exc` says WHICH size blew the limit: the raw upload or the re-encode.
        raise api_error(413, "too_large", str(exc)) from exc
    except storage.UnsupportedImage as exc:
        raise api_error(415, "unsupported_media", str(exc)) from exc
    async with open_db(settings.db_path) as db:
        repo = Chat2Repo(db)
        if await repo.get_chat(user_id, chat_id) is None:
            raise api_error(404, "not_found", "chat not found")
        if await repo.refcount(user_id, stored.sha256) == 0:
            try:
                await check_quota(repo, settings, user_id, stored.size_bytes)
            except QuotaExceeded as exc:
                raise api_error(413, "quota_exceeded", str(exc)) from exc
            await run_in_threadpool(
                storage.write_file,
                storage.file_path(settings.data_dir, user_id, stored.sha256, stored.ext),
                data,
            )
        row = await repo.create_attachment(user_id, chat_id=chat_id, stored=stored)
    return _serialize(request.app.state.jwt_secret, row)


@router.get("/{attachment_id}")
async def serve(attachment_id: str, request: Request, t: str = Query(...)) -> Response:
    settings = request.app.state.settings
    try:
        user_id = int(t.split(".", 1)[0])
    except ValueError:
        raise api_error(401, "bad_token", "malformed signed-url token") from None
    if not verify_attachment(
        request.app.state.jwt_secret, user_id, attachment_id, t, int(time.time())
    ):
        raise api_error(401, "bad_token", "bad or expired signed-url token")
    async with open_db(settings.db_path) as db:
        row = await Chat2Repo(db).get_attachment(user_id, attachment_id)
    if row is None or row.evicted_at is not None:
        raise api_error(404, "not_found", "attachment not found")
    path = storage.file_path(settings.data_dir, user_id, row.sha256, ALLOWED_IMAGE_MIMES[row.mime])
    if not path.exists():
        raise api_error(404, "not_found", "attachment file is missing")
    return FileResponse(
        path,
        media_type=row.mime,
        headers={
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": "inline",
            "Cache-Control": "private, max-age=600",
            "Content-Security-Policy": "sandbox",
        },
    )


@router.delete("/{attachment_id}", status_code=204)
async def delete_draft(
    attachment_id: str, request: Request, user_id: int = Depends(current_user_id)
) -> Response:
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        repo = Chat2Repo(db)
        row = await repo.delete_draft(user_id, attachment_id)
        if row is None:
            raise api_error(404, "not_found", "draft attachment not found")
        remaining = await repo.refcount(user_id, row.sha256)
    if remaining == 0:
        storage.unlink_quiet(
            storage.file_path(settings.data_dir, user_id, row.sha256, ALLOWED_IMAGE_MIMES[row.mime])
        )
    return Response(status_code=204)
