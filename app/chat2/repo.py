from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import aiosqlite

from app.chat2.clock import now_iso
from app.chat2.ids import new_id

DEFAULT_SETTINGS: dict[str, Any] = {
    "temperature": 0.7,
    "max_tokens": 1024,
    "top_p": 1.0,
    "system_prompt": "",
    # `present_options` is the one client-fulfilled tool (routes_turn.CLIENT_TOOLS)
    # and the Tools checkbox in the settings panel toggles exactly this entry, so
    # a fresh chat has to start with it on for the checkbox to read as checked.
    "enabled_tools": ["present_options"],
    "enabled_skills": [],
    # Reasoning models burn most of a turn's budget on a preamble nobody asked
    # for. `False` sends vLLM `chat_template_kwargs.enable_thinking = False`;
    # `True` (the default) sends nothing at all, which is the only safe "on"
    # for chat templates that have never heard of the flag.
    "enable_thinking": True,
}

# `title_prev` (0026) is appended, not slotted next to `title`: the positional
# indices in `_chat_row` and the placeholder count in `create_chat` are pinned
# to this order, so the list is append-only.
_CHAT_COLS = (
    "id, user_id, org_id, title, title_source, model, settings_json, forked_from_chat_id, "
    "forked_at_seq, cost_micros_total, created_at, updated_at, last_message_at, title_prev"
)


@dataclass(frozen=True)
class ChatRow:
    id: str
    user_id: int
    org_id: str | None
    title: str
    title_source: str
    model: str | None
    settings: dict[str, Any]
    forked_from_chat_id: str | None
    forked_at_seq: int | None
    cost_micros_total: int
    created_at: str
    updated_at: str
    last_message_at: str | None
    # The title a generated one displaced, for the sidebar's undo affordance.
    # NULL until an auto-title reassessment actually renames the chat.
    title_prev: str | None = None


def _chat_row(r: Any) -> ChatRow:
    return ChatRow(
        id=r[0],
        user_id=r[1],
        org_id=r[2],
        title=r[3],
        title_source=r[4],
        model=r[5],
        settings=json.loads(r[6]),
        forked_from_chat_id=r[7],
        forked_at_seq=r[8],
        cost_micros_total=r[9],
        created_at=r[10],
        updated_at=r[11],
        last_message_at=r[12],
        title_prev=r[13],
    )


_MSG_COLS = (
    "id, chat_id, seq, role, parts_json, model, settings_snapshot_json, usage_json, "
    "finish_reason, error_json, created_at"
)


@dataclass(frozen=True)
class MessageRow:
    id: str
    chat_id: str
    seq: int
    role: str
    parts: list[dict[str, Any]]
    model: str | None
    settings_snapshot: dict[str, Any]
    usage: dict[str, Any] | None
    finish_reason: str | None
    error: dict[str, Any] | None
    created_at: str


def _msg_row(r: Any) -> MessageRow:
    return MessageRow(
        id=r[0],
        chat_id=r[1],
        seq=r[2],
        role=r[3],
        parts=json.loads(r[4]),
        model=r[5],
        settings_snapshot=json.loads(r[6]),
        usage=json.loads(r[7]) if r[7] else None,
        finish_reason=r[8],
        error=json.loads(r[9]) if r[9] else None,
        created_at=r[10],
    )


class _MessagesMixin:
    db: aiosqlite.Connection

    async def append_message(
        self,
        chat_id: str,
        *,
        role: str,
        parts: list[dict[str, Any]],
        settings_snapshot: dict[str, Any],
        model: str | None = None,
        usage: dict[str, Any] | None = None,
        finish_reason: str | None = None,
        error: dict[str, Any] | None = None,
        message_id: str | None = None,
    ) -> MessageRow:
        mid = message_id or new_id()
        await self.db.execute("BEGIN IMMEDIATE")
        try:
            cur = await self.db.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE chat_id = ?", (chat_id,)
            )
            row = await cur.fetchone()
            assert row is not None
            seq = row[0]
            await self.db.execute(
                f"INSERT INTO messages({_MSG_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    mid,
                    chat_id,
                    seq,
                    role,
                    json.dumps(parts),
                    model,
                    json.dumps(settings_snapshot),
                    json.dumps(usage) if usage else None,
                    finish_reason,
                    json.dumps(error) if error else None,
                    now_iso(),
                ),
            )
            await self.db.commit()
        except Exception:
            await self.db.rollback()
            raise
        return await self._get_message(mid)

    async def _get_message(self, message_id: str) -> MessageRow:
        cur = await self.db.execute(f"SELECT {_MSG_COLS} FROM messages WHERE id = ?", (message_id,))
        r = await cur.fetchone()
        assert r is not None
        return _msg_row(r)

    async def list_messages(self, chat_id: str) -> list[MessageRow]:
        cur = await self.db.execute(
            f"SELECT {_MSG_COLS} FROM messages WHERE chat_id = ? ORDER BY seq", (chat_id,)
        )
        return [_msg_row(r) for r in await cur.fetchall()]

    async def delete_tail_assistant(self, chat_id: str) -> int | None:
        """Delete the trailing assistant row, plus any tool rows appended after it.

        A no-op (returns None, deletes nothing) unless the tail of the chat is an
        assistant row or a run of tool rows immediately following one -- e.g. a new
        user message after the assistant reply means that turn is closed and must
        not be touched.
        """
        cur = await self.db.execute(
            "SELECT seq, role FROM messages WHERE chat_id = ? ORDER BY seq DESC", (chat_id,)
        )
        tail = await cur.fetchall()
        if not tail:
            return None
        assistant_seq: int | None = None
        for seq, role in tail:
            if role == "assistant":
                assistant_seq = int(seq)
                break
            if role != "tool":
                return None
        if assistant_seq is None:
            return None
        await self.db.execute(
            "DELETE FROM messages WHERE chat_id = ? AND seq >= ?", (chat_id, assistant_seq)
        )
        await self.db.commit()
        return assistant_seq

    async def set_options_answered(
        self, message_id: str, call_id: str, selected: list[str]
    ) -> None:
        row = await self._get_message(message_id)
        parts = row.parts
        for p in parts:
            if p.get("type") == "options" and p.get("callId") == call_id:
                p["answered"] = selected
        await self.db.execute(
            "UPDATE messages SET parts_json = ? WHERE id = ?", (json.dumps(parts), message_id)
        )
        await self.db.commit()

    async def fork(
        self, user_id: int, chat_id: str, *, at_seq: int, edited_text: str | None = None
    ) -> ChatRow | None:
        src = await self.get_chat(user_id, chat_id)  # type: ignore[attr-defined]
        if src is None:
            return None
        rows = await self.list_messages(chat_id)
        if edited_text is not None:
            keep = [r for r in rows if r.seq < at_seq]
        else:
            keep = [r for r in rows if r.seq <= at_seq]
        new = await self.create_chat(  # type: ignore[attr-defined]
            user_id,
            model=src.model,
            settings=src.settings,
            title=f"{src.title} (fork)",
            org_id=src.org_id,
            forked_from=(chat_id, at_seq),
        )
        try:
            att_map: dict[str, str] = {}
            for r in keep:
                parts = json.loads(json.dumps(r.parts))
                for p in parts:
                    if p.get("type") == "image" and p.get("attachmentId"):
                        old = p["attachmentId"]
                        if old not in att_map:
                            att_map[old] = await self._copy_attachment(old, new.id)
                        p["attachmentId"] = att_map[old]
                new_row = await self.append_message(
                    new.id,
                    role=r.role,
                    parts=parts,
                    settings_snapshot=r.settings_snapshot,
                    model=r.model,
                    usage=r.usage,
                    finish_reason=r.finish_reason,
                    error=r.error,
                )
                ids = [p["attachmentId"] for p in parts if p.get("type") == "image"]
                if ids:
                    await self.db.execute(
                        f"UPDATE attachments SET message_id = ? "
                        f"WHERE id IN ({','.join('?' * len(ids))})",
                        (new_row.id, *ids),
                    )
                    await self.db.commit()
            if edited_text is not None:
                await self.append_message(
                    new.id,
                    role="user",
                    parts=[{"type": "text", "text": edited_text}],
                    settings_snapshot=src.settings,
                )
            await self.touch_chat(new.id)  # type: ignore[attr-defined]
        except Exception:
            await self.delete_chat(user_id, new.id)  # type: ignore[attr-defined]
            raise
        return await self.get_chat(user_id, new.id)  # type: ignore[attr-defined,no-any-return]

    async def _copy_attachment(self, attachment_id: str, new_chat_id: str) -> str:
        """Copy an attachment row into the forked chat (same sha256 -> same file)."""
        cur = await self.db.execute(
            "SELECT user_id, org_id, kind, mime, size_bytes, sha256, width, height, evicted_at "
            "FROM attachments WHERE id = ?",
            (attachment_id,),
        )
        r = await cur.fetchone()
        if r is None:
            return attachment_id
        nid = new_id()
        await self.db.execute(
            "INSERT INTO attachments(id, user_id, org_id, chat_id, message_id, kind, mime, "
            "size_bytes, sha256, width, height, created_at, evicted_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                nid,
                r[0],
                r[1],
                new_chat_id,
                None,
                r[2],
                r[3],
                r[4],
                r[5],
                r[6],
                r[7],
                now_iso(),
                r[8],
            ),
        )
        await self.db.commit()
        return nid


_ATT_COLS = (
    "id, user_id, org_id, chat_id, message_id, kind, mime, size_bytes, sha256, width, height, "
    "created_at, evicted_at"
)


@dataclass(frozen=True)
class AttachmentRow:
    id: str
    user_id: int
    org_id: str | None
    chat_id: str | None
    message_id: str | None
    kind: str
    mime: str
    size_bytes: int
    sha256: str
    width: int | None
    height: int | None
    created_at: str
    evicted_at: str | None


def _att_row(r: Any) -> AttachmentRow:
    return AttachmentRow(*r)


class _AttachmentsMixin:
    db: aiosqlite.Connection

    async def create_attachment(
        self, user_id: int, *, chat_id: str, stored: Any, org_id: str | None = None
    ) -> AttachmentRow:
        aid = new_id()
        await self.db.execute(
            f"INSERT INTO attachments({_ATT_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                aid,
                user_id,
                org_id,
                chat_id,
                None,
                "image",
                stored.mime,
                stored.size_bytes,
                stored.sha256,
                stored.width,
                stored.height,
                now_iso(),
                None,
            ),
        )
        await self.db.commit()
        row = await self.get_attachment(user_id, aid)
        assert row is not None
        return row

    async def get_attachment(self, user_id: int, attachment_id: str) -> AttachmentRow | None:
        cur = await self.db.execute(
            f"SELECT {_ATT_COLS} FROM attachments WHERE user_id = ? AND id = ?",
            (user_id, attachment_id),
        )
        r = await cur.fetchone()
        return _att_row(r) if r else None

    async def list_attachments(self, chat_id: str) -> list[AttachmentRow]:
        cur = await self.db.execute(
            f"SELECT {_ATT_COLS} FROM attachments WHERE chat_id = ? ORDER BY created_at",
            (chat_id,),
        )
        return [_att_row(r) for r in await cur.fetchall()]

    async def link_attachments(
        self, user_id: int, chat_id: str, message_id: str, ids: list[str]
    ) -> list[AttachmentRow]:
        out: list[AttachmentRow] = []
        for aid in ids:
            row = await self.get_attachment(user_id, aid)
            if row is None or row.chat_id != chat_id or row.message_id is not None:
                raise KeyError(aid)
            await self.db.execute(
                "UPDATE attachments SET message_id = ? WHERE id = ?", (message_id, aid)
            )
            out.append(row)
        await self.db.commit()
        return out

    async def delete_draft(self, user_id: int, attachment_id: str) -> AttachmentRow | None:
        row = await self.get_attachment(user_id, attachment_id)
        if row is None or row.message_id is not None:
            return None
        await self.db.execute("DELETE FROM attachments WHERE id = ?", (attachment_id,))
        await self.db.commit()
        return row

    async def user_bytes(self, user_id: int) -> int:
        cur = await self.db.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) FROM (SELECT DISTINCT sha256, size_bytes "
            "FROM attachments WHERE user_id = ? AND evicted_at IS NULL)",
            (user_id,),
        )
        row = await cur.fetchone()
        assert row is not None
        return int(row[0])

    async def refcount(self, user_id: int, sha256: str) -> int:
        cur = await self.db.execute(
            "SELECT COUNT(*) FROM attachments WHERE user_id = ? AND sha256 = ? "
            "AND evicted_at IS NULL",
            (user_id, sha256),
        )
        row = await cur.fetchone()
        assert row is not None
        return int(row[0])


class Chat2Repo(_MessagesMixin, _AttachmentsMixin):
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    # ---- chats -------------------------------------------------------------
    async def create_chat(
        self,
        user_id: int,
        *,
        model: str | None,
        settings: dict[str, Any],
        title: str = "New chat",
        org_id: str | None = None,
        forked_from: tuple[str, int] | None = None,
    ) -> ChatRow:
        cid, ts = new_id(), now_iso()
        ffc, ffs = forked_from if forked_from else (None, None)
        await self.db.execute(
            f"INSERT INTO chats({_CHAT_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                cid,
                user_id,
                org_id,
                title,
                "auto",
                model,
                json.dumps(settings),
                ffc,
                ffs,
                0,
                ts,
                ts,
                None,
                None,
            ),
        )
        await self.db.commit()
        chat = await self.get_chat(user_id, cid)
        assert chat is not None
        return chat

    async def list_chats(self, user_id: int) -> list[ChatRow]:
        cur = await self.db.execute(
            f"SELECT {_CHAT_COLS} FROM chats WHERE user_id = ? "
            "ORDER BY COALESCE(last_message_at, created_at) DESC",
            (user_id,),
        )
        return [_chat_row(r) for r in await cur.fetchall()]

    async def get_chat(self, user_id: int, chat_id: str) -> ChatRow | None:
        cur = await self.db.execute(
            f"SELECT {_CHAT_COLS} FROM chats WHERE user_id = ? AND id = ?", (user_id, chat_id)
        )
        r = await cur.fetchone()
        return _chat_row(r) if r else None

    async def update_chat(
        self,
        user_id: int,
        chat_id: str,
        *,
        title: str | None = None,
        title_source: str | None = None,
        title_prev: str | None = None,
        clear_title_prev: bool = False,
        model: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> ChatRow | None:
        sets: list[str] = ["updated_at = ?"]
        vals: list[Any] = [now_iso()]
        if title is not None:
            sets.append("title = ?")
            vals.append(title)
        if title_prev is not None:
            sets.append("title_prev = ?")
            vals.append(title_prev)
        elif clear_title_prev:
            # `title_prev=None` means "leave it alone" like every other field
            # here, so retiring the undo target needs its own explicit flag.
            sets.append("title_prev = NULL")
        if title_source is not None:
            sets.append("title_source = ?")
            vals.append(title_source)
        if model is not None:
            sets.append("model = ?")
            vals.append(model)
        if settings is not None:
            sets.append("settings_json = ?")
            vals.append(json.dumps(settings))
        vals += [user_id, chat_id]
        cur = await self.db.execute(
            f"UPDATE chats SET {', '.join(sets)} WHERE user_id = ? AND id = ?", vals
        )
        await self.db.commit()
        if cur.rowcount == 0:
            return None
        return await self.get_chat(user_id, chat_id)

    async def replace_auto_title(
        self, user_id: int, chat_id: str, *, title: str, keep_prev: bool
    ) -> bool:
        """Swap a still-auto-titled chat's title in a single statement.

        The title task decides to rename, then writes; because the generation
        call sits at the lowest proxy priority, the gap between those two is
        seconds wide, and a user rename can land inside it. Re-reading
        `title_source` before the write narrows that window but cannot close
        it -- the `AND title_source = 'auto'` predicate does, and the boolean
        return lets the caller see when the user won.

        `title_prev = title` is evaluated against the row's PRE-update values
        (standard SQL, and SQLite's behaviour), so the displaced title is
        captured atomically rather than from a stale read. `keep_prev=False`
        skips it: the first LLM title replaces the first-line fallback, which
        is not a title anyone chose and makes a poor undo target.
        """
        prev = "title_prev = title, " if keep_prev else ""
        cur = await self.db.execute(
            f"UPDATE chats SET {prev}title = ?, updated_at = ? "
            "WHERE user_id = ? AND id = ? AND title_source = 'auto'",
            (title, now_iso(), user_id, chat_id),
        )
        await self.db.commit()
        return (cur.rowcount or 0) > 0

    async def touch_chat(self, chat_id: str, *, add_cost_micros: int = 0) -> None:
        ts = now_iso()
        await self.db.execute(
            "UPDATE chats SET updated_at = ?, last_message_at = ?, "
            "cost_micros_total = cost_micros_total + ? WHERE id = ?",
            (ts, ts, add_cost_micros, chat_id),
        )
        await self.db.commit()

    async def delete_chat(self, user_id: int, chat_id: str) -> bool:
        cur = await self.db.execute(
            "DELETE FROM chats WHERE user_id = ? AND id = ?", (user_id, chat_id)
        )
        await self.db.commit()
        return (cur.rowcount or 0) > 0

    async def delete_all_chats(self, user_id: int) -> int:
        cur = await self.db.execute("DELETE FROM chats WHERE user_id = ?", (user_id,))
        await self.db.commit()
        return cur.rowcount or 0

    # ---- defaults ----------------------------------------------------------
    async def get_defaults(self, user_id: int) -> tuple[str | None, dict[str, Any]] | None:
        cur = await self.db.execute(
            "SELECT model, settings_json FROM user_chat_defaults WHERE user_id = ?", (user_id,)
        )
        r = await cur.fetchone()
        return (r[0], json.loads(r[1])) if r else None

    async def put_defaults(
        self, user_id: int, *, model: str | None, settings: dict[str, Any]
    ) -> None:
        await self.db.execute(
            "INSERT INTO user_chat_defaults(user_id, model, settings_json, updated_at) "
            "VALUES (?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET model = excluded.model, "
            "settings_json = excluded.settings_json, updated_at = excluded.updated_at",
            (user_id, model, json.dumps(settings), now_iso()),
        )
        await self.db.commit()
