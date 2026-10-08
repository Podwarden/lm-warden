"""Merged template accessor: built-in presets (registry) + user templates (DB).

The rest of the app calls ``list_templates(db)`` / ``get_template(db, id)`` and
is agnostic to origin (spec D3). User templates are JSON blobs in the
``engine_templates`` table; built-ins are code-defined and read-only.
"""

from __future__ import annotations

import json

import aiosqlite

from app.templates import registry
from app.templates.registry import ModelTemplate


async def save_user_template(db: aiosqlite.Connection, t: ModelTemplate) -> None:
    payload = json.dumps(registry.template_to_dict(t))
    await db.execute(
        "INSERT INTO engine_templates(id, label, payload, source) "
        "VALUES (?,?,?,'user') "
        "ON CONFLICT(id) DO UPDATE SET label=excluded.label, "
        "payload=excluded.payload",
        (t.id, t.label, payload),
    )
    await db.commit()


async def list_user_templates(db: aiosqlite.Connection) -> list[ModelTemplate]:
    cur = await db.execute("SELECT payload FROM engine_templates ORDER BY created_at")
    return [registry.template_from_dict(json.loads(r[0])) for r in await cur.fetchall()]


async def list_templates(db: aiosqlite.Connection) -> list[ModelTemplate]:
    return registry.list_builtin_templates() + await list_user_templates(db)


async def get_template(db: aiosqlite.Connection, template_id: str) -> ModelTemplate | None:
    builtin = registry.get_builtin_template(template_id)
    if builtin is not None:
        return builtin
    cur = await db.execute("SELECT payload FROM engine_templates WHERE id = ?", (template_id,))
    row = await cur.fetchone()
    return registry.template_from_dict(json.loads(row[0])) if row else None


async def delete_user_template(db: aiosqlite.Connection, template_id: str) -> None:
    if registry.get_builtin_template(template_id) is not None:
        raise ValueError(f"template '{template_id}' is built-in and cannot be deleted")
    await db.execute("DELETE FROM engine_templates WHERE id = ?", (template_id,))
    await db.commit()
