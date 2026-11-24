"""Model aliases in the proxy (D4a-D4e).

An enabled alias wins over a model with the same served name (D4a); a request
is routed to the first LOADED target in the alias's ordered list (D4b). A
disabled alias is invisible: it never shadows a model and never routes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from app.proxy.model_cache import alias_by_name

if TYPE_CHECKING:
    from app.db.repos.model_aliases import AliasRow
    from app.db.repos.models import ModelRow
    from app.proxy.model_cache import ModelCache


async def resolve_alias(
    cache: ModelCache | None, db_path: str | Path, name: str
) -> AliasRow | None:
    """The enabled alias named ``name`` (a copy), or None when absent/disabled."""
    alias = await alias_by_name(cache, db_path, name)
    if alias is None or not alias.enabled:
        return None
    return alias


def first_loaded_target(alias: AliasRow, models: list[ModelRow]) -> ModelRow | None:
    """The first target in the alias's ordered list whose row is loaded."""
    by_id = {m.id: m for m in models}
    for target_id in alias.targets:
        row = by_id.get(target_id)
        if row is not None and row.status == "loaded":
            return row
    return None
