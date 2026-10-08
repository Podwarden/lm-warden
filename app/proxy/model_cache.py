"""The models table cached for a few seconds (issue #293).

Every /v1 request used to open a connection to read its model row
(``_resolve_target``), and so did ``/v1/models`` and the 5xx envelope hint. The
whole table is now cached as one snapshot for MODEL_CACHE_TTL_S: it is a few
dozen rows, so one ``list_all`` serves every lookup by id or served name.

Every write to ``models`` goes through ModelRepo, which calls invalidate_all()
after each commit, so a status change (loading -> loaded -> failed, unload), an
edit, a rename or a delete applies to the very next request in this process.
The 404/503 decision is therefore never taken from a superseded row. The TTL is
a backstop for anything that writes the table outside ModelRepo (a hand edit
with sqlite3); the supervisor is in-process, so there is no other worker whose
writes the generation cannot see.

Same race rule as app/proxy/token_cache.py: a caller snapshots generation()
BEFORE it reads and passes it to put(), so a write committing while the read is
in flight makes the snapshot dead on arrival. Lookups return deep copies (rows
carry lists and dicts) so a caller can never mutate the shared snapshot.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from app.db.repos.models import ModelRow

logger = logging.getLogger(__name__)

MODEL_CACHE_TTL_S = 5.0
MISS: Any = object()
_generation = 0


def generation() -> int:
    return _generation


def invalidate_all() -> None:
    global _generation
    _generation += 1


class _Snapshot:
    __slots__ = ("at", "by_id", "by_name", "gen", "rows")

    def __init__(self, rows: list[ModelRow], at: float, gen: int) -> None:
        self.rows = tuple(rows)
        self.by_id = {r.id: r for r in rows}
        self.by_name = {r.served_model_name: r for r in rows}
        self.at = at
        self.gen = gen


class ModelCache:
    def __init__(
        self,
        ttl_s: float = MODEL_CACHE_TTL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_s
        self._clock = clock
        self._snap: _Snapshot | None = None

    def _live(self) -> _Snapshot | None:
        snap = self._snap
        if snap is None:
            return None
        if snap.gen != _generation or self._clock() - snap.at > self._ttl:
            self._snap = None
            return None
        return snap

    def rows(self) -> Any:
        """Every row in ``created_at`` order (a copy), or MISS."""
        snap = self._live()
        return MISS if snap is None else copy.deepcopy(list(snap.rows))

    def by_id(self, model_id: str) -> Any:
        """The row (a copy), None when the snapshot has no such id, or MISS."""
        snap = self._live()
        return MISS if snap is None else copy.deepcopy(snap.by_id.get(model_id))

    def by_served_name(self, served: str) -> Any:
        snap = self._live()
        return MISS if snap is None else copy.deepcopy(snap.by_name.get(served))

    def put(self, rows: list[ModelRow], gen: int) -> None:
        self._snap = _Snapshot(rows, self._clock(), gen)


async def _read(cache: ModelCache | None, db_path: str | Path) -> list[ModelRow]:
    from app.db.database import open_db
    from app.db.repos.models import ModelRepo

    # Snapshot BEFORE the read: a write committing while it is in flight bumps
    # the generation, and the stale table is then dead on arrival.
    gen = generation()
    async with open_db(Path(db_path)) as db:
        rows = await ModelRepo(db).list_all()
    if cache is not None:
        try:
            cache.put(rows, gen)
        except Exception:  # noqa: BLE001 -- fail open: the read already succeeded
            logger.exception("model_cache: put failed")
    return rows


def _cached(cache: ModelCache | None, method: str, *args: str) -> Any:
    # Fail open: any cache trouble is a plain DB read.
    if cache is None:
        return MISS
    try:
        return getattr(cache, method)(*args)
    except Exception:  # noqa: BLE001
        logger.exception("model_cache: %s failed", method)
        return MISS


async def all_models(cache: ModelCache | None, db_path: str | Path) -> list[ModelRow]:
    hit = _cached(cache, "rows")
    return await _read(cache, db_path) if hit is MISS else cast("list[ModelRow]", hit)


async def model_by_id(
    cache: ModelCache | None, db_path: str | Path, model_id: str
) -> ModelRow | None:
    hit = _cached(cache, "by_id", model_id)
    if hit is not MISS:
        return cast("ModelRow | None", hit)
    return next((r for r in await _read(cache, db_path) if r.id == model_id), None)


async def model_by_served_name(
    cache: ModelCache | None, db_path: str | Path, served: str
) -> ModelRow | None:
    hit = _cached(cache, "by_served_name", served)
    if hit is not MISS:
        return cast("ModelRow | None", hit)
    rows = await _read(cache, db_path)
    return next((r for r in rows if r.served_model_name == served), None)
