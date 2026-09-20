import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import aiosqlite

from app.runtime.backends.registry import DEFAULT_BACKEND

#: Columns persisted as a JSON document in a TEXT column. One constant, because
#: ``insert`` and ``update_fields`` both have to agree with ``_decode_row``.
_JSON_COLUMNS = frozenset({"gpu_indices", "extra_args", "extra_env"})


@dataclass
class ModelRow:
    id: str
    served_model_name: str
    hf_repo: str
    hf_revision: str
    gpu_indices: list[int]
    tensor_parallel_size: int
    dtype: str | None
    max_model_len: int | None
    gpu_memory_utilization: float
    trust_remote_code: bool
    extra_args: list[str]
    status: str
    pulled_bytes: int
    pulled_total: int | None
    last_error: str | None
    extra_env: dict[str, str]
    # New for #85 — see migration 0014. ``filename`` is None for legacy / whole-repo
    # pulls. ``parallelism_strategy`` and ``max_batch_size`` have defaults that
    # mirror the pre-#85 wizard's implicit behaviour so existing rows decode
    # to a sensible shape.
    filename: str | None = None
    parallelism_strategy: str = "auto"
    max_batch_size: int = 1
    # New for #106 — see migration 0015. ``hf_config_repo`` populates the
    # vLLM ``--hf-config-path`` flag for GGUF repos that omit ``config.json``
    # (common for unsloth republishes). ``tokenizer_repo`` populates
    # ``--tokenizer`` for the same upstream-vs-quant split. Both default to
    # None so non-GGUF rows and self-contained GGUF repos behave identically
    # to v17.17.
    hf_config_repo: str | None = None
    tokenizer_repo: str | None = None
    # Added for #114 — exposed on every SELECT so the cache-management GC
    # sweep can decide whether a ``status=failed`` row is "stale enough"
    # to collect WITHOUT an N+1 follow-up per row. Defaults to None for
    # ergonomic construction in tests; production reads from SQLite always
    # populate it (the column is NOT NULL with a ``datetime('now')`` default).
    updated_at: str | None = None
    # New for #162 — see migration 0022. Per-model engine axis. None on legacy
    # rows means the supervisor falls back to the in-container engine.
    engine_channel: str | None = None
    engine_vllm_version: str | None = None
    engine_image: str | None = None
    # New for migration 0024. The status that was interrupted when a serving
    # engine died -- the machine-readable half of what last_error used to
    # carry alone. Set on the crash path, cleared once the model is serving
    # again or an operator unloads it. The restart sweep keys on this.
    prior_status: str | None = None
    # New for migration 0025 (chat2). Capability flags the chat2 catalog
    # reads to decide what a served model can do -- tool calling, image
    # input, and reasoning/thinking output.
    #
    # TRI-state, and the third state is load-bearing: 1/0 is an operator's
    # explicit answer (set via PATCH /api/models/{id}/settings, which accepts
    # true/false/null and is exempt from that endpoint's unload-first 409),
    # while NULL means nobody has stated one. app/chat2/catalog.py auto-detects
    # `supports_vision` from the on-disk HF config, and `supports_reasoning`
    # from the on-disk chat template (True when it references
    # `enable_thinking`), only while each is NULL -- collapsing NULL into 0
    # here is what silently served every pasted image as "[image omitted]" on
    # a model that could read it (#106), and what hid the "Enable thinking"
    # toggle on every reasoning model until an operator found and set the flag
    # by hand (#239). `supports_tools` has no auto-detection and reads as
    # False when NULL.
    #
    # Decode these RAW (never `bool(...)`): the distinction is the point.
    supports_tools: int | None = None
    supports_vision: int | None = None
    supports_reasoning: int | None = None
    # New for migration 0027 (sub-project B). Which backend serves this model.
    # NULL on every pre-B row and decoded to 'vllm' by _decode_row via
    # app.runtime.backends.registry -- decision D6, no backfill.
    backend: str | None = None
    # New for migration 0028 (sub-project C). Both NULL on every pre-C row and
    # on every vLLM row. See the migration for why n_gpu_layers=None means
    # "let mainline choose" rather than "zero layers on the GPU".
    mmproj_filename: str | None = None
    n_gpu_layers: int | None = None


# Column list shared by insert + every SELECT so we can't drift them.
# ``updated_at`` is appended at the end so the column-index map in
# ``_decode_row`` stays append-only — never reorder, the indices below
# pin the layout.
_MODEL_COLS = (
    "id, served_model_name, hf_repo, hf_revision, gpu_indices, "
    "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
    "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error, "
    "extra_env, filename, parallelism_strategy, max_batch_size, "
    "hf_config_repo, tokenizer_repo, updated_at, "
    "engine_channel, engine_vllm_version, engine_image, prior_status, "
    "supports_tools, supports_vision, supports_reasoning, backend, "
    "mmproj_filename, n_gpu_layers"
)


def _decode_row(row: tuple) -> ModelRow:
    return ModelRow(
        id=row[0], served_model_name=row[1], hf_repo=row[2], hf_revision=row[3],
        gpu_indices=json.loads(row[4]), tensor_parallel_size=row[5], dtype=row[6],
        max_model_len=row[7], gpu_memory_utilization=row[8],
        trust_remote_code=bool(row[9]), extra_args=json.loads(row[10]), status=row[11],
        pulled_bytes=row[12], pulled_total=row[13], last_error=row[14],
        extra_env=json.loads(row[15]),
        filename=row[16],
        parallelism_strategy=row[17] if row[17] is not None else "auto",
        max_batch_size=row[18] if row[18] is not None else 1,
        hf_config_repo=row[19],
        tokenizer_repo=row[20],
        updated_at=row[21],
        engine_channel=row[22],
        engine_vllm_version=row[23],
        engine_image=row[24],
        prior_status=row[25],
        supports_tools=row[26],
        supports_vision=row[27],
        supports_reasoning=row[28],
        # D6: NULL means vLLM. registry.get owns that default so it lives in
        # exactly one place when a second backend lands.
        backend=row[29] or DEFAULT_BACKEND,
        # Migration 0028. NULL on every row that is not a llama.cpp vision
        # model / does not want partial CPU offload -- see the migration.
        mmproj_filename=row[30],
        n_gpu_layers=row[31],
    )


class ModelRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def insert(self, row: ModelRow) -> None:
        await self.db.execute(
            """INSERT INTO models(
                id, served_model_name, hf_repo, hf_revision, gpu_indices,
                tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization,
                trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error,
                extra_env, filename, parallelism_strategy, max_batch_size,
                hf_config_repo, tokenizer_repo,
                engine_channel, engine_vllm_version, engine_image,
                backend, mmproj_filename, n_gpu_layers
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                row.id, row.served_model_name, row.hf_repo, row.hf_revision,
                json.dumps(row.gpu_indices),
                row.tensor_parallel_size, row.dtype, row.max_model_len, row.gpu_memory_utilization,
                int(row.trust_remote_code), json.dumps(row.extra_args), row.status, row.pulled_bytes,
                row.pulled_total, row.last_error, json.dumps(row.extra_env),
                row.filename, row.parallelism_strategy, row.max_batch_size,
                row.hf_config_repo, row.tokenizer_repo,
                row.engine_channel, row.engine_vllm_version, row.engine_image,
                row.backend, row.mmproj_filename, row.n_gpu_layers,
            ),
        )
        await self.db.commit()

    async def get(self, model_id: str) -> ModelRow | None:
        cur = await self.db.execute(
            f"SELECT {_MODEL_COLS} FROM models WHERE id = ?",
            (model_id,),
        )
        row = await cur.fetchone()
        if not row:
            return None
        return _decode_row(row)

    async def list_all(self) -> list[ModelRow]:
        cur = await self.db.execute(
            f"SELECT {_MODEL_COLS} FROM models ORDER BY created_at"
        )
        rows = await cur.fetchall()
        return [_decode_row(r) for r in rows]

    async def list_by_repo(self, hf_repo: str) -> list[ModelRow]:
        """Return every row whose ``hf_repo`` exactly matches ``hf_repo``.

        Used by the HF cache management API (vllm-warden#114) to decide
        whether a candidate cache directory is currently owned by one or
        more live model rows. The same ``hf_repo`` can legitimately back
        multiple rows (different served_model_names, different GPU
        mappings), so this MUST return a list and not an Optional.
        Empty list means the cache dir is orphaned and safe to delete.
        """
        cur = await self.db.execute(
            f"SELECT {_MODEL_COLS} FROM models WHERE hf_repo = ? ORDER BY created_at",
            (hf_repo,),
        )
        rows = await cur.fetchall()
        return [_decode_row(r) for r in rows]

    async def updated_at(self, model_id: str) -> str | None:
        """Return the ``updated_at`` timestamp (ISO text) for a row, or None.

        Pulled out so the GC sweep in app/cache/routes_api.py can decide
        whether a status=failed row is "stale enough" to garbage collect
        its cache without exposing the freshness signal to non-cache
        callers.
        """
        cur = await self.db.execute(
            "SELECT updated_at FROM models WHERE id = ?", (model_id,)
        )
        row = await cur.fetchone()
        return row[0] if row else None

    async def get_by_served_name(self, served: str) -> ModelRow | None:
        cur = await self.db.execute("SELECT id FROM models WHERE served_model_name = ?", (served,))
        row = await cur.fetchone()
        return await self.get(row[0]) if row else None

    async def update_status(
        self, model_id: str, status: str, last_error: str | None = None
    ) -> None:
        """Set status/last_error, and retire ``prior_status`` once it is spent.

        ``prior_status`` is the machine-readable "was serving when it died"
        flag recovery keys on. It must be cleared the moment it stops being
        true, or a model an operator deliberately unloaded would be
        resurrected on the next watchdog pass. Anything that is not a
        failure clears it: reaching ``loaded`` means recovery succeeded,
        and ``pulled``/``idle`` means a human took it down on purpose.
        ``failed`` deliberately preserves it — that is the state the
        restart sweep is looking for.
        """
        if status == "failed":
            await self.db.execute(
                "UPDATE models SET status = ?, last_error = ?, "
                "updated_at = datetime('now') WHERE id = ?",
                (status, last_error, model_id),
            )
        else:
            await self.db.execute(
                "UPDATE models SET status = ?, last_error = ?, prior_status = NULL, "
                "updated_at = datetime('now') WHERE id = ?",
                (status, last_error, model_id),
            )
        await self.db.commit()

    async def set_prior_status(self, model_id: str, prior: str | None) -> None:
        """Record which status was interrupted, independent of ``last_error``.

        Kept separate from ``update_status`` because the crash path writes the
        two at different moments and for different audiences: ``last_error`` is
        for a human reading the UI, ``prior_status`` is for the watchdog
        deciding whether to restart. Coupling them is the bug this column
        exists to remove.
        """
        await self.db.execute(
            "UPDATE models SET prior_status = ? WHERE id = ?", (prior, model_id)
        )
        await self.db.commit()

    async def update_extra_fields(
        self, model_id: str, *, extra_env: dict[str, str], extra_args: list[str]
    ) -> None:
        """Overwrite ``extra_env``/``extra_args`` verbatim. #266 boot reconcile.

        The only caller today is ``app.runtime.boot_reconcile`` cleaning up a
        row poisoned by a settings-PATCH door that (before tonight's fix)
        accepted a hard-locked ``extra_env`` key, or either column in a shape
        ``POST /api/models`` would have refused (not ``dict[str, str]`` /
        ``list[str]``). Writes both columns unconditionally -- the caller
        always has both values in hand (unchanged ones round-trip through
        unmodified) -- rather than building a partial SET clause for two
        columns that are only ever touched together here.
        """
        await self.db.execute(
            "UPDATE models SET extra_env = ?, extra_args = ?, "
            "updated_at = datetime('now') WHERE id = ?",
            (json.dumps(extra_env), json.dumps(extra_args), model_id),
        )
        await self.db.commit()

    async def update_pull_progress(
        self, model_id: str, pulled_bytes: int, pulled_total: int | None
    ) -> None:
        await self.db.execute(
            "UPDATE models SET pulled_bytes = ?, pulled_total = ?, updated_at = datetime('now') "
            "WHERE id = ?",
            (pulled_bytes, pulled_total, model_id),
        )
        await self.db.commit()

    async def update_fields(self, model_id: str, values: Mapping[str, Any]) -> None:
        """Write operator-writable columns by name.

        **Not a public write path.** Callers go through
        ``app/models/writer.py::apply_model_change``, which is where the per-
        column policy, the principal check, the loaded guard, the full-row
        ``ModelSpec`` validation and the cross-row checks live. This method only
        turns a validated mapping into SQL, which is why it asserts rather than
        raising something an HTTP layer could answer with: a violation is a
        programming error inside the writer, not a bad request.

        The assert is cheap and it names the invariant the raw ``UPDATE models``
        statements this method replaced could not state -- one in
        ``patch_model_settings``, one in ``try_stack`` and one in stress-apply,
        each with its own idea of what a value should look like (#262, #265).
        """
        # Imported here, not at module scope: app/models/policy.py reads
        # ``ModelRow`` from this module to prove its table is complete.
        from app.models.policy import OPERATOR_FIELDS

        assert set(values) <= OPERATOR_FIELDS, (
            f"update_fields may only write operator columns; "
            f"got {sorted(set(values) - OPERATOR_FIELDS)}"
        )
        if not values:
            return
        columns = sorted(values)
        params: list[Any] = []
        for column in columns:
            value = values[column]
            if column in _JSON_COLUMNS:
                params.append(json.dumps(value))
            elif column == "trust_remote_code":
                params.append(int(bool(value)))
            else:
                params.append(value)
        params.append(model_id)
        await self.db.execute(
            f"UPDATE models SET {', '.join(f'{c} = ?' for c in columns)}, "
            f"updated_at = datetime('now') WHERE id = ?",
            tuple(params),
        )
        await self.db.commit()

    async def delete(self, model_id: str) -> None:
        await self.db.execute("DELETE FROM models WHERE id = ?", (model_id,))
        await self.db.commit()

    async def mark_runtime_dead_on_startup(self) -> int:
        """Wipe any row whose status presumes a live backing process to
        ``failed`` on app startup. Returns count updated.

        Status set wiped: ``loaded``, ``loading``, ``unloading`` (live
        vLLM subprocess) and ``pulling`` (live HF download task). After a
        warden restart NONE of those processes exists — the supervisor
        and pull-task state lives in-process only — so leaving any of
        those statuses in the DB strands the row in a state with no
        operator-actionable affordance (#11).

        For ``pulling`` rows we additionally zero ``pulled_bytes`` and
        ``pulled_total`` so the UI does not show stale progress for a
        pull that no longer has a backing task. Re-pulling appends to a
        fresh counter from byte 0 (the pull-task does not resume — it
        re-fetches), so the zeroed counters reflect reality. Wiping
        progress on non-``pulling`` rows would be wrong (e.g. a
        ``loaded`` row's ``pulled_total`` is the persisted weights
        size); the CASE expression below guards that.
        """
        cur = await self.db.execute(
            "UPDATE models SET status = 'failed', "
            # Record WHICH prior status was interrupted. Without it the watchdog
            # cannot tell a model that was serving (restore it) from one that was
            # merely mid-pull (do not).
            #
            # This now goes in its OWN column. It used to be encoded into
            # last_error and matched back as an exact string, which meant any
            # diagnosis written on the crash path displaced it and silently
            # disabled recovery -- see migration 0024. last_error keeps the
            # human-readable sentence; prior_status is what the watchdog reads.
            "prior_status = status, "
            "last_error = 'process not running after restart (was ' "
            "              || status || ')', "
            "pulled_bytes = CASE WHEN status = 'pulling' THEN 0 "
            "                    ELSE pulled_bytes END, "
            "pulled_total = CASE WHEN status = 'pulling' THEN 0 "
            "                    ELSE pulled_total END, "
            "updated_at = datetime('now') "
            "WHERE status IN ('loaded', 'loading', 'unloading', 'pulling')"
        )
        await self.db.commit()
        return cur.rowcount
