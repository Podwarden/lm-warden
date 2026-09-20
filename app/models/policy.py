"""Per-column write policy for ``ModelRow`` — who may write each column.

The ``ModelRow`` is what the runtime trusts: ``backend.plan(row)`` turns it into
argv and env, ``Supervisor._resolve_engine_image(row)`` turns it into a container
image, the watchdog re-reads it and restarts it with nobody at the keyboard, and
the proxy reads ``row.trust_remote_code`` / ``row.tokenizer_repo`` on the hot
path. Until this module existed, every rule about what may be *on* a row was
attached to a *request* instead — the full ruleset in ``ModelCreate``, a thinner
copy in ``TemplateCreate``, a hand-written subset in ``patch_model_settings``,
and none at all in ``try_stack`` and stress-apply. A new column was therefore
HTTP-writable by default, because the settings PATCH derived its allowlist from
``dataclasses.fields(ModelRow)`` minus a blocklist someone had to remember to
extend (#110's fix, and how ``trust_remote_code`` and ``prior_status`` arrived on
the PATCH surface — #256 / #260).

``MODEL_FIELD_POLICY`` below names **every** ``ModelRow`` column exactly once, and
the module **fails at import** if it does not. Import time means app startup and
the top of the test run, so the next migration that adds a column cannot get a
writable field by forgetting: the app does not boot until someone classifies it.
The diff to this table *is* the security review that a derived allowlist skipped.

The rule to apply when classifying a new column, carried over from the #256/#260
reviews (it used to live in a comment in
``tests/unit/settings/test_patch_allowlist_derivation.py``):

    a column is ``OPERATOR`` only if an admin token can ALREADY set it at
    ``POST /api/models`` (so no other write path grants anything register does
    not) and it is not control state some part of the runtime owns.

``trust_remote_code`` is the one column that fails the first half — register
refuses a truthy value from a token — so it is ``SESSION_TO_SET`` rather than
being locked outright: a session may set it, anyone may clear it.
``prior_status`` fails the second half and is ``RUNTIME``. Everything else is a
plain user setting that register already accepts from a token.

Scope note: this module is the *table*. It is deliberately data plus one
completeness check, with no enforcement of its own — the callers derive their
existing constants from it (``app/settings/routes_api.py``). A single
``apply_model_change`` writer that consults it, and the ``ModelSpec`` full-row
validator, are later steps of the same plan.

``created_at`` is intentionally absent: it exists in the SQL schema only, is
never decoded into ``ModelRow``, and has no place in a ``ModelRow`` policy. The
old ``_NEVER_PATCH`` carried it "should the dataclass ever grow it", which needed
a ``SQL_ONLY_EXCEPTIONS`` escape hatch; the completeness check below covers that
case exactly — promote the column and the app stops booting until it is named.
"""
from dataclasses import dataclass, fields
from enum import Enum

from app.db.repos.models import ModelRow


class Writer(Enum):
    """Who is permitted to write a column."""

    #: Any ``/api`` credential — a signed-in session or an admin token.
    OPERATOR = "operator"
    #: Any credential may write a falsy value; only a session may write a truthy
    #: one. "A session may set it, anyone may clear it" (#256).
    SESSION_TO_SET = "session_to_set"
    #: The state machine only: the pull task, the load runner, the crash path,
    #: the watchdog's restart sweep, boot reconciliation. No HTTP writer, with
    #: no principal that could authorise one.
    RUNTIME = "runtime"
    #: Never written by application code at all — set at insert, or maintained
    #: by the UPDATE statements themselves.
    DB = "db"


@dataclass(frozen=True)
class FieldPolicy:
    """The policy for one ``ModelRow`` column."""

    writer: Writer
    #: May this column be written while ``status == 'loaded'``? The settings
    #: PATCH's capabilities-only carve-out: the column is inert metadata that
    #: neither the load runner nor the engine ever reads, so requiring an
    #: unload to fix a label would be backwards.
    editable_while_loaded: bool = False
    #: Is the column's wire contract TRI-state (``true`` / ``false`` / ``null``
    #: where null means "nobody has stated an answer") rather than a plain
    #: value? Separate from ``editable_while_loaded`` on purpose: today the two
    #: sets coincide exactly, and conflating them would silently apply boolean
    #: coercion to the next inert-but-not-boolean column.
    tristate: bool = False
    #: One line of rationale. This is where the reasoning that used to be split
    #: across ``_NEVER_PATCH``'s comments and a test's comment block lives.
    why: str = ""


MODEL_FIELD_POLICY: dict[str, FieldPolicy] = {
    # -- identity and source -------------------------------------------------
    "id": FieldPolicy(Writer.DB, why="opaque identifier, set once at insert"),
    "served_model_name": FieldPolicy(Writer.OPERATOR, why="UNIQUE in SQL"),
    "hf_repo": FieldPolicy(Writer.OPERATOR),
    "hf_revision": FieldPolicy(Writer.OPERATOR),
    "filename": FieldPolicy(Writer.OPERATOR, why="#85; missed by the pre-#110 hand-written allowlist"),
    "hf_config_repo": FieldPolicy(Writer.OPERATOR, why="#106; likewise missed pre-#110"),
    "tokenizer_repo": FieldPolicy(Writer.OPERATOR, why="#106; likewise missed pre-#110"),
    "mmproj_filename": FieldPolicy(Writer.OPERATOR, why="llama.cpp vision projector (migration 0028)"),

    # -- sizing and placement ------------------------------------------------
    "gpu_indices": FieldPolicy(
        Writer.OPERATOR,
        why="must stay a subset of setup's allowed_gpu_indices; checked by the writer, not here",
    ),
    "tensor_parallel_size": FieldPolicy(Writer.OPERATOR),
    "parallelism_strategy": FieldPolicy(Writer.OPERATOR, why="#85; missed by the pre-#110 hand-written allowlist"),
    "max_batch_size": FieldPolicy(Writer.OPERATOR, why="#85; missed by the pre-#110 hand-written allowlist"),
    "max_model_len": FieldPolicy(Writer.OPERATOR),
    "gpu_memory_utilization": FieldPolicy(Writer.OPERATOR),
    "dtype": FieldPolicy(Writer.OPERATOR),
    "n_gpu_layers": FieldPolicy(Writer.OPERATOR, why="partial CPU offload (migration 0028)"),

    # -- engine axis ---------------------------------------------------------
    "backend": FieldPolicy(
        Writer.OPERATOR,
        why="registry.is_known is the source of truth; an unknown value must fail at write time, not inside Supervisor.load",
    ),
    "engine_channel": FieldPolicy(Writer.OPERATOR),
    "engine_vllm_version": FieldPolicy(Writer.OPERATOR),
    "engine_image": FieldPolicy(
        Writer.OPERATOR,
        why="code execution for any principal that can load; register already takes it from a token, so left open on purpose (API.md)",
    ),

    # -- free-form engine start-up inputs ------------------------------------
    # Both are operator-writable because POST /api/models already takes them
    # from an admin token verbatim: fencing them on one route would move the
    # same power one route over, not remove it. See API.md's "treat an admin
    # token as equivalent to code execution in the warden container".
    "extra_args": FieldPolicy(Writer.OPERATOR, why="appended to argv verbatim; same standing as engine_image"),
    "extra_env": FieldPolicy(
        Writer.OPERATOR,
        why=(
            "per-backend key allowlist, hard-locked keys refused (LD_PRELOAD, "
            "PYTHONPATH, HF_TOKEN, PATH); enforced on every write path as of "
            "#261 -- a non-allowlisted key that reaches the row makes the "
            "model refuse to load, because filter_extra_env fails closed"
        ),
    ),

    # -- the one column a token may clear but not set -------------------------
    "trust_remote_code": FieldPolicy(
        Writer.SESSION_TO_SET,
        why="the proxy imports the target repo's tokenizer Python inside the warden process (#256)",
    ),

    # -- inert capability metadata, editable on a loaded model ----------------
    # TRI-state (1 / 0 / NULL) and the third state is load-bearing:
    # app/chat2/catalog.py only auto-detects vision from the on-disk HF config
    # and reasoning from the chat template while the column is NULL, so
    # collapsing NULL into 0 is what served every pasted image as
    # "[image omitted]" on a model that could read it (#106) and hid the
    # "Enable thinking" toggle on every reasoning model (#239).
    "supports_tools": FieldPolicy(Writer.OPERATOR, editable_while_loaded=True, tristate=True),
    "supports_vision": FieldPolicy(Writer.OPERATOR, editable_while_loaded=True, tristate=True),
    "supports_reasoning": FieldPolicy(Writer.OPERATOR, editable_while_loaded=True, tristate=True),

    # -- state machine -------------------------------------------------------
    # Owned by the pull task / load runner / crash path. Mutating these
    # out-of-band corrupts the state machine (#11, #29).
    "status": FieldPolicy(Writer.RUNTIME, why="authoritative lifecycle state, set on supervisor transitions"),
    "pulled_bytes": FieldPolicy(Writer.RUNTIME, why="pull-task progress"),
    "pulled_total": FieldPolicy(Writer.RUNTIME, why="pull-task progress"),
    "last_error": FieldPolicy(Writer.RUNTIME, why="written by whatever failed"),
    "prior_status": FieldPolicy(
        Writer.RUNTIME,
        why=(
            "the watchdog's restart flag (migration 0024): an HTTP writer could arm "
            "restart_crashed_models on a failed row and have the sweep call start_engine "
            "with whatever argv the row carries, with no /load at all (#260). #256 "
            "framed that as pairing with a patched-on trust_remote_code; #264 found that "
            "column never reaches argv, so extra_args is the argv the sweep would honour "
            "-- the hazard is the unasked-for start either way"
        ),
    ),

    # -- DB-managed ----------------------------------------------------------
    "updated_at": FieldPolicy(Writer.DB, why="bumped by every UPDATE in app/settings/routes_api.py"),
}


def _check_policy_covers_model_row() -> None:
    """Fail at import if the table and ``ModelRow`` have drifted.

    Import time is app startup and the top of the test run, so an unclassified
    column is a hard stop rather than a silently writable field. This is the
    whole point of the module.
    """
    row_columns = {f.name for f in fields(ModelRow)}
    unclassified = sorted(row_columns - MODEL_FIELD_POLICY.keys())
    stale = sorted(MODEL_FIELD_POLICY.keys() - row_columns)
    if not unclassified and not stale:
        return
    problems: list[str] = []
    if unclassified:
        problems.append(
            f"ModelRow column(s) {unclassified} have no entry in MODEL_FIELD_POLICY. "
            f"Add one to app/models/policy.py and say who may write the column: "
            f"Writer.OPERATOR for a plain user setting register already accepts from an "
            f"admin token, Writer.SESSION_TO_SET if a token may clear but not set it, "
            f"Writer.RUNTIME if the state machine owns it, Writer.DB if application code "
            f"never writes it. Until then nothing may write the column and the app will "
            f"not start -- that is deliberate: an unclassified column must not become "
            f"HTTP-writable by default."
        )
    if stale:
        problems.append(
            f"MODEL_FIELD_POLICY names {stale}, which are not ModelRow columns. Did a "
            f"column get renamed or dropped? Update app/models/policy.py to match "
            f"app/db/repos/models.py::ModelRow."
        )
    raise RuntimeError(
        "MODEL_FIELD_POLICY must name every ModelRow column exactly once. "
        + " ".join(problems)
    )


_check_policy_covers_model_row()


#: Columns an operator-facing write path may set: ``OPERATOR`` plus
#: ``SESSION_TO_SET`` (which is operator-writable, with the truthy value
#: reserved to a session). This is the settings PATCH's allowlist.
OPERATOR_FIELDS: frozenset[str] = frozenset(
    name
    for name, policy in MODEL_FIELD_POLICY.items()
    if policy.writer in (Writer.OPERATOR, Writer.SESSION_TO_SET)
)

#: Columns whose truthy value only a session may write; any credential may
#: clear them. Today exactly ``trust_remote_code``.
SESSION_TO_SET_FIELDS: frozenset[str] = frozenset(
    name for name, policy in MODEL_FIELD_POLICY.items() if policy.writer is Writer.SESSION_TO_SET
)

#: Columns writable while the model is ``loaded`` — the settings PATCH's
#: capabilities-only carve-out from its unload-first 409.
LOADED_EDITABLE_FIELDS: frozenset[str] = frozenset(
    name for name, policy in MODEL_FIELD_POLICY.items() if policy.editable_while_loaded
)

#: Columns whose wire contract is tri-state (``true`` / ``false`` / ``null``).
TRISTATE_FIELDS: frozenset[str] = frozenset(
    name for name, policy in MODEL_FIELD_POLICY.items() if policy.tristate
)

#: Columns the state machine owns. No HTTP write path may name them.
RUNTIME_FIELDS: frozenset[str] = frozenset(
    name for name, policy in MODEL_FIELD_POLICY.items() if policy.writer is Writer.RUNTIME
)
