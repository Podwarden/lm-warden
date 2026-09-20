import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.runtime.backends import registry

SLUG_RE = re.compile(r"^[a-zA-Z0-9._-]+$")

#: A Hugging Face ``owner/name`` repo id. One constant, because four columns
#: use it (``hf_repo``, ``hf_config_repo``, ``tokenizer_repo``) and so does
#: ``TemplateCreate`` -- a template is a register-time prefill, so a repo id it
#: accepts is a repo id that lands on a row.
HF_REPO_PATTERN = r"^[\w.-]+/[\w.-]+$"

# The shared constrained types. They exist so ``ModelSpec`` and
# ``TemplateCreate`` cannot hold the same column to different rules: the
# template schema drifted from the model schema once already (#261's
# ``extra_env`` gap), and a duplicated `Field(gt=0)` is exactly how that starts.
# ``tests/unit/models/test_template_shares_model_spec_rules.py`` pins the pairing.
HfRepoId = Annotated[str, Field(pattern=HF_REPO_PATTERN)]
#: A weights / projector file name inside the repo.
RepoFilename = Annotated[str, Field(min_length=1, max_length=512)]
MaxModelLen = Annotated[int, Field(gt=0)]
GpuMemoryUtilisation = Annotated[float, Field(gt=0, le=1.0)]


def validate_extra_env_for_backend(extra_env: dict[str, str], backend: str | None) -> None:
    """Raise ``ValueError`` if any ``extra_env`` key is hard-locked, or not on
    the given backend's allowlist.

    Every way the API can write ``extra_env`` shares this one check, because
    #261 found the PATCH validating nothing at all -- a hard-locked key like
    ``LD_PRELOAD`` or ``HF_TOKEN`` was a 200 there and a 422 on register. There
    are **four** such doors, the same shape #256 had to close for
    ``trust_remote_code`` (see ``app/models/routes_api.py``'s module comment):

    * ``ModelCreate._extra_env_allowlist`` -- the body key on ``POST
      /api/models``;
    * ``create_model``'s merged value -- a ``template_id`` prefills
      ``extra_env`` whenever the body does not set it, so the body check alone
      was a one-request bypass;
    * ``TemplateCreate._extra_env_allowlist`` -- ``POST
      /api/models/templates``, so a refused payload cannot be left lying
      around as a stored instruction for a later register to trip over;
    * ``patch_model_settings`` -- ``PATCH /api/models/{id}/settings``
      (``app/settings/routes_api.py``).

    ``registry.get`` resolves ``None``/``""`` to the default backend, so
    callers may pass the model's current/effective backend verbatim. A template
    has no backend axis at all and uses
    :func:`validate_extra_env_for_any_backend` instead.
    """
    caps = registry.get(backend).capabilities
    _check(
        extra_env,
        prefixes=caps.env_prefixes,
        exact=caps.env_exact,
        allowlist=f"backend '{caps.name}'",
    )


def validate_extra_env_for_any_backend(extra_env: dict[str, str]) -> None:
    """Like :func:`validate_extra_env_for_backend`, against the union of every
    registered backend's allowlist.

    For templates, which have no backend axis: the backend is chosen at
    register time, by the body, so at template-create time there is no per-
    backend answer to give. Checking against the default backend instead was
    wrong in a way that cost a feature -- the two allowlists are disjoint
    (vLLM's ``VLLM_``/``TRITON_``/``NCCL_``/... vs llama.cpp's ``GGML_``), so
    it made llama.cpp's documented ``extra_env`` allowlist unreachable through
    templates, #170's "save the working combo" flow included.

    The security property is unchanged: ``HARD_LOCKED_ENV_KEYS`` is global and
    is refused here exactly as everywhere else. What the union gives up is only
    the *per-backend* rule, and ``create_model`` still enforces that on the
    merged value -- at the one point where the backend is actually known.
    """
    prefixes: tuple[str, ...] = ()
    exact: frozenset[str] = frozenset()
    for name in registry.available():
        caps = registry.get(name).capabilities
        prefixes += caps.env_prefixes
        exact |= caps.env_exact
    _check(extra_env, prefixes=prefixes, exact=exact, allowlist="any backend")


def _check(
    extra_env: dict[str, str],
    *,
    prefixes: tuple[str, ...],
    exact: frozenset[str],
    allowlist: str,
) -> None:
    from app.runtime.backends.vllm.env import HARD_LOCKED_ENV_KEYS

    for key in extra_env:
        if key in HARD_LOCKED_ENV_KEYS:
            raise ValueError(
                f"extra_env key '{key}' is hard-locked and cannot be set via API"
            )
        if not (any(key.startswith(p) for p in prefixes) or key in exact):
            raise ValueError(
                f"extra_env key '{key}' is not in the allowlist for {allowlist} "
                f"(prefix one of {sorted(set(prefixes))} "
                f"or exact match in {sorted(exact)})"
            )


def coerce_tristate_storage(field: str, v: Any) -> int | None:
    """Normalise a capability flag to its stored form: ``1`` / ``0`` / ``None``.

    Accepts what a row can hold (``1`` / ``0`` / NULL, straight off
    ``ModelRow``) as well as what a client sends (a JSON boolean, or its
    case-insensitive string spelling for form-encoded clients and ``curl``
    recipes). Anything else raises, because ``int(bool(v))`` reads the string
    ``"no"`` -- a perfectly plausible thing for a shell script to send -- as an
    explicit YES.

    Deliberately WIDER than the settings PATCH's wire coercer
    (``app/models/writer.py::_coerce_tristate_wire``, which refuses a bare
    ``1``): this one also has to accept the integer the DATABASE holds, because
    ``ModelSpec`` validates merged rows. The wire coercer answers "what may a
    client send", this answers "what may a row hold", and the two are not the
    same question.
    """
    if v is None:
        return None
    if isinstance(v, bool):  # must precede the int check -- bool IS an int
        return int(v)
    if isinstance(v, int) and v in (0, 1):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return int(v.strip().lower() == "true")
    raise ValueError(f"{field} must be true, false, or null (got {v!r})")


class ModelSpec(BaseModel):
    """Every operator-writable ``ModelRow`` column, with the rules a row must
    satisfy.

    Validated as a WHOLE ROW. This is the half of the write path that the
    per-column policy table (``app/models/policy.py``) does not cover: the table
    says *who* may write a column, this says *what* may be on it. Both are
    complete by construction -- the table fails at import if a ``ModelRow``
    column is unclassified, and
    ``tests/unit/models/test_model_spec.py::test_model_spec_covers_exactly_the_operator_fields``
    fails if an operator-writable column has no rule here. Between them, a
    migration cannot add a column that is writable-by-default *or*
    unvalidated-by-default.

    Why a row shape rather than a request shape (#262): the runtime trusts the
    row. ``backend.plan(row)`` builds argv and env from it, ``Supervisor.
    _resolve_engine_image(row)`` picks the container image, and the watchdog
    re-reads it and restarts it with nobody at the keyboard. Attaching the rules
    to ``ModelCreate`` meant every other producer of a row re-derived them from
    memory: the settings PATCH kept a hand-written subset (which skipped every
    bound below), ``TemplateCreate`` kept a thinner copy, and ``try_stack`` and
    stress-apply kept none. ``app/models/writer.py::apply_model_change`` merges
    a change into the current row and validates the RESULT here, so the
    cross-field checks run on an update as well as on a register.

    Only ``app.models.writer`` should validate merged rows with this; routes
    keep taking their own request models (``ModelCreate`` below), which is where
    a wire shape belongs.
    """

    # -- identity and source -------------------------------------------------
    served_model_name: str = Field(..., min_length=1, max_length=100)
    hf_repo: HfRepoId
    hf_revision: str = "main"
    # ``filename`` narrows the pull to one weights file + tokenizer/config;
    # unset (None) preserves the legacy "pull the whole repo" path (#85).
    filename: RepoFilename | None = None
    # #106 — see migration 0015. ``hf_config_repo`` populates the vLLM
    # ``--hf-config-path`` flag (required when a GGUF repo omits config.json,
    # e.g. unsloth republishes). ``tokenizer_repo`` populates ``--tokenizer``
    # for the same upstream-vs-quant split. Both optional; the wizard defaults
    # them from a single "Base repo" Input, and empty strings normalise to None
    # so a cleared field round-trips correctly.
    hf_config_repo: HfRepoId | None = None
    tokenizer_repo: HfRepoId | None = None
    # Migration 0028 (sub-project C) -- llama.cpp only. ``mmproj_filename`` is
    # the multimodal projector GGUF that sits beside the weights in the same
    # repo; llama.cpp takes it as a separate --mmproj file because, unlike vLLM,
    # its vision tower is not inside the checkpoint.
    mmproj_filename: RepoFilename | None = None

    # -- sizing and placement ------------------------------------------------
    gpu_indices: list[int] = Field(..., min_length=1)
    tensor_parallel_size: int | None = None
    # ``parallelism_strategy`` records the operator's choice; ``auto`` is the
    # safe default — single GPU is no-parallelism, multi-GPU defaults to TP.
    parallelism_strategy: Literal["tp", "pp", "auto"] = "auto"
    # ``max_batch_size`` feeds the KV-reserve math (app/models/fit.py). 1 =
    # single-request, the wizard's default; 64 is a sane upper bound that
    # already overruns most consumer cards' KV budget.
    max_batch_size: int = Field(default=1, ge=1, le=64)
    max_model_len: MaxModelLen | None = None
    gpu_memory_utilization: GpuMemoryUtilisation = 0.9
    dtype: str | None = None
    # ``n_gpu_layers`` = None means "omit the flag" and let llama.cpp's own
    # auto/--fit sizing choose. An explicit value is a deliberate PARTIAL
    # offload -- spilling layers to CPU RAM to fit a model that does not fit the
    # card. 0 is legal and means "everything on the CPU"; the upper bound is
    # loose because it is a sanity check, not a model fact.
    n_gpu_layers: int | None = Field(None, ge=0, le=999)

    # -- engine axis ---------------------------------------------------------
    # NULL is the D6 default (``_decode_row`` resolves it to the registry's
    # default) and is how an operator puts a row back on the default without
    # naming it, so a ROW may hold None. ``ModelCreate`` narrows this to the
    # Literal, because a register always states a backend.
    backend: Literal["vllm", "llamacpp"] | None = None
    engine_channel: str | None = None
    engine_vllm_version: str | None = None
    # Deliberately unvalidated beyond being a string: `POST /api/models` already
    # accepts any image from an admin token, so constraining it here would move
    # that power one route over rather than remove it. See
    # ``app/models/policy.py``'s ``why=`` for this column, and API.md's "treat an
    # admin token as equivalent to code execution in the warden container".
    engine_image: str | None = None

    # -- free-form engine start-up inputs ------------------------------------
    # ``list[str]`` is the load-bearing part (#262): the settings PATCH used to
    # ``json.dumps`` whatever arrived, and a plain string round-tripped as a
    # string, which ``args += extra_args`` then appended one CHARACTER at a time.
    extra_args: list[str] = Field(default_factory=list)
    extra_env: dict[str, str] = Field(default_factory=dict)

    # -- the one column a token may clear but not set ------------------------
    # Who may write this is the policy table's business (``SESSION_TO_SET``),
    # enforced by the writer; the type is this file's.
    trust_remote_code: bool = False

    # -- inert capability metadata -------------------------------------------
    # TRI-state (1 / 0 / NULL) and the third state is load-bearing:
    # app/chat2/catalog.py only auto-detects vision from the on-disk HF config
    # and reasoning from the chat template while the column is NULL. Kept as the
    # raw integer rather than a bool, so "the operator said no" stays
    # distinguishable from "nobody said".
    supports_tools: int | None = None
    supports_vision: int | None = None
    supports_reasoning: int | None = None

    @field_validator("served_model_name")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not SLUG_RE.match(v):
            raise ValueError("served_model_name must be alphanumeric/dot/dash/underscore")
        return v

    @field_validator("hf_config_repo", "tokenizer_repo", mode="before")
    @classmethod
    def _empty_string_to_none(cls, v: str | None) -> str | None:
        """Treat empty strings as unset (#106).

        The wizard's "Base repo" Input ships an empty string when the operator
        clears the field — without this the ``pattern=`` regex would reject
        it. Normalising here means the API surface accepts both ``""`` (UI)
        and ``null`` (programmatic) for "no override".
        """
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("supports_tools", "supports_vision", "supports_reasoning",
                     mode="before")
    @classmethod
    def _tristate(cls, v: Any, info: Any) -> int | None:
        return coerce_tristate_storage(str(info.field_name), v)

    @field_validator("gpu_indices")
    @classmethod
    def _unique_gpus(cls, v: list[int]) -> list[int]:
        if len(set(v)) != len(v):
            raise ValueError("gpu_indices must be unique")
        if any(i < 0 for i in v):
            raise ValueError("gpu_indices must be >= 0")
        # Canonical order, so every write path produces the same list. Register
        # sorted in the route; the rule belongs to the row.
        return sorted(v)

    @model_validator(mode="after")
    def _tp_consistent(self) -> "ModelSpec":
        if self.tensor_parallel_size is None:
            object.__setattr__(self, "tensor_parallel_size", len(self.gpu_indices))
        elif self.tensor_parallel_size != len(self.gpu_indices):
            raise ValueError(
                f"tensor_parallel_size={self.tensor_parallel_size} "
                f"must equal len(gpu_indices)={len(self.gpu_indices)}"
            )
        return self

    @model_validator(mode="after")
    def _extra_env_allowlist(self) -> "ModelSpec":
        """Enforce the same allowlist that runs at subprocess-spawn time.

        Defense-in-depth: at load time filter_extra_env raises on hard-locked
        keys and silently drops unknown ones. Enforcing here means the API
        rejects bad extra_env synchronously at write time, so operators get an
        immediate, descriptive error instead of mysteriously-missing env.

        A model_validator rather than a field_validator because the allowlist
        is per-backend and ``self.backend`` must already be resolved.
        HARD_LOCKED_ENV_KEYS stays global -- every key in it is locked for
        reasons unrelated to which server runs. ``registry.get`` resolves
        None/"" to the default backend, so a NULL-backend row is checked
        against the allowlist it will actually run under.

        Because this runs on the MERGED row, the four doors #261 had to close
        one at a time (the register body key, register's template prefill, the
        template create, the settings PATCH) are one check at one place. See
        ``validate_extra_env_for_backend`` above for that history.
        """
        validate_extra_env_for_backend(self.extra_env, self.backend)
        return self


class ModelCreate(ModelSpec):
    """``POST /api/models``'s body: a ``ModelSpec`` plus the request-shaped
    parts.

    Same wire contract as before ``ModelSpec`` was extracted, plus the three
    ``supports_*`` capability flags, which are ``ModelRow`` columns an operator
    may write and were previously settable only by a follow-up PATCH.
    """

    # #162 — ``hf_repo`` is optional when a template supplies it. The route
    # enforces presence after merging template + body (explicit body wins), and
    # ``ModelSpec`` requires it on the merged row.
    # A pydantic field override, so mypy's Liskov check does not apply: the
    # request may omit the repo, the merged ROW may not, and ``ModelSpec`` is
    # never asked to validate a ``ModelCreate`` instance -- the writer validates
    # the merged dict.
    hf_repo: HfRepoId | None = None  # type: ignore[assignment]
    # Which inference program serves this model (sub-project B, decision D6).
    # Defaulted, so a client that never sends it behaves exactly as before.
    # Sub-project C widens the Literal to include "llamacpp"; the registry is
    # the source of truth and tests/unit/models/test_schemas.py asserts the two
    # never drift. Narrower than ``ModelSpec.backend``, which also admits None
    # because a legacy ROW may hold NULL.
    backend: Literal["vllm", "llamacpp"] = "vllm"
    # #162 — engine templates. ``template_id`` selects a builtin/user template
    # whose fields prefill the model; the ``engine_*`` trio (on ``ModelSpec``)
    # overrides the template's engine axis.
    template_id: str | None = None


class EngineSpecBody(BaseModel):
    channel: str
    vllm_version: str
    image: str | None = None


class TemplateCreate(BaseModel):
    """``POST /api/models/templates``'s body.

    A template is not a row -- it has an ``id``, a ``label``, an ``engine`` and
    no GPU axis at all -- so it keeps its own wire shape rather than inheriting
    ``ModelSpec``. What it must NOT keep is its own *rules*: a template is a
    register-time prefill, so every value it stores is a value that lands on a
    row, and the one time the two schemas were allowed to drift, a template
    could store an ``extra_env`` key register refused (#261).

    The columns it shares with ``ModelSpec`` therefore use the same constrained
    types (``HfRepoId``, ``MaxModelLen``, ``GpuMemoryUtilisation``), and
    ``tests/unit/models/test_template_shares_model_spec_rules.py`` fails if a
    shared column is held to different rules on the two schemas. Defaults still
    differ on purpose: a template states a value, a row may leave one unset.
    """

    # ``model_id`` (below) sits in pydantic's reserved ``model_`` namespace;
    # opt out of the protection so the field name stays meaningful and no
    # spurious UserWarning is emitted at import.
    model_config = ConfigDict(protected_namespaces=())

    id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9._-]+$")
    label: str = Field(..., min_length=1, max_length=200)
    hf_repo: HfRepoId
    hf_revision: str = "main"
    dtype: str = "auto"
    # Defaults let a try-stack "save working combo" POST omit these — that flow
    # only captures the engine axis + repo. The UI sends the model's actual
    # values when known; these fallbacks apply when the model left them unset.
    max_model_len: MaxModelLen = 8192
    tensor_parallel_size: int = Field(1, ge=1)
    gpu_memory_utilization: GpuMemoryUtilisation = 0.9
    trust_remote_code: bool = False
    extra_args: list[str] = Field(default_factory=list)
    extra_env: dict[str, str] = Field(default_factory=dict)
    engine: EngineSpecBody | None = None
    # #170 — the try-stack "save working combo" flow references the live model
    # whose combo was just validated. When set, the route sources the model's
    # ACTUAL ``extra_args`` + ``gpu_memory_utilization`` (and other tuning) from
    # the live model row, so a saved AWQ template keeps e.g. --enforce-eager +
    # gpu_memory_utilization=0.92 instead of silently falling back to defaults.
    # Explicit body fields still win over the live row.
    model_id: str | None = None

    @model_validator(mode="after")
    def _extra_env_allowlist(self) -> "TemplateCreate":
        """Hold a template's ``extra_env`` to the same allowlist as a model's.

        A template is a register-time prefill, so an unvalidated one is a stored
        instruction to write ``extra_env`` onto every model made from it:
        ``POST /api/models/templates`` followed by ``POST /api/models
        {template_id}`` persisted a hard-locked ``LD_PRELOAD`` in two
        authenticated calls -- a payload ``POST /api/models`` refuses outright,
        and one that then fails the model closed at launch
        (``filter_extra_env``), so the row cannot be loaded again until an
        operator hand-edits it. Refusing here keeps the instruction from being
        stored at all; ``create_model`` refuses the merged value as well, the
        same belt and braces #256 needed for ``trust_remote_code``.

        Templates have no backend axis, so the check is against the union of
        every backend's allowlist: which one a template ends up serving is
        decided by the register body, and holding a template to the *default*
        backend's list would make a llama.cpp-only key unsavable. The
        per-backend rule is still enforced on the merged value at register,
        where the backend is known. Hard-locked keys are global and refused
        here either way -- which is the key this door exists to stop.
        """
        validate_extra_env_for_any_backend(self.extra_env)
        return self


class TryStackRequest(BaseModel):
    channel: str = Field(..., min_length=1)
    vllm_version: str = Field(..., min_length=1)
    image: str | None = None


class TryStackResult(BaseModel):
    result: Literal["ok", "failed"]
    error: str | None = None


class ModelEngine(BaseModel):
    """The pinned engine; ``null`` on a model until a channel is pinned."""

    channel: str
    vllm_version: str | None
    image: str | None


class ModelOut(BaseModel):
    """One model as GET /api/models and GET /api/models/{id} return it.

    Mirrors app/models/serialisation.py::_serialise field for field
    (tests/unit/models/test_model_serialisation.py compares the two), so the
    OpenAPI spec types what clients actually receive. ``extra="allow"`` keeps
    a column that _serialise publishes before it is declared here in the
    response instead of silently dropping it -- the drift that module exists
    to prevent.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    served_model_name: str
    hf_repo: str
    hf_revision: str
    gpu_indices: list[int]
    tensor_parallel_size: int | None
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
    filename: str | None
    parallelism_strategy: str
    max_batch_size: int
    hf_config_repo: str | None
    tokenizer_repo: str | None
    supports_tools: bool | None
    supports_vision: bool | None
    supports_reasoning: bool | None
    backend: str
    mmproj_filename: str | None
    n_gpu_layers: int | None
    engine: ModelEngine | None


class ModelList(BaseModel):
    models: list[ModelOut]
