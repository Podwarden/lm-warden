"""Model VARIANTS: which exact configuration of a model served a request.

A ``models`` row is edited in place -- the try-stack rewrites its engine
image/version (app/models/routes_api.py), the settings form its quantization,
dtype, context length, extra args and revision -- so ``models.id`` alone would
merge "vLLM 0.9 FP8" and "vLLM 0.11 AWQ" into one line of the token page's
"Usage by model" card. A variant is a hash of what decides how the served model
BEHAVES, taken from what the engine was actually LAUNCHED as:

  * the row's identity fields (``VARIANT_FIELDS``), with the load overrides
    applied the way the argv builders apply them, and ``extra_args`` values of
    secret-looking flags replaced by a digest (``redact_args``);
  * RUNTIME facts the row does not say (``RUNTIME_FIELDS``): the engine image
    the driver actually ran -- the pin, or the driver's default when the row
    pins nothing -- or, for the in-container engine, the engine version baked
    into this warden image; and the commit ``hf_revision`` resolved to in the
    HF cache. So a warden upgrade that moves the default engine, or a floating
    ``main`` that moved, is a new variant too.

``Supervisor.load`` computes it after the launch plan and keeps it per model id
until the engine exits or is unloaded, so editing the row while the engine runs
does not change attribution; ``start_engine`` records it in ``model_variants``.
The proxy reads the cached one; only for an engine this supervisor did not
launch does it compute one from the row (without runtime facts, not cached).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import aiosqlite

from app.runtime.backends.registry import DEFAULT_BACKEND

#: The row's part of a variant's identity: every field that changes what the
#: served model IS or how it answers. Excluded on purpose:
#:   * placement and throughput knobs -- gpu_indices, tensor_parallel_size,
#:     gpu_memory_utilization, parallelism_strategy, max_batch_size,
#:     n_gpu_layers -- which move the same model around the hardware;
#:   * ``extra_env``, NEVER: it holds secrets such as HF_TOKEN, and the
#:     descriptor is stored in plain text and hashed into an id.
#: Adding a field splits every running model into a new variant on its next
#: start, so treat this list as append-only and deliberate.
VARIANT_FIELDS: tuple[str, ...] = (
    "backend",
    "engine_channel",
    "engine_vllm_version",
    "engine_image",
    "hf_repo",
    "hf_revision",
    "filename",
    "mmproj_filename",
    "hf_config_repo",
    "tokenizer_repo",
    "quantization",
    "dtype",
    "max_model_len",
    "max_num_seqs",
    "trust_remote_code",
    "extra_args",
)

#: The launch's part: facts only the running engine knows (see module doc).
#: ``engine_image_used`` -- the image the driver ran, None in-container;
#: ``engine_version`` -- the in-container engine's baked version, None when an
#: image says it; ``hf_commit`` -- the commit ``hf_revision`` resolved to, or
#: the ref itself when the cache has no entry for it.
RUNTIME_FIELDS: tuple[str, ...] = ("engine_image_used", "engine_version", "hf_commit")

#: Load-config overrides (``Supervisor.load(overrides=...)``) win over the row
#: for these, exactly as the backends' argv builders apply them.
_OVERRIDABLE = frozenset({"quantization", "max_model_len", "max_num_seqs"})

#: An ``extra_args`` flag whose NAME looks like it carries a credential
#: (``--api-key``, ``--hf-token``, ``--ssl-keyfile-password``...), matched on
#: dash-delimited segments so ``--max-num-batched-tokens`` and ``--tokenizer*``
#: stay readable. Still broad on ``secret*``/``pass*``/``auth*``/``cred*``: a
#: false positive only hides a value that still splits variants through its
#: digest; a false negative stores a secret forever.
_SECRET_FLAG = re.compile(
    r"(?i)(?:^|-)(?:keys?|token|secret\w*|pass\w*|auth\w*|cred\w*)(?:-|$)"
)
#: What looks like the next flag rather than a value (a negative number does not).
_FLAG = re.compile(r"^--?[A-Za-z]")


@dataclass(frozen=True)
class Variant:
    id: str
    model_id: str
    served_model_name: str
    #: ``VARIANT_FIELDS`` + ``RUNTIME_FIELDS`` -> value; JSON-serialisable.
    descriptor: dict[str, Any]


def canonical_json(obj: Any) -> str:
    """Sorted keys, no whitespace: the same dict always gives the same bytes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _redacted(value: str) -> str:
    return "<redacted:" + hashlib.sha256(value.encode()).hexdigest()[:8] + ">"


def redact_args(args: list[str]) -> list[str]:
    """``args`` with the value of every secret-looking flag replaced by
    ``<redacted:sha256[:8]>``, in both ``--flag=value`` and ``--flag value``
    forms. Different secrets keep different digests, so they still make
    different variants; the secret itself is never stored."""
    out: list[str] = []
    redact_next = False
    for arg in args:
        if redact_next:
            redact_next = False
            if not _FLAG.match(arg):
                out.append(_redacted(arg))
                continue
        if arg.startswith("-"):
            name, eq, value = arg.partition("=")
            if _SECRET_FLAG.search(name.lstrip("-")):
                if eq:
                    out.append(f"{name}={_redacted(value)}")
                    continue
                redact_next = True
        out.append(arg)
    return out


def descriptor_of(
    model: Any,
    overrides: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The variant descriptor of ``model`` as launched with ``overrides``,
    plus the ``runtime`` facts of that launch (all None when not known).

    Reads with ``getattr`` like the argv builders do: ``quantization`` and
    ``max_num_seqs`` are DB columns ModelRow does not carry, so for a row they
    are whatever the load overrides said -- which is also what the engine was
    started with.
    """
    ov = overrides or {}
    out: dict[str, Any] = {}
    for field in VARIANT_FIELDS:
        if field in _OVERRIDABLE and field in ov:
            value = ov[field]
        else:
            value = getattr(model, field, None)
        out[field] = _scalar(value)
    out["backend"] = out["backend"] or DEFAULT_BACKEND
    out["trust_remote_code"] = bool(out["trust_remote_code"])
    raw_args = getattr(model, "extra_args", None)
    out["extra_args"] = (
        redact_args([str(a) for a in raw_args]) if isinstance(raw_args, list | tuple) else []
    )
    rt = runtime or {}
    for field in RUNTIME_FIELDS:
        out[field] = _scalar(rt.get(field))
    return out


def _scalar(value: Any) -> Any:
    """A JSON scalar as is; anything else as its string, so computing a variant
    can never fail the engine start it rides on."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def variant_id(model_id: str, descriptor: dict[str, Any]) -> str:
    """sha256 of the canonical JSON, hex, first 16 characters.

    The model id is hashed in alongside the descriptor (and stored in its own
    column, not in the descriptor): two model rows serving the same repo with
    the same settings are still two models, and usage rows are keyed on
    ``(token_id, variant_id, minute)``, so a shared id would merge them.
    """
    payload = canonical_json({"model_id": model_id, "descriptor": descriptor})
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def variant_of(
    model: Any,
    overrides: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
) -> Variant:
    descriptor = descriptor_of(model, overrides, runtime)
    return Variant(
        id=variant_id(model.id, descriptor),
        model_id=model.id,
        served_model_name=getattr(model, "served_model_name", None) or model.id,
        descriptor=descriptor,
    )


# ---- runtime facts -------------------------------------------------------------

_HEX_SHA = re.compile(r"^[0-9a-f]{40}$")


def resolve_hf_commit(hf_cache_dir: str | Path | None, hf_repo: str | None,
                      revision: str | None) -> str | None:
    """The commit ``revision`` points at in the HF cache
    (``models--org--name/refs/<revision>``, at the cache root or under
    ``hub/`` -- the same two places app/runtime/backends/paths.py looks), else
    ``revision`` itself. A revision that already is a commit is returned as is.
    Never raises: an unreadable cache means "not resolved"."""
    if not isinstance(revision, str):
        return None
    if not revision or not isinstance(hf_repo, str) or _HEX_SHA.match(revision) or not hf_cache_dir:
        return revision
    slug = f"models--{hf_repo.replace('/', '--')}"
    root = Path(hf_cache_dir)
    for base in (root, root / "hub"):
        try:
            sha = (base / slug / "refs" / revision).read_text().strip()
        except (OSError, ValueError):
            continue
        if sha:
            return sha
    return revision


@lru_cache(maxsize=4)
def baked_engine_version(backend: str) -> str | None:
    """The engine version compiled into THIS warden image, for the in-container
    engine. Declared values, not probes -- the same sources as
    ``GET /api/system/backends`` (app/system/routes_engine.py): vLLM's package
    metadata, and the llama.cpp build tag the Dockerfile stamps (the binary is
    built into the image, D2). Cached: it cannot change without a restart."""
    if backend == "vllm":
        try:
            return version("vllm")
        except PackageNotFoundError:
            return None
    if backend == "llamacpp":
        from app.runtime.backends.llamacpp.version import version_string

        return version_string()
    return None


def runtime_facts(
    *, backend: str, image: str | None, hf_cache_dir: str | Path | None,
    hf_repo: str | None, revision: str | None,
) -> dict[str, Any]:
    """``RUNTIME_FIELDS`` for one launch. ``image`` is the image the driver
    will run (the pin, or its default), None for the in-container engine."""
    return {
        "engine_image_used": image,
        "engine_version": None if image else baked_engine_version(backend),
        "hf_commit": resolve_hf_commit(hf_cache_dir, hf_repo, revision),
    }


def running_variant(state: Any, model: Any) -> Variant:
    """The variant ``model`` is being served as right now: the one the
    supervisor fixed when it launched the engine. For an engine it did not
    launch, one computed from the row and the overrides in effect, without the
    launch's runtime facts (and not cached -- the supervisor owns that cache)."""
    sup = getattr(state, "supervisor", None)
    get = getattr(sup, "get_variant", None)
    cached = get(model.id) if get is not None else None
    if isinstance(cached, Variant):
        return cached
    get_overrides = getattr(sup, "get_overrides", None)
    overrides = get_overrides(model.id) if get_overrides is not None else None
    if not isinstance(overrides, dict):
        overrides = None
    return variant_of(model, overrides)


# ---- persistence ---------------------------------------------------------------

#: (db_path, variant_id) pairs already committed to ``model_variants`` by this
#: process, so the proxy's hot path skips the INSERT OR IGNORE after the first
#: request. Keyed on the database too: tests (and only tests) open many.
# Process-lifetime memo keyed by (db path, variant id). Replacing the DB file
# at the same path without a restart leaves stale entries (their
# model_variants rows are then not re-inserted until the process restarts).
_ENSURED: set[tuple[str, str]] = set()


def is_recorded(db_path: str | Path, variant_id: str) -> bool:
    return (str(db_path), variant_id) in _ENSURED


def mark_recorded(db_path: str | Path, variant_id: str) -> None:
    """Call only once the INSERT has COMMITTED."""
    _ENSURED.add((str(db_path), variant_id))


class ModelVariantRepo:
    """``model_variants`` (migration 0036): one row per variant ever served."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def ensure(self, variant: Variant, *, commit: bool = True) -> None:
        """INSERT OR IGNORE: ``first_seen`` stays the first time it was seen."""
        await self.db.execute(
            "INSERT OR IGNORE INTO model_variants"
            "(id, model_id, served_model_name, descriptor, first_seen) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                variant.id,
                variant.model_id,
                variant.served_model_name,
                canonical_json(variant.descriptor),
                time.time(),
            ),
        )
        if commit:
            await self.db.commit()
