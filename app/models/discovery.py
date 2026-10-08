"""HuggingFace repo discovery (#84, parent #82).

Pure helpers + a single I/O wrapper for listing files + reading config.json
on an HF Hub repo. The helpers are filename-only string ops so unit tests
don't need any network or fixtures, and the I/O wrapper is injectable so the
route tests can pass a fake HfApi.

The result shape is intentionally schema-shaped (`files`, `config`, `repo`,
`errors`) and stable — it crosses the wire to the FE in #86/#87 and is
snapshotted in ``tests/fixtures/discovery/`` so we'll notice drift.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

logger = logging.getLogger(__name__)

# ---- Pure helpers ---------------------------------------------------------

FileKind = Literal[
    "safetensors_single",
    "safetensors_sharded",
    "gguf",
    # New in sub-project C. A multimodal projector is a GGUF, but it is not
    # WEIGHTS: llama.cpp takes it via --mmproj alongside -m. Classifying it as
    # `gguf` puts it in the wizard's weights-file radio list, where selecting it
    # produces a load failure that reads like a corrupt download.
    "mmproj",
    "pytorch_bin",
    "config",
    "tokenizer",
    "other",
]

# Upstream's convention for a projector file, PREFIX-anchored so a weights file
# whose name merely contains the substring ("model-with-mmproj-support-Q4.gguf")
# is not misfiled.
_MMPROJ_RE = re.compile(r"^mmproj([.\-_]|\.gguf$)", re.IGNORECASE)

# The config.json keys we surface to the FE wizard. #82 committed to the
# first six (VRAM-fit math); #176 adds ``quantization_config`` so the
# capability-warning layer (and the AWQ KV-cache heuristic in suggest.py)
# can read ``quantization_config.quant_method`` instead of relying solely on
# repo-name / filename markers. Anything else stays opaque.
#
# The last three are the KV-shape keys. They are here because the fit math is
# this projection's whole reason to exist and it was wrong without them: with
# only the original six, the route could not tell that gpt-oss-20b declares
# ``head_dim: 64`` (rather than the 2880//64 = 45 the derivation gives) or
# that 12 of its 24 layers use a 128-token sliding window. Between them those
# two are worth about 2x on the KV term. Widening the projection is the only
# way to fix that -- the route reads ``discovery["config"]``, so a key absent
# here is a key the math cannot see.
_CONFIG_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "torch_dtype",
    "quantization_config",
    "head_dim",
    "sliding_window",
    "layer_types",
)

_TOKENIZER_NAMES = {
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
}

# GGUF architectures vLLM is known to load without warnings (#101). Lower-case
# canonical names match the ``general.architecture`` field emitted by
# llama.cpp's GGUF tooling and the common filename slugs on the HF Hub.
# This is an ALLOWLIST — any arch not in this set surfaces a soft warning so
# the operator sees it BEFORE picking the file (most failures here are silent
# vLLM-side init crashes that are very hard to diagnose post-pull).
KNOWN_GGUF_ARCHES: frozenset[str] = frozenset(
    {
        "llama",
        "llama2",
        "llama3",
        "mistral",
        "mixtral",
        "qwen",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen3_moe",
        "phi3",
        "phi3_5",
        "gemma",
        "gemma2",
        "gemma3",
        "deepseek",
        "deepseek_v2",
        "deepseek_v3",
        "yi",
        "command_r",
        "starcoder2",
    }
)

# Filename heuristic for inferring GGUF arch when ``config.json`` is absent
# (very common for unsloth/TheBloke republishes that ship raw GGUF only).
# Order matters — longer / more specific matches first so "llama3" hits
# before "llama".
_GGUF_ARCH_FILENAME_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("qwen3_moe", re.compile(r"qwen.?3.*moe", re.IGNORECASE)),
    ("qwen3", re.compile(r"qwen.?3", re.IGNORECASE)),
    ("qwen2_moe", re.compile(r"qwen.?2.*moe", re.IGNORECASE)),
    ("qwen2", re.compile(r"qwen.?2", re.IGNORECASE)),
    ("qwen", re.compile(r"qwen", re.IGNORECASE)),
    ("llama3", re.compile(r"llama.?3", re.IGNORECASE)),
    ("llama2", re.compile(r"llama.?2", re.IGNORECASE)),
    ("llama", re.compile(r"llama", re.IGNORECASE)),
    ("mixtral", re.compile(r"mixtral", re.IGNORECASE)),
    ("mistral", re.compile(r"mistral", re.IGNORECASE)),
    ("phi3_5", re.compile(r"phi.?3\.?5", re.IGNORECASE)),
    ("phi3", re.compile(r"phi.?3", re.IGNORECASE)),
    ("gemma3", re.compile(r"gemma.?3", re.IGNORECASE)),
    ("gemma2", re.compile(r"gemma.?2", re.IGNORECASE)),
    ("gemma", re.compile(r"gemma", re.IGNORECASE)),
    ("deepseek_v3", re.compile(r"deepseek.*v.?3", re.IGNORECASE)),
    ("deepseek_v2", re.compile(r"deepseek.*v.?2", re.IGNORECASE)),
    ("deepseek", re.compile(r"deepseek", re.IGNORECASE)),
    ("yi", re.compile(r"\byi\b", re.IGNORECASE)),
    ("command_r", re.compile(r"command.?r", re.IGNORECASE)),
    ("starcoder2", re.compile(r"starcoder.?2", re.IGNORECASE)),
)


def _infer_gguf_arch(filename: str, config: dict[str, Any] | None) -> str | None:
    """Best-effort guess of a GGUF file's architecture.

    Strategy: prefer ``config.get('general.architecture')`` if a config.json
    accompanies the GGUF (rare for raw-quant repos), else scan the filename
    against ``_GGUF_ARCH_FILENAME_PATTERNS``. Returns a lower-case canonical
    slug or None when nothing matches — the caller treats None as
    "couldn't infer; emit a `gguf_arch_unknown` warning".
    """
    if config is not None:
        # config.json reduction in ``_select_config_keys`` drops everything
        # except six numeric fields, so check the raw dict by passing it in
        # via the second argument from ``discover_repo_files`` (which has
        # the raw config in hand). The reduced dict has no architecture
        # key — fall through to the filename heuristic if we're called
        # with the trimmed dict.
        for key in ("general.architecture", "model_type", "architectures"):
            v = config.get(key)
            if isinstance(v, list) and v:
                v = v[0]
            if isinstance(v, str) and v.strip():
                return v.strip().lower()
    base = filename.rsplit("/", 1)[-1]
    for arch, pat in _GGUF_ARCH_FILENAME_PATTERNS:
        if pat.search(base):
            return arch
    return None


# Sharded safetensors filenames look like ``model-00001-of-00004.safetensors``.
_SAFETENSORS_SHARD_RE = re.compile(r"^.*-\d{5}-of-\d{5}\.safetensors$")
# Sharded pytorch_model filenames look like
# ``pytorch_model-00001-of-00002.bin``. We classify both shapes as
# ``pytorch_bin`` (no separate "sharded" bucket — bins are legacy enough that
# the FE doesn't need to split them out).
_PYTORCH_BIN_RE = re.compile(r"^pytorch_model(-\d{5}-of-\d{5})?\.bin$")

# GGUF tag, case-insensitive. The K-quant family ("Q4_K_M", "Q5_K_M", "Q3_K_S",
# etc.) plus the legacy "Q8_0"/"Q4_0" forms. Tag is normalised UPPERCASE on
# return so the FE doesn't have to.
_GGUF_QUANT_RE = re.compile(r"[.\-_](Q\d_K(?:_[SML])?|Q\d_\d)(?=\.|$|[._-])", re.IGNORECASE)
# Safetensors / generic quant markers — AWQ, GPTQ, FP16 — separated by a
# dot/dash/underscore. Order matters: AWQ before GPTQ before FP16 so a name
# like "model-awq-gptq" returns "AWQ" deterministically (multiple-quant
# filenames are pathological and shouldn't happen in practice).
_OTHER_QUANT_RE = re.compile(r"[.\-_](AWQ|GPTQ|FP16|INT8|INT4)(?=\.|$|[._-])", re.IGNORECASE)

# Parameter count: matches "{number}{unit}" where unit is B/M (case-insensitive)
# and number may be fractional ("0.5B", "1.3B"). Token must be word-boundary
# so "embeddings" (ending in "s") doesn't false-match. Hint-only: expect
# false positives on filenames that happen to contain disconnected
# digit-letter sequences (e.g. "foo-2B-bar" inside an unrelated filename).
# The FE treats ``params`` as a display hint, not authoritative metadata —
# real param counts come from the safetensors header.
_PARAMS_RE = re.compile(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)([BM])(?![A-Za-z])", re.IGNORECASE)


def _classify(filename: str) -> FileKind:
    """Classify a filename into one of the FileKind buckets.

    Pure string op — no I/O. Used both server-side (in ``discover_repo_files``)
    and (planned) client-side via the same canonical bucket names so the FE
    can colour-code rows without recomputing the mapping.
    """
    base = filename.rsplit("/", 1)[-1]
    if base == "config.json":
        return "config"
    if base in _TOKENIZER_NAMES:
        return "tokenizer"
    if base.endswith(".gguf"):
        # Checked INSIDE the .gguf branch, so nothing that is not a GGUF can be
        # classified as a projector -- --mmproj takes a GGUF, and the kind is
        # about the file, not about the name.
        if _MMPROJ_RE.match(base):
            return "mmproj"
        return "gguf"
    if _SAFETENSORS_SHARD_RE.match(base):
        return "safetensors_sharded"
    if base.endswith(".safetensors"):
        return "safetensors_single"
    if _PYTORCH_BIN_RE.match(base):
        return "pytorch_bin"
    return "other"


def _parse_quant(filename: str) -> str | None:
    """Extract a canonical quant tag from a filename, or None if absent.

    Returns the GGUF K-quant family ("Q4_K_M", "Q5_K_M", "Q8_0", "Q4_0", ...)
    or AWQ / GPTQ / FP16 / INT8 / INT4 markers. Case-insensitive input,
    UPPERCASE output. Returns None when no recognisable tag is present
    (rather than guessing) so the FE never gets a fabricated quant label.
    """
    base = filename.rsplit("/", 1)[-1]
    m = _GGUF_QUANT_RE.search(base)
    if m:
        return m.group(1).upper()
    m = _OTHER_QUANT_RE.search(base)
    if m:
        return m.group(1).upper()
    return None


def _parse_params(filename: str) -> int | None:
    """Extract a parameter-count hint from a filename.

    Returns an integer (e.g. ``7_000_000_000`` for "7B", ``500_000_000`` for
    "0.5B", ``125_000_000`` for "125m"). Returns None when no
    ``{number}{B|M}`` token surrounded by separators is present.

    This is a HINT, not ground truth — for a real model param count we'd
    have to read the safetensors header. The HF UI uses the same filename
    convention so it's good enough for the wizard's display row.
    """
    base = filename.rsplit("/", 1)[-1]
    m = _PARAMS_RE.search(base)
    if not m:
        return None
    n = float(m.group(1))
    unit = m.group(2).upper()
    if unit == "B":
        return int(n * 1_000_000_000)
    if unit == "M":
        return int(n * 1_000_000)
    return None


# ---- I/O wrapper ----------------------------------------------------------


class HfApiLike(Protocol):
    """Subset of ``huggingface_hub.HfApi`` we depend on.

    Declared as a Protocol so the route tests can pass a hand-rolled fake
    without import-time coupling to the real HfApi class.
    """

    def model_info(
        self, repo_id: str, *, revision: str | None = None, files_metadata: bool = False
    ) -> Any: ...


HfApiFactory = Any  # callable: (token: str | None) -> HfApiLike
ConfigFetcher = Any  # callable: (repo_id, revision, token) -> dict | None


@dataclass(frozen=True)
class DiscoveredFile:
    filename: str
    size: int
    kind: FileKind
    quant: str | None
    params: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "size": self.size,
            "kind": self.kind,
            "quant": self.quant,
            "params": self.params,
        }


@dataclass(frozen=True)
class DiscoveryWarning:
    """A soft, per-file signal surfaced to the FE wizard (#101).

    Distinct from ``DiscoveryResult.errors`` which lists repo-level hard
    failures (config_not_found, config_fetch_failed). Warnings are
    file-scoped and don't block selection — they show up as an amber
    banner inline with the file row so the operator can decide whether
    to proceed.
    """

    type: str  # "gguf_arch_unsupported" | "gguf_arch_unknown"
    filename: str
    arch: str | None  # inferred lower-case arch, or None when unknown

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "filename": self.filename, "arch": self.arch}


@dataclass(frozen=True)
class DiscoveryResult:
    files: list[DiscoveredFile]
    config: dict[str, Any] | None
    repo: dict[str, Any]
    errors: list[str] = field(default_factory=list)
    warnings: list[DiscoveryWarning] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": [f.to_dict() for f in self.files],
            "config": self.config,
            "repo": self.repo,
            "errors": list(self.errors),
            "warnings": [w.to_dict() for w in self.warnings],
        }


class DiscoveryAuthRequired(Exception):
    """HF Hub returned 401/403 for a gated or private repo we lack rights to."""


class DiscoveryNotFound(Exception):
    """HF Hub returned 404 — repo_id is wrong or the revision doesn't exist."""


def _default_hf_api_factory(token: str | None) -> HfApiLike:
    # Imported lazily so unit tests for the pure helpers don't pay the
    # huggingface_hub import cost.
    from huggingface_hub import HfApi  # noqa: PLC0415

    return HfApi(token=token)


def _default_config_fetcher(
    repo_id: str, revision: str | None, token: str | None
) -> dict[str, Any] | None:
    """Fetch and parse ``config.json`` from the HF Hub.

    Uses ``hf_hub_download`` so the file gets cached the same way the rest of
    the pull flow caches things. Returns None on 404 (caller decides whether
    that's an error or just "this isn't a transformers-shaped repo").

    Raises HF errors (RepositoryNotFoundError, GatedRepoError, etc.) up to the
    caller so they can be classified into our typed envelopes.
    """
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    from huggingface_hub.errors import EntryNotFoundError  # noqa: PLC0415

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            revision=revision,
            token=token,
        )
    except EntryNotFoundError:
        return None
    with open(path, encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
        return data


def _select_config_keys(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    """Reduce a raw ``config.json`` to the keys the FE wizard needs.

    Returns None if input is None. Returns a dict with every key (defaulting
    to None when absent) so the FE can assume the shape.
    """
    if raw is None:
        return None
    return {k: raw.get(k) for k in _CONFIG_KEYS}


def _file_from_sibling(sibling: Any) -> DiscoveredFile:
    filename = str(getattr(sibling, "rfilename", None) or getattr(sibling, "filename", "") or "")
    size = int(getattr(sibling, "size", 0) or 0)
    return DiscoveredFile(
        filename=filename,
        size=size,
        kind=_classify(filename),
        quant=_parse_quant(filename),
        params=_parse_params(filename),
    )


async def discover_repo_files(
    repo_id: str,
    revision: str | None,
    token: str | None,
    *,
    hf_api_factory: HfApiFactory = _default_hf_api_factory,
    config_fetcher: ConfigFetcher = _default_config_fetcher,
) -> DiscoveryResult:
    """List files + config for an HF Hub repo, classified for the FE wizard.

    Returns a ``DiscoveryResult`` with a stable wire shape. Network/auth
    errors are surfaced as typed exceptions (``DiscoveryAuthRequired``,
    ``DiscoveryNotFound``) so the route can translate them to JSON envelopes
    — everything else propagates and the route returns 502.

    The ``hf_api_factory`` / ``config_fetcher`` seams exist purely for tests:
    the route always uses the default factories.
    """
    # Imported lazily so this module loads cleanly even if huggingface_hub
    # is missing in a tools-only environment.
    from huggingface_hub.errors import (  # noqa: PLC0415
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )

    def _sync() -> tuple[Any, dict[str, Any] | None, list[str]]:
        errors: list[str] = []
        api = hf_api_factory(token)
        try:
            info = api.model_info(repo_id, revision=revision, files_metadata=True)
        except GatedRepoError as e:
            raise DiscoveryAuthRequired(str(e)) from e
        except RepositoryNotFoundError as e:
            raise DiscoveryNotFound(str(e)) from e
        except HfHubHTTPError as e:
            # 401/403 surface as HfHubHTTPError when the token is missing
            # entirely (vs. wrong) — classify by status code.
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (401, 403):
                raise DiscoveryAuthRequired(str(e)) from e
            if status == 404:
                raise DiscoveryNotFound(str(e)) from e
            raise

        try:
            raw_config = config_fetcher(repo_id, revision, token)
        except GatedRepoError as e:
            raise DiscoveryAuthRequired(str(e)) from e
        # Defensive: race window if the repo is deleted between model_info()
        # and hf_hub_download(). Dead code in normal flow — model_info would
        # have raised RepositoryNotFoundError above — but kept so a deletion
        # between the two HF calls degrades gracefully instead of 500ing.
        except RepositoryNotFoundError:
            # File-level 404 on config.json — repo exists but no transformers
            # config (e.g. pure GGUF repo).
            raw_config = None
            errors.append("config_not_found")
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (401, 403):
                raise DiscoveryAuthRequired(str(e)) from e
            if status == 404:
                raw_config = None
                errors.append("config_not_found")
            else:
                # Any other transport error for config.json is non-fatal —
                # the file list is the primary product of this endpoint.
                logger.warning("config.json fetch failed for %s@%s: %s", repo_id, revision, e)
                raw_config = None
                errors.append("config_fetch_failed")
        except Exception:  # noqa: BLE001
            logger.exception("config.json fetch raised for %s@%s", repo_id, revision)
            raw_config = None
            errors.append("config_fetch_failed")

        if (
            raw_config is None
            and "config_not_found" not in errors
            and "config_fetch_failed" not in errors
        ):
            errors.append("config_not_found")

        return info, raw_config, errors

    info, raw_config, errors = await asyncio.to_thread(_sync)

    siblings = list(getattr(info, "siblings", None) or [])
    files = [_file_from_sibling(s) for s in siblings]

    # #101: emit per-file GGUF-arch warnings AFTER classification. We use the
    # RAW (un-trimmed) config so the architectures/model_type fields survive,
    # then fall back to the filename heuristic for raw-quant repos that ship
    # no config.json at all. Each GGUF file gets at most one warning row.
    warnings: list[DiscoveryWarning] = []
    for f in files:
        if f.kind != "gguf":
            continue
        arch = _infer_gguf_arch(f.filename, raw_config)
        if arch is None:
            warnings.append(
                DiscoveryWarning(
                    type="gguf_arch_unknown",
                    filename=f.filename,
                    arch=None,
                )
            )
        elif arch not in KNOWN_GGUF_ARCHES:
            warnings.append(
                DiscoveryWarning(
                    type="gguf_arch_unsupported",
                    filename=f.filename,
                    arch=arch,
                )
            )

    repo_payload = {
        "id": repo_id,
        "revision": revision or "main",
        "private": bool(getattr(info, "private", False)),
        "gated": bool(getattr(info, "gated", False)),
    }
    return DiscoveryResult(
        files=files,
        config=_select_config_keys(raw_config),
        repo=repo_payload,
        errors=errors,
        warnings=warnings,
    )


# ---- Token loader ---------------------------------------------------------


async def load_hf_token(settings: Any) -> str | None:
    """Read ``Settings.hf_token_path`` off the event loop.

    Mirrors ``app.models.pull_task._read_hf_token`` so the discovery path
    silently reuses whatever the operator set during onboarding — gated-repo
    access doesn't get a second confirmation prompt (per CTO decision in #84).
    """

    def _read() -> str | None:
        p = settings.hf_token_path
        if p.exists():
            v = p.read_text().strip()
            return v or None
        return None

    return await asyncio.to_thread(_read)
