"""Load admission for unified-memory hosts (Metal design s10, "The GpuOwnership trap").

The supervisor admits every load through one collaborator with
``claim(model_id, gpu_indices, *, label, model)`` / ``release(model_id)``.
On discrete GPUs that is ``GpuOwnership`` (one model per card, unchanged).
On a Mac every row is ``gpu_indices == [0]``, so ``GpuOwnership`` would
refuse the SECOND model and silently serialise the product to one model per
machine. There is nothing to own on unified memory -- the question is bytes:
``UnifiedMemoryAdmission`` refuses only when ``resident + estimate > budget``.

Choices (kept deliberately small; each is a seam, not a final answer):

* **Resident bytes are the admission's own ledger** of the estimates it
  admitted, released through the same ``release`` calls the supervisor
  already makes on unload, crash and failed spawn. The claim happens before
  the engine exists, so there is no pid to measure yet; measured
  ``phys_footprint`` (``app/system/apple/proc.py``) can replace the ledger
  later without changing the seam.
* **The estimate is injected** (``model -> bytes``). The profile factory
  sizes the newest snapshot's files that the row's pull patterns select --
  the pinned GGUF, its shard siblings, the mmproj projector and the
  config/tokenizer text files (the whole snapshot for a row that pins no
  filename) -- following the snapshot's symlinks into ``blobs/``.
* **The KV cache is a second injected estimate** (``(model, overrides) ->
  KvReserve | None``), sized by ``kv_cache_reserve`` from the snapshot's
  ``config.json`` with the same formula as the load preflight
  (``app/models/fit.kv_reserve_bytes``). One rule serves MLX and llama.cpp;
  only llama.cpp has ``--cache-type-k/-v`` to narrow the element width.

  - The context is the live override (the preflight's cap, a reload-with-
    config) when one is set, else the row's ``max_model_len``. Either is
    FIRM: weights + KV must fit, because the engine is told that context.
  - llama.cpp's own ``-c``/``--ctx-size`` in ``extra_args`` beats both
    (extra_args are appended last, so the engine obeys it); ``0`` there
    means the model max, as it does to llama.cpp.
  - With none of them, the model's own ``max_position_embeddings`` is
    charged SOFTLY. Neither engine allocates it at load -- llama.cpp's
    ``--fit`` shrinks an unflagged context to what is free, mlx_lm grows
    lazily -- so it is never the cause of a refusal. Admission counts only
    FIRM bytes (weights + firm KV); soft wants then share whatever the firm
    bytes leave, first-claimed first, recomputed after every claim and
    release. A later load therefore reclaims soft room instead of being
    refused by it, and the ledger total never exceeds the budget.
  - An unreadable shape (no config.json -- most GGUF-only repos) falls back
    to a soft quarter of the weights (``KV_FALLBACK_DIVISOR``): never 0, and
    never a refusal the weights-only rule would not have made.
* **Fail closed.** A budget <= 0 (the capacity reading is 0 when hw.memsize is
  unreadable; ``unified_weights_budget`` can go negative) and an estimate
  that is unknown (<= 0) or raises both refuse the load, with a message
  saying why, rather than admitting at zero cost and overcommitting the Mac.
"""

from __future__ import annotations

import fnmatch
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from threading import Lock
from typing import Any

from app.runtime.gpu_ownership import GpuConflict, GpuOwnership

logger = logging.getLogger(__name__)

_GIB = 1024**3


# Unknown KV shape -> a soft reserve of weights // 4. A quarter is about what
# an 8B model's KV needs at a 32k context against its 4-bit weights.
KV_FALLBACK_DIVISOR = 4

# Bytes per element of llama.cpp's --cache-type-k/-v values (ggml block sizes:
# q8_0 = 34 bytes per 32 elements, ...). Anything else is charged f32 -- the
# widest -- so an unrecognised type is never under-counted.
_GGML_KV_WIDTH = {
    "f32": Fraction(4),
    "f16": Fraction(2),
    "bf16": Fraction(2),
    "q8_0": Fraction(34, 32),
    "q5_1": Fraction(24, 32),
    "q5_0": Fraction(22, 32),
    "q4_1": Fraction(20, 32),
    "q4_0": Fraction(18, 32),
    "iq4_nl": Fraction(18, 32),
}
_CACHE_K_FLAGS = ("--cache-type-k", "-ctk")
_CACHE_V_FLAGS = ("--cache-type-v", "-ctv")
_CTX_SIZE_FLAGS = ("--ctx-size", "-c")


def _gib(n: int) -> str:
    return f"{n / _GIB:.1f} GiB"


@dataclass(frozen=True)
class KvReserve:
    """The KV cache a load will hold: ``nbytes`` for ``context`` tokens.

    ``firm`` means the engine is TOLD this context, so it must fit; a soft
    reserve is held only up to the headroom left after the weights.
    """

    nbytes: int
    context: int
    firm: bool


def _positive_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _flag_value(extra_args: Any, flags: Any) -> str | None:
    """The last value given to any of ``flags`` (``--f v`` or ``--f=v``), or None."""
    found = None
    args = [a for a in (extra_args or []) if isinstance(a, str)]
    for i, arg in enumerate(args):
        flag, eq, value = arg.partition("=")
        if flag in flags:
            found = value if eq else (args[i + 1] if i + 1 < len(args) else "")
    return found


def _kv_widths(extra_args: Any, default: Fraction) -> tuple[Fraction, Fraction]:
    """K and V element widths: llama.cpp's cache-type flags (last wins), else ``default``."""
    widths = []
    for flags in (_CACHE_K_FLAGS, _CACHE_V_FLAGS):
        value = _flag_value(extra_args, flags)
        widths.append(
            default if value is None else _GGML_KV_WIDTH.get(value.strip().lower(), Fraction(4))
        )
    return widths[0], widths[1]


def kv_cache_reserve(
    config: dict[str, Any] | None, model: Any, overrides: dict[str, Any] | None
) -> KvReserve | None:
    """Size ``model``'s KV cache from its HF ``config.json``; None when unknowable."""
    from app.models.fit import dtype_bytes_from_torch_dtype, kv_reserve_bytes

    if not isinstance(config, dict):
        return None
    cfg = config
    text = config.get("text_config")
    if not _positive_int(config.get("num_hidden_layers")) and isinstance(text, dict):
        cfg = {"torch_dtype": config.get("torch_dtype"), **text}
    hidden = _positive_int(cfg.get("hidden_size"))
    layers = _positive_int(cfg.get("num_hidden_layers"))
    heads = _positive_int(cfg.get("num_attention_heads"))
    kv_heads = _positive_int(cfg.get("num_key_value_heads")) or heads
    if not (hidden and layers and heads and kv_heads):
        return None

    ov = overrides or {}
    flagged = _flag_value(getattr(model, "extra_args", None), _CTX_SIZE_FLAGS)
    if flagged is not None:
        digits = flagged.strip()
        # ASCII only: str.isdigit() accepts "²", which int() then rejects.
        context = _positive_int(int(digits)) if digits.isascii() and digits.isdigit() else 0
    else:
        context = _positive_int(
            ov["max_model_len"] if "max_model_len" in ov else getattr(model, "max_model_len", None)
        )
    firm = bool(context)
    if not firm:
        context = _positive_int(cfg.get("max_position_embeddings"))
        if not context:
            return None

    # Same sliding-window reading as fit_preview: the explicit per-layer
    # declaration only, and only with a window to charge.
    layer_types = cfg.get("layer_types")
    window = _positive_int(cfg.get("sliding_window"))
    sliding = 0
    if isinstance(layer_types, list) and window:
        sliding = min(sum(1 for t in layer_types if t == "sliding_attention"), layers)

    # kv_reserve_bytes at 1 byte/element counts K+V elements (its factor 2);
    # half are K, half are V, each at its own width.
    elements = kv_reserve_bytes(
        hidden_size=hidden,
        num_layers=layers,
        num_kv_heads=kv_heads,
        num_attention_heads=heads,
        max_model_len=context,
        dtype_bytes=1,
        max_batch_size=_positive_int(getattr(model, "max_batch_size", None)) or 1,
        head_dim=_positive_int(cfg.get("head_dim")) or None,
        sliding_layers=sliding,
        sliding_window=window or None,
    )
    k, v = _kv_widths(
        getattr(model, "extra_args", None),
        Fraction(dtype_bytes_from_torch_dtype(cfg.get("torch_dtype"))),
    )
    return KvReserve(nbytes=int(elements * (k + v) / 2), context=context, firm=firm)


class UnifiedMemoryAdmission:
    """Admit loads against a unified-memory weights budget, in bytes."""

    def __init__(
        self,
        *,
        budget_bytes: int,
        estimate: Callable[[Any], int],
        kv_estimate: Callable[[Any, dict[str, Any] | None], KvReserve | None] | None = None,
    ) -> None:
        self.budget_bytes = budget_bytes
        self._estimate = estimate
        self._kv_estimate = kv_estimate
        # model_id -> firm bytes (weights + firm KV), the only bytes admission
        # counts; soft wants and the share of the free room each one holds.
        self._firm: dict[str, int] = {}
        self._soft_want: dict[str, int] = {}
        self._soft_held: dict[str, int] = {}
        self._labels: dict[str, str] = {}
        self._lock = Lock()

    def claim(
        self,
        model_id: str,
        gpu_indices: list[int],
        *,
        label: str | None = None,
        model: Any = None,
        overrides: dict[str, Any] | None = None,
    ) -> None:
        name = label or model_id
        with self._lock:
            if model_id in self._firm:
                return  # re-claim by the holder: already counted
            if self.budget_bytes <= 0:
                raise GpuConflict(
                    f"Cannot load '{name}': the unified-memory budget could not be "
                    "determined (machine memory unreadable, or the OS reserve leaves "
                    "nothing for models). Refusing rather than risk overcommitting."
                )
            try:
                needed = int(self._estimate(model))
            except Exception as error:
                logger.warning("unified admission: estimate for %s failed: %s", model_id, error)
                needed = 0
            if needed <= 0:
                raise GpuConflict(
                    f"Cannot load '{name}': its size could not be measured (are the "
                    "model files downloaded?). Refusing rather than admit it at zero cost."
                )
            kv = self._kv(model_id, model, overrides, needed)
            firm = needed + (kv.nbytes if kv.firm else 0)
            used = sum(self._firm.values())
            if used + firm > self.budget_bytes:
                raise GpuConflict(self._refusal(name, firm, used, needed, kv))
            self._firm[model_id] = firm
            self._soft_want[model_id] = 0 if kv.firm else kv.nbytes
            self._share_soft_room()
            if label:
                self._labels[model_id] = label

    def _share_soft_room(self) -> None:
        """Hand the room firm bytes leave to soft wants, first-claimed first. Caller holds the lock."""
        free = max(0, self.budget_bytes - sum(self._firm.values()))
        for model_id, want in self._soft_want.items():
            self._soft_held[model_id] = min(want, free)
            free -= self._soft_held[model_id]

    def _kv(
        self, model_id: str, model: Any, overrides: dict[str, Any] | None, weights: int
    ) -> KvReserve:
        """The KV reserve, or the soft fallback when it cannot be sized."""
        if self._kv_estimate is None:
            return KvReserve(nbytes=0, context=0, firm=False)
        try:
            kv = self._kv_estimate(model, overrides)
        except Exception as error:
            logger.warning("unified admission: KV estimate for %s failed: %s", model_id, error)
            kv = None
        if kv is None:
            logger.info(
                "unified admission: KV shape of %s unknown; reserving %s softly",
                model_id,
                _gib(weights // KV_FALLBACK_DIVISOR),
            )
            return KvReserve(nbytes=weights // KV_FALLBACK_DIVISOR, context=0, firm=False)
        return kv

    def _refusal(self, name: str, needed: int, used: int, weights: int, kv: KvReserve) -> str:
        """Name the occupants and the way out. Caller holds the lock."""
        if self._firm:
            listed = ", ".join(
                f"'{self._labels.get(m, m)}' ({_gib(b)})" for m, b in self._firm.items()
            )
            occupied = f"; {_gib(used)} is already used by {listed}"
            remedy = "unload a model first, or load a smaller one."
        else:
            occupied = ""
            remedy = "load a smaller model."
        if kv.firm and kv.nbytes:
            needs = (
                f"{_gib(needed)} ({_gib(weights)} weights + {_gib(kv.nbytes)} KV cache "
                f"for a {kv.context}-token context)"
            )
            remedy = remedy[:-1] + ", or lower max_model_len."
        else:
            needs = _gib(needed)
        return (
            f"Not enough unified memory to load '{name}': it needs about {needs} "
            f"of a {_gib(self.budget_bytes)} model budget{occupied} — {remedy}"
        )

    def release(self, model_id: str) -> None:
        with self._lock:
            self._firm.pop(model_id, None)
            self._soft_want.pop(model_id, None)
            self._soft_held.pop(model_id, None)
            self._labels.pop(model_id, None)
            self._share_soft_room()

    def resident(self) -> dict[str, int]:
        with self._lock:
            return {m: b + self._soft_held.get(m, 0) for m, b in self._firm.items()}


def _loaded_files_estimate(hf_cache_dir: Path) -> Callable[[Any], int]:
    from app.models.pull_task import allow_patterns_for
    from app.runtime.backends.paths import _snapshot_dir

    def estimate(model: Any) -> int:
        repo = getattr(model, "hf_repo", None)
        if not repo:
            return 0
        filename = getattr(model, "filename", None)
        snap = _snapshot_dir(Path(hf_cache_dir), repo)
        if snap is None:
            return 0
        patterns = allow_patterns_for(
            filename, mmproj_filename=getattr(model, "mmproj_filename", None)
        )
        if filename is not None and not (snap / filename).is_file():
            # Fail closed: a pinned weights file that is missing must not be
            # admitted at the size of its tokenizer.
            return 0
        total = 0
        for p in sorted(snap.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(snap).as_posix()
            if patterns is not None and not any(fnmatch.fnmatch(rel, pat) for pat in patterns):
                continue
            try:
                total += p.stat().st_size
            except (FileNotFoundError, OSError):
                continue
        return total

    return estimate


def _config_kv_estimate(hf_cache_dir: Path) -> Callable[[Any, dict[str, Any] | None], Any]:
    from app.runtime.backends.paths import _snapshot_dir

    def kv_estimate(model: Any, overrides: dict[str, Any] | None) -> KvReserve | None:
        # #106: the config may live in a separate repo from the weights.
        repo = getattr(model, "hf_config_repo", None) or getattr(model, "hf_repo", None)
        if not isinstance(repo, str) or not repo:
            return None
        snap = _snapshot_dir(Path(hf_cache_dir), repo)
        if snap is None or not (snap / "config.json").is_file():
            return None
        try:
            config = json.loads((snap / "config.json").read_text())
        except (OSError, ValueError):
            return None
        return kv_cache_reserve(config, model, overrides)

    return kv_estimate


async def admission_for_profile(profile: str, *, hf_cache_dir: Any) -> Any:
    """Choose the supervisor's admission once, from the deployment profile.

    ``linux`` -> ``GpuOwnership`` (no probe runs). ``apple`` ->
    ``UnifiedMemoryAdmission`` whose budget is ``unified_weights_budget`` of
    the live capacity reading (``gpu._apple_capacity``, the same reading the
    GPU table and the shell feed use): the device ceiling
    ``recommendedMaxWorkingSetSize`` lowered by ``iogpu.wired_limit_mb``,
    with no OS reserve on top (the ceiling already leaves the OS its share).
    Fail closed: no ceiling -> hw.memsize less the OS reserve taken of
    ``hw.memsize_usable`` (the pre-P6 stand-in); no hw.memsize -> budget 0,
    which refuses every load.
    """
    if profile != "apple":
        return GpuOwnership()
    from app.system import gpu

    budget = (await gpu._apple_capacity()).weights_budget()
    return UnifiedMemoryAdmission(
        budget_bytes=budget,
        estimate=_loaded_files_estimate(hf_cache_dir),
        kv_estimate=_config_kv_estimate(hf_cache_dir),
    )
