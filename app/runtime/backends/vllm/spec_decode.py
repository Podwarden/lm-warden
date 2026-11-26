"""How a vLLM engine's prefix hits are quantised: the match unit, and whether
it drops the last matched unit on a hit.

``--prefix-match-unit N`` (CacheConfig ``prefix_match_unit``, vLLM 0.32)
matches prefixes every N tokens instead of every physical block; N must
divide the block size. Unset, the unit is the block.

vLLM 0.32.0 does, whenever its speculative method uses the target model's
hidden states (``SpeculativeConfig.use_eagle()``: eagle, eagle3, mtp, dflash,
dspark; every ``*_mtp`` model-specific name is normalised to ``mtp``). The
drafter needs the tokens just before the generation point recomputed, so the
KV cache manager pops one unit off every prefix hit (one whole block when the
match unit is the block; ``hit_length -= min(alignment_tokens, block_size)``)
(``v1/core/single_type_kv_cache_manager.py``, ``drop_eagle_block``; wired in
``v1/core/sched/scheduler.py`` from ``use_eagle_block_drop()``), and a hybrid
model's Mamba checkpoints follow the same boundary
(``kv_cache_coordinator.py``, ``drop_eagle_checkpoint_block``). The
experimental ``disable_eagle_block_drop`` turns it off.

On 16-token pages that is noise. On a hybrid model's large aligned blocks
(1,584 tokens on lmwarden.com's Qwen) it is a whole block per request, which
the cache-outcome classifier must not count as reusable
(app/cache_obs/classify.py).

The answer comes from the launch args, the only place the warden sees the
speculative config. No evidence (no flag, an unparseable value, no method)
answers False, which keeps the classifier's previous rules.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

#: ``SpeculativeConfig.use_eagle()`` at vLLM v0.32.0 (config/speculative.py).
_EAGLE_LIKE = frozenset(("eagle", "eagle3", "mtp", "dflash", "dspark"))
_FLAGS = ("--speculative-config", "--speculative_config")
_UNIT_FLAGS = ("--prefix-match-unit", "--prefix_match_unit")


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _spec_config(args: Sequence[str]) -> dict[str, Any]:
    """The speculative config from argv: one JSON value, or dotted
    ``--speculative-config.<key>`` flags (vLLM accepts both forms)."""
    out: dict[str, Any] = {}
    i = 0
    while i < len(args):
        arg = str(args[i])
        name, eq, value = arg.partition("=")
        for flag in _FLAGS:
            if name == flag or name.startswith(flag + "."):
                if not eq:
                    if i + 1 >= len(args):
                        return out
                    i += 1
                    value = str(args[i])
                key = name[len(flag) + 1 :]
                if key:
                    out[key] = value
                else:
                    try:
                        parsed = json.loads(value)
                    except ValueError:
                        parsed = None
                    if isinstance(parsed, dict):
                        out.update(parsed)
                break
        i += 1
    return out


def drops_last_cached_block(extra_args: Sequence[str] | None) -> bool:
    """True when an engine launched with ``extra_args`` drops one whole
    prefix-cache block from every hit (EAGLE-like speculative decoding)."""
    if not extra_args:
        return False
    cfg = _spec_config(list(extra_args))
    method = cfg.get("method")
    if not isinstance(method, str):
        return False
    method = method.strip().lower()
    if method not in _EAGLE_LIKE and not method.endswith("_mtp"):
        return False
    return not _truthy(cfg.get("disable_eagle_block_drop", False))


def prefix_match_unit(extra_args: Sequence[str] | None, block_size: int) -> int:
    """The engine's effective prefix-match unit in tokens: ``--prefix-match-unit``
    when it is a positive divisor of ``block_size`` no larger than it, else the
    block size (vLLM's default, and what an unusable value leaves us assuming)."""
    raw: str | None = None
    args = [str(a) for a in extra_args or ()]
    for i, arg in enumerate(args):
        name, eq, value = arg.partition("=")
        if name in _UNIT_FLAGS:
            if eq:
                raw = value
            elif i + 1 < len(args):
                raw = args[i + 1]
    try:
        unit = int(raw) if raw is not None else 0
    except ValueError:
        unit = 0
    if 0 < unit <= block_size and block_size % unit == 0:
        return unit
    return block_size


def speculative_method(extra_args: Sequence[str] | None) -> str | None:
    """The speculative-decoding method an engine launched with ``extra_args`` runs, lower case
    (a model-specific ``*_mtp`` name reads ``mtp``, as vLLM normalises it), or None when the
    args name none."""
    if not extra_args:
        return None
    method = _spec_config(list(extra_args)).get("method")
    if not isinstance(method, str) or not method.strip():
        return None
    method = method.strip().lower()
    return "mtp" if method.endswith("_mtp") else method
