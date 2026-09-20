"""Turn a vLLM engine-log tail into an actionable, operator-facing error.

When a load fails, the control plane historically reported only a generic
``last_error`` ("vllm subprocess exited unexpectedly (rc=1)"). The real cause
is in the engine log (``{data_dir}/logs/{model_id}.log``). This pure parser
scans the tail and, for the failure modes we can recognise, produces a message
that tells the operator exactly what to change.

Matching is on STABLE tokens, not exact phrasing — vLLM's wording drifts across
versions and even contains typos (the real observed string says ``models's``). We key
off invariants and capture numbers opportunistically. No match → ``None`` so
the caller keeps its existing generic message.

The failure modes we distinguish carefully (all observed live):

  0. **The card was already occupied.** vLLM's pre-flight free-memory check
     fires before it profiles anything ("Free memory on device cuda:0
     (1.65/15.6 GiB) on startup is less than desired GPU memory utilization").
     This is the most likely first-run failure on a host that already runs
     something, and it used to fall through every rule here, leaving the
     operator with a bare "vllm subprocess exited unexpectedly (rc=1)".

  1. **KV cache too small for context.** vLLM prints its OWN profiled fit
     estimate ("the estimated maximum model length is N"). That estimate beats
     any static heuristic, so we surface N as ``recommended_max_model_len`` and
     tell the operator to cap to it (or raise gpu_memory_utilization). This is
     the ``tencent/Hy-MT2-1.8B`` footgun.

  2. **No room for the cache blocks at all** ("No available memory for the
     cache blocks", typically with a NEGATIVE "Available KV cache memory"). The
     weights + overhead already exceed the budget; lowering ``max_model_len``
     does NOT help. We must NOT suggest it — only more/larger GPUs or a higher
     gpu_memory_utilization. ``recommended_max_model_len`` stays ``None``.

There is deliberately NO auto-retry here: the post-crash path only REPORTS.
Auto-relaunching on a parsed estimate risks oscillation; the static pre-spawn
preflight (``app/models/load_preflight.py``) is the only place we adjust a
launch, and only for the NULL-max_model_len case.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EngineDiagnosis:
    """An actionable diagnosis extracted from an engine log.

    ``message`` is operator-facing (goes straight into ``last_error``).
    ``recommended_max_model_len`` is vLLM's own profiled fit estimate when the
    failure was a context/KV overflow and we could capture it, else ``None``.
    """

    message: str
    recommended_max_model_len: int | None = None


# --- Variant 0: another process already holds the card (pre-flight). --------
# vLLM's very first memory check, before it profiles anything:
#   ValueError: Free memory on device cuda:0 (1.65/15.6 GiB) on startup is less
#   than desired GPU memory utilization (0.13, 2.03 GiB). Decrease GPU memory
#   utilization or reduce GPU memory used by other processes.
# It is the single most likely first-run failure on a box that already runs
# something -- and it used to fall through every rule below, so the operator
# saw only "vllm subprocess exited unexpectedly (rc=1)" with no hint that the
# card was simply occupied. Checked FIRST: the check aborts startup before
# profiling, so no other variant's vocabulary can be present, and the numbers
# it prints are the whole diagnosis.
# Deliberately single-line (no DOTALL) and length-bounded between the two
# halves: vLLM prints this as one sentence, and an unbounded gap would let a
# stray "Free memory on device ..." INFO line pair up with a "less than
# desired" sentence hundreds of lines later and invent a diagnosis.
_FREE_MEM_PREFLIGHT_RE = re.compile(
    r"free\s+memory\s+on\s+device\s+(\S+?)\s*"
    r"\(\s*([\d.]+)\s*/\s*([\d.]+)\s*GiB\s*\)"
    r"[^\n]{0,80}?less\s+than\s+desired",
    re.IGNORECASE,
)
# The requested side of the same sentence: "(0.13, 2.03 GiB)".
_FREE_MEM_DESIRED_RE = re.compile(
    r"less\s+than\s+desired\s+GPU\s+memory\s+utilization\s*"
    r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*GiB\s*\)",
    re.IGNORECASE,
)

# --- Variant 2: no room for the cache blocks (weights/util, NOT context). ----
# Checked FIRST because this line co-occurs with "KV cache" text and would
# otherwise be misclassified as a context overflow. Lowering max_model_len does
# not help here.
_NO_CACHE_BLOCKS_RE = re.compile(
    r"no\s+available\s+memory\s+for\s+the\s+cache\s+blocks", re.IGNORECASE
)

# --- Variant 1: KV cache too small for the requested context. ----------------
# Stable tokens: "max seq len" + "KV cache", or an explicit decrease-max_model_len
# instruction. vLLM's authoritative estimate, when present:
#   "the estimated maximum model length is 157216"
_MAX_SEQ_LEN_RE = re.compile(r"max\s+seq\s+len", re.IGNORECASE)
_KV_CACHE_RE = re.compile(r"kv\s+cache", re.IGNORECASE)
_DECREASE_MAXLEN_RE = re.compile(
    r"(decreas|lower|reduc)\w*\s+`?max_model_len", re.IGNORECASE
)
_ESTIMATED_MAX_LEN_RE = re.compile(
    r"estimated\s+maximum\s+model\s+length\s+is\s+(\d+)", re.IGNORECASE
)
# Model's declared max seq len, e.g. "max seq len (262144)".
_MODEL_SEQ_LEN_RE = re.compile(r"max\s+seq\s+len\s*\((\d+)\)", re.IGNORECASE)
# KV-cache GiB figures: "16.0 GiB KV cache is needed" / "available KV cache
# memory (9.6 GiB)". Captured loosely for the message only.
_KV_NEEDED_GIB_RE = re.compile(
    r"([\d.]+)\s*GiB\s+KV\s+cache\s+is\s+needed", re.IGNORECASE
)
_KV_AVAIL_GIB_RE = re.compile(
    r"available\s+KV\s+cache\s+memory\s*\(([\d.]+)\s*GiB\)", re.IGNORECASE
)

# --- CUDA OOM — torch exception class or the human string. -------------------
_CUDA_OOM_RE = re.compile(r"(cuda out of memory|outofmemoryerror)", re.IGNORECASE)

# --- trust_remote_code required prompt. --------------------------------------
# Match the SENTENCE transformers/vLLM emit when the flag is genuinely
# required -- never the flag name on its own.
#
# The bare substring `trust_remote_code` was a false positive on essentially
# every crash: vLLM's startup banner and every `dump_input.py` ERROR line echo
# the full engine config, which always contains `trust_remote_code=False`.
# Because this rule sits last, it became the default diagnosis for any
# unrecognised failure, and it told operators to enable arbitrary remote code
# execution to fix unrelated faults. On 2026-08-18 a TP worker hang was
# reported as "This model requires trust_remote_code to load" for 1h43m.
#
# Matching `trust_remote_code=True` would reintroduce the bug the moment an
# operator legitimately enables the flag, because then the config echo says
# True. Anchor on prose only.
_TRUST_REMOTE_CODE_RE = re.compile(
    r"requires you to execute"           # transformers: "...the configuration file"
    r"|requires you to load"             # transformers variant
    r"|set the option\s+`?trust_remote_code`?\s*=\s*True"
    r"|--trust-remote-code",             # the CLI flag as advice, not as config echo
    re.IGNORECASE,
)


def diagnose_engine_log(text: str) -> EngineDiagnosis | None:
    """Scan an engine-log tail and return an actionable diagnosis, or None."""
    if not text or not text.strip():
        return None

    # Variant 0 — the card was already occupied when the engine started. This
    # never reaches profiling, so it cannot collide with the variants below.
    preflight = _FREE_MEM_PREFLIGHT_RE.search(text)
    if preflight is not None:
        device, free_gib, total_gib = preflight.groups()
        parts = [
            f"Not enough free VRAM on {device} to start: "
            f"{free_gib} GiB free of {total_gib} GiB"
        ]
        desired = _FREE_MEM_DESIRED_RE.search(text)
        if desired is not None:
            util, want_gib = desired.groups()
            parts.append(
                f", but this model asks for {want_gib} GiB "
                f"(gpu_memory_utilization {util})"
            )
        parts.append(
            ". Something else is holding the card — free it, pick another GPU, "
            "or lower gpu_memory_utilization."
        )
        return EngineDiagnosis(
            message="".join(parts), recommended_max_model_len=None
        )

    # Variant 2 first — it shares vocabulary with Variant 1 but the fix differs.
    if _NO_CACHE_BLOCKS_RE.search(text):
        return EngineDiagnosis(
            message=(
                "GPU has no room for the KV cache after loading the weights — "
                "the model is too large for the selected GPU(s) at this "
                "gpu_memory_utilization. Use more/larger GPUs or raise "
                "gpu_memory_utilization."
            ),
            recommended_max_model_len=None,
        )

    # Variant 1 — KV / context overflow.
    has_max_seq = bool(_MAX_SEQ_LEN_RE.search(text))
    has_kv_cache = bool(_KV_CACHE_RE.search(text))
    has_decrease = bool(_DECREASE_MAXLEN_RE.search(text))
    if (has_max_seq and has_kv_cache) or has_decrease:
        est_m = _ESTIMATED_MAX_LEN_RE.search(text)
        recommended = int(est_m.group(1)) if est_m else None

        model_len_m = _MODEL_SEQ_LEN_RE.search(text)
        model_len = model_len_m.group(1) if model_len_m else None
        kv_needed_m = _KV_NEEDED_GIB_RE.search(text)
        kv_avail_m = _KV_AVAIL_GIB_RE.search(text)

        if recommended is not None:
            parts = ["Context too long for GPU memory:"]
            if model_len and kv_needed_m:
                parts.append(
                    f" model wants {model_len} tokens (needs "
                    f"{kv_needed_m.group(1)} GiB KV cache)"
                )
            elif model_len:
                parts.append(f" model wants {model_len} tokens")
            if kv_avail_m:
                parts.append(f" but only {kv_avail_m.group(1)} GiB is available.")
            else:
                parts.append(".")
            parts.append(
                f" vLLM estimates the max workable context here is {recommended} "
                f"— set max_model_len <= {recommended}, or raise "
                f"gpu_memory_utilization."
            )
            return EngineDiagnosis(
                message="".join(parts), recommended_max_model_len=recommended
            )

        # No authoritative estimate to capture — still actionable.
        return EngineDiagnosis(
            message=(
                "KV cache too small for the requested context. Lower "
                "max_model_len or raise gpu_memory_utilization for the selected "
                "GPU(s)."
            ),
            recommended_max_model_len=None,
        )

    # CUDA out of memory.
    if _CUDA_OOM_RE.search(text):
        return EngineDiagnosis(
            message=(
                "GPU ran out of memory loading the model. Use fewer/larger GPUs, "
                "lower gpu_memory_utilization, or reduce max_model_len."
            ),
            recommended_max_model_len=None,
        )

    # trust_remote_code required.
    #
    # #264: this used to say "enable trust_remote_code for this model", which
    # could not fix the error it responds to -- the `trust_remote_code` column
    # is never emitted to the engine (app/runtime/backends/vllm/args.py builds
    # no `--trust-remote-code`, pinned by tests/unit/runtime/backends/
    # test_launch_characterisation.py::test_no_case_sets_trust_remote_code), so
    # the operator would toggle it, retry, and hit the identical failure. The
    # only thing that actually reaches argv is `extra_args`, which is appended
    # verbatim and last. Say that instead. (Making the column emit the flag is
    # a behaviour change and a separate decision.)
    if _TRUST_REMOTE_CODE_RE.search(text):
        return EngineDiagnosis(
            message=(
                "This model requires trust_remote_code to load. Add "
                '"--trust-remote-code" to this model\'s extra_args, then retry '
                "— the model's trust_remote_code setting is not passed to the "
                "engine. Only do this for repos you trust: it makes the engine "
                "execute code from the model repo."
            ),
            recommended_max_model_len=None,
        )

    return None
