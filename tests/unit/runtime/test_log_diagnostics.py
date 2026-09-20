"""Unit tests for the engine-log diagnostic parser.

``diagnose_engine_log`` scans an engine-log tail and turns a raw vLLM crash
traceback into an actionable, operator-facing message. It matches on STABLE
tokens (not exact phrasing) because vLLM's wording drifts across versions.
Returns ``None`` when nothing matches so the caller keeps its generic message.

The two KV-related fixtures below are the EXACT strings observed in production (real
vLLM output), including their quirks (``models's`` typo, a stray ``(`` before
``16.0 GiB``). The parser must match these tolerantly — never anchor on exact
punctuation.
"""
from __future__ import annotations

from app.runtime.backends.vllm.diagnostics import diagnose_engine_log

# --- Variant 1: KV cache too small for the requested context. ---------------
# vLLM prints its OWN authoritative fit estimate ("the estimated maximum model
# length is 157216"). That estimate is far better than our static math, so the
# parser MUST capture it and use it as the recommendation.
_KV_OVERFLOW_LOG = (
    "ValueError: To serve at least one request with the models's max seq len "
    "(262144), (16.0 GiB KV cache is needed, which is larger than the available "
    "KV cache memory (9.6 GiB). Based on the available memory, the estimated "
    "maximum model length is 157216. Try increasing `gpu_memory_utilization` or "
    "decreasing `max_model_len` when initializing the engine. See "
    "https://docs.vllm.ai/en/latest/configuration/conserving_memory/ for more "
    "details."
)

# --- Variant 2: no room for cache blocks at all (weights/util, NOT context). -
# Seen alongside "Available KV cache memory: -1.06 GiB" — negative means weights
# + overhead already exceed the budget; lowering max_model_len will NOT help.
_NO_CACHE_BLOCKS_LOG = (
    "INFO 05-25 worker.py:1 Available KV cache memory: -1.06 GiB\n"
    "ValueError: No available memory for the cache blocks. Try increasing "
    "`gpu_memory_utilization` when initializing the engine. See "
    "https://docs.vllm.ai/en/latest/configuration/conserving_memory/ for more "
    "details."
)

_CUDA_OOM_LOG = (
    "INFO 05-25 12:00:01 model_runner.py:1 Loading weights\n"
    "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.\n"
    "GPU 0 has a total capacity of 15.99 GiB of which 1.20 GiB is free.\n"
)

_TRUST_REMOTE_CODE_LOG = (
    "ValueError: Loading this model requires you to execute the configuration "
    "file in that repo on your local machine. Make sure you have read the code "
    "there to avoid malicious use, then set the option `trust_remote_code=True` "
    "to remove this error.\n"
)

_UNRELATED_LOG = (
    "INFO 05-25 12:00:01 api_server.py:1 Started server process\n"
    "INFO 05-25 12:00:02 api_server.py:1 Application startup complete.\n"
    "INFO 05-25 12:00:03 api_server.py:1 Uvicorn running on http://0.0.0.0:8000\n"
)


def test_kv_overflow_uses_vllm_estimated_max_len():
    """Variant 1: vLLM's own profiled estimate (157216) wins over static math."""
    diag = diagnose_engine_log(_KV_OVERFLOW_LOG)
    assert diag is not None
    # The recommendation is vLLM's authoritative estimate, NOT a parenthesized
    # token count.
    assert diag.recommended_max_model_len == 157216
    # Message names the numbers the operator needs and what to change.
    assert "262144" in diag.message
    assert "157216" in diag.message
    assert "max_model_len" in diag.message
    assert "gpu_memory_utilization" in diag.message


def test_no_cache_blocks_does_not_recommend_lowering_max_len():
    """Variant 2: weights already overflow the budget; lowering max_model_len
    will NOT help, so no recommendation and the message must NOT suggest it."""
    diag = diagnose_engine_log(_NO_CACHE_BLOCKS_LOG)
    assert diag is not None
    assert diag.recommended_max_model_len is None
    msg = diag.message.lower()
    # Points at the real fix (bigger/more GPUs or higher util)...
    assert "gpu_memory_utilization" in msg or "gpu" in msg
    # ...and crucially does NOT tell the operator to lower max_model_len.
    assert "max_model_len" not in msg


def test_cuda_oom():
    diag = diagnose_engine_log(_CUDA_OOM_LOG)
    assert diag is not None
    assert "out of memory" in diag.message.lower()
    assert "gpu_memory_utilization" in diag.message
    assert diag.recommended_max_model_len is None


def test_trust_remote_code():
    diag = diagnose_engine_log(_TRUST_REMOTE_CODE_LOG)
    assert diag is not None
    assert "trust_remote_code" in diag.message
    assert diag.recommended_max_model_len is None


def test_trust_remote_code_advice_points_at_extra_args_not_the_column():
    """#264 — the advice must name something that reaches the engine.

    The `trust_remote_code` COLUMN is never emitted to argv (see
    tests/unit/runtime/backends/test_launch_characterisation.py::
    test_no_case_sets_trust_remote_code), so "enable trust_remote_code for this
    model" sent the operator round a loop that ended in the identical failure.
    `extra_args` is appended to argv verbatim, so `--trust-remote-code` there
    is the only thing that fixes this error.
    """
    diag = diagnose_engine_log(_TRUST_REMOTE_CODE_LOG)
    assert diag is not None
    assert "--trust-remote-code" in diag.message
    assert "extra_args" in diag.message
    # And it must not tell the operator to flip the column instead.
    assert "enable trust_remote_code" not in diag.message.lower()


def test_no_match_returns_none():
    assert diagnose_engine_log(_UNRELATED_LOG) is None


def test_empty_text_returns_none():
    assert diagnose_engine_log("") is None
    assert diagnose_engine_log("   \n  ") is None


def test_kv_overflow_without_estimate_still_actionable():
    """A KV-overflow line with no 'estimated maximum model length' phrase still
    produces an actionable message, just without a numeric recommendation."""
    text = (
        "ValueError: The model's max seq len is larger than the maximum number "
        "of tokens that can be stored in KV cache. Try decreasing "
        "`max_model_len`."
    )
    diag = diagnose_engine_log(text)
    assert diag is not None
    assert "max_model_len" in diag.message
    assert diag.recommended_max_model_len is None


def test_kv_overflow_case_insensitive_tokens():
    """The KV/context branch keys off STABLE tokens case-insensitively so it
    survives vLLM wording drift."""
    text = (
        "Error: Model's MAX SEQ LEN exceeds the KV CACHE capacity. The "
        "estimated maximum model length is 9000. Decrease max_model_len."
    )
    diag = diagnose_engine_log(text)
    assert diag is not None
    assert diag.recommended_max_model_len == 9000


def test_oom_via_outofmemoryerror_token():
    text = "RuntimeError: OutOfMemoryError raised during weight allocation"
    diag = diagnose_engine_log(text)
    assert diag is not None
    assert "out of memory" in diag.message.lower()


# --- regression: the config echo is not a diagnosis --------------------------

# A trimmed but faithful slice of what vLLM writes on EVERY startup and on
# every dump_input.py ERROR line. Note `trust_remote_code=False` in the middle:
# the old matcher was a bare substring, so this text alone was enough to make
# the warden tell an operator to enable remote code execution.
_CONFIG_ECHO = (
    "INFO 08-18 11:09:32 [core.py:116] Initializing a V1 LLM engine (v0.26.0) "
    "with config: model='Qwen/Qwen3.8-27B-FP8', tokenizer_mode=auto, "
    "revision=main, trust_remote_code=False, dtype=torch.bfloat16, "
    "max_seq_len=262144, tensor_parallel_size=4, enforce_eager=False\n"
)

_TP_HANG = (
    "ERROR 08-18 13:11:56 [dump_input.py:72] Dumping input data for V1 LLM "
    "engine (v0.26.0) with config: model='Qwen/Qwen3.8-27B-FP8', "
    "trust_remote_code=False, tensor_parallel_size=4\n"
    "ERROR 08-18 13:11:56 [core.py:1332] EngineCore encountered a fatal error.\n"
    "ERROR 08-18 13:11:56 [core.py:1332] TimeoutError: RPC call to sample_tokens "
    "timed out.\n"
)


def test_config_echo_alone_is_not_a_trust_remote_code_diagnosis():
    """`trust_remote_code=False` appears in every vLLM startup banner."""
    assert diagnose_engine_log(_CONFIG_ECHO) is None


def test_a_tp_hang_is_not_reported_as_trust_remote_code():
    """The 2026-08-18 outage. A worker stopped answering and the engine timed
    out; the warden reported "This model requires trust_remote_code to load"
    for 1h43m because the crash dump echoed the config. Anything is better
    than advice to execute untrusted code for an unrelated fault."""
    diag = diagnose_engine_log(_TP_HANG)
    assert diag is None or "trust_remote_code" not in diag.message


def test_operator_enabled_flag_does_not_false_positive():
    """Matching `trust_remote_code=True` would reintroduce the bug the moment
    an operator legitimately turns the flag on, because the echo then says
    True on every single startup."""
    enabled_echo = _CONFIG_ECHO.replace(
        "trust_remote_code=False", "trust_remote_code=True"
    )
    assert diagnose_engine_log(enabled_echo) is None


def test_real_requirement_still_detected_among_noise():
    """The true error must still be found when the log also carries the echo."""
    diag = diagnose_engine_log(_CONFIG_ECHO + _TRUST_REMOTE_CODE_LOG)
    assert diag is not None
    assert "trust_remote_code" in diag.message


# --- Variant 0: the card was already occupied (pre-flight) -------------------

# VERBATIM from a real first-run failure on a host whose GPUs were already
# held by another service. This is the single most likely first failure a new
# operator hits, and until this rule existed it produced nothing better than
# "vllm subprocess exited unexpectedly (rc=1)".
_FREE_MEM_PREFLIGHT_LOG = (
    "(EngineCore pid=640064) ERROR 09-06 17:03:09 [core.py:1330] "
    "ValueError: Free memory on device cuda:0 (1.65/15.6 GiB) on startup is "
    "less than desired GPU memory utilization (0.13, 2.03 GiB). Decrease GPU "
    "memory utilization or reduce GPU memory used by other processes.\n"
)


def test_preflight_free_memory_names_the_actual_shortfall():
    diag = diagnose_engine_log(_FREE_MEM_PREFLIGHT_LOG)
    assert diag is not None
    msg = diag.message
    # Which card, how much was free, how big the card is, and how much the
    # model asked for -- the four numbers that make this actionable.
    assert "cuda:0" in msg
    assert "1.65" in msg
    assert "15.6" in msg
    assert "2.03" in msg
    assert "0.13" in msg
    # It is not a context problem; never suggest capping max_model_len.
    assert "max_model_len" not in msg
    assert diag.recommended_max_model_len is None


def test_preflight_free_memory_survives_wording_drift():
    """Keyed on stable tokens, not on vLLM's exact sentence."""
    diag = diagnose_engine_log(
        "ValueError: FREE MEMORY ON DEVICE cuda:1 ( 0.4 / 24.0 GiB ) is less "
        "than desired GPU memory utilization (0.90, 21.6 GiB).\n"
    )
    assert diag is not None
    assert "cuda:1" in diag.message
    assert "0.4" in diag.message


def test_preflight_free_memory_without_the_desired_half_still_diagnoses():
    """The requested figures are captured opportunistically, never required."""
    diag = diagnose_engine_log(
        "ValueError: Free memory on device cuda:0 (1.65/15.6 GiB) on startup "
        "is less than desired.\n"
    )
    assert diag is not None
    assert "1.65" in diag.message


def test_preflight_rule_does_not_pair_unrelated_lines():
    """A free-memory INFO line and a far-away "less than desired" sentence are
    not one sentence. The bounded, single-line match is what stops the parser
    inventing a diagnosis out of two unrelated log lines."""
    text = (
        "INFO 09-06 17:03:00 Free memory on device cuda:0 (14.9/15.6 GiB)\n"
        + "INFO 09-06 17:03:01 warming up\n" * 40
        + "ValueError: something is less than desired somewhere else\n"
    )
    assert diagnose_engine_log(text) is None


def test_preflight_beats_the_cuda_oom_rule_when_both_appear():
    """Ordering guard. The pre-flight check aborts before any allocation, so
    when both strings are present the pre-flight sentence is the real cause
    and the generic OOM advice would send the operator the wrong way."""
    diag = diagnose_engine_log(
        _FREE_MEM_PREFLIGHT_LOG
        + "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 22.00 MiB\n"
    )
    assert diag is not None
    assert "cuda:0" in diag.message
    assert "1.65" in diag.message
