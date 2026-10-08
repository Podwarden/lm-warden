"""llama.cpp's log grammar -> an actionable operator message.

Driven by the REAL logs Task 2 captured, not by paraphrase: a diagnoser tested
only against strings someone typed from memory passes forever and never fires in
production. Two of the cases below are the distinction that matters most, and it
is the same one app/runtime/backends/vllm/diagnostics.py draws for vLLM:

  * ``log_ctx_capped``  -- the requested context did not fit and llama.cpp said
    so. Lowering max_model_len IS the fix, so we say so and, where the log gives
    the capped number, hand it back as recommended_max_model_len.
  * ``log_oom``         -- the allocator failed while placing WEIGHTS or KV.
    Lowering max_model_len does NOT help; the answer is fewer offloaded layers,
    a smaller quant, or a bigger card. Suggesting a context reduction here sends
    the operator round a loop that cannot terminate.

Matching is on STABLE TOKENS, not exact phrasing -- llama.cpp's wording drifts
across builds, and these fixtures are one build.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.runtime.backends.llamacpp.diagnostics import diagnose_llamacpp_log

FIX = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "llamacpp"


def _tail(name: str) -> str:
    return (FIX / name).read_text()


def test_successful_startup_yields_no_diagnosis():
    """No match must be None, so the caller keeps its generic message. A
    diagnoser that invents a cause for a healthy log is worse than none."""
    assert diagnose_llamacpp_log(_tail("log_startup_ok.txt")) is None


def test_empty_log_yields_no_diagnosis():
    assert diagnose_llamacpp_log("") is None


def test_oom_is_reported_and_does_not_suggest_lowering_context():
    d = diagnose_llamacpp_log(_tail("log_oom.txt"))
    assert d is not None
    assert "memory" in d.message.lower()
    assert d.recommended_max_model_len is None
    for wrong in ("max_model_len", "context size", "ctx-size"):
        assert wrong not in d.message


def test_oom_names_the_lever_that_actually_helps():
    d = diagnose_llamacpp_log(_tail("log_oom.txt"))
    assert "n_gpu_layers" in d.message or "offload" in d.message.lower()


def test_context_capping_is_reported_as_a_context_problem():
    d = diagnose_llamacpp_log(_tail("log_ctx_capped.txt"))
    assert d is not None
    assert "context" in d.message.lower()


def test_context_capping_recommends_the_training_context():
    """The captured log says "the slot context (8192) exceeds the training
    context of the model (2048) - capping", so the number to hand back is 2048.
    Recommending the number the log itself printed is what turns a diagnosis
    into a one-click fix."""
    d = diagnose_llamacpp_log(_tail("log_ctx_capped.txt"))
    assert d.recommended_max_model_len == 2048


def test_unknown_architecture_is_named():
    d = diagnose_llamacpp_log(_tail("log_unknown_arch.txt"))
    assert d is not None
    assert "architecture" in d.message.lower()
    assert "notarealarch" in d.message


def test_wrong_shard_says_which_file_to_point_at():
    """llama.cpp discovers the rest of a shard set itself, but ONLY from shard
    one. The message has to say that, because 'point -m at a different file' is
    not an obvious remedy."""
    d = diagnose_llamacpp_log(_tail("log_wrong_shard.txt"))
    assert d is not None
    assert "first" in d.message.lower()


def test_oom_is_checked_before_context():
    """An OOM log frequently also contains context-sized numbers, and
    misclassifying it sends the operator to lower max_model_len forever. Same
    ordering rule, and the same reason, as the vLLM diagnoser's
    _NO_CACHE_BLOCKS_RE being checked first."""
    text = "llama_context: n_ctx_seq (8192) > n_ctx_train (4096)\n" + _tail("log_oom.txt")
    d = diagnose_llamacpp_log(text)
    assert d.recommended_max_model_len is None
    assert "memory" in d.message.lower()


def test_a_cuda_oom_matches_the_same_rule():
    """log_oom.txt is a CPU-side allocation failure -- see the fixture README's
    known gap: the only NVIDIA host in scope was read-only until acceptance. The
    CUDA spellings come from the same pinned b10731 tree
    (ggml/src/ggml-cuda/ggml-cuda.cu), and they have to match the same rule or
    the diagnoser is untested for the case it exists to catch."""
    cuda = (
        "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 11256.00 MiB on "
        "device 0: cudaMalloc failed: out of memory\n"
        "alloc_tensor_range: failed to allocate CUDA0 buffer of size 11803033600\n"
        "llama_model_load: error loading model: unable to allocate CUDA0 buffer\n"
    )
    d = diagnose_llamacpp_log(cuda)
    assert d is not None
    assert "memory" in d.message.lower()
    assert d.recommended_max_model_len is None


def test_a_generic_load_failure_still_says_something_useful():
    d = diagnose_llamacpp_log(
        "llama_model_load: error loading model: tensor 'blk.0.attn_q.weight' "
        "data is not within the file bounds, model is corrupted or incomplete\n"
    )
    assert d is not None
    assert "re-pull" in d.message or "complete" in d.message


@pytest.mark.parametrize("noise", ["", "\n\n", "some unrelated line\n" * 50])
def test_noise_alone_never_matches(noise):
    assert diagnose_llamacpp_log(noise) is None


def test_the_backend_routes_to_this_grammar():
    """LlamaCppBackend.diagnose must reach it -- Task 7 left a stub returning
    None, and a stub that survives is a diagnoser nobody notices is dead."""
    from app.runtime.backends.llamacpp import LlamaCppBackend

    d = LlamaCppBackend().diagnose(_tail("log_unknown_arch.txt"))
    assert d is not None


def test_the_vllm_grammar_does_not_claim_llamacpp_logs():
    """And the converse, which is why routing by the row's backend matters: the
    vLLM diagnoser must not produce a confident vLLM-flavoured explanation for a
    llama.cpp failure."""
    from app.runtime.backends.vllm import VllmBackend

    assert VllmBackend().diagnose(_tail("log_unknown_arch.txt")) is None
