"""The characterisation corpus for sub-project B.

Sub-project B's defining property is that NOTHING changes. These 22 cases are
the mechanical proof: each is a model row + launch context that exercises one
branch of the vLLM argv/env builders. The exact argv and env they produce today
are frozen into ``tests/fixtures/launch_golden.json`` and re-asserted after
every refactor step.

Run ``python -m tests.unit.runtime.backends.corpus`` from the repo root to
REGENERATE the golden file. Regenerating is a deliberate act: the diff on
``launch_golden.json`` is the behaviour change, and it must be empty for every
task in this plan.
"""
from __future__ import annotations

import json
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path

GOLDEN_PATH = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "launch_golden.json"

Case = namedtuple(
    "Case", "name model port driver overrides hf_token hf_cache_dir"
)


@dataclass
class StubModel:
    """Stand-in for ``app.db.repos.models.ModelRow``.

    Deliberately a separate type, and deliberately NOT importing ModelRow: the
    builders read their inputs with ``getattr(model, name, default)`` for the
    columns added by migrations 0011/0014/0015/0022, and the corpus must keep
    exercising those defensive reads.

    WARNING -- THIS STUB IS CURRENTLY *MORE CAPABLE* THAN THE REAL ModelRow.
    ``quantization`` and ``max_num_seqs`` are real columns (migration 0011) but
    are absent from ``ModelRow``, ``_MODEL_COLS`` and ``_decode_row``, so on a
    production row ``getattr(model, "quantization", None)`` returns None and the
    flag is never emitted. Declaring them here means the ``quantization_set``
    and ``max_num_seqs_set`` cases below exercise a path production cannot
    currently reach. That divergence is deliberate and pinned by
    ``test_stub_mirrors_model_row`` -- do NOT quietly delete the fields, and do
    NOT quietly delete the test when the bug is fixed. See Task 1 Step 6
    precondition 5.
    """

    id: str = "m1"
    served_model_name: str = "demo"
    hf_repo: str = "org/model"
    hf_revision: str = ""
    gpu_indices: list[int] = field(default_factory=lambda: [0])
    tensor_parallel_size: int = 1
    dtype: str | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = False
    extra_args: list[str] = field(default_factory=list)
    extra_env: dict[str, str] = field(default_factory=dict)
    filename: str | None = None
    parallelism_strategy: str = "auto"
    max_batch_size: int = 1
    hf_config_repo: str | None = None
    tokenizer_repo: str | None = None
    quantization: str | None = None
    max_num_seqs: int | None = None


def _c(name, model, *, port=10000, driver="local", overrides=None,
       hf_token="hf_tok", hf_cache_dir="/cache") -> Case:
    return Case(name, model, port, driver, overrides, hf_token, hf_cache_dir)


CASES: list[Case] = [
    # --- the baseline -------------------------------------------------------
    _c("minimal", StubModel()),
    # --- parallelism (D4: the column counts GPUs; the flag is backend-owned) -
    _c("tp2", StubModel(gpu_indices=[0, 1], tensor_parallel_size=2)),
    _c("pp2", StubModel(gpu_indices=[0, 1], tensor_parallel_size=2,
                        parallelism_strategy="pp")),
    _c("tp_explicit", StubModel(gpu_indices=[2, 3], tensor_parallel_size=2,
                                parallelism_strategy="tp")),
    _c("gpu_order_reversed", StubModel(gpu_indices=[3, 1],
                                       tensor_parallel_size=2)),
    # --- optional flags, each present and absent ----------------------------
    _c("dtype_set", StubModel(dtype="bfloat16")),
    _c("dtype_none_is_omitted", StubModel(dtype=None)),
    _c("max_model_len_set", StubModel(max_model_len=8192)),
    # LATENT-BUG CASES. Both emit their flag here because StubModel declares
    # the attribute; a production row does NOT (see the StubModel warning), so
    # today these two document the POST-FIX behaviour of the row path and the
    # already-working override path -- not what production does. Verified
    # 2026-09-01: StubModel(quantization="awq") -> "--quantization awq";
    # the equivalent real row -> no flag at all.
    _c("quantization_set", StubModel(quantization="awq")),
    _c("max_num_seqs_set", StubModel(max_num_seqs=16)),
    _c("revision_set", StubModel(hf_revision="abc1234")),
    _c("hf_config_and_tokenizer",
       StubModel(hf_config_repo="org/base", tokenizer_repo="org/tok")),
    _c("gpu_mem_util", StubModel(gpu_memory_utilization=0.92)),
    # --- GGUF repo:QUANT addressing (#100) ----------------------------------
    _c("gguf_quant_tag", StubModel(filename="Model-Q5_K_M.gguf")),
    _c("gguf_extended_quant", StubModel(filename="Model-UD-Q4_K_XL.gguf")),
    _c("gguf_unmatched_filename", StubModel(filename="Model-f16.gguf")),
    _c("non_gguf_filename_ignored", StubModel(filename="model.safetensors")),
    # --- overrides (the ad-hoc reload path) ---------------------------------
    _c("overrides_full", StubModel(gpu_indices=[0, 1], tensor_parallel_size=2),
       overrides={"quantization": "gptq", "tensor_parallel_size": 2,
                  "gpu_memory_utilization": 0.85, "max_model_len": 4096,
                  "max_num_seqs": 8}),
    _c("overrides_max_model_len_explicit_none",
       StubModel(max_model_len=8192), overrides={"max_model_len": None}),
    # --- extra_args append-last ordering ------------------------------------
    _c("extra_args", StubModel(extra_args=["--enforce-eager",
                                           "--scheduling-policy", "fcfs"])),
    # --- extra_env: allowed, dropped, and the docker bind host --------------
    _c("extra_env_mixed",
       StubModel(extra_env={"VLLM_LOGGING_LEVEL": "DEBUG",
                            "NCCL_DEBUG": "INFO",
                            "PYTHONFAULTHANDLER": "0",
                            "SOME_RANDOM_KEY": "dropped"})),
    _c("docker_driver_bind", StubModel(), driver="docker"),
]

# ---------------------------------------------------------------------------
# The live-baseline cases. THESE THREE ARE NOT INVENTED.
# ---------------------------------------------------------------------------
# Each is a verbatim transcription of a model row that was serving in
# production on 2026-09-01, captured before the `bonus` wipe
# (~/llm-warden-baselines-2026-09-01/, committed here as
# tests/fixtures/live_baseline_argv.json).
#
# They matter for two different reasons:
#
#   1. COVERAGE. Only `bonus` is wiped and reinstalled; `home` is a production
#      dependency and is never touched, so there is no post-release `home` to
#      diff against. `home` is also the only install exercising the
#      tensor-parallel path (TP=4), the large-context path (262144) and an
#      fp8 KV cache. Transcribing its row keeps those argv branches covered
#      even though that install never runs the new build.
#   2. CORRECTNESS, which is the stronger of the two. Every other case in this
#      corpus is self-referential: the golden file records whatever the code
#      does, so it proves the refactor changed nothing -- not that the output
#      was ever right. These three have an EXTERNAL oracle: the argv a live
#      install actually produced. See test_live_baseline_oracle.py.
#
# Transcribe rows EXACTLY. A "tidied" value silently converts an oracle into
# another self-referential case.
LIVE_CASES: list[Case] = [
    # home install -- Qwen3.8-27B-FP8, TP=4 across all four GPUs.
    # The most configured row in the fleet and the only multi-GPU one.
    _c("live_home_qwen3_8_27b_fp8", StubModel(
        id="e29871dc4eeb0850",
        served_model_name="qwen3.8-27b-fp8-model",
        hf_repo="Qwen/Qwen3.8-27B-FP8",
        hf_revision="main",
        gpu_indices=[0, 1, 2, 3],
        tensor_parallel_size=4,
        dtype=None,
        max_model_len=262144,
        gpu_memory_utilization=0.95,
        # NOTE: there is no kv_cache_dtype COLUMN. The fp8 KV cache arrives as
        # two of the eight extra_args tokens below. Do not add a column for it.
        extra_args=["--enable-prefix-caching", "--kv-cache-dtype", "fp8",
                    "--reasoning-parser", "qwen3",
                    "--enable-auto-tool-choice",
                    "--tool-call-parser", "qwen3_xml"],
    )),
    # bonus -- Llama 3.1 8B AWQ-INT4, single GPU, no extra_args.
    _c("live_bonus_llama31_8b_awq", StubModel(
        id="873c3ed5860d1204",
        served_model_name="llama-3.1-8b",
        hf_repo="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        hf_revision="main",
        gpu_indices=[1],
        tensor_parallel_size=1,
        dtype="float16",
        max_model_len=8192,
        gpu_memory_utilization=0.9,
    )),
    # bonus -- the GSQ model from spec section 3.1, the one vLLM cannot serve.
    # Captured in `failed` state; the argv is still what the row produces.
    _c("live_bonus_qwen3_8_27b_gsq", StubModel(
        id="da98f08a0a3a826d",
        served_model_name="qwen3.8-27b",
        hf_repo="ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ",
        hf_revision="main",
        gpu_indices=[0],
        tensor_parallel_size=1,
        dtype="float16",
        max_model_len=8192,
        gpu_memory_utilization=0.94,
        extra_args=["--enforce-eager"],
    )),
]

CASES += LIVE_CASES

DIAGNOSIS_GOLDEN_PATH = GOLDEN_PATH.with_name("diagnosis_golden.json")

# Engine-log tails, one per branch of the diagnosis grammar. The strings are
# deliberately paraphrased vLLM output, not verbatim copies: the parser matches
# on STABLE TOKENS, not exact phrasing (log_diagnostics' module docstring), and
# a corpus of verbatim lines would hide a regression that tightened the regexes.
LOG_CASES: list[tuple[str, str]] = [
    ("empty", ""),
    ("whitespace_only", "   \n\t  "),
    ("unrecognised", "ERROR 08-31 10:00:00 something entirely unfamiliar\n"),
    # Variant 0 -- vLLM's pre-flight free-memory check, which fires before it
    # profiles anything. Paraphrased like its neighbours (different card,
    # different numbers) so a tightened regex shows up here.
    ("preflight_free_memory_on_startup",
     "ValueError: Free memory on device cuda:1 (0.8/24.0 GiB) on startup is "
     "less than desired GPU memory utilization (0.90, 21.6 GiB).\n"),
    ("no_cache_blocks",
     "ValueError: No available memory for the cache blocks. "
     "Available KV cache memory (-1.2 GiB)\n"),
    ("kv_overflow_with_estimate",
     "The model's max seq len (262144) is larger than the maximum number of "
     "tokens that can be stored in KV cache. 16.0 GiB KV cache is needed but "
     "available KV cache memory (9.6 GiB). Based on the actual memory usage, "
     "the estimated maximum model length is 157216.\n"),
    ("kv_overflow_no_estimate",
     "max seq len exceeds what fits in the KV cache on this device\n"),
    ("decrease_maxlen_phrasing",
     "Try to decrease `max_model_len` or increase gpu_memory_utilization.\n"),
    ("cuda_oom",
     "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n"),
    # The 2026-08-18 false-positive guard: a config echo containing
    # trust_remote_code=False must NOT be diagnosed as needing the flag.
    ("trust_remote_code_config_echo_is_not_a_match",
     "EngineArgs(model='org/m', tokenizer='org/m', trust_remote_code=False, "
     "dtype='auto')\n"),
    ("trust_remote_code_genuinely_required",
     "The repository contains custom code which must be executed to correctly "
     "load the model. You can inspect the repository content at the given "
     "path. Please pass the argument `--trust-remote-code`.\n"),
]


def _diagnosis_snapshot() -> dict:
    from app.runtime.backends.vllm.diagnostics import diagnose_engine_log

    out: dict[str, dict | None] = {}
    for name, text in LOG_CASES:
        d = diagnose_engine_log(text)
        out[name] = (
            None if d is None
            else {"message": d.message,
                  "recommended_max_model_len": d.recommended_max_model_len}
        )
    return out


def load_diagnosis_golden() -> dict:
    return json.loads(DIAGNOSIS_GOLDEN_PATH.read_text())


LIVE_BASELINE_PATH = GOLDEN_PATH.with_name("live_baseline_argv.json")


def load_live_baseline() -> dict:
    return json.loads(LIVE_BASELINE_PATH.read_text())


def _branch_point() -> str:
    """The commit the corpus is being frozen against.

    Recorded INTO the artefact because the corpus is only a regression net for
    the tree it was captured from, and B's branch point is not obvious: an
    operator release (v2026.09.01.1), its backmerge, and the whole of
    sub-project A all land between the design's reference commit and B's start.
    A corpus frozen against the wrong tree fails silently -- it keeps passing,
    against the wrong baseline.

    THE FIRST FREEZE WINS. Once the golden file records a SHA, regenerating
    REUSES it instead of stamping the current HEAD. Every later task in this
    plan regenerates (Task 2 adds the log corpus) and then asserts
    ``git diff --stat tests/fixtures/launch_golden.json`` is EMPTY -- if this
    function returned HEAD each time, that check could never pass and the
    field would drift from "the tree we froze against" to "the last time
    someone ran the generator", which is not a fact anyone needs.

    To deliberately re-freeze against a different tree, delete
    ``launch_golden.json`` first. That is a big, visible diff, which is the
    correct weight for that decision.
    """
    import subprocess

    if GOLDEN_PATH.exists():
        try:
            recorded = json.loads(GOLDEN_PATH.read_text())["_meta"]["captured_from"]
        except Exception:  # noqa: BLE001 - a malformed file re-stamps from git
            recorded = "unknown"
        if recorded and recorded != "unknown":
            return recorded

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - never block regeneration on git
        return "unknown"


def _snapshot() -> dict:
    from app.runtime.backends.vllm.args import build_vllm_args
    from app.runtime.backends.vllm.env import build_subprocess_env

    out: dict[str, dict] = {
        "_meta": {
            "captured_from": _branch_point(),
            "note": (
                "Frozen from the tree named above. If you are reading this on a "
                "tree with a different history, the corpus is policing the "
                "wrong baseline -- see Task 1 Step 6."
            ),
        }
    }
    for c in CASES:
        out[c.name] = {
            "argv": build_vllm_args(
                c.model, port=c.port, overrides=c.overrides, driver=c.driver
            ),
            "env": build_subprocess_env(
                c.model,
                hf_token=c.hf_token,
                hf_cache_dir=c.hf_cache_dir,
                engine_driver=c.driver,
            ),
        }
    return out


def load_golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text())


if __name__ == "__main__":  # pragma: no cover - regeneration helper
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(_snapshot(), indent=2, sort_keys=True) + "\n")
    DIAGNOSIS_GOLDEN_PATH.write_text(
        json.dumps(_diagnosis_snapshot(), indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {len(CASES)} launch cases and {len(LOG_CASES)} log cases")
