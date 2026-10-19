"""Tests for the engine load-log parser (Task 1 of the model-load-progress plan)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import ANY

from app.runtime.load_progress.parse import (
    LLAMACPP_STEPS,
    STEP_LABELS,
    VLLM_STEPS,
    LogEvent,
    Step,
    parse_llamacpp_line,
    parse_vllm_line,
    split_log_chunk,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "engine_logs"


def FIXTURE(name: str) -> str:
    """Read a fixture log preserving raw bytes (``\r`` must survive)."""
    # The brief's test names the vLLM file without the ``.log`` suffix and the
    # llama.cpp file without the ``.synthetic.log`` suffix; accept both shapes.
    candidates = [
        _FIXTURE_DIR / name,
        _FIXTURE_DIR / f"{name}.log",
        _FIXTURE_DIR / f"{name}.synthetic.log",
    ]
    for p in candidates:
        if p.is_file():
            # Read in binary and decode so raw ``\r`` bytes survive — Python's
            # universal-newlines text mode would convert them to ``\n``.
            return p.read_bytes().decode("utf-8", "replace")
    raise FileNotFoundError(f"fixture not found: {name!r} under {_FIXTURE_DIR}")


# ---------------------------------------------------------------------------
# Step labels and step tuples
# ---------------------------------------------------------------------------


def test_step_labels_pinned():
    assert STEP_LABELS == {
        Step.START: "Start engine",
        Step.WEIGHTS: "Load weights",
        Step.COMPILE: "Compile kernels",
        Step.KV: "Size KV cache",
        Step.TUNE: "Tune kernels",
        Step.GRAPHS: "Capture CUDA graphs",
        Step.WARMUP: "Warm-up request",
    }


def test_step_tuples():
    assert VLLM_STEPS == (
        Step.START,
        Step.WEIGHTS,
        Step.COMPILE,
        Step.KV,
        Step.TUNE,
        Step.GRAPHS,
        Step.WARMUP,
    )
    assert LLAMACPP_STEPS == (Step.START, Step.WEIGHTS, Step.WARMUP)


# ---------------------------------------------------------------------------
# split_log_chunk
# ---------------------------------------------------------------------------


def test_tqdm_carriage_returns_and_glued_prefixes_split():
    chunk = (
        "Capturing CUDA graphs (decode, FULL):  50%|█| 5/10 [..]\r"
        "Capturing CUDA graphs (decode, FULL):  60%|█| 6/10 [..]"
        "(Worker_TP0 pid=9) INFO 10-05 13:55:07 [x.py:1] "
        "Graph capturing finished in 3 secs, took 0.19 GiB\n"
    )
    assert len(split_log_chunk(chunk)) == 3


def test_bare_engine_core_glued_to_tqdm_bar_splits():
    """A bare ``(EngineCore pid=…)`` prefix glued to a tqdm bar tail must split."""
    chunk = (
        "\rCapturing CUDA graphs (FULL):  80%|████████  | 8/10 [..]"
        "(EngineCore pid=1) INFO 10-18 17:11:31 [core.py:402] "
        "init engine (profile, create kv cache, warmup model) took 1.0 s\n"
    )
    lines = split_log_chunk(chunk)
    # The EngineCore line must be its own segment so the engine_ready event fires.
    ready = [e for line in lines for e in parse_vllm_line(line) if e.kind == "engine_ready"]
    assert len(ready) == 1
    assert ready[0].replica == "EngineCore"


# ---------------------------------------------------------------------------
# vLLM line parsing — replica / source ids
# ---------------------------------------------------------------------------


def test_weights_counter_and_end():
    ev = parse_vllm_line(
        "(Worker_TP0 pid=931964) Loading safetensors checkpoint shards:  "
        "45% Completed | 30/66 [00:05<..]"
    )
    assert ev == [LogEvent("Worker_TP0:931964", "counter", Step.WEIGHTS, 30, 66, ANY)]
    assert (
        parse_vllm_line(
            "(Worker_TP0 pid=931964) INFO 10-05 13:50:55 [default_loader.py:430] "
            "Loading weights took 10.45 seconds"
        )[0].kind
        == "step_end"
    )


def test_kv_size_line_ends_compile_and_kv():
    kinds = {
        (e.kind, e.step)
        for e in parse_vllm_line(
            "(EngineCore_DP2 pid=929431) INFO 10-05 13:54:45 "
            "[kv_cache_utils.py:2177] GPU KV cache size: 683,853 tokens"
        )
    }
    assert ("step_end", Step.KV) in kinds and ("step_end", Step.COMPILE) in kinds


def test_source_ids_for_all_prefix_types():
    """Replica is the raw source process id for every prefix type."""
    # EngineCore DP
    r, _ = _source_and_body("(EngineCore_DP0 pid=100) INFO x")
    assert r == "EngineCore_DP0"
    # EngineCore bare
    r, _ = _source_and_body("(EngineCore pid=101) INFO x")
    assert r == "EngineCore"
    # Worker TP
    r, _ = _source_and_body("(Worker_TP0 pid=9) INFO x")
    assert r == "Worker_TP0:9"
    # Worker TP+EP
    r, _ = _source_and_body("(Worker_TP0_EP0 pid=9) INFO x")
    assert r == "Worker_TP0_EP0:9"
    # Worker bare
    r, _ = _source_and_body("(Worker pid=9) INFO x")
    assert r == "Worker:9"
    # APIServer (v0.32)
    r, _ = _source_and_body("(APIServer pid=356343) INFO x")
    assert r == "APIServer"
    # ApiServer_n (v0.26)
    r, _ = _source_and_body("(ApiServer_1 pid=929809) INFO x")
    assert r == "ApiServer_1"
    # MainProcess (v0.32 data-parallel top-level coordinator)
    r, _ = _source_and_body("(MainProcess pid=2044944) INFO x")
    assert r == "MainProcess"
    # unprefixed
    r, _ = _source_and_body("INFO something")
    assert r == ""


def test_prefixed_body_has_prefix_stripped():
    """For every prefixed line, the event ``text`` must NOT include the prefix."""
    ev = parse_vllm_line(
        "(Worker_TP0_EP0 pid=357962) INFO 10-18 17:03:34 "
        "[model_runner.py:424] Model loading took 55.42 GiB memory and 42.850489 seconds"
    )
    assert ev
    assert not ev[0].text.startswith("(")
    assert ev[0].text.startswith("INFO ")


def test_layout_event_carries_dp_size():
    """The layout event sets dp_size from data_parallel_size in the args dict."""
    # DP3 fixture's non-default args line has 'data_parallel_size': 3
    ev = parse_vllm_line(
        "INFO 10-05 13:49:11 [api_utils.py:273] non-default args: "
        "{'tensor_parallel_size': 2, 'data_parallel_size': 3}"
    )
    assert len(ev) == 1
    assert ev[0].kind == "layout"
    assert ev[0].dp_size == 3


def test_layout_event_dp_size_defaults_to_one_when_absent():
    """When data_parallel_size is absent, dp_size is 1 for a layout event."""
    ev = parse_vllm_line(
        "(APIServer pid=356343) INFO 10-18 17:01:11 [api_utils.py:286] "
        "non-default args: {'tensor_parallel_size': 8}"
    )
    assert len(ev) == 1
    assert ev[0].kind == "layout"
    assert ev[0].dp_size == 1


def test_unknown_line_yields_nothing_and_never_raises():
    for line in ["", "\x00\xff garbage", "INFO something new in vLLM 0.30", "(Worker_TP"]:
        assert parse_vllm_line(line) == []


# ---------------------------------------------------------------------------
# Fixture-level tests
# ---------------------------------------------------------------------------


def test_fixture_dp3_tp2_three_distinct_engine_core_sources():
    """DP3×TP2 fixture: engine_ready from 3 distinct EngineCore_DP0..2 sources,
    plus a layout event with dp_size 3."""
    events = [
        e
        for line in split_log_chunk(FIXTURE("vllm-0.26.0-dp3-tp2-fp8.log"))
        for e in parse_vllm_line(line)
    ]
    ready = [e for e in events if e.kind == "engine_ready"]
    assert len(ready) == 3
    assert {e.replica for e in ready} == {"EngineCore_DP0", "EngineCore_DP1", "EngineCore_DP2"}
    layouts = [e for e in events if e.kind == "layout"]
    assert len(layouts) == 1
    assert layouts[0].dp_size == 3
    assert {e.step for e in events if e.kind == "counter"} == {Step.WEIGHTS, Step.GRAPHS}


def test_llamacpp_fixture_reaches_ready():
    events = [
        e for line in split_log_chunk(FIXTURE("llamacpp-b10731")) for e in parse_llamacpp_line(line)
    ]
    assert any(e.kind == "step_end" and e.step == Step.WEIGHTS for e in events)
    assert any(e.kind == "engine_ready" for e in events)


def test_llamacpp_engine_ready_case_insensitive():
    """llama.cpp may log "Model loaded" (capital M) or "Server is listening"
    (capital S) on some builds.  The parser must match case-insensitively."""
    # Capital "Model loaded"
    ev = parse_llamacpp_line("INFO 10-18 12:00:00 [main] Model loaded")
    assert len(ev) == 1
    assert ev[0].kind == "engine_ready"
    # Capital "Server is listening"
    ev = parse_llamacpp_line("INFO 10-18 12:00:01 [main] Server is listening on port 8080")
    assert len(ev) == 1
    assert ev[0].kind == "engine_ready"
    # Original lowercase still works
    ev = parse_llamacpp_line("model loaded")
    assert len(ev) == 1
    assert ev[0].kind == "engine_ready"
    ev = parse_llamacpp_line("server is listening")
    assert len(ev) == 1
    assert ev[0].kind == "engine_ready"


def test_v032_glm_fixture_single_engine_one_ready():
    """The v0.32 GLM fixture is a single engine — one engine_ready from 'EngineCore',
    layout dp_size 1."""
    events = [
        e
        for line in split_log_chunk(FIXTURE("vllm-0.32.0-tp8-glm52-nvfp4-b300.log"))
        for e in parse_vllm_line(line)
    ]
    ready = [e for e in events if e.kind == "engine_ready"]
    assert len(ready) == 1
    assert ready[0].replica == "EngineCore"
    layouts = [e for e in events if e.kind == "layout"]
    assert len(layouts) == 1
    assert layouts[0].dp_size == 1
    assert {e.step for e in events if e.kind == "counter"} == {
        Step.WEIGHTS,
        Step.GRAPHS,
        Step.TUNE,
    }
    assert any(e.kind == "step_end" and e.step == Step.KV for e in events)


def test_v026_glm_fixture_single_engine_one_ready():
    """The v0.26 GLM fixture is also a single engine — one engine_ready from 'EngineCore',
    layout dp_size 1."""
    events = [
        e
        for line in split_log_chunk(FIXTURE("vllm-0.26.0-tp8-glm52-nvfp4-b300.log"))
        for e in parse_vllm_line(line)
    ]
    ready = [e for e in events if e.kind == "engine_ready"]
    assert len(ready) == 1
    assert ready[0].replica == "EngineCore"
    layouts = [e for e in events if e.kind == "layout"]
    assert len(layouts) == 1
    assert layouts[0].dp_size == 1
    assert {e.step for e in events if e.kind == "counter"} == {
        Step.WEIGHTS,
        Step.GRAPHS,
        Step.TUNE,
    }
    assert any(e.kind == "step_end" and e.step == Step.KV for e in events)


# ---------------------------------------------------------------------------
# AutoTuner (TUNE step) parsing
# ---------------------------------------------------------------------------


def test_autotuner_line_parsed():
    """A bare ``[AutoTuner]: Tuning <op>: NN%|…| x/y […]`` line produces a
    TUNE counter event with the op name (short form), count, and total."""
    ev = parse_vllm_line(
        "[AutoTuner]: Tuning trtllm_batch_decode_mla:   5%|▍  | 1/22 [00:00<00:07, 2.92profile/s]"
    )
    assert len(ev) == 1
    assert ev[0].kind == "counter"
    assert ev[0].step == Step.TUNE
    assert ev[0].count == 1
    assert ev[0].total == 22
    assert ev[0].text == "trtllm_batch_decode_mla"
    assert ev[0].replica == ""  # no (Worker… pid=) prefix


def test_autotuner_line_strips_flashinfer_prefix():
    """The op name is stored in short form: a leading ``flashinfer::`` is
    stripped."""
    ev = parse_vllm_line(
        "[AutoTuner]: Tuning flashinfer::trtllm_fp4_block_scale_moe: "
        "100%|██████████| 22/22 [04:37<00:00, 1.63s/profile]"
    )
    assert len(ev) == 1
    assert ev[0].step == Step.TUNE
    assert ev[0].text == "trtllm_fp4_block_scale_moe"
    assert ev[0].count == 22
    assert ev[0].total == 22


def test_autotuner_line_with_glued_worker_tail():
    """An AutoTuner line can have a ``(Worker_TP2_EP2 pid=…)`` prefix glued
    after the tqdm tail.  ``split_log_chunk`` splits on the prefix, so the
    AutoTuner line arrives without the prefix and parses cleanly."""
    chunk = (
        "[AutoTuner]: Tuning trtllm_batch_decode_mla:   5%|▍  | 1/22 [00:00<00:07, 2.92profile/s]"
        "(Worker_TP2_EP2 pid=252475) INFO 10-18 13:54:31 [x.py:1] something\n"
    )
    lines = split_log_chunk(chunk)
    autotuner_lines = [ln for ln in lines if ln.startswith("[AutoTuner]")]
    assert len(autotuner_lines) == 1
    ev = parse_vllm_line(autotuner_lines[0])
    assert len(ev) == 1
    assert ev[0].kind == "counter"
    assert ev[0].step == Step.TUNE
    assert ev[0].text == "trtllm_batch_decode_mla"
    assert ev[0].count == 1
    assert ev[0].total == 22


# ---------------------------------------------------------------------------
# vLLM 0.32 data-parallel: ``(MainProcess pid=…)`` top-level prefix
# ---------------------------------------------------------------------------


def test_mainprocess_prefix_parsed_and_stripped():
    """A ``(MainProcess pid=…)`` line is parsed with the prefix stripped and the
    source id set to ``"MainProcess"`` — the body is then parsed like any other
    vLLM line.  Uses a real line from the v0.32 DP3 fixture."""
    ev = parse_vllm_line(
        "(MainProcess pid=2044944) INFO 10-18 17:41:33 [api_utils.py:286] "
        "non-default args: {'tensor_parallel_size': 2, 'data_parallel_size': 3}"
    )
    assert len(ev) == 1
    assert ev[0].replica == "MainProcess"
    assert ev[0].kind == "layout"
    assert ev[0].dp_size == 3
    # The prefix must be stripped from the event text.
    assert not ev[0].text.startswith("(MainProcess")
    assert ev[0].text.startswith("INFO ")


def test_mainprocess_prefix_splits_when_glued_to_tqdm_bar():
    """A ``(MainProcess pid=…)`` prefix glued to the tail of a tqdm progress bar
    with no separator is split into its own line by ``split_log_chunk``."""
    chunk = (
        "Capturing CUDA graphs (FULL):  80%|████████  | 8/10 [..]"
        "(MainProcess pid=2044944) INFO 10-18 17:46:16 [utils.py:551] "
        "Waiting for API servers to complete ...\n"
    )
    lines = split_log_chunk(chunk)
    main_process_lines = [ln for ln in lines if ln.startswith("(MainProcess")]
    assert len(main_process_lines) == 1
    # The MainProcess line must parse without raising.
    ev = parse_vllm_line(main_process_lines[0])
    assert ev == []


def test_v032_dp3_mainprocess_fixture_replay():
    """The v0.32 DP3 fixture has 18 ``(MainProcess pid=…)`` lines — the parser
    strips the prefix and the layout event carries dp_size 3, and the engine
    reaches ready from the three EngineCore_DPn replicas."""
    events = [
        e
        for line in split_log_chunk(FIXTURE("vllm-0.32.0-dp3-tp2-fp8.log"))
        for e in parse_vllm_line(line)
    ]
    # The layout event from the MainProcess non-default-args line.
    layouts = [e for e in events if e.kind == "layout"]
    assert len(layouts) == 1
    assert layouts[0].replica == "MainProcess"
    assert layouts[0].dp_size == 3
    # Three engine_ready events from the EngineCore_DP0..2 replicas.
    ready = [e for e in events if e.kind == "engine_ready"]
    assert len(ready) == 3
    assert {e.replica for e in ready} == {"EngineCore_DP0", "EngineCore_DP1", "EngineCore_DP2"}


# ---------------------------------------------------------------------------
# Synthetic: single-engine TP=4 without EP — four distinct Worker sources,
# one bare EngineCore engine_ready
# ---------------------------------------------------------------------------


def test_single_engine_tp4_without_ep_four_distinct_workers():
    """Single-engine TP=4 (no EP suffix): four Worker_TPn lines produce four
    distinct source ids; one bare EngineCore produces the single engine_ready."""
    lines = [
        f"(Worker_TP{rank} pid={100 + rank}) INFO 10-18 17:03:34 "
        f"[model_runner.py:424] Model loading took 55.42 GiB memory and 42.0 seconds"
        for rank in range(4)
    ]
    all_events = [e for line in lines for e in parse_vllm_line(line)]
    weight_ends = [e for e in all_events if e.kind == "step_end" and e.step == Step.WEIGHTS]
    assert len(weight_ends) == 4
    assert {e.replica for e in weight_ends} == {
        "Worker_TP0:100",
        "Worker_TP1:101",
        "Worker_TP2:102",
        "Worker_TP3:103",
    }

    engine_events = parse_vllm_line(
        "(EngineCore pid=200) INFO 10-18 17:11:31 "
        "[core.py:402] init engine (profile, create kv cache, warmup model) took 454.96 s"
    )
    ready = [e for e in engine_events if e.kind == "engine_ready"]
    assert len(ready) == 1
    assert ready[0].replica == "EngineCore"


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------


def _source_and_body(line: str) -> tuple[str, str]:
    """Call the internal _vllm_replica and return (source_id, body)."""
    from app.runtime.load_progress.parse import _vllm_replica

    return _vllm_replica(line)
