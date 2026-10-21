"""The MLX engine log grammar -> an actionable operator message.

The linux unit part of Metal design section 5's diagnose() grammar: OOM first,
and NO message may ever say "reduce n_gpu_layers" because on Apple silicon the
weights and the KV cache live in the same unified memory either way -- there is
no partial offload and no dial to turn. The llama.cpp backend's OOM advice is
correct for its engine and false for this one, so the ban is asserted on every
text this module can produce, not only the OOM message.
"""

from __future__ import annotations

import pytest

# --- realistic log tails, one per class. ------------------------------------
# Synthetic (this host is linux and the fixtures ship with the macos lane) but
# each one is built from the stable anchors the rule actually matches, the way
# the tests/fixtures/llamacpp/ captures do for diagnose_llamacpp_log.

METAL_OOM_LOG = """\
INFO 07-11 10:02:11 mlx_loader.py:44] loading weights from /models/Llama-3.2-1B-mlx
ERROR 07-11 10:02:11 mlx_loader.py:57] failed to allocate 5.21 GB for weights: out of memory
"""

METAL_OOM_WITH_TRACEBACK_LOG = """\
INFO 07-11 10:02:11 mlx_loader.py:44] loading weights from /models/Llama-3.2-1B-mlx
mlx.metal: failed to allocate 5.21 GB for weights: out of memory
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/mlx/nn/layers/base.py", line 199, in __getattr__
    raise AttributeError(f'no attribute {name!r}')
  File "/usr/local/bin/mlx_server", line 118, in load_model
    model = load_weights(path, model)
  File "/usr/local/bin/mlx_server", line 96, in load_weights
    return model.load_weights(list(mx.load(path).items()))
FileNotFoundError: [Errno 2] No such file or directory: '/models/Llama-3.2-1B-mlx/weights.npz'
"""

JETKILL_LOG = """\
INFO 07-11 10:02:03 mlx_loader.py:44] loading weights from /models/Llama-3.2-3B-mlx
warden: engine exited rc=-9
"""

NOT_JETKILL_LOG = """\
INFO 07-11 10:02:03 mlx_loader.py:44] loading weights from /models/Llama-3.2-3B-mlx
warden: engine exited rc=-1
"""

UNSUPPORTED_ARCH_LOG = """\
Traceback (most recent call last):
  File "/usr/local/bin/mlx_server", line 118, in load_model
    model = load_weights(path, model)
  File "/usr/local/bin/mlx_server", line 88, in load_weights
    raise ValueError(f"Unsupported MLX model architecture: {arch}")
ValueError: Unsupported MLX model architecture: notamlxarch
"""

NOT_MLX_LOG = """\
Traceback (most recent call last):
  File "/usr/local/bin/mlx_server", line 118, in load_model
    model = load_weights(path, model)
  File "/usr/local/bin/mlx_server", line 84, in load_weights
    raise FileNotFoundError(f"missing MLX weights at {path}: expected weights.npz")
FileNotFoundError: missing MLX weights at /models/Llama-3.2-1B-mlx: expected weights.npz
"""

MISSING_CHAT_TEMPLATE_LOG = """\
Traceback (most recent call last):
  File "/usr/local/bin/mlx_server", line 131, in start
    template = chat_template_for(config)
  File "/usr/local/bin/mlx_server", line 142, in chat_template_for
    raise ValueError("missing chat template")
ValueError: missing chat template
"""

GENERIC_TRACEBACK_LOG = """\
INFO 07-11 10:02:11 mlx_server.py:12] serving on 127.0.0.1:8110
INFO 07-11 10:02:11 mlx_loader.py:44] loading weights from /models/Llama-3.2-1B-mlx
Traceback (most recent call last):
  File "/usr/local/bin/mlx_server", line 118, in load_model
    model = load_weights(path, model)
  File "/usr/local/bin/mlx_server", line 134, in load_weights
    raise RuntimeError("mlx engine failed to load the model")
RuntimeError: mlx engine failed to load the model
"""

BOTH_OOM_AND_TRACEBACK_LOG = """\
INFO 07-11 10:02:11 mlx_server.py:12] serving on 127.0.0.1:8110
INFO 07-11 10:02:11 mlx_loader.py:44] loading weights from /models/Llama-3.2-1B-mlx
mlx.metal: failed to allocate 5.21 GB for weights: out of memory
Traceback (most recent call last):
  File "/usr/local/bin/mlx_server", line 118, in load_model
    model = load_weights(path, model)
  File "/usr/local/bin/mlx_server", line 131, in load_weights
    raise RuntimeError("mlx out of memory while placing weights")
RuntimeError: mlx out of memory while placing weights
"""

# --- mlx / mlx_lm's own spellings of the same classes (mlx_lm.server is the
# engine, Metal design section 5). The anchors above are the load runner's
# synthetic ones; these are what the real engine prints.
METAL_MALLOC_LIMIT_LOG = """\
Traceback (most recent call last):
  File "/opt/lmw/bin/mlx_lm.server", line 8, in <module>
    sys.exit(main())
RuntimeError: [metal::malloc] Attempting to allocate 21474836480 bytes which is greater than the maximum allowed buffer size of 17179869184 bytes.
"""

METAL_COMMAND_BUFFER_OOM_LOG = """\
2026-10-19 10:02:11,004 - INFO - Starting httpd at 127.0.0.1 on port 8110...
libc++abi: terminating due to uncaught exception of type std::runtime_error: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
"""

MLX_LM_MODEL_TYPE_LOG = """\
ERROR:root:Model type notamlxarch not supported.
Traceback (most recent call last):
  File "/opt/lmw/lib/python3.12/site-packages/mlx_lm/utils.py", line 70, in _get_classes
    raise ValueError(msg)
ValueError: Model type notamlxarch not supported.
"""

MLX_LM_NO_SAFETENSORS_LOG = """\
Traceback (most recent call last):
  File "/opt/lmw/lib/python3.12/site-packages/mlx_lm/utils.py", line 160, in load_model
    raise FileNotFoundError(f"No safetensors found in {model_path}")
FileNotFoundError: No safetensors found in /models/Llama-3.2-1B
"""

MLX_LM_NO_CONFIG_LOG = """\
ERROR:root:Config file not found in /models/Llama-3.2-1B
Traceback (most recent call last):
FileNotFoundError: [Errno 2] No such file or directory: '/models/Llama-3.2-1B/config.json'
"""

MLX_LM_NO_CHAT_TEMPLATE_LOG = """\
Traceback (most recent call last):
  File "/opt/lmw/lib/python3.12/site-packages/mlx_lm/server.py", line 640, in handle_chat_completions
    prompt = self.tokenizer.apply_chat_template(messages)
ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed!
"""

PLAIN_CHATTER_LOG = """\
INFO 07-11 10:02:11 mlx_server.py:12] serving on 127.0.0.1:8110
INFO 07-11 10:02:11 mlx_loader.py:44] loading weights from /models/Llama-3.2-1B-mlx
INFO 07-11 10:02:12 mlx_loader.py:61] model loaded
"""


def _diagnose(text: str):
    from app.runtime.backends.mlx.diagnostics import diagnose_mlx_log

    return diagnose_mlx_log(text)


def test_metal_allocation_refused_is_diagnosed_as_oom():
    """Red today: ImportError for diagnose_mlx_log inside the test. The Metal
    allocator line is the OOM anchor; an OOM is classified from the line alone,
    with no traceback present at all."""
    d = _diagnose(METAL_OOM_LOG)
    assert d is not None
    assert "memory" in d.message.lower()
    # Weights did not fit: the context is not the lever on a 16 GB Mac.
    assert d.recommended_max_model_len is None
    for wrong in ("max_model_len", "context size", "ctx-size"):
        assert wrong not in d.message
    assert "n_gpu_layers" not in d.message
    assert "ngl" not in d.message


def test_jetsam_kill_is_diagnosed_from_the_synthetic_line():
    """The load runner appends 'warden: engine exited rc=-9' when the OS kills
    the engine and the log carries no exception of its own; the text-only
    Protocol still has to see it as what it is -- the system, not the model."""
    d = _diagnose(JETKILL_LOG)
    assert d is not None
    assert "jetsam" in d.message.lower()
    assert d.recommended_max_model_len is None


def test_non_negative_rc_is_not_jetsam():
    """Boundary leg: rc=-9 is the SIGKILL jetsam leaves behind; any other exit
    code is an ordinary crash the traceback rule (or the caller's generic
    message) handles. Diagnosing every nonzero exit as a system kill is a
    claim the log does not support."""
    assert _diagnose(NOT_JETKILL_LOG) is None


def test_unsupported_architecture_names_the_llamacpp_backend():
    """Two backends pay off here: the architecture this MLX build cannot load
    is usually loadable by llama.cpp from a GGUF of the same model, so the
    diagnosis says so instead of leaving the operator to discover it."""
    d = _diagnose(UNSUPPORTED_ARCH_LOG)
    assert d is not None
    assert "llama.cpp" in d.message
    assert "GGUF" in d.message
    assert "notamlxarch" in d.message
    assert d.recommended_max_model_len is None


def test_not_an_mlx_conversion_is_diagnosed():
    """The row is not an MLX conversion: the weights file a converted
    directory always carries is missing, and the message says the file is
    the problem, not the model."""
    d = _diagnose(NOT_MLX_LOG)
    assert d is not None
    assert "conversion" in d.message.lower()
    assert d.recommended_max_model_len is None


def test_missing_chat_template_is_diagnosed():
    for text in (MISSING_CHAT_TEMPLATE_LOG, MLX_LM_NO_CHAT_TEMPLATE_LOG):
        d = _diagnose(text)
        assert d is not None
        assert "chat template" in d.message.lower()
        # The rule's own advice, not the traceback catch-all echoing the
        # exception line (which also says "chat template").
        assert "llama.cpp" in d.message
        assert "raised before serving" not in d.message
        assert d.recommended_max_model_len is None


def test_mlx_own_oom_spellings_are_diagnosed_as_oom():
    """One leg per arm of the OOM rule in mlx's own words: the allocator
    refusing a buffer over the device limit (raised, so a traceback follows
    -- OOM must still win), and the Metal command buffer aborted for want of
    memory (a C++ abort, no Python traceback at all)."""
    for text in (METAL_MALLOC_LIMIT_LOG, METAL_COMMAND_BUFFER_OOM_LOG):
        d = _diagnose(text)
        assert d is not None, text
        assert "unified memory" in d.message
        assert "RuntimeError" not in d.message
        assert "n_gpu_layers" not in d.message
        assert "ngl" not in d.message
        assert d.recommended_max_model_len is None


def test_mlx_lm_model_type_not_supported_names_the_llamacpp_backend():
    d = _diagnose(MLX_LM_MODEL_TYPE_LOG)
    assert d is not None
    assert "'notamlxarch'" in d.message
    assert "llama.cpp" in d.message and "GGUF" in d.message


def test_mlx_lm_missing_weights_or_config_is_not_an_mlx_conversion():
    for text in (MLX_LM_NO_SAFETENSORS_LOG, MLX_LM_NO_CONFIG_LOG):
        d = _diagnose(text)
        assert d is not None, text
        assert "conversion" in d.message.lower()
        assert "raised before serving" not in d.message


def test_generic_python_traceback_is_diagnosed():
    """The catch-all: a traceback with no other rule firing still gets the
    engine's own final exception line, which beats a bare 'exited with rc=1'."""
    d = _diagnose(GENERIC_TRACEBACK_LOG)
    assert d is not None
    assert "RuntimeError" in d.message
    assert "mlx engine failed to load the model" in d.message
    assert d.recommended_max_model_len is None


def test_oom_wins_when_a_traceback_is_also_present():
    """Order is load-bearing, the same reason as in the llamacpp module: an
    OOM log routinely also carries a traceback, and the traceback checked
    first would hand back an exception line while the real answer (the weights
    did not fit unified memory) sits unused. OOM is checked first."""
    d = _diagnose(BOTH_OOM_AND_TRACEBACK_LOG)
    assert d is not None
    assert "memory" in d.message.lower()
    assert "RuntimeError" not in d.message


def test_every_message_bans_the_partial_offload_advice():
    """On unified memory there is no partial offload: 'reduce n_gpu_layers' /
    'ngl' is llama.cpp's lever and false for this engine. The ban is asserted
    over every text this module can produce, so a future rule cannot smuggle
    the advice in through its own message."""
    for text in (
        METAL_OOM_LOG,
        JETKILL_LOG,
        UNSUPPORTED_ARCH_LOG,
        NOT_MLX_LOG,
        MISSING_CHAT_TEMPLATE_LOG,
        GENERIC_TRACEBACK_LOG,
        BOTH_OOM_AND_TRACEBACK_LOG,
    ):
        d = _diagnose(text)
        assert d is not None
        assert "n_gpu_layers" not in d.message
        assert "ngl" not in d.message


def test_empty_and_plain_chatter_yield_no_diagnosis():
    assert _diagnose("") is None
    assert _diagnose(PLAIN_CHATTER_LOG) is None


# --- Fail-fast: the engine's own fatal-load line (BACKLOG-macos P5) ----------
# mlx_lm.server does not exit when its model fails to load; it keeps answering
# 503 on /health. These pin the one line the load-progress watcher fails an
# MLX load early on, against logs captured from the real engine on the M1.


@pytest.mark.parametrize("case", ["empty_dir", "corrupt_safetensors", "unsupported_arch"])
def test_captured_fatal_load_line_is_recognised(case):
    from app.runtime.backends.mlx.diagnostics import fatal_mlx_load_line
    from tests.unit.runtime.backends.mlx_load_failure_logs import CAPTURED

    log, fatal = CAPTURED[case]
    hits = [r for r in (fatal_mlx_load_line(line) for line in log.splitlines()) if r]
    # Exactly one line of each captured log is fatal -- the traceback frames,
    # the final exception line and the /health 503 chatter are not.
    assert len(hits) == 1
    # The timestamp/level prefix is dropped; the engine's words are kept.
    assert hits[0] == fatal[fatal.index("mlx_lm.server generation thread died:") :]


@pytest.mark.parametrize("case", ["empty_dir", "corrupt_safetensors", "unsupported_arch"])
def test_captured_fatal_logs_still_get_a_diagnosis(case):
    """The fail-fast's last_error carries the diagnosis too: every captured
    log must keep being recognised by diagnose_mlx_log."""
    from tests.unit.runtime.backends.mlx_load_failure_logs import CAPTURED

    d = _diagnose(CAPTURED[case][0])
    assert d is not None
    assert "n_gpu_layers" not in d.message


def test_unsupported_arch_capture_names_the_architecture():
    from tests.unit.runtime.backends.mlx_load_failure_logs import UNSUPPORTED_ARCH_LOG

    d = _diagnose(UNSUPPORTED_ARCH_LOG)
    assert d is not None
    assert "frobnicator" in d.message
    assert "llama.cpp" in d.message


@pytest.mark.parametrize(
    "line",
    [
        "",
        '127.0.0.1 - - [21/Oct/2026 01:18:41] "GET /health HTTP/1.1" 503 -',
        "2026-10-21 01:18:41,122 - INFO - Starting httpd at 127.0.0.1 on port 18951...",
        "Traceback (most recent call last):",
        "ValueError: Model type frobnicator not supported.",
        "RuntimeError: generation thread died",  # the per-request error, not the load
        "Fetching 9 files: 100%|##########| 9/9 [00:00<00:00, 9.1it/s]",
        "INFO vllm generation thread died: something",  # wrong program
    ],
)
def test_non_fatal_lines_are_not_fatal(line):
    from app.runtime.backends.mlx.diagnostics import fatal_mlx_load_line

    assert fatal_mlx_load_line(line) is None


def test_fatal_line_is_bounded():
    from app.runtime.backends.mlx.diagnostics import fatal_mlx_load_line

    got = fatal_mlx_load_line("ERROR - mlx_lm.server generation thread died: " + "x" * 5000)
    assert got is not None
    assert len(got) <= 301
    assert got.endswith("…")
