"""Turn an MLX engine log tail into an actionable, operator-facing error.

The same job app/runtime/backends/vllm/diagnostics.py does for vLLM and
app/runtime/backends/llamacpp/diagnostics.py does for llama-server, with the
same discipline: match on STABLE TOKENS rather than exact phrasing, return
None on no match so the caller keeps its generic message.

METAL HAS NO PARTIAL OFFLOAD, and this module exists to keep saying so. On
Apple silicon the weights and the KV cache live in the same unified memory
whether or not a layer is "offloaded", so the one lever every other backend's
OOM message reaches for -- n_gpu_layers -- does not exist here. An operator
who turns a knob that is not a knob wastes an hour and files the row as
"model too big" when the answer was a smaller quant, fewer concurrent
decodes, or more memory. Both its spellings are therefore banned from every
message in this module, by the test that scans all of them.

ORDER IS LOAD-BEARING, and for the same reason it is in the two sibling
modules. An OOM log routinely also contains a traceback -- the engine raises
out of its allocator -- so the traceback rule, checked first, would hand the
operator an exception line while the real answer (the weights did not fit
unified memory) sat unused. OOM is checked FIRST and its message
deliberately never mentions the context.

THE JETKILL LINE. When macOS kills the engine the log ends mid-load with no
exception of its own, and the load runner appends a synthetic
``warden: engine exited rc=-9`` line so the text-only Protocol still sees
what happened. rc=-9 (SIGKILL from the system, not the engine's own exit
path) is jetsam; every other exit code is an ordinary crash the traceback
rule -- or the caller's generic message -- handles, and claiming a system
kill on their behalf would send the operator after a fix the log does not
support.
"""

from __future__ import annotations

import re

from app.runtime.backends.vllm.diagnostics import EngineDiagnosis

# --- 1. Out of memory. Checked FIRST -- see the module docstring. ------------
# The Metal allocator's refusal, with the numbers it prints, is the anchor:
# the size is what did not fit, and the line is present even when the engine
# then raises (which is why this rule must run before the traceback rule).
# The bare "out of memory" alternative catches the engine's own spelling of
# the same event when it raises first. The last two are MLX's own spellings
# (mlx/backend/metal): the allocator refusing a buffer over the device's
# limit -- "[metal::malloc] Attempting to allocate N bytes which is greater
# than the maximum allowed buffer size" -- and the command buffer the GPU
# aborted for want of memory, "Insufficient Memory
# (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)".
_MLX_OOM_RE = re.compile(
    r"failed\s+to\s+allocate[^\n]*out\s+of\s+memory"
    r"|out\s+of\s+memory\s*\(attempted\s+to\s+allocate"
    r"|\[metal::malloc\]\s+attempting\s+to\s+allocate"
    r"|kIOGPUCommandBufferCallbackErrorOutOfMemory",
    re.IGNORECASE,
)

# --- 2. Killed by the system (jetsam), not by its own exception. ------------
_JETKILL_RE = re.compile(r"warden:\s+engine\s+exited\s+rc=-9\b")

# --- 3. Architecture this build of MLX cannot load. --------------------------
# mlx_lm's own refusal is "Model type <type> not supported." (mlx_lm.utils).
_ARCH_RE = re.compile(
    r"unsupported\s+mlx\s+model\s+architecture:\s*'?\"?([\w./+-]+)"
    r"|model\s+type\s+'?([\w./+-]+?)'?\s+not\s+supported",
    re.IGNORECASE,
)

# --- 4. The row is not an MLX conversion. ------------------------------------
# mlx_lm's spellings: "No safetensors found in <dir>" and "Config file not
# found in <dir>" (mlx_lm.utils load_model / load_config).
_NOT_MLX_RE = re.compile(
    r"missing\s+mlx\s+weights|no\s+such\s+file\s+or\s+directory[^\n]*\.npz"
    r"|no\s+safetensors\s+found\s+in|config\s+file\s+not\s+found\s+in",
    re.IGNORECASE,
)

# --- 5. The conversion is fine but says nothing about how to talk to it. -----
# transformers' refusal, raised through mlx_lm's tokenizer wrapper:
# "... because tokenizer.chat_template is not set ...".
_CHAT_TEMPLATE_RE = re.compile(
    r"missing\s+chat\s+template|chat_template\s+is\s+not\s+set", re.IGNORECASE
)

# --- 6. Generic Python traceback, the catch-all. -----------------------------
# Better than llama.cpp's equivalent: the engine is a Python process, so its
# own final exception line names the failure, and we hand it back verbatim.
_TRACEBACK_RE = re.compile(r"^Traceback \(most recent call last\):", re.MULTILINE)
_EXCEPTION_RE = re.compile(r"^[ \t]*([\w.]+(?:Error|Exception|Interrupt)\w*:.*)$", re.MULTILINE)


# --- Fail-fast: the load is dead but the server is not. ----------------------
# mlx_lm.server (0.32.x, captured on the M1 2026-10-21) loads the model on a
# background "generation thread" AFTER the HTTP server starts. When that load
# raises -- an empty dir (FileNotFoundError on config.json), a corrupt
# safetensors ("[load_safetensors] Invalid json header length"), an
# architecture this build lacks ("Model type X not supported.") -- the thread
# logs exactly one line through logging.exception and dies, and the process
# keeps answering 503 on /health forever. Nothing else on the load path would
# notice before load_stall_timeout_s. The anchor is the server's own
# ``_run_generate`` except-clause, so it fires for EVERY exception the load
# raises and for nothing else: one stable token, not a list of guesses at the
# individual failure strings (those stay diagnose_mlx_log's job). It is
# checked only while the load is in flight, so the same line during serving
# is out of scope here.
_FATAL_LOAD_RE = re.compile(r"mlx_lm\.server generation thread died:.*")

# The matched line is quoted in last_error; bound it so a pathological
# exception message cannot balloon the row.
_FATAL_LINE_CHARS = 300


def fatal_mlx_load_line(line: str) -> str | None:
    """The engine's own fatal-load line when ``line`` is one, else None.

    Pure and line-at-a-time: the load-progress watcher feeds it each log line
    of an MLX row while the load is in flight (scoped there, so vLLM and
    llama.cpp rows never see it). Returns the line from the anchor on --
    without the logging timestamp prefix -- bounded to a few hundred chars.
    """
    m = _FATAL_LOAD_RE.search(line)
    if m is None:
        return None
    found = m.group(0).strip()
    if len(found) > _FATAL_LINE_CHARS:
        found = found[:_FATAL_LINE_CHARS] + "…"
    return found


def diagnose_mlx_log(text: str) -> EngineDiagnosis | None:
    """Best-effort diagnosis of an MLX engine failure. None when unrecognised."""
    if not text or not text.strip():
        return None

    if _MLX_OOM_RE.search(text):
        return EngineDiagnosis(
            message=(
                "The MLX engine ran out of unified memory placing the model. "
                "The weights and KV cache themselves did not fit, and unlike "
                "a discrete-GPU engine there is no layer-offload dial on "
                "Apple silicon to turn: pick a smaller quantisation, lower "
                "the decode concurrency, or give the row more memory. "
                "A smaller context will not help."
            ),
            recommended_max_model_len=None,
        )

    if _JETKILL_RE.search(text):
        return EngineDiagnosis(
            message=(
                "macOS killed the engine mid-load (exit -9, the system's "
                "jetsam, not a crash the engine raised). The load took more "
                "of the machine's memory than the system would give it. "
                "Pick a smaller quantisation or a smaller model for this "
                "machine; a smaller context will not help."
            ),
            recommended_max_model_len=None,
        )

    m = _ARCH_RE.search(text)
    if m:
        return EngineDiagnosis(
            message=(
                f"This build of MLX does not know the model architecture "
                f"'{m.group(1) or m.group(2)}'. The file itself is readable, "
                "so this is a "
                "gap in the MLX side, not a bad download: try the llama.cpp "
                "backend with a GGUF of the same model, which usually loads "
                "what MLX cannot."
            ),
            recommended_max_model_len=None,
        )

    if _NOT_MLX_RE.search(text):
        return EngineDiagnosis(
            message=(
                "The weights are not an MLX conversion: the files the engine "
                "loads (the weights and config of an mlx-converted directory) "
                "are missing. Convert the model to MLX and pull the converted "
                "directory, or switch this row to the llama.cpp backend with "
                "a GGUF of the same model."
            ),
            recommended_max_model_len=None,
        )

    if _CHAT_TEMPLATE_RE.search(text):
        return EngineDiagnosis(
            message=(
                "The MLX conversion has no chat template, so the engine "
                "cannot frame prompts. Add one to the model's conversion "
                "metadata, or switch this row to the llama.cpp backend with "
                "a GGUF of the same model, which carries its own."
            ),
            recommended_max_model_len=None,
        )

    if _TRACEBACK_RE.search(text):
        exceptions = _EXCEPTION_RE.findall(text)
        if exceptions:
            return EngineDiagnosis(
                message=(
                    "The MLX engine raised before serving. Its own final "
                    f"error: {exceptions[-1]}"
                ),
                recommended_max_model_len=None,
            )

    return None
