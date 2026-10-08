"""Run-boundary sentinels scope engine-log diagnosis to the CURRENT run (#234).

The engine log is append-only across load attempts, so before this the
200-line diagnosis window straddled runs: a previous attempt's recognised
traceback (OOM / KV overflow) was matched and reported as the cause of the
current attempt, replacing the accurate generic message. Observed twice live
on 2026-09-03 -- once a run that got all the way through profiling
(``Available KV cache memory: 2.31 GiB``, ``init engine ... took 123.13 s``)
was reported as "GPU ran out of memory loading the model".

The fixtures below are shaped exactly like that incident: run 1 fails with a
pattern the parser recognises, run 2 does not contain it. The parser must see
only run 2.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

from app.models.routes_api import _read_engine_log_tail
from app.runtime.backends.vllm.diagnostics import diagnose_engine_log
from app.runtime.engine.run_marker import (
    RUN_SENTINEL_PREFIX,
    format_run_sentinel,
    is_run_sentinel,
    tail_since_last_run,
)

# Run 1: the real KV-overflow string from evidence 2 in the issue -- the
# parser recognises this one and produces a confident "wants 262144 tokens".
_RUN_1_KV_OVERFLOW = (
    "INFO 09-03 19:58:01 core.py:1 Starting vLLM engine\n"
    "ValueError: To serve at least one request with the models's max seq len "
    "(262144), (2.54 GiB KV cache is needed, which is larger than the available "
    "KV cache memory (0.3 GiB). Based on the available memory, the estimated "
    "maximum model length is 32768. Try increasing `gpu_memory_utilization` or "
    "decreasing `max_model_len` when initializing the engine.\n"
)

# Run 2: max_model_len was lowered and the engine profiled fine. Nothing here
# matches any grammar, so the caller must keep its own generic message.
_RUN_2_HEALTHY = (
    "INFO 09-03 20:32:00 core.py:1 vLLM engine args {'max_model_len': 32768, "
    "'enforce_eager': True, 'gpu_memory_utilization': 0.95}\n"
    "INFO 09-03 20:33:10 gpu_worker.py:1 Available KV cache memory: 2.9 GiB\n"
    "INFO 09-03 20:33:11 core.py:1 GPU KV cache size: 269,633 tokens\n"
    "INFO 09-03 20:34:03 core.py:1 init engine (profile, create kv cache, "
    "warmup model) took 123.13 s\n"
)

_RUN_1_OOM = (
    "INFO 09-03 19:58:20 model_runner.py:1 Loading weights\n"
    "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.\n"
)


def _two_run_log(run1: str, run2: str) -> str:
    return f"{format_run_sentinel('run1')}\n" f"{run1}" f"{format_run_sentinel('run2')}\n" f"{run2}"


# --- the sentinel itself -----------------------------------------------------


def test_sentinel_is_recognisable_and_carries_id_and_timestamp():
    line = format_run_sentinel("abc123")
    assert line.startswith(RUN_SENTINEL_PREFIX)
    assert "abc123" in line
    assert is_run_sentinel(line)
    # An ISO-8601 UTC timestamp, not vLLM's year-less "MM-DD HH:MM:SS".
    assert re.search(r"started \d{4}-\d{2}-\d{2}T[\d:.]+\+00:00 =====$", line)


def test_sentinel_ids_are_unique_per_run():
    assert format_run_sentinel() != format_run_sentinel()


def test_engine_log_lines_are_not_mistaken_for_sentinels():
    for line in _RUN_1_KV_OVERFLOW.splitlines() + _RUN_2_HEALTHY.splitlines():
        assert not is_run_sentinel(line)


# --- tail_since_last_run -----------------------------------------------------


def test_tail_starts_at_last_sentinel():
    tail = tail_since_last_run(_two_run_log(_RUN_1_KV_OVERFLOW, _RUN_2_HEALTHY), max_lines=200)
    assert "262144" not in tail
    assert "269,633 tokens" in tail
    # Warden bookkeeping never reaches a backend grammar.
    assert RUN_SENTINEL_PREFIX not in tail


def test_parser_sees_only_run_two_kv_overflow():
    """Evidence 2 from #234: run 1's 'wants 262144 tokens' must not be reported
    for run 2, which ran at max_model_len=32768 and reached a real KV cache."""
    tail = tail_since_last_run(_two_run_log(_RUN_1_KV_OVERFLOW, _RUN_2_HEALTHY), max_lines=200)
    assert diagnose_engine_log(tail) is None


def test_parser_sees_only_run_two_oom():
    """Evidence 1 from #234: run 1 OOM'd, run 2 profiled fine and was still
    reported as 'GPU ran out of memory loading the model'."""
    tail = tail_since_last_run(_two_run_log(_RUN_1_OOM, _RUN_2_HEALTHY), max_lines=200)
    assert diagnose_engine_log(tail) is None


def test_current_run_failure_is_still_diagnosed():
    """Scoping must not disable the feature: a fault in the CURRENT run is
    exactly what we want reported."""
    tail = tail_since_last_run(_two_run_log(_RUN_2_HEALTHY, _RUN_1_KV_OVERFLOW), max_lines=200)
    diag = diagnose_engine_log(tail)
    assert diag is not None
    assert diag.recommended_max_model_len == 32768


def test_max_lines_still_caps_a_chatty_run():
    noisy = "".join(f"line {i}\n" for i in range(1000))
    tail = tail_since_last_run(f"{format_run_sentinel('r')}\n{noisy}", max_lines=200)
    lines = tail.splitlines()
    assert len(lines) == 200
    assert lines[-1] == "line 999"


def test_sentinel_as_final_line_yields_empty_tail():
    """A run that has produced no output yet has nothing to diagnose -- and
    must NOT inherit the previous run's."""
    log = f"{_RUN_1_OOM}{format_run_sentinel('r2')}\n"
    assert tail_since_last_run(log, max_lines=200) == ""


def test_sentinel_glued_to_an_unterminated_previous_line_still_splits():
    """Defence in depth: if a run's last chunk lacked a trailing newline the
    sentinel could land mid-line. The boundary must still be found."""
    log = (
        f"torch.cuda.OutOfMemoryError: CUDA out of memory.{format_run_sentinel('r2')}\n"
        f"{_RUN_2_HEALTHY}"
    )
    tail = tail_since_last_run(log, max_lines=200)
    assert "OutOfMemoryError" not in tail
    assert diagnose_engine_log(tail) is None


# --- legacy logs (no sentinel) ----------------------------------------------


def test_legacy_log_without_sentinel_falls_back_to_whole_tail():
    """Logs already on disk have no sentinels. Returning "" there would
    silently delete the diagnosis feature for every existing install; a
    possibly-stale diagnosis still beats none."""
    tail = tail_since_last_run(_RUN_1_KV_OVERFLOW, max_lines=200)
    diag = diagnose_engine_log(tail)
    assert diag is not None


def test_legacy_log_without_sentinel_is_still_capped():
    noisy = "".join(f"line {i}\n" for i in range(500))
    assert len(tail_since_last_run(noisy, max_lines=200).splitlines()) == 200


def test_empty_log_is_empty():
    assert tail_since_last_run("", max_lines=200) == ""


# --- the reader on the route side -------------------------------------------


def _settings(tmp_path):
    return SimpleNamespace(logs_dir=tmp_path)


def test_read_engine_log_tail_scopes_to_current_run(tmp_path):
    (tmp_path / "m1.log").write_text(_two_run_log(_RUN_1_OOM, _RUN_2_HEALTHY))
    tail = _read_engine_log_tail(_settings(tmp_path), "m1")
    assert "OutOfMemoryError" not in tail
    assert "269,633 tokens" in tail


def test_read_engine_log_tail_legacy_file(tmp_path):
    (tmp_path / "m1.log").write_text(_RUN_1_OOM)
    assert "OutOfMemoryError" in _read_engine_log_tail(_settings(tmp_path), "m1")


def test_read_engine_log_tail_missing_file_is_empty(tmp_path):
    assert _read_engine_log_tail(_settings(tmp_path), "nope") == ""
