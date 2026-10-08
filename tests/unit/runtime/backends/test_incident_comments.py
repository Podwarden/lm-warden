"""Nine production incidents live as comments beside the code they explain.

Sub-project B relocates the three modules that carry them. Comments do not have
tests, so a "helpful" rewrite during the move would delete institutional memory
silently and nobody would notice until the incident recurred. This test pins
one stable marker per incident to the file that must contain it.

If a marker moves to a different file, update the table -- deliberately, in a
commit that says why. If a marker DISAPPEARS, the comment was deleted and the
change is wrong.
"""

from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[4] / "app"
ARGS = APP / "runtime" / "backends" / "vllm" / "args.py"
ENV = APP / "runtime" / "backends" / "vllm" / "env.py"
DIAG = APP / "runtime" / "backends" / "vllm" / "diagnostics.py"

# (incident, file, marker that must appear verbatim)
INCIDENTS = [
    (
        "2026-05-08 CUDA_VISIBLE_DEVICES inherited from the parent",
        ENV,
        "THIS FIXES THE 2026-05-08 BUG",
    ),
    (
        "2026-06-15 ENOSPC: engine wrote to /data instead of the cache volume",
        ENV,
        "the 2026-06-15 ENOSPC crash",
    ),
    (
        "#210 uncatchable SIGBUS in shm_broadcast on an undersized /dev/shm",
        ENV,
        "uncatchable SIGBUS",
    ),
    ("v2026.05.15.5 block-buffered stdout swallowed fast-crash tracebacks", ENV, "v2026.05.15.5"),
    # NB: the comment in env.py uses an EM DASH after the date. Anchor on the
    # prose only -- a marker containing "--" would never match.
    (
        "2026-08-18 EngineCore died 20 times leaving no traceback",
        ENV,
        "EngineCore died 20 times in one night leaving NOTHING",
    ),
    (
        "#211 engine bound 0.0.0.0 and republished an unauthenticated API",
        ARGS,
        "COMPLETELY UNAUTHENTICATED",
    ),
    (
        "--dtype None crashed create_subprocess_exec before vLLM ever started",
        ARGS,
        "expected str, bytes or os.PathLike object, not NoneType",
    ),
    ("#100 GGUF repo:QUANT addressing regression", ARGS, "every GGUF deployment died at vllm"),
    ("2026-08-18 false trust_remote_code diagnosis for 1h43m", DIAG, "1h43m"),
]


@pytest.mark.parametrize(
    "incident,path,marker",
    INCIDENTS,
    ids=[i[0][:40] for i in INCIDENTS],
)
def test_incident_comment_survived_the_move(incident, path, marker):
    assert path.exists(), f"{path} is missing"
    assert (
        marker in path.read_text()
    ), f"the comment recording {incident!r} is gone from {path.name}"
