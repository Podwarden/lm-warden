"""The baked llama.cpp build id is declared, not discovered.

Read from VW_LLAMACPP_BUILD, which the Dockerfile sets from its
LLAMACPP_BUILD ARG -- the same single-source-of-truth discipline VW_BASE_DIGEST
already has, and for the same reason: CI's post-build cleanup greps the
Dockerfile, and a cleanup that reads a stale value deletes a base you are still
using.

Deliberately NOT ``llama-server --version``: the control plane answers capability
questions from a declared value (the pattern app/system/routes_engine.py set),
and an import-time subprocess is both a startup cost and a new failure mode for
a metadata read. tests/integration/test_llamacpp_smoke.py checks the declared
value against the real binary, where a subprocess is free.

The plan sketched this as a module-level ``BUILD`` constant. It is a FUNCTION
instead, for one concrete reason: a constant is bound at import time, so any
test -- or any caller -- that changes the environment has to reload the module
to see it, and a stale import elsewhere in the process would silently keep the
old value. An env read is cheap enough to do per call.
"""

from __future__ import annotations

from app.runtime.backends.llamacpp.version import STAMP_NAME, baked_build, version_string


def test_build_is_none_outside_a_built_image(monkeypatch):
    monkeypatch.delenv("VW_LLAMACPP_BUILD", raising=False)
    assert baked_build() is None
    assert version_string() is None


def test_build_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("VW_LLAMACPP_BUILD", "b10731")
    assert baked_build() == "b10731"
    assert version_string() == "b10731"


def test_empty_env_reads_as_none(monkeypatch):
    """An exported-but-blank build arg must not surface as an empty-string
    version in the API -- the same 'empty counts as unset' rule
    engine_bind_host applies to VW_ENGINE_BIND_HOST."""
    monkeypatch.setenv("VW_LLAMACPP_BUILD", "")
    assert baked_build() is None


def test_whitespace_only_env_reads_as_none(monkeypatch):
    monkeypatch.setenv("VW_LLAMACPP_BUILD", "   ")
    assert baked_build() is None


def test_linux_never_reads_a_stamp_file(monkeypatch, tmp_path):
    """The image's behaviour is unchanged: on linux only the env var counts,
    even if a stamp happens to sit beside the interpreter."""
    monkeypatch.delenv("VW_LLAMACPP_BUILD", raising=False)
    (tmp_path / STAMP_NAME).write_text("b11719\n")
    assert baked_build(bin_dir=str(tmp_path)) is None
    assert version_string(profile="linux", bin_dir=str(tmp_path)) is None


def test_apple_falls_back_to_the_bundle_stamp(monkeypatch, tmp_path):
    monkeypatch.delenv("VW_LLAMACPP_BUILD", raising=False)
    (tmp_path / STAMP_NAME).write_text("b11719\n")
    assert version_string(profile="apple", bin_dir=str(tmp_path)) == "b11719"


def test_apple_env_var_still_wins_over_the_stamp(monkeypatch, tmp_path):
    monkeypatch.setenv("VW_LLAMACPP_BUILD", "b10731")
    (tmp_path / STAMP_NAME).write_text("b11719\n")
    assert baked_build(profile="apple", bin_dir=str(tmp_path)) == "b10731"


def test_apple_without_a_stamp_is_none(monkeypatch, tmp_path):
    monkeypatch.delenv("VW_LLAMACPP_BUILD", raising=False)
    assert baked_build(profile="apple", bin_dir=str(tmp_path)) is None
    (tmp_path / STAMP_NAME).write_text("  \n")
    assert baked_build(profile="apple", bin_dir=str(tmp_path)) is None


def test_apple_stamp_tolerates_crlf_and_no_trailing_newline(monkeypatch, tmp_path):
    monkeypatch.delenv("VW_LLAMACPP_BUILD", raising=False)
    for body in (b"b11719\r\n", b"b11719", b"  b11719  \nextra\n"):
        (tmp_path / STAMP_NAME).write_bytes(body)
        assert baked_build(profile="apple", bin_dir=str(tmp_path)) == "b11719", body


def test_apple_garbage_stamp_reads_as_none_and_never_raises(monkeypatch, tmp_path):
    """A stale or corrupt stamp is an informational field gone wrong, not a 500:
    non-ASCII, binary, control characters, a token with spaces, an overlong
    line (bounded read: never slurped whole) and a directory all read as None."""
    monkeypatch.delenv("VW_LLAMACPP_BUILD", raising=False)
    stamp = tmp_path / STAMP_NAME
    for body in (
        "b11719\u00e9\n".encode(),
        b"\xff\xfe\x00\x01",
        b"b117\x1b[31m19\n",
        b"b11719 rc1\n",
        b"b" + b"1" * 10_000_000,
        b"b" + b"1" * 40 + b"\n",
    ):
        stamp.write_bytes(body)
        assert baked_build(profile="apple", bin_dir=str(tmp_path)) is None, body[:20]
    # Only the first _STAMP_MAX_BYTES (64) bytes are read: a token followed by
    # padding past that bound parses from the bounded prefix alone.
    stamp.write_bytes(b"b11719" + b" " * 100 + b"tail\n")
    assert baked_build(profile="apple", bin_dir=str(tmp_path)) == "b11719"
    stamp.unlink()
    stamp.mkdir()
    assert baked_build(profile="apple", bin_dir=str(tmp_path)) is None


def test_apple_fifo_stamp_reads_as_none_without_blocking(monkeypatch, tmp_path):
    """A FIFO under the stamp name must not hang the backends route: opening it
    for reading would block until a writer appears, so only a regular file counts."""
    import os
    import threading

    monkeypatch.delenv("VW_LLAMACPP_BUILD", raising=False)
    fifo = tmp_path / STAMP_NAME
    os.mkfifo(fifo)
    out: list[str | None] = []
    t = threading.Thread(
        target=lambda: out.append(baked_build(profile="apple", bin_dir=str(tmp_path))),
        daemon=True,
    )
    t.start()
    t.join(timeout=5)
    blocked = t.is_alive()
    if blocked:  # release the blocked reader before failing
        fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        os.close(fd)
        t.join(timeout=5)
    assert not blocked, "baked_build blocked opening a FIFO stamp"
    assert out == [None]


def test_apple_default_bin_dir_is_the_interpreters_own(monkeypatch, tmp_path):
    """The bundle places llama-server (and its stamp) in dirname(sys.executable),
    Contents/Resources/python/bin -- the default needs no launcher setting."""
    monkeypatch.delenv("VW_LLAMACPP_BUILD", raising=False)
    (tmp_path / STAMP_NAME).write_text("b11719\n")
    monkeypatch.setattr("sys.executable", str(tmp_path / "python3"))
    assert baked_build(profile="apple") == "b11719"


def test_apple_backend_reports_the_stamp(monkeypatch, tmp_path):
    from app.runtime.backends.llamacpp import LlamaCppBackend

    monkeypatch.delenv("VW_LLAMACPP_BUILD", raising=False)
    (tmp_path / STAMP_NAME).write_text("b11719\n")
    monkeypatch.setattr("sys.executable", str(tmp_path / "python3"))
    assert LlamaCppBackend(profile="apple").version_string() == "b11719"
    assert LlamaCppBackend().version_string() is None


def test_the_dockerfile_pins_the_same_tag_the_fixtures_were_captured_from():
    """The fixtures in tests/fixtures/llamacpp/ are what every parser in this
    sub-project was written against. If the Dockerfile's pin drifts from them,
    the argv, the log grammar and the metric names are all being tested against
    a build the product does not ship."""
    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    dockerfile = (repo / "Dockerfile").read_text()
    m = re.search(r"^ARG LLAMACPP_BUILD=(\S+)", dockerfile, re.M)
    assert m, "Dockerfile must declare ARG LLAMACPP_BUILD as a greppable line"
    pinned = m.group(1)

    readme = (repo / "tests" / "fixtures" / "llamacpp" / "README.md").read_text()
    assert pinned in readme, (
        f"Dockerfile pins llama.cpp {pinned}, which the fixture README does not "
        f"mention. Re-capture tests/fixtures/llamacpp/ against {pinned}, or "
        f"revert the pin."
    )
