"""I1 (round 4) regression: a SECOND signal must not be able to skip
restore.

Reproduced by the reviewer (scratchpad/r3repro): with the OLD fixture
structure, `isolation`'s own finalizer ran an up-to-30s wait BEFORE
`run_state`'s finalizer (where SIGINT/SIGTERM get ignored) ever started --
pytest's fixture teardown machinery only catches the test's own outcome,
not an arbitrary second KeyboardInterrupt, so a signal landing during that
window escaped straight out and `run_state`'s restore never ran at all.

The fix removed `isolation`'s finalizer entirely (see conftest.py) so
there is no fixture teardown anywhere in the chain that runs before
signals are ignored.

Two layers of evidence, per the round-5 review (minor 3: the original
version of this file only checked a HAND-WRITTEN COPY of the fixture
layout, which could silently drift from the real conftest.py):

1. `TestRealConftestFixtureStructure` below imports the REAL
   `tests.live.conftest` module and asserts the structural invariant
   directly via `inspect.isasyncgenfunction` -- `isolation` (and
   `model_info`/`test_tokens`) must NOT be async-generator fixtures (no
   `yield` => no finalizer to sit in an unprotected window), and
   `run_state` MUST be one (it owns the protected finalizer). This fails
   automatically the moment conftest.py's structure regresses, with no
   duplicate to fall out of sync.
2. The subprocess-based test below proves the FIX's runtime BEHAVIOR under
   two real OS signals, using the real `cancel_pending_tasks` /
   `restore_paused_tokens` from `tests.live._isolation` (still a small
   hand-written conftest for the subprocess itself, since driving the real
   one end to end would need a live admin server -- (1) is what guards
   against that duplicate drifting).
"""

from __future__ import annotations

import inspect
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


class TestRealConftestFixtureStructure:
    """Static, drift-proof evidence: imports the ACTUAL tests/live/conftest.py
    module (not a copy) and inspects its fixtures directly."""

    @staticmethod
    def _raw(func):
        # pytest's fixture decorator wraps with functools.wraps, so the
        # original function (whose *-ness inspect.isasyncgenfunction cares
        # about) is reachable via __wrapped__.
        return getattr(func, "__wrapped__", func)

    def test_isolation_has_no_yield_therefore_no_finalizer(self):
        from tests.live import conftest as real_conftest

        raw = self._raw(real_conftest.isolation)
        assert not inspect.isasyncgenfunction(raw), (
            "isolation must not be an async-generator fixture (no `yield`) -- "
            "any code after a yield here is a finalizer that runs BEFORE "
            "run_state's, in a window where signals are not yet ignored (I1)"
        )

    def test_run_state_has_a_yield_and_owns_the_finalizer(self):
        from tests.live import conftest as real_conftest

        raw = self._raw(real_conftest.run_state)
        assert inspect.isasyncgenfunction(raw), (
            "run_state must be an async-generator fixture -- it is the one "
            "place SIGINT/SIGTERM get ignored and restore actually runs"
        )

    def test_model_info_and_test_tokens_have_no_yield(self):
        # Round 2 consolidated their cleanup into run_state too -- neither
        # should have crept a finalizer back in since.
        from tests.live import conftest as real_conftest

        for name in ("model_info", "test_tokens"):
            raw = self._raw(getattr(real_conftest, name))
            assert not inspect.isasyncgenfunction(raw), (
                f"{name} must not be an async-generator fixture (no finalizer)"
            )

    def test_run_state_finalizer_ignores_signals_before_any_await(self):
        """A cheap source-level guard against the OTHER historical bug (round
        3/C3): the very first statement after `yield` in run_state must be
        the signal.signal(...) call, not an `await` -- an `await` there is
        exactly the gap a signal could land in before protection exists."""
        from tests.live import conftest as real_conftest

        raw = self._raw(real_conftest.run_state)
        src = inspect.getsource(raw)
        after_yield = src.split("yield state", 1)[1]
        # Find the first non-blank, non-comment statement after the yield.
        first_stmt = next(
            line.strip()
            for line in after_yield.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
        assert "signal.signal" in first_stmt, (
            f"expected the first statement after `yield state` in run_state to "
            f"install a signal handler; got: {first_stmt!r}"
        )
        assert "await" not in first_stmt


_CONFTEST = textwrap.dedent(
    """
    import asyncio, os, signal, sys
    import pytest, pytest_asyncio

    from tests.live._isolation import StateFile, cancel_pending_tasks, restore_paused_tokens

    signal.signal(signal.SIGTERM, signal.default_int_handler)

    LOG = os.environ["DOUBLE_SIGNAL_LOG"]


    def log(msg: str) -> None:
        with open(LOG, "a") as f:
            f.write(msg + "\\n")


    class _StubAdmin:
        \"\"\"No entries to restore -- this test is about whether run_state's
        finalizer BODY runs to completion under a second signal, not about
        restore's own pause-tracking correctness (covered elsewhere).\"\"\"

        async def get_token(self, token_id):
            return None

        async def set_paused(self, token_id, paused):
            return None

        async def list_tokens(self):
            return []


    @pytest_asyncio.fixture(scope="session", loop_scope="session")
    async def run_state():
        state = StateFile.new(run_id="dbl", base_url="https://example.invalid", model="m")
        admin = _StubAdmin()
        log("run_state setup")

        yield state

        # Mirrors conftest.py's run_state fixture EXACTLY: ignore signals
        # as the very first thing after yield, before any await.
        prev_term = signal.signal(signal.SIGTERM, signal.SIG_IGN)
        prev_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        log("signals ignored")
        try:
            await cancel_pending_tasks()
            # Widened so a second signal sent right after the first has a
            # reliable window to land inside (this test controls both
            # signals' timing from the outside).
            await asyncio.sleep(1.5)
            failures = await restore_paused_tokens(
                admin, state=state, state_dir=os.environ["DOUBLE_SIGNAL_STATE_DIR"]
            )
            log(f"RESTORE RAN failures={failures}")
        finally:
            signal.signal(signal.SIGTERM, prev_term)
            signal.signal(signal.SIGINT, prev_int)


    @pytest_asyncio.fixture(scope="session", loop_scope="session")
    async def isolation(run_state):
        # I1 fix: NO finalizer here at all -- a `return`, never a `yield`
        # with trailing code. Mirrors conftest.py's real isolation fixture.
        log("isolation setup (no teardown)")
        return {}
    """
)

_TEST_X = textwrap.dedent(
    """
    import asyncio
    import pytest

    @pytest.mark.asyncio(loop_scope="session")
    async def test_load(isolation):
        await asyncio.sleep(8)
    """
)


def _wait_for_log_line(log_path, needle: str, *, timeout: float, proc: subprocess.Popen) -> None:
    """Poll the log file for `needle` instead of guessing a fixed sleep --
    robust under heavy parallel load (e.g. `pytest -n 8`), where a blind
    `time.sleep(0.4)` can land before OR after the subprocess actually
    reaches the point it's meant to synchronize on."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if log_path.exists() and needle in log_path.read_text():
            return
        if proc.poll() is not None:
            break  # subprocess already exited -- stop polling, let the caller report it
        time.sleep(0.02)
    text = log_path.read_text() if log_path.exists() else ""
    raise AssertionError(f"timed out waiting for {needle!r} in the log; log so far:\n{text}")


def test_i1_double_signal_subprocess_does_not_skip_restore(tmp_path):
    log_path = tmp_path / "log.txt"
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (tmp_path / "conftest.py").write_text(_CONFTEST)
    (tmp_path / "test_x.py").write_text(_TEST_X)

    env = dict(os.environ)
    env["DOUBLE_SIGNAL_LOG"] = str(log_path)
    env["DOUBLE_SIGNAL_STATE_DIR"] = str(state_dir)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        [
            sys.executable, "-u", "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "--no-header", str(tmp_path),
        ],
        cwd=str(tmp_path),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        # First "kill": synchronized on "isolation setup" actually having
        # logged (not a guessed delay) -- lands once test_load is running.
        _wait_for_log_line(log_path, "isolation setup (no teardown)", timeout=30.0, proc=proc)
        proc.send_signal(signal.SIGTERM)
        # Second "kill": synchronized on "signals ignored" -- guarantees
        # this lands INSIDE run_state's protected window (previously a
        # blind time.sleep(0.3), which could arrive too early or too late
        # under heavy parallel load and flake either way).
        _wait_for_log_line(log_path, "signals ignored", timeout=30.0, proc=proc)
        proc.send_signal(signal.SIGTERM)
        stdout, _ = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, _ = proc.communicate()
        raise AssertionError(f"subprocess hung; output so far:\n{stdout.decode(errors='replace')}") from None

    log_text = log_path.read_text() if log_path.exists() else ""
    assert "RESTORE RAN" in log_text, (
        f"restore did not run under a double signal -- this is the I1 bug.\n"
        f"log:\n{log_text}\nsubprocess output:\n{stdout.decode(errors='replace')}"
    )
