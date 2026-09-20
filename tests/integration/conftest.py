# The aiosqlite "wedge" this file used to work around now has a real fix.
#
# #43: the CI integration job printed `47 passed` and then hung 25+ minutes
# until the runner killed it, because an abandoned `aiosqlite.Connection` --
# a non-daemon thread parked on its work queue -- blocked CPython's interpreter
# shutdown, so the container's PID 1 never exited. The workaround here was a
# process-wide monkey-patch of `aiosqlite.core.Connection.__init__` that marked
# every connection thread a daemon, plus a thread census and a faulthandler
# dump armed at `pytest_unconfigure`.
#
# Both are gone. The connection is no longer abandoned in the first place:
# `app/db/database.py::open_db` closes it even when the connect or the body is
# cancelled (#260), and sets the daemon flag itself as a last-resort guard, in
# app code, where it also covers production. The diagnostics moved to
# `tests/conftest.py::pytest_sessionfinish`, which names any lingering
# non-daemon thread and arms a faulthandler dump that *exits*, for the whole
# test tree rather than just this directory -- keeping the patch here would have
# overridden that with a 30-second dump that hung afterwards anyway, and it hid
# #260 from every run that collected this directory.
import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "tests/integration" in str(item.fspath) or "tests\\integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
