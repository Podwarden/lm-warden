"""Child-process entry point for a model download: ``python -m app.models.pull_worker``.

WHY A SEPARATE PROCESS
----------------------
On lmwarden.com (2026-11-04) the api process died about a minute into a 27B
pull, with no Python traceback in the log, and took the whole API down with
it (502s for every user, and the pull lost). The download had been running
in a thread inside the api process. huggingface_hub hands the transfer to
hf_xet, which is native (Rust) code: a crash or an OOM kill there ends the
process it runs in, and no ``except`` can catch it. In a child process the
same failure ends the pull, which ``run_pull`` records as ``failed`` with the
exit status, and the API stays up.

PROTOCOL
--------
stdin: one JSON object — ``repo_id``, ``revision``, ``cache_dir``, ``token``,
``allow_patterns``. The token comes in on stdin, never argv: argv is readable
by every process on the host through /proc.

stdout: zero or more lines of anything, then one line that starts with
``WORKER_RESULT_MARKER`` followed by JSON:
``{"ok": true, "path": "..."}`` or
``{"ok": false, "auth": bool, "error": "Type: message"}``.
No marker line at all means the process died before it could say why;
the parent reports the exit code or signal.

stderr: logs and huggingface_hub's progress bars; inherited from the parent
so they land in the container log.

The worker exits when its parent does (see ``_die_with_parent``), so a
restarted warden never races a leftover downloader for the same files.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

# Kept in sync with app.models.pull_task.WORKER_RESULT_MARKER (asserted by
# the tests); duplicated so this module imports nothing heavy before it must.
WORKER_RESULT_MARKER = "@@VW_PULL_RESULT@@ "


def _emit(result: dict[str, object]) -> None:
    sys.stdout.write(WORKER_RESULT_MARKER + json.dumps(result) + "\n")
    sys.stdout.flush()


def _die_with_parent(parent_pid: int, interval_s: float = 1.0) -> None:
    """Exit as soon as the parent is gone (we get re-parented)."""

    def watch() -> None:
        while True:
            if os.getppid() != parent_pid:
                os._exit(3)
            time.sleep(interval_s)

    threading.Thread(target=watch, name="pull-worker-parent-watch", daemon=True).start()


def main() -> int:
    _die_with_parent(os.getppid())
    req = json.loads(sys.stdin.read())
    try:
        from huggingface_hub import snapshot_download

        kwargs = {
            "repo_id": req["repo_id"],
            "revision": req["revision"],
            "cache_dir": req["cache_dir"],
            "token": req.get("token"),
        }
        if req.get("allow_patterns") is not None:
            kwargs["allow_patterns"] = req["allow_patterns"]
        path = snapshot_download(**kwargs)
    except BaseException as exc:  # noqa: BLE001 -- every failure must reach the parent
        auth = False
        try:
            from app.models.pull_task import _classify_hf_auth_error

            auth = _classify_hf_auth_error(exc) is not None
        except Exception:  # noqa: BLE001 -- classification is best effort
            pass
        _emit({"ok": False, "auth": auth, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    _emit({"ok": True, "path": str(path)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
