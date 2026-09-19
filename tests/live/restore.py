"""Standalone crash-recovery: restore tokens a killed live-test run paused.

    python -m tests.live.restore --list
    python -m tests.live.restore --state <path>
    python -m tests.live.restore --state <path> --sweep
    python -m tests.live.restore --state <path> --force

Reads credentials from the same VW_LIVE_* env vars the test uses -- source
the private env file first:

    set -a; . /path/outside/repo/vw-live.env; set +a
    python -m tests.live.restore --list

Never prints credentials, a JWT, or a token's plaintext. Shares
tests/live/_isolation.py with the `isolation` pytest fixture so the two
recovery paths cannot drift.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import os
import sys

from tests.live._admin import AdminSession
from tests.live._env import load_config, missing_vars
from tests.live._isolation import (
    StateFile,
    delete_created_tokens,
    is_our_token_name,
    restore_paused_tokens,
)


def _list_state_files(state_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(state_dir, "*.json")))


async def _run(args: argparse.Namespace) -> int:
    missing = missing_vars()
    if missing:
        print(f"missing env vars: {missing}", file=sys.stderr)
        return 2
    cfg = load_config()

    if args.list:
        files = _list_state_files(cfg.state_dir)
        if not files:
            print(f"no state files in {cfg.state_dir}")
        for f in files:
            st = StateFile.load(f)
            pending = [p for p in st.paused if not p.get("restored")]
            print(
                f"{f}: run={st.run_id} phase={st.phase} "
                f"paused={len(st.paused)} pending_restore={len(pending)}"
            )
        return 0

    if not args.state:
        print("--state <path> is required unless --list", file=sys.stderr)
        return 2

    state = StateFile.load(args.state)
    if state.base_url != cfg.base_url:
        print(
            "refusing: state file targets a different base_url than "
            "VW_LIVE_BASE_URL currently points at", file=sys.stderr,
        )
        return 2

    admin = AdminSession(cfg.base_url, cfg.admin_user, cfg.admin_password, verify=cfg.httpx_verify)
    try:
        await admin.login()
        try:
            # Same ~10-minute defensive cap as the pytest fixture's finalizer
            # (conftest.py) -- restore_paused_tokens's own pass loop can run
            # long with many entries; this is a last-resort ceiling, not the
            # expected case.
            failures = await asyncio.wait_for(
                restore_paused_tokens(admin, state=state, state_dir=cfg.state_dir, force=args.force),
                timeout=600.0,
            )
        except TimeoutError:
            print(
                "restore did not finish within 10 minutes -- the state file is "
                "accurate and lists exactly which tokens are still unrestored; "
                "re-run this command to continue",
                file=sys.stderr,
            )
            return 1
        for line in failures:
            print(f"RESTORE FAILED: {line}", file=sys.stderr)
        if not failures:
            actually_restored = sum(1 for p in state.paused if p.get("restored"))
            print(f"restored {actually_restored} token(s) for run {state.run_id}")

        if args.sweep:
            del_failures = await delete_created_tokens(admin, state=state)
            for line in del_failures:
                print(f"DELETE FAILED: {line}", file=sys.stderr)
            tokens = await admin.list_tokens()
            leftovers = [t for t in tokens if is_our_token_name(t["name"], cfg.token_prefix)]
            for t in leftovers:
                try:
                    await admin.delete_token(t["id"])
                    print(f"swept leftover token {t['id']} ({t['name']})")
                except Exception as exc:  # noqa: BLE001
                    print(f"sweep failed for {t['id']}: {type(exc).__name__}", file=sys.stderr)
        return 1 if failures else 0
    finally:
        await admin.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", help="path to a run's state JSON file")
    parser.add_argument("--list", action="store_true", help="list state files in VW_LIVE_STATE_DIR")
    parser.add_argument("--sweep", action="store_true", help="also delete leftover <prefix>-* tokens")
    parser.add_argument(
        "--force",
        action="store_true",
        help="skip the paused_at equality check, and reopen any entry a previous "
        "run left alone only because of it (a 'paused_at changed' line)",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
