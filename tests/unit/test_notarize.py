"""Notarization wrapper for the macOS app (BACKLOG-macos P8).

Selected only under ``-m macos`` (see ``pyproject.toml``); the default
``addopts`` deselects it. ``scripts/macos/notarize.sh`` submits a .dmg or a
dittied .app to ``xcrun notarytool submit --wait`` through a keychain
profile only (no --apple-id/--password/--team-id/--key/--key-id/--issuer and
no credential env var), fetches the rejection log on a non-Accepted status,
and staples + validates the original artifact on Accepted.

The tests shell out only to ``bash``, ``ditto`` and ``python3``. The stub
``xcrun`` is always first on PATH, so the real notarytool/stapler are never
reached; no Apple account, keychain profile or network is needed. The .app
legs are skipped when ``ditto`` is absent. No app code is imported.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.macos

REPO = Path(__file__).resolve().parents[2]

NOTARIZE = REPO / "scripts" / "macos" / "notarize.sh"

PROFILE = "test-profile"

NEEDS_DITTO = pytest.mark.skipif(shutil.which("ditto") is None, reason="ditto not on PATH")

# Argument flags the wrapper must never pass to any xcrun tool (credentials
# live only in the keychain profile, created once with store-credentials).
FORBIDDEN = ("--apple-id", "--password", "--team-id", "--key", "--key-id", "--issuer")

_STUB = """#!/bin/bash
{ for a in "$@"; do printf '%s\\t' "$a"; done; printf '\\n'; } >> "$STUB_LOG"
if [ "$1" = "notarytool" ]; then
    case "$2" in
        submit)
            if [ -n "${STUB_SUBMIT_RAW:-}" ]; then
                printf '%s\\n' "$STUB_SUBMIT_RAW"
            else
                printf '{"id":"11111111-2222-3333-4444-555555555555","status":"%s"}\\n' "${STUB_STATUS:-Accepted}"
            fi
            exit "${STUB_SUBMIT_RC:-0}"
            ;;
        log)
            echo "stub notary log for $3"
            exit "${STUB_LOG_RC:-0}"
            ;;
    esac
fi
# The real stapler talks on stdout ("Processing: ...", "The staple and
# validate action worked!"); the stub does too, so the one-line stdout
# contract is tested against that chatter.
if [ "$1" = "stapler" ] && [ "$2" = "staple" ]; then
    echo "stub stapler staple chatter"
    exit "${STUB_STAPLE_RC:-0}"
fi
if [ "$1" = "stapler" ] && [ "$2" = "validate" ]; then
    echo "stub stapler validate chatter"
    exit "${STUB_VALIDATE_RC:-0}"
fi
exit 0
"""


def _notarize() -> Path:
    assert NOTARIZE.is_file(), f"missing {NOTARIZE}"
    return NOTARIZE


def _build_stub(tmp_path: Path) -> Path:
    bindir = tmp_path / "stubbin"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "xcrun"
    stub.write_text(_STUB)
    stub.chmod(0o755)
    return bindir


def _run(
    tmp_path: Path,
    *argv: str,
    **stub_env: str,
) -> tuple[subprocess.CompletedProcess, list[list[str]]]:
    """Run the wrapper with the stub xcrun; ``stub_env`` keys are passed as
    environment variables (never as argv: an ``X=y`` shell word looks like a
    credential export to the scrubbers) -- STUB_STATUS, STUB_SUBMIT_RC,
    STUB_SUBMIT_RAW, STUB_STAPLE_RC, STUB_VALIDATE_RC."""
    log = tmp_path / "stub.log"
    log.write_text("")
    env = dict(os.environ)
    env["PATH"] = f"{_build_stub(tmp_path)}{os.pathsep}{env.get('PATH', '')}"
    env["STUB_LOG"] = str(log)
    for key in (
        "STUB_STATUS",
        "STUB_SUBMIT_RC",
        "STUB_SUBMIT_RAW",
        "STUB_STAPLE_RC",
        "STUB_VALIDATE_RC",
        "STUB_LOG_RC",
    ):
        env.pop(key, None)
    env.update(stub_env)
    res = subprocess.run(
        ["bash", str(_notarize()), *argv],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    # Each argv line ends with a tab: drop the empty field after it.
    calls = [ln.split("\t")[:-1] for ln in log.read_text().splitlines() if ln]
    return res, calls


def test_dmg_accepted_submits_staples_validates(tmp_path: Path) -> None:
    dmg = tmp_path / "LM-Warden-x.dmg"
    dmg.write_bytes(b"fake dmg")
    res, calls = _run(tmp_path, str(dmg), PROFILE)
    assert res.returncode == 0, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    assert calls, "xcrun never called"
    assert calls[0][:2] == ["notarytool", "submit"] and str(dmg) in calls[0], calls
    assert calls[1][:2] == ["stapler", "staple"] and str(dmg) in calls[1], calls
    assert calls[2][:2] == ["stapler", "validate"] and str(dmg) in calls[2], calls
    assert len(calls) == 3, f"unexpected extra xcrun calls:\n{calls}"
    assert calls[0] == [
        "notarytool",
        "submit",
        str(dmg),
        "--keychain-profile",
        PROFILE,
        "--wait",
        "--output-format",
        "json",
    ], calls[0]
    assert calls[1] == ["stapler", "staple", str(dmg)], calls[1]
    assert calls[2] == ["stapler", "validate", str(dmg)], calls[2]
    assert "stub stapler staple chatter" in res.stderr, res.stderr
    assert res.stdout == (
        f"notarize.sh: accepted and stapled {dmg.name} (id 11111111-2222-3333-4444-555555555555)\n"
    ), res.stdout


@NEEDS_DITTO
def test_app_accepted_zips_and_staples_the_app(tmp_path: Path) -> None:
    app = tmp_path / "staging" / "LM Warden.app"
    (app / "Contents" / "MacOS").mkdir(parents=True)
    (app / "Contents" / "MacOS" / "LMWarden").write_text("fake binary\n")
    res, calls = _run(tmp_path, str(app), PROFILE)
    assert res.returncode == 0, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    submit = calls[0]
    assert submit[:2] == ["notarytool", "submit"], calls
    zips = [a for a in submit if a.endswith(".zip")]
    assert len(zips) == 1 and zips[0].endswith("LM Warden.zip"), calls
    tmpdir = Path(zips[0]).parent
    assert not tmpdir.exists(), f"temp zip dir not removed: {tmpdir}"
    assert calls[1][:2] == ["stapler", "staple"] and str(app) in calls[1], calls
    assert calls[2][:2] == ["stapler", "validate"] and str(app) in calls[2], calls
    assert len(calls) == 3, f"unexpected extra xcrun calls:\n{calls}"
    assert res.stdout == (
        "notarize.sh: accepted and stapled LM Warden.app "
        "(id 11111111-2222-3333-4444-555555555555)\n"
    ), res.stdout


def test_not_accepted_logs_and_skips_stapler(tmp_path: Path) -> None:
    dmg = tmp_path / "LM-Warden-x.dmg"
    dmg.write_bytes(b"fake dmg")
    res, calls = _run(tmp_path, str(dmg), PROFILE, STUB_STATUS="Invalid")
    assert res.returncode == 1, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    assert calls[1][:2] == ["notarytool", "log"], calls
    assert "11111111-2222-3333-4444-555555555555" in calls[1], calls
    assert not any(c[:2] == ["stapler", "staple"] for c in calls), (
        "stapler called on rejection:\n" + repr(calls)
    )
    assert not any(c[:2] == ["stapler", "validate"] for c in calls), (
        "stapler called on rejection:\n" + repr(calls)
    )
    assert (
        "notarize.sh: notarization Invalid (id 11111111-2222-3333-4444-555555555555)" in res.stderr
    ), res.stderr
    assert "stub notary log" in res.stderr, "rejection log output not on stderr:\n" + res.stderr


def test_submit_failure_exits_1(tmp_path: Path) -> None:
    dmg = tmp_path / "LM-Warden-x.dmg"
    dmg.write_bytes(b"fake dmg")
    res, calls = _run(tmp_path, str(dmg), PROFILE, STUB_SUBMIT_RC="7")
    assert res.returncode == 1, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    assert "notarize.sh: notarytool submit failed" in res.stderr, res.stderr
    assert len(calls) == 1 and calls[0][:2] == ["notarytool", "submit"], calls


def test_unparseable_submit_output_exits_1(tmp_path: Path) -> None:
    dmg = tmp_path / "LM-Warden-x.dmg"
    dmg.write_bytes(b"fake dmg")
    res, calls = _run(tmp_path, str(dmg), PROFILE, STUB_SUBMIT_RAW="not json at all")
    assert res.returncode == 1, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    assert "notarize.sh: could not parse notarytool output" in res.stderr, res.stderr
    assert "not json at all" in res.stderr, "raw output not on stderr:\n" + res.stderr
    assert not any(
        c[:2] == ["stapler", "staple"] for c in calls
    ), "stapler called after unparseable output"


def test_staple_failure_exits_1_naming_step(tmp_path: Path) -> None:
    dmg = tmp_path / "LM-Warden-x.dmg"
    dmg.write_bytes(b"fake dmg")
    res, calls = _run(tmp_path, str(dmg), PROFILE, STUB_STAPLE_RC="3")
    assert res.returncode == 1, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    assert "staple" in res.stderr, res.stderr
    assert not any(
        c[:2] == ["stapler", "validate"] for c in calls
    ), "validate ran after staple failed"
    assert "accepted and stapled" not in res.stdout, res.stdout


def test_validate_failure_exits_1_naming_step(tmp_path: Path) -> None:
    dmg = tmp_path / "LM-Warden-x.dmg"
    dmg.write_bytes(b"fake dmg")
    res, calls = _run(tmp_path, str(dmg), PROFILE, STUB_VALIDATE_RC="4")
    assert res.returncode == 1, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    assert "validate" in res.stderr, res.stderr
    assert len(calls) == 3, calls
    assert "accepted and stapled" not in res.stdout, res.stdout


def test_usage_legs_exit_2_without_xcrun(tmp_path: Path) -> None:
    dmg = tmp_path / "ok.dmg"
    dmg.write_bytes(b"fake dmg")
    zipf = tmp_path / "ok.zip"
    zipf.write_bytes(b"fake zip")
    other = tmp_path / "LM-Other.bundle"
    other.mkdir()
    cases = [
        (),
        (str(dmg),),
        (str(dmg), PROFILE, "extra"),
        (str(dmg), ""),
        (str(tmp_path / "missing.dmg"), PROFILE),
        (str(zipf), PROFILE),
        (str(other), PROFILE),
    ]
    for argv in cases:
        res, calls = _run(tmp_path, *argv)
        assert res.returncode == 2, f"argv={argv} rc={res.returncode}, stderr={res.stderr}"
        assert "usage: notarize.sh" in res.stderr, f"argv={argv} no usage line:\n{res.stderr}"
        assert not calls, f"argv={argv} xcrun called:\n{calls}"


def test_no_credential_args_and_profile_on_every_notarytool_call(tmp_path: Path) -> None:
    dmg = tmp_path / "LM-Warden-x.dmg"
    dmg.write_bytes(b"fake dmg")
    res, calls = _run(tmp_path, str(dmg), PROFILE)
    assert res.returncode == 0, res.stderr
    assert calls, "xcrun never called"
    for c in calls:
        for flag in FORBIDDEN:
            assert flag not in c, f"credential flag {flag} in call: {c}"
    for c in calls:
        if c[0] == "notarytool":
            i = c.index("--keychain-profile")
            assert c[i + 1] == PROFILE, f"notarytool call without the profile: {c}"
    # the script itself must not pass credential flags either (checked as
    # whole tokens so --key does not match --keychain-profile)
    src = _notarize().read_text(encoding="utf-8")
    for i, ln in enumerate(src.splitlines(), start=1):
        stripped = ln.strip()
        if stripped.startswith("#"):
            continue
        tokens = stripped.split()
        for flag in FORBIDDEN:
            assert flag not in tokens, f"notarize.sh:{i} passes {flag}: {stripped!r}"


ID = "11111111-2222-3333-4444-555555555555"


def _dmg(tmp_path: Path) -> Path:
    dmg = tmp_path / "LM-Warden-x.dmg"
    dmg.write_bytes(b"fake dmg")
    return dmg


def test_log_failure_still_exits_1_with_message(tmp_path: Path) -> None:
    # The log fetch's own failure is ignored: the rejection still reports
    # exit 1 and its message, not the log's exit code.
    res, calls = _run(
        tmp_path, str(_dmg(tmp_path)), PROFILE, STUB_STATUS="Invalid", STUB_LOG_RC="9"
    )
    assert res.returncode == 1, f"rc={res.returncode}\n{res.stderr}"
    assert f"notarize.sh: notarization Invalid (id {ID})" in res.stderr, res.stderr
    assert not any(c[0] == "stapler" for c in calls), calls


def test_wait_gave_up_in_progress_exits_1_without_stapler(tmp_path: Path) -> None:
    # A --wait that returns before Apple finished reports "In Progress": that
    # is not an acceptance, so no staple.
    res, calls = _run(tmp_path, str(_dmg(tmp_path)), PROFILE, STUB_STATUS="In Progress")
    assert res.returncode == 1, f"rc={res.returncode}\n{res.stderr}"
    assert f"notarize.sh: notarization In Progress (id {ID})" in res.stderr, res.stderr
    assert not any(c[0] == "stapler" for c in calls), calls


def test_submit_nonzero_with_rejected_json_fetches_log(tmp_path: Path) -> None:
    # notarytool may exit non-zero for a finished rejection; the log is still
    # fetched so the CI job shows why.
    res, calls = _run(
        tmp_path, str(_dmg(tmp_path)), PROFILE, STUB_STATUS="Invalid", STUB_SUBMIT_RC="1"
    )
    assert res.returncode == 1, f"rc={res.returncode}\n{res.stderr}"
    assert "notarize.sh: notarytool submit failed" in res.stderr, res.stderr
    assert [c[:3] for c in calls] == [
        ["notarytool", "submit", calls[0][2]],
        ["notarytool", "log", ID],
    ], calls
    assert calls[1][3:] == ["--keychain-profile", PROFILE], calls[1]
    assert "stub notary log" in res.stderr, res.stderr
