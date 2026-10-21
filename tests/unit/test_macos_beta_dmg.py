"""The beta DMG path (BACKLOG-macos P8, OWNER DECISION P8-4).

A beta DMG is an ad-hoc-signed, un-notarized build of a release tag, offered
as a GitHub PRE-RELEASE download only. Overseer rule: it NEVER carries the
Sparkle feed keys (SUFeedURL / SUPublicEDKey) and nothing ever writes an
update-feed (appcast) entry for it, so a beta can never auto-update anyone
and no stable install is ever offered a beta.

Not marked ``macos``: everything here is bash, plistlib, PyYAML or text, so it
runs in Linux CI. The DMG-level guard (mounting a real DMG) is in
test_beta_dmg_seal.py, which is ``macos``-marked.

What it pins:

  (p1) write-info-plist.sh with LMW_BETA_BUILD=1 leaves every SU* key out,
       even for a release tag with both Sparkle vars set; any value other
       than "", "0" or "1" is exit 2 with no file.
  (m1) `make macos-beta-dmg` exists, wants a strict release tag, builds
       through macos-dmg with LMW_BETA_BUILD=1 and both Sparkle vars unset,
       names the DMG LM-Warden-<tag>-beta.dmg and seals it (the DMG guard +
       the .sha256) with beta-dmg-seal.sh.
  (c1) build:macos-beta: M1, tag pipelines only, manual, hard-needs
       test:macos, runs the make target from the tag, unsets the Sparkle
       vars, keeps the DMG and its .sha256 as artifacts.
  (c2) publish:macos-beta: tag pipelines only, manual, hard-needs
       build:macos-beta, runs publish/github-beta-release.sh; forwards only
       the tag and the token into its container.
  (c3) the stable jobs never pick up a beta: build:macos does not build one,
       publish:macos does not need the beta build.
  (g1) nothing writes an update-feed entry for a beta: no beta job, beta
       recipe or beta script mentions the appcast or the Sparkle vars outside
       comments, and no non-comment line of any script, make file or CI job
       mentions both the appcast and a beta artifact.
"""

from __future__ import annotations

import os
import plistlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WRITE_PLIST = REPO / "scripts" / "macos" / "write-info-plist.sh"
SEAL = REPO / "scripts" / "macos" / "beta-dmg-seal.sh"
PUBLISH = REPO / "publish" / "github-beta-release.sh"
MACOS_MK = REPO / "make" / "macos.mk"
CI_FILE = REPO / ".gitlab-ci.yml"

TAG = "v2026.10.21.1"
FEED = "https://lmwarden.com/macos/appcast.xml"
ED_KEY = "A" * 43 + "="
SPARKLE_VARS = ("LMW_SPARKLE_FEED_URL", "LMW_SPARKLE_PUBLIC_ED_KEY")
RELEASE_TAG_RULE = "$CI_COMMIT_TAG =~ /^v\\d{4}\\.\\d{2}\\.\\d{2}\\.\\d+$/"


# ---------------------------------------------------------------- helpers


def _env(**extra: str) -> dict[str, str]:
    env = {**os.environ}
    for k in (*SPARKLE_VARS, "LMW_BETA_BUILD"):
        env.pop(k, None)
    env.update(extra)
    return env


def _plist(tmp_path: Path, **extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(WRITE_PLIST), str(tmp_path / "Info.plist"), TAG, "LMWarden"],
        capture_output=True,
        text=True,
        env=_env(**extra),
    )


def _code_lines(text: str) -> list[str]:
    """Non-empty lines that are not `#` comments (shell, make and YAML)."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _recipe(target: str) -> str:
    """The rule line and recipe of a make target in make/macos.mk."""
    lines = MACOS_MK.read_text(encoding="utf-8").splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith(f"{target}:")), None)
    assert start is not None, f"make/macos.mk has no {target} target"
    body = [lines[start]]
    for ln in lines[start + 1 :]:
        if ln.startswith("\t"):
            body.append(ln)
        else:
            break
    return "\n".join(body)


def _ci() -> dict:
    doc = yaml.safe_load(CI_FILE.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    return doc


def _job(name: str) -> dict:
    job = _ci().get(name)
    assert isinstance(job, dict), f"job {name!r} is missing from .gitlab-ci.yml"
    return job


def _job_text(job: dict) -> str:
    parts: list[str] = []
    for key in ("before_script", "script", "after_script"):
        for entry in job.get(key) or []:
            parts.append(str(entry))
    parts.append(yaml.safe_dump(job.get("variables") or {}))
    return "\n".join(parts)


def _needs(job: dict) -> dict[str, bool]:
    """need name -> optional?"""
    out: dict[str, bool] = {}
    for entry in job.get("needs") or []:
        if isinstance(entry, str):
            out[entry] = False
        else:
            out[entry["job"]] = bool(entry.get("optional"))
    return out


# ------------------------------------------------- (p1) write-info-plist


def test_beta_build_drops_the_sparkle_keys_from_a_release_tag(tmp_path: Path) -> None:
    proc = _plist(
        tmp_path,
        LMW_SPARKLE_FEED_URL=FEED,
        LMW_SPARKLE_PUBLIC_ED_KEY=ED_KEY,
        LMW_BETA_BUILD="1",
    )
    assert proc.returncode == 0, proc.stderr
    info = plistlib.loads((tmp_path / "Info.plist").read_bytes())
    su = [k for k in info if k.startswith("SU")]
    assert su == [], f"a beta plist carries Sparkle keys: {su}"
    assert "beta" in proc.stderr.lower(), "the beta drop is silent; say so on stderr"
    # It is still the tag's build otherwise.
    assert info["CFBundleShortVersionString"] == TAG
    assert info["CFBundleVersion"] == "2026102101"


def test_beta_build_is_byte_identical_to_a_feedless_release(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    beta = tmp_path / "beta"
    plain.mkdir()
    beta.mkdir()
    assert _plist(plain).returncode == 0
    assert (
        _plist(
            beta,
            LMW_SPARKLE_FEED_URL=FEED,
            LMW_SPARKLE_PUBLIC_ED_KEY=ED_KEY,
            LMW_BETA_BUILD="1",
        ).returncode
        == 0
    )
    assert (beta / "Info.plist").read_bytes() == (plain / "Info.plist").read_bytes()


@pytest.mark.parametrize("value", ["", "0"])
def test_non_beta_release_still_carries_the_keys(tmp_path: Path, value: str) -> None:
    # Control: the beta switch does not turn the stable path off.
    proc = _plist(
        tmp_path,
        LMW_SPARKLE_FEED_URL=FEED,
        LMW_SPARKLE_PUBLIC_ED_KEY=ED_KEY,
        LMW_BETA_BUILD=value,
    )
    assert proc.returncode == 0, proc.stderr
    info = plistlib.loads((tmp_path / "Info.plist").read_bytes())
    assert info["SUFeedURL"] == FEED


@pytest.mark.parametrize("value", ["yes", "true", "2", " 1"])
def test_malformed_beta_switch_exits_2_and_writes_nothing(tmp_path: Path, value: str) -> None:
    proc = _plist(tmp_path, LMW_BETA_BUILD=value)
    assert proc.returncode == 2, proc.stderr
    assert not (tmp_path / "Info.plist").exists()


# ------------------------------------------------------- (m1) make target


def test_make_beta_target_builds_through_macos_dmg_as_a_beta() -> None:
    recipe = _recipe("macos-beta-dmg")
    assert (
        "macos-beta-dmg" in MACOS_MK.read_text(encoding="utf-8").split(".PHONY:")[1].splitlines()[0]
    )
    assert "LMW_BETA_BUILD=1" in recipe
    for var in SPARKLE_VARS:
        assert f"-u {var}" in recipe, f"macos-beta-dmg must unset {var} for the build"
    assert "macos-dmg VERSION=" in recipe
    assert 'OUT="$(DIST_DIR)/LM-Warden-$(VERSION)-beta.dmg"' in recipe
    assert "scripts/macos/beta-dmg-seal.sh" in recipe
    # The seal is its own recipe command on the beta DMG, not a word inside
    # another one (a `true scripts/...` or an `echo` would still match above).
    seal_cmds = [
        ln.strip()
        for ln in recipe.splitlines()[1:]
        if ln.strip().lstrip("@-+").startswith("scripts/macos/beta-dmg-seal.sh ")
    ]
    assert seal_cmds == [
        'scripts/macos/beta-dmg-seal.sh "$(DIST_DIR)/LM-Warden-$(VERSION)-beta.dmg" "$(VERSION)"'
    ], recipe
    # The seal (the DMG guard + checksum) runs after the build.
    assert recipe.index("macos-dmg VERSION=") < recipe.index("beta-dmg-seal.sh")


def test_make_beta_target_wants_a_release_tag() -> None:
    recipe = _recipe("macos-beta-dmg")
    assert "bundle-version.sh" in recipe
    # The tag guard comes before the build.
    assert recipe.index("bundle-version.sh") < recipe.index("macos-dmg VERSION=")


@pytest.mark.skipif(shutil.which("make") is None, reason="needs make")
def test_make_beta_target_refuses_a_non_tag_before_building(tmp_path: Path) -> None:
    # The guard is the recipe's first lines: a dev/sha/malformed VERSION is
    # exit 2 before the first build command runs.
    for bad in ("dev", "abc1234", "v2026.13.01.1", ""):
        proc = subprocess.run(
            [
                "make",
                "--no-print-directory",
                "macos-beta-dmg",
                f"VERSION={bad}",
                f"DIST_DIR={tmp_path}",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            env=_env(),
        )
        assert proc.returncode == 2, f"VERSION={bad!r}: rc={proc.returncode} {proc.stderr}"
        assert "swift build" not in proc.stdout, f"VERSION={bad!r} started a build"
        assert "macos-beta-dmg" in proc.stderr


# --------------------------------------------------------- (c1)-(c3) CI


def _only_manual_on_release_tags(job: dict, name: str) -> None:
    rules = job.get("rules")
    assert isinstance(rules, list) and len(rules) == 1, f"{name}: want exactly one rule"
    rule = rules[0]
    assert rule.get("if") == RELEASE_TAG_RULE, f"{name}: must run on release tags only"
    assert rule.get("when") == "manual", f"{name}: a beta is an overseer call, so manual"
    assert (
        rule.get("allow_failure") is True
    ), f"{name}: a manual beta must not block the tag pipeline"


def test_build_macos_beta_job() -> None:
    job = _job("build:macos-beta")
    assert job.get("stage") == "macos"
    assert job.get("tags") == ["macos", "arm64"]
    assert job.get("interruptible") is False
    _only_manual_on_release_tags(job, "build:macos-beta")
    assert _needs(job) == {"test:macos": False}, "a beta DMG hard-needs a green test:macos"
    cmds = _code_lines(_job_text(job))
    assert 'make macos-beta-dmg VERSION="$CI_COMMIT_TAG"' in cmds
    unset = [c for c in cmds if c.startswith("unset ")]
    assert unset, "build:macos-beta must unset the Sparkle vars"
    for var in SPARKLE_VARS:
        assert any(var in c for c in unset), f"build:macos-beta does not unset {var}"
    paths = job["artifacts"]["paths"]
    assert paths == ["dist/LM-Warden-*-beta.dmg", "dist/LM-Warden-*-beta.dmg.sha256"]


def test_build_macos_beta_body_stays_one_locked_single_quoted_string() -> None:
    script = _job("build:macos-beta")["script"]
    assert len(script) == 1
    body = script[0]
    assert "portable.py lock /Users/Shared/lmw-live-model.lock -- sh -c '" in body
    inner = body[body.find("sh -c '") + len("sh -c '") : body.rfind("'")]
    assert "'" not in inner, "an apostrophe inside the sh -c body breaks the job"


def test_publish_macos_beta_job() -> None:
    job = _job("publish:macos-beta")
    assert job.get("stage") == "macos-publish"
    assert job.get("interruptible") is False
    assert job.get("dependencies") is None or job.get("dependencies") == ["build:macos-beta"]
    _only_manual_on_release_tags(job, "publish:macos-beta")
    assert _needs(job) == {"build:macos-beta": False}
    text = _job_text(job)
    assert "publish/github-beta-release.sh" in text
    assert '"dist/LM-Warden-$CI_COMMIT_TAG-beta.dmg"' in text
    # Only the tag and the token cross into the publish container.
    forwarded = re.findall(r"-e ([A-Z_]+)", text)
    assert sorted(forwarded) == ["CI_COMMIT_TAG", "GITHUB_PUBLISH_TOKEN"]
    assert "set -x" not in text


def test_stable_jobs_never_touch_the_beta() -> None:
    assert "build:macos-beta" not in _needs(_job("publish:macos"))
    assert "macos-beta-dmg" not in _job_text(_job("build:macos"))
    assert "-beta" not in str(_job("build:macos").get("artifacts"))


# ------------------------------------------- (g1) no update feed for a beta


def test_beta_surfaces_never_mention_the_feed_outside_comments() -> None:
    surfaces = {
        "make macos-beta-dmg": _recipe("macos-beta-dmg"),
        "beta-dmg-seal.sh": SEAL.read_text(encoding="utf-8"),
        "github-beta-release.sh": PUBLISH.read_text(encoding="utf-8"),
        "build:macos-beta": _job_text(_job("build:macos-beta")),
        "publish:macos-beta": _job_text(_job("publish:macos-beta")),
    }
    for where, text in surfaces.items():
        for line in _code_lines(text):
            low = line.lower()
            assert "appcast" not in low, f"{where} mentions the appcast: {line!r}"
            assert "generate_appcast" not in low
            if where not in ("make macos-beta-dmg", "build:macos-beta"):
                # Only the build may NAME the vars, to unset them.
                assert "LMW_SPARKLE" not in line, f"{where} uses a Sparkle var: {line!r}"


def test_nothing_writes_a_feed_entry_for_a_beta_artifact() -> None:
    # Any shell script, make file, publish script or CI job that (outside
    # comments) mentions the appcast must not mention a beta artifact.
    files = [
        *sorted((REPO / "scripts").rglob("*.sh")),
        *sorted((REPO / "publish").rglob("*.sh")),
        *sorted((REPO / "make").glob("*.mk")),
        REPO / "Makefile",
    ]
    for path in files:
        code = [
            ln.lower() for ln in _code_lines(path.read_text(encoding="utf-8", errors="replace"))
        ]
        if any("appcast" in ln for ln in code):
            hits = [ln for ln in code if "-beta" in ln or "beta.dmg" in ln]
            assert not hits, f"{path.relative_to(REPO)} writes the feed and names a beta: {hits}"
    for name, job in _ci().items():
        if not isinstance(job, dict) or "script" not in job:
            continue
        code = [ln.lower() for ln in _code_lines(_job_text(job))]
        if any("appcast" in ln for ln in code):
            assert "build:macos-beta" not in _needs(
                job
            ), f"{name} feeds the appcast from the beta build"
            assert not [ln for ln in code if "-beta" in ln], f"{name} feeds the appcast a beta"
