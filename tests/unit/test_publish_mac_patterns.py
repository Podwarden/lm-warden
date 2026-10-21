"""Publish safety for the new ``macos/`` Swift/Xcode tree (Task 8).

The public GitHub publish (``publish/publish-github.sh``) stages the repo
minus ``publish/exclude.txt``, then scans the staged tree against
``publish/denylist-block.txt`` (case-insensitive extended regexes; ANY match
FAILS the publish) and ``publish/denylist-warn.txt`` (reported, not fatal).

Xcode files can carry Apple team IDs, signing identities and internal host
names that today's block list does not catch. These tests pin the new block
patterns and the new exclude rules so a future macOS file that leaks any of
them fails the publish before it reaches GitHub.

This module uses the SAME matcher the publish script uses: it strips
comments from the denylist the same way (``grep -vE '^[[:space:]]*(#|$)'``),
then calls ``grep -rliE -e <pattern> <dir>`` on a tmp tree. Nothing about the
script's matching is factored out or approximated -- the test reproduces it
line for line so a change to the script's matcher breaks this test too.

IMPORTANT: this test file is itself published (``tests/`` is not in
``exclude.txt``), and the denylist patterns are case-insensitive. Every
literal leak string and pattern string in this source is therefore assembled
from fragments at runtime so the published source does not itself trip the
block scan. The technique mirrors ``test_bench_figures.py``'s
``"lmw" + "48"`` idiom: the bytes that would match never appear as a
contiguous literal in the file.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

#: Repo root (the worktree root for this test run).
REPO = Path(__file__).resolve().parents[2]
PUBLISH_DIR = REPO / "publish"
BLOCK_LIST = PUBLISH_DIR / "denylist-block.txt"
EXCLUDE_LIST = PUBLISH_DIR / "exclude.txt"
PUBLISH_SCRIPT = PUBLISH_DIR / "publish-github.sh"

# --- fragment-assembled leak strings ----------------------------------------
#
# Every string below is built from fragments so the contiguous literal never
# appears in this source file. The publish block scan is case-insensitive
# and this file ships in the public tree, so a contiguous pair of fragments
# is safe but the full leak string written as one literal would fail the
# publish on this test file itself.
#
# ``_PAT`` fragments assemble to the denylist regex; ``_FIX`` fragments
# assemble to a fixture line the regex must match. The patterns are also
# read back from the denylist file at runtime (see ``block_patterns``) so
# the test does not rely on the fragment assembly for correctness -- it
# only relies on it to keep this source file off the block list.

_DT_PREFIX = "DEVELOPMENT" + "_"
_DT_ASSIGN = " " + "=" + " "
_DT_VALUE = "ABCDE" + "12345"
_DT_FIX = _DT_PREFIX + "TEAM" + _DT_ASSIGN + _DT_VALUE  # the team-id leak line
_DT_PAT = _DT_PREFIX + "TEAM" + " " + "*" + "=" + " " + "*" + "[A-Z0-9]{10}"

_DEVID_FIX = (
    "CODE_SIGN_IDENTITY = "
    + '"'
    + "Developer ID "
    + "Application: "
    + "Someone ("
    + _DT_VALUE
    + ')"'
)
_DEVID_PAT = "Developer ID " + "Application:"

_IP_FIX = "ssh ip@" + "10" + ".10." + "0.141"
# Tightened to an IP-ish form: a non-digit/dot/v char (or start of line)
# before the IP, then a full four-octet form. This avoids matching version
# strings (v10.10.1, "10.10.0" in a lockfile) while still catching the
# fixture (@ is not [0-9.v]).
_IP_PAT = "(^|[^0-9.v])" + "10" + r"\." + "10" + r"\." + "[0-9]{1,3}" + r"\." + "[0-9]{1,3}"

_HOST_FIX = "Host " + "mac" + "-m1"
_HOST_PAT = "mac" + "-m1"

#: The four new block patterns Task 8 adds, each paired with a fixture line
#: the pattern must match. Patterns and fixtures are fragment-assembled above
#: so this source file does not itself trip the block scan.
NEW_BLOCK_PATTERNS = {
    _DT_PAT: _DT_FIX,
    _DEVID_PAT: _DEVID_FIX,
    _IP_PAT: _IP_FIX,
    _HOST_PAT: _HOST_FIX,
}


def block_patterns() -> list[str]:
    """Read the non-comment patterns from ``denylist-block.txt`` -- the same
    ``strip_comments`` the publish script uses. Returned in file order.
    """
    out: list[str] = []
    for line in BLOCK_LIST.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(line)
    return out


# --- helpers ----------------------------------------------------------------


def grep_block_hit(pattern: str, root: Path) -> bool:
    """True when ``pattern`` matches any file under ``root`` using the SAME
    invocation the publish script's block scan uses:
    ``grep -rliE -e <pattern> <root>``. Returns True on any hit, False on
    no hits, and propagates grep exit 2 (a usage / malformed-regex error)
    as a test failure -- the publish script fails closed on exit 2 too.
    """
    proc = subprocess.run(
        ["grep", "-rliE", "-e", pattern, str(root)],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 2:
        pytest.fail(f"grep could not evaluate pattern {pattern!r}: {proc.stderr}")
    return proc.returncode == 0


needs_rsync = pytest.mark.skipif(
    shutil.which("rsync") is None,
    reason="rsync not installed (the publish script's staging tool; covered where rsync exists)",
)


def stage_repo(repo: Path, dest: Path) -> None:
    """Dry-stage the repo into ``dest`` using ``publish/exclude.txt`` -- the
    same ``rsync -a --exclude-from`` the publish script uses. Strips any
    surviving ``.git`` exactly as the script does.

    NOTE: this stages the WORKING TREE (including untracked and gitignored
    content). Use ``stage_tracked_files`` for the real-tree guard test,
    which must stage only what the publish actually ships.
    """
    subprocess.run(
        [
            "rsync",
            "-a",
            f"--exclude-from={EXCLUDE_LIST}",
            f"{repo}/",
            f"{dest}/",
        ],
        check=True,
    )
    shutil.rmtree(dest / ".git", ignore_errors=True)


def stage_tracked_files(repo: Path, dest: Path, exclude_from: Path) -> None:
    """Stage TRACKED files only from ``repo`` into ``dest``, then apply
    ``exclude_from`` (``publish/exclude.txt``) in a second rsync pass.

    The publish script stages the working tree, but in CI the checkout is
    a fresh clone with no untracked or gitignored content. Locally the
    working tree has ``macos/.build`` (157 MB), ``.venv``, ``.git``, and
    scratch files that CI never has -- staging those makes the real-tree
    guard test slow and environment-dependent. This helper stages exactly
    what the publish would ship: tracked files, minus the exclude list.

    Two rsync passes are needed because ``--files-from`` disables the
    anchored-exclude semantics of ``--exclude-from``: the first pass
    copies tracked files (via ``git ls-files -z`` piped to ``rsync
    --from0 --files-from=-``), the second applies ``--exclude-from`` with
    its root-anchored ``/`` patterns intact.

    Raises ``RuntimeError`` if ``repo`` is not inside a git checkout
    (``git ls-files`` fails) -- callers should ``pytest.skip`` in that
    case.
    """
    tracked = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"],
        capture_output=True,
        check=False,
    )
    if tracked.returncode != 0:
        raise RuntimeError(
            f"git ls-files failed in {repo} -- not a git checkout. "
            f"stderr: {tracked.stderr.decode()}"
        )
    # Step 1: copy tracked files only (NUL-delimited, preserves all paths).
    mkdir(dest)
    subprocess.run(
        [
            "rsync",
            "-a",
            "--from0",
            "--files-from=-",
            str(repo) + "/",
            str(dest) + "/",
        ],
        input=tracked.stdout,
        check=True,
    )
    # Step 2: apply the exclude list with anchored patterns intact.
    excluded = dest.parent / (dest.name + "-excluded")
    subprocess.run(
        [
            "rsync",
            "-a",
            "--delete",
            f"--exclude-from={exclude_from}",
            f"{dest}/",
            f"{excluded}/",
        ],
        check=True,
    )
    shutil.rmtree(excluded / ".git", ignore_errors=True)
    # Replace dest with the excluded tree.
    shutil.rmtree(dest)
    excluded.rename(dest)


def mkdir(path: Path) -> None:
    """``mkdir -p`` for a Path."""
    path.mkdir(parents=True, exist_ok=True)


# --- pattern presence in the denylist ---------------------------------------


def test_new_block_patterns_are_listed() -> None:
    """Every new Task 8 block pattern appears verbatim in
    ``publish/denylist-block.txt``. A typo in the pattern (or a dropped
    line) breaks this test loudly.
    """
    listed = set(block_patterns())
    for pattern in NEW_BLOCK_PATTERNS:
        assert pattern in listed, (
            f"pattern {pattern!r} is not in {BLOCK_LIST}. " "Task 8 requires it to be added."
        )


def test_existing_block_patterns_still_listed() -> None:
    """The pre-Task-8 block patterns are still present -- the new patterns
    are ADDITIVE, not a rewrite.
    """
    listed = set(block_patterns())
    # Fragment-assembled so this source does not trip the block scan itself.
    existing = (
        "co-" + "authored-" + "by",
        "noreply" + "@" + "anthropic" + r"\." + "com",
        "prot" + "rener",
    )
    for pat in existing:
        assert pat in listed, f"pre-Task-8 pattern {pat!r} was removed from the block list."


# --- each new pattern matches a fixture line ---------------------------------


@pytest.mark.parametrize("pattern,fixture_line", list(NEW_BLOCK_PATTERNS.items()))
def test_new_block_pattern_matches_fixture(tmp_path: Path, pattern: str, fixture_line: str) -> None:
    """Each new block pattern matches its fixture line when grep is run
    with the publish script's exact flags (``-rliE -e``). A pattern that
    compiles but does not match the leak it is supposed to catch is a
    no-op denylist entry; this test pins that the pattern is meaningful.
    """
    (tmp_path / "leak.txt").write_text(fixture_line + "\n")
    assert grep_block_hit(pattern, tmp_path), (
        f"pattern {pattern!r} did not match its fixture line "
        "under the publish script's `grep -rliE -e` invocation."
    )


@pytest.mark.parametrize("pattern,fixture_line", list(NEW_BLOCK_PATTERNS.items()))
def test_new_block_pattern_clean_when_absent(
    tmp_path: Path, pattern: str, fixture_line: str
) -> None:
    """The inverse: a tree that does NOT contain the leak has no hit for
    the pattern. Guards against a pattern that matches everything (e.g.
    a stray ``.*`` or a bare dot).
    """
    (tmp_path / "clean.txt").write_text("nothing to see here\n")
    assert not grep_block_hit(pattern, tmp_path), (
        f"pattern {pattern!r} matched a file that does not contain the " f"leak -- it is too broad."
    )


# --- the block scan reads patterns the same way the script does --------------


def test_block_scan_finds_xcconfig_team_id(tmp_path: Path) -> None:
    """A dry-stage of a tmp tree with ``macos/x.xcconfig`` containing a
    DEVELOPMENT_TEAM line fails the block scan (i.e. the scan finds at
    least one hit). This is the headline Task 8 scenario: an Xcode
    build-settings file leaking an Apple team ID must never reach the
    public mirror.
    """
    macos = tmp_path / "macos"
    macos.mkdir()
    (macos / "x.xcconfig").write_text(_DT_FIX + "\n" + "CODE_SIGN_STYLE = Automatic\n")
    # Run the block scan the same way the script does: for each pattern
    # in the denylist (comments stripped), grep -rliE -e <pattern> <dir>.
    hit = False
    for pattern in block_patterns():
        if grep_block_hit(pattern, tmp_path):
            hit = True
            break
    assert hit, (
        "A tmp tree containing macos/x.xcconfig with a DEVELOPMENT_TEAM line "
        "was NOT caught by the block scan. The publish would ship an Apple "
        "team ID to GitHub."
    )


def test_block_scan_passes_real_package_swift(tmp_path: Path) -> None:
    """The real ``macos/Package.swift`` (the one checked into the repo)
    passes the block scan: no block pattern matches it. This is the
    other half of the Task 8 headline: the legitimate Swift package
    skeleton must not be false-failed by the new patterns.
    """
    src = REPO / "macos" / "Package.swift"
    assert src.is_file(), f"{src} does not exist; the macos skeleton is Task 6."
    shutil.copy2(src, tmp_path / "Package.swift")
    for pattern in block_patterns():
        assert not grep_block_hit(pattern, tmp_path), (
            "a block pattern matches the real macos/Package.swift. "
            "A legitimate public file is being false-failed; the pattern is "
            "too broad or the file leaks something it should not."
        )


# --- exclude rules ----------------------------------------------------------


@needs_rsync
def test_restart_factory_sh_is_excluded(tmp_path: Path) -> None:
    """``restart-factory.sh`` (the factory's session launcher) is excluded
    from the public snapshot. Same mechanism: an ``rsync --exclude-from`` dry stage drops it.
    """
    src_repo = tmp_path / "src"
    src_repo.mkdir()
    (src_repo / "restart-factory.sh").write_text("#!/bin/sh\n# operator script\n")
    (src_repo / "README.md").write_text("# public\n")
    stage = tmp_path / "stage"
    stage_repo(src_repo, stage)
    assert not (stage / "restart-factory.sh").exists(), (
        "restart-factory.sh was NOT excluded by publish/exclude.txt. " "It must be listed there."
    )


def test_exclude_list_names_restart_scripts() -> None:
    """``publish/exclude.txt`` explicitly names the restart script.
    Belt-and-braces against the rsync behaviour test above: a future
    edit that breaks the rsync anchoring (e.g. dropping the leading
    ``/``) would slip past the rsync test if the file happened to be
    absent; this assertion catches the textual omission directly.
    """
    text = EXCLUDE_LIST.read_text()
    assert "/restart-factory.sh" in text, "publish/exclude.txt does not name /restart-factory.sh."


# --- publish script matcher contract ----------------------------------------


def test_publish_script_uses_grep_rliE() -> None:
    """The publish script's block scan uses ``grep -rliE -e <pattern>``
    (recursive, list-files, case-insensitive, extended-regex, -e for
    pattern safety). If the script's matcher changes, these tests must
    be revisited so they do not keep passing against a different
    matcher. This test pins the contract.
    """
    text = PUBLISH_SCRIPT.read_text()
    assert "grep -rliE -e" in text, (
        "publish-github.sh no longer uses `grep -rliE -e <pattern>` for "
        "the block scan. The matcher this test reproduces has changed; "
        "update test_publish_mac_patterns.py to match."
    )
    assert "strip_comments()" in text, (
        "publish-github.sh no longer defines strip_comments(). The "
        "comment-stripping this test reproduces has changed."
    )


# --- self-defense: this test file does not trip the block scan --------------


def test_this_test_file_does_not_trip_block_scan(tmp_path: Path) -> None:
    """This test file is published (``tests/`` is not excluded), and the
    block scan is case-insensitive. If a future edit introduces a
    contiguous literal that matches a block pattern, the publish would
    fail on this file itself. This test copies the module into a tmp
    tree and runs the full block scan against it -- it must be clean.
    """
    shutil.copy2(Path(__file__), tmp_path / "test_publish_mac_patterns.py")
    for pattern in block_patterns():
        assert not grep_block_hit(pattern, tmp_path), (
            f"block pattern {pattern!r} matches THIS test file. The publish "
            "would fail on tests/unit/test_publish_mac_patterns.py itself. "
            "Assemble the leak string from fragments (see the top of this "
            "module) so the contiguous literal never appears in the source."
        )


# --- the real tracked tree is clean for the new patterns --------------------


@needs_rsync
def test_real_tree_clean_for_new_patterns(tmp_path: Path) -> None:
    """A dry-stage of the REAL TRACKED tree (``git ls-files`` minus
    ``publish/exclude.txt``) has ZERO hits for each of the four new Task 8
    block patterns. This is the operational invariant: the publish must
    not fail on a legitimate public file.

    Stages TRACKED files only (not the working tree) so untracked /
    gitignored content (``macos/.build``, ``.venv``, scratch files) does
    not make the test slow or environment-dependent -- CI's fresh clone
    has none of that, and neither does this stage. Skips cleanly when
    the repo is not inside a git checkout.

    A future leak (a new ``.xcconfig`` with a team ID, a comment naming
    the M1 worker) or a removed exclusion (e.g. ``tests/unit/farm/``
    dropped from ``exclude.txt``) turns this test red. The pre-existing
    GitLab-registry hit on ``frontend/package-lock.json`` is NOT tested
    here: the publish script rewrites that ``resolved`` URL to npmjs
    before the scan runs (lines 177-182), so it is a false positive of a
    local dry-stage and not a real publish failure. Only the four new
    patterns are asserted clean.
    """
    stage = tmp_path / "stage"
    try:
        stage_tracked_files(REPO, stage, EXCLUDE_LIST)
    except (RuntimeError, FileNotFoundError) as exc:
        pytest.skip(f"not in a git checkout (or git absent): {exc}")
    for pattern in NEW_BLOCK_PATTERNS:
        assert not grep_block_hit(pattern, stage), (
            f"new block pattern {pattern!r} matches a file in the dry-staged "
            "tracked tree. The publish would fail on a legitimate public "
            "file. Either clean the source (split the literal, as "
            "test_bench_figures.py does for 'lmw' + '48'), or extend "
            "publish/exclude.txt (as tests/unit/farm/ is excluded because "
            "scripts/farm/ is)."
        )
