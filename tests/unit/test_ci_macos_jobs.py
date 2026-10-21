"""CI-lint net for the `test:macos` job's requirements lock check.

BACKLOG-macos P8 pinned `requirements-macos.txt` (a `uv pip compile` lock of
the Python the app bundle embeds) with `--exclude-newer`, so `--check` and
re-locks are reproducible. The `macos-requirements-check` make target
re-resolves `requirements-macos.in` and diffs against the committed lock, so a
lock that drifted (a `.in` or a requirements.txt pin changed without a
`make macos-requirements` re-lock) is catchable. This file pins that the
`test:macos` CI job actually runs that check on the M1, where the lock only
means anything (the resolution is pinned to the bundle's arm64 interpreter).

What it pins:

  (d1) the `test:macos` script is still ONE entry, and it still runs under the
      single `portable.py lock ... -- sh -c` wrapper (the lock cannot span
      separate `script:` list items, and a split would let the requirements
      check race the live model);
  (d2) `make macos-requirements-check` runs, exactly once;
  (d3) it runs AFTER `uv pip install` (uv must be on PATH) and BEFORE
      `make macos-test` (fail fast on drift, cheap, before the Swift suite);
  (d4) the whole body is still one single-quoted `sh -c` string — the check
      line and its comment carry no single quote of their own (an apostrophe
      would end the string and break the job);
  (d5) an MR touching only the lock (`requirements-macos.in` / `.txt`)
      triggers the job, so the drift check runs exactly when the lock could
      move;
  (d6) only `test:macos` runs the check — not `build:macos` (the check needs
      PyPI network and would slow the artifact build for nothing);
  (d7) the same body still exports `LMW_DMG_BUILD=1` before `make macos-test`
      (the opt-in for test_make_dmg.py's full bundle + DMG leg).

`.gitlab-ci.yml` is parsed with PyYAML (`safe_load`, the safe parser) — the
same approach as the sibling `test_ci_tag_pipeline.py`; PyYAML is available
transitively via `transformers`, so this file adds no dependency. The parse
resolves the `*needs-gate` anchor itself; only the `rules:` evaluation is
GitLab-specific and is NOT re-implemented here (the `changes:` rule is
inspected structurally, not evaluated).
"""

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_FILE = REPO_ROOT / ".gitlab-ci.yml"

CHECK = "make macos-requirements-check"


def _ci() -> dict:
    doc = yaml.safe_load(CI_FILE.read_text())
    assert isinstance(doc, dict) and doc, f"{CI_FILE.name} did not parse to a mapping"
    return doc


def _job(ci: dict, name: str) -> dict:
    job = ci.get(name)
    assert isinstance(job, dict), f"job {name!r} is missing from .gitlab-ci.yml"
    return job


def _script_body(job: dict) -> str:
    script = job.get("script")
    assert isinstance(script, list), f"job {job!r} has no `script` list"
    assert len(script) == 1, f"expected exactly one script entry, got {len(script)}"
    body = script[0]
    assert isinstance(body, str), "the single script entry is not a string"
    return body


def _script_commands(body: str) -> list[str]:
    """The stripped non-empty lines of a script body, minus `#` comments.

    A line whose first non-space character is `#` is a shell comment and is
    dropped; the rest are returned stripped, in order."""
    out: list[str] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def _mr_rule(job: dict) -> dict:
    """The `merge_request_event` rule, with its `changes:` list."""
    rules = job.get("rules")
    assert isinstance(rules, list) and rules, "test:macos has no rules"
    for rule in rules:
        assert isinstance(rule, dict), f"unsupported rules entry: {rule!r}"
        if rule.get("if") == '$CI_PIPELINE_SOURCE == "merge_request_event"':
            assert "changes" in rule, "the MR rule has no `changes:` list"
            return rule
    raise AssertionError("no merge_request_event rule in test:macos")


def test_test_macos_keeps_single_locked_script():
    # (d1) one script entry, still the portable.py lock wrapper.
    body = _script_body(_job(_ci(), "test:macos"))
    assert body.lstrip().startswith(
        "python3 scripts/farm/portable.py lock /Users/Shared/lmw-live-model.lock -- sh -c"
    ), "test:macos no longer runs its whole script under one portable.py lock"


def test_test_macos_runs_requirements_check_once():
    # (d2) the check runs, exactly once.
    cmds = _script_commands(_script_body(_job(_ci(), "test:macos")))
    assert cmds.count(CHECK) == 1, f"{CHECK!r} should run exactly once, runs {cmds.count(CHECK)}"


def test_test_macos_check_sits_between_install_and_test():
    # (d3) after `uv pip install`, before `make macos-test`.
    cmds = _script_commands(_script_body(_job(_ci(), "test:macos")))
    uv_idx = next(
        (i for i, c in enumerate(cmds) if c.startswith("uv pip install --python .venv-macos")),
        None,
    )
    test_idx = next((i for i, c in enumerate(cmds) if c == "make macos-test"), None)
    check_idx = next((i for i, c in enumerate(cmds) if c == CHECK), None)
    assert uv_idx is not None, "expected the `uv pip install` line"
    assert test_idx is not None, "expected the `make macos-test` line"
    assert check_idx is not None, f"missing {CHECK!r}"
    assert uv_idx < check_idx, "the check must run after uv is installed (uv on PATH)"
    assert check_idx < test_idx, "the check must run before the Swift suite (fail fast)"


def test_test_macos_keeps_the_dmg_build_opt_in():
    # (d7) the check landed in the same body as 1af91749's `export LMW_DMG_BUILD=1`;
    # the merge must not drop it: still exported once, still before `make macos-test`
    # (test_make_dmg.py's full bundle + DMG leg only runs with it set).
    cmds = _script_commands(_script_body(_job(_ci(), "test:macos")))
    assert cmds.count("export LMW_DMG_BUILD=1") == 1, "test:macos lost its LMW_DMG_BUILD=1 opt-in"
    assert cmds.index("export LMW_DMG_BUILD=1") < cmds.index(
        "make macos-test"
    ), "LMW_DMG_BUILD=1 must be exported before `make macos-test`"


def test_test_macos_body_stays_one_single_quoted_string():
    # (d4) between the first `sh -c '` and the final `'` there is no other `'`.
    body = _script_body(_job(_ci(), "test:macos"))
    start = body.find("sh -c '")
    assert start != -1, "the body does not open an sh -c string"
    inner_start = start + len("sh -c '")
    end = body.rfind("'")
    assert end > inner_start, "the body does not close its sh -c string"
    inner = body[inner_start:end]
    assert "'" not in inner, (
        "an apostrophe inside the sh -c body would end the single-quoted string "
        "and break the job"
    )


def test_mr_rule_watches_the_macos_lock_files():
    # (d5) an MR touching only the lock triggers the job.
    changes = _mr_rule(_job(_ci(), "test:macos"))["changes"]
    assert "requirements-macos.in" in changes, "MR lock trigger missing requirements-macos.in"
    assert "requirements-macos.txt" in changes, "MR lock trigger missing requirements-macos.txt"


def test_only_test_macos_runs_requirements_check():
    # (d6) across all jobs, only test:macos runs the check (build:macos does not).
    ci = _ci()
    runners = []
    for name, job in ci.items():
        if not isinstance(job, dict) or not isinstance(job.get("script"), list):
            continue
        for entry in job["script"]:
            if not isinstance(entry, str):
                continue
            if any(c == CHECK for c in _script_commands(entry)):
                runners.append(name)
    assert runners == ["test:macos"], f"expected only test:macos to run the check, got {runners}"
    build_cmds = _script_commands(_script_body(_job(ci, "build:macos")))
    assert CHECK not in build_cmds, "build:macos must not run the requirements check"


def test_script_commands_helper_boundaries():
    # (e) helper legs: comments and indentation excluded, exact match, empty.
    assert _script_commands("# make macos-requirements-check") == []
    assert _script_commands("   # make macos-requirements-check") == []
    assert _script_commands("make macos-requirements-check  ") == ["make macos-requirements-check"]
    assert _script_commands("make macos-requirements-checkx") == ["make macos-requirements-checkx"]
    assert "make macos-requirements-check" not in _script_commands("make macos-requirements-checkx")
    assert _script_commands("") == []


# ---------------------------------------------------------------------------
# (f) BACKLOG-macos P8 webui-default-on: the bundle carries the web UI export
# by default, so both M1 jobs install frontend/ deps inside the lock before
# anything builds the bundle, keep PYTHON_CACHE outside the checkout, and the
# body never prints the environment (NPM_TOKEN reaches npm through it).
# ---------------------------------------------------------------------------

NPM_CI = "(cd frontend && npm ci --no-audit --no-fund --prefer-offline)"
PY_CACHE = 'export PYTHON_CACHE="$HOME/Library/Caches/lmw-macos-python"'
# Job name -> the command that builds the bundle (needs node_modules).
# `set -x`, `set -ex`, `set -o xtrace`, `sh -x`, `bash -xc` ...
XTRACE = re.compile(r"\b(?:set|sh|bash)\s+(?:-[a-wyzA-Z]*x|-o\s+xtrace)")
ENV_DUMPS = ("env", "printenv", "export", "export -p", "set")
BUILDERS = {"test:macos": "make macos-test", "build:macos": 'make macos-dmg VERSION="$VERSION"'}


def test_macos_jobs_install_frontend_deps_before_the_bundle():
    # (f1) npm ci runs once, inside the locked body, before the bundle build.
    ci = _ci()
    for name, build in BUILDERS.items():
        cmds = _script_commands(_script_body(_job(ci, name)))
        assert cmds.count(NPM_CI) == 1, f"{name}: expected one {NPM_CI!r}, got {cmds.count(NPM_CI)}"
        assert build in cmds, f"{name}: missing {build!r}"
        assert cmds.index(NPM_CI) < cmds.index(build), f"{name}: npm ci must run before {build!r}"


def test_macos_jobs_keep_python_cache_outside_the_checkout():
    # (f2) PYTHON_CACHE is exported before the bundle build in both jobs.
    ci = _ci()
    for name, build in BUILDERS.items():
        cmds = _script_commands(_script_body(_job(ci, name)))
        assert cmds.count(PY_CACHE) == 1, f"{name}: expected one {PY_CACHE!r}"
        before = cmds.index(PY_CACHE) < cmds.index(build)
        assert before, f"{name}: PYTHON_CACHE must be set before {build!r}"


def test_macos_jobs_never_print_the_environment():
    # (f3) NPM_TOKEN is in the job env: no xtrace, no env dump, no token echo,
    # no cat of an npmrc, in either job's before_script or script.
    ci = _ci()
    for name in BUILDERS:
        job = _job(ci, name)
        lines = list(job.get("before_script") or []) + [_script_body(job)]
        cmds = [c for entry in lines for c in _script_commands(entry)]
        for c in cmds:
            assert not XTRACE.search(c), f"{name}: xtrace in {c!r}"
            assert c not in ENV_DUMPS, f"{name}: env dump {c!r}"
            assert "NPM_TOKEN" not in c, f"{name}: a command names NPM_TOKEN: {c!r}"
            assert "npmrc" not in c, f"{name}: a command touches an npmrc: {c!r}"


def test_build_macos_body_stays_one_single_quoted_string():
    # (f4) the comment lines added to build:macos carry no apostrophe.
    body = _script_body(_job(_ci(), "build:macos"))
    start = body.find("sh -c '")
    assert start != -1, "the body does not open an sh -c string"
    inner = body[start + len("sh -c '") : body.rfind("'")]
    assert "'" not in inner, "an apostrophe inside the build:macos sh -c body breaks the job"
