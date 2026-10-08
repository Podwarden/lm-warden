"""Regression net for R21 — a release tag must be tested before it is published.

`publish:images` builds and pushes the release images on a CalVer tag
pipeline, but until R21 nothing in that pipeline gated it on a test job:
every test job was gated on `$CI_PIPELINE_SOURCE == "merge_request_event"` or
`$CI_COMMIT_BRANCH`, and a tag pipeline sets neither, so the tag pipeline ran
no tests at all and a tag could ship an image whose exact commit was never
checked in its own pipeline (CI audit A23, issue #270).

The fix has two halves, and this file pins both:

  (a) `publish:images` lists the release test jobs in `needs:` as HARD needs —
      a red test job then fails the pipeline before a byte is pushed, and a
      pipeline missing a needed job fails creation rather than publishing
      anyway (an `optional: true` need is dropped when the job is absent, so
      it cannot gate anything);
  (b) each of those jobs' `rules` admit a release-tag pipeline, so the needs
      in (a) are satisfiable there in the first place.

`.gitlab-ci.yml` is parsed with PyYAML (`safe_load`, which never constructs
objects — the safe parser). PyYAML is in the deps image transitively via
`transformers`; nothing pins it directly, so this file must not grow a new
dependency to parse the file. The YAML anchors (`*needs-gate-and-deps` and
friends) resolve in the parse itself; only the two GitLab-specific behaviours
— the `extends:` merge and the `rules:` evaluation — are re-implemented here,
against the condition shapes this file actually uses (`$VAR`,
`$VAR == "literal"`, `$VAR =~ /regex/`, joined by `&&`). Any other shape makes
the test fail loudly instead of passing silently.
"""

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_FILE = REPO_ROOT / ".gitlab-ci.yml"

# A CalVer release tag, the shape of publish:images' own rule. The test jobs
# carry the same rule, so the pipeline runs the tests exactly when it would
# publish.
RELEASE_TAG = "v2026.09.28.1"

# The jobs whose green is the precondition for publishing a release.
RELEASE_TEST_JOBS = (
    "lint",
    "typecheck:mypy",
    "unit-tests",
    "integration-tests",
    "vitest:frontend",
    "conformance:chat2",
)

# A release-tag pipeline as GitLab fills it: CI_COMMIT_TAG set,
# CI_COMMIT_BRANCH UNSET (the whole point of a tag), source `push`.
TAG_ENV: dict[str, str] = {
    "CI_COMMIT_TAG": RELEASE_TAG,
    "CI_COMMIT_REF_NAME": RELEASE_TAG,
    "CI_PIPELINE_SOURCE": "push",
}


def _ci() -> dict:
    doc = yaml.safe_load(CI_FILE.read_text())
    assert isinstance(doc, dict) and doc, f"{CI_FILE.name} did not parse to a mapping"
    return doc


def _job(ci: dict, name: str) -> dict:
    job = ci.get(name)
    assert isinstance(job, dict), f"job {name!r} is missing from .gitlab-ci.yml"
    return job


def _extends_targets(job: dict) -> list[str]:
    ext = job.get("extends")
    if ext is None:
        return []
    return [ext] if isinstance(ext, str) else list(ext)


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolved(job: dict, ci: dict) -> dict:
    """The job with its `extends:` templates merged under it (child wins per
    key — the GitLab extends semantics for the keys this test reads)."""
    merged: dict = {}
    for target in _extends_targets(job):
        merged = _merge(_resolved(_job(ci, target), ci), merged)
    return _merge(merged, job)


_ATOM = re.compile(
    r"^\$(?P<var>[A-Za-z0-9_]+)" r"(?:\s*(?P<op>==|!=|=~|!~)\s*(?P<operand>\"[^\"]*\"|/[^/]*\/))?$"
)


def _eval_atom(atom: str, env: dict[str, str]) -> bool:
    m = _ATOM.match(atom.strip())
    assert m, f"unsupported rule condition in .gitlab-ci.yml: {atom!r}"
    value = env.get(m.group("var"), "")  # unset variables read as the empty string
    op = m.group("op")
    if op is None:
        return bool(value)  # bare $VAR: true when set and non-empty
    operand = m.group("operand")
    if op in ("==", "!="):
        result = value == operand[1:-1]
    else:
        result = re.search(operand[1:-1], value) is not None
    return result if op in ("==", "=~") else not result


def _eval_if(expr: str, env: dict[str, str]) -> bool:
    return all(_eval_atom(part, env) for part in re.split(r"\s*&&\s*", expr) if part)


def _first_matching_when(rules: list, env: dict[str, str]) -> str | None:
    """GitLab evaluates rules in order and the FIRST matching one wins."""
    for rule in rules:
        assert isinstance(rule, dict), f"unsupported rules entry: {rule!r}"
        if "if" in rule and not _eval_if(rule["if"], env):
            continue
        return rule.get("when", "on_success")
    return None


def _runs_automatically_on_release_tag(job: dict, ci: dict, name: str) -> bool:
    """True when a release-tag pipeline creates the job and starts it without
    a manual play (so it can actually go red and gate something)."""
    rules = _resolved(job, ci).get("rules")
    assert isinstance(rules, list) and rules, f"job {name!r} has no rules"
    when = _first_matching_when(rules, TAG_ENV)
    return when in ("on_success", "always")


def _hard_needs(ci: dict) -> set[str]:
    """publish:images' needs, keeping only the HARD ones: a string entry or a
    mapping without `optional: true`. An optional need is dropped when the
    job is absent from the pipeline, so it gates nothing."""
    needs = _job(ci, "publish:images").get("needs")
    assert isinstance(needs, list) and needs, (
        "publish:images has no `needs:` at all — a tag pipeline publishes "
        "whatever it contains, gated on nothing (R21)"
    )
    hard: set[str] = set()
    for entry in needs:
        if isinstance(entry, str):
            hard.add(entry)
        elif isinstance(entry, dict) and "job" in entry:
            if not entry.get("optional"):
                hard.add(entry["job"])
        else:
            raise AssertionError(f"unsupported needs entry on publish:images: {entry!r}")
    return hard


def test_publish_images_hard_needs_every_release_test_job():
    """(a) A red release test must stop the publish: publish:images needs all
    six test jobs, and hard."""
    hard = _hard_needs(_ci())
    missing = [name for name in RELEASE_TEST_JOBS if name not in hard]
    assert not missing, (
        "publish:images does not hard-need the release test job(s): "
        + ", ".join(missing)
        + " — a tag can then ship an image whose exact commit was never "
        "checked in its own pipeline (R21, CI audit A23, issue #270)."
    )


def test_every_release_test_job_runs_on_release_tag_pipelines():
    """(b) The needs in (a) are satisfiable only if each job's rules admit a
    tag pipeline: with CI_COMMIT_TAG set and CI_COMMIT_BRANCH unset, the
    first matching rule must start the job automatically."""
    ci = _ci()
    for name in RELEASE_TEST_JOBS:
        assert _runs_automatically_on_release_tag(_job(ci, name), ci, name), (
            f"{name} has no rule that admits a release-tag pipeline "
            f"(CI_COMMIT_TAG set, CI_COMMIT_BRANCH unset), so "
            f"publish:images' need on it can never be satisfied there."
        )


def test_publish_images_still_admits_the_release_tag():
    """The coupling the gate rests on: publish:images itself still admits the
    release tag. If its rule drifted (e.g. to a branch or a variable), the
    test jobs' release-tag rules would no longer match the pipelines that
    publish, and the gate would silently die."""
    ci = _ci()
    assert _runs_automatically_on_release_tag(
        _job(ci, "publish:images"), ci, "publish:images"
    ), "publish:images no longer runs on the release tag; the R21 gate is orphaned"
