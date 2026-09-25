"""Regression net for vllm-warden#45 — `vdev · unknown` banner.

The runtime version surfacing is already covered by
`tests/unit/system/test_routes_version.py`. This file guards the *release
procedure* side — where the build-time identity flows in:

  1. `docs/releasing.md` — the manual `docker buildx --push` recipe must
     pass `--build-arg VW_BUILD_VERSION=…` so the backend image bakes the
     real tag instead of the Dockerfile default `dev`. Both the backend and
     UI commands carry the arg (the UI Dockerfile currently ignores it but
     the doc keeps the convention symmetric and forward-compatible).

(The old `deploy/hub/compose.yaml` check is gone with that stale mirror of the
catalogue row, removed 2026-09-24; CI's publish:images bakes both build args
into every release image.)

These are file-content asserts, not behavioural tests — pytest is just a
convenient harness because vllm-warden already runs unit tests via pytest
on every push. Failing tests here mean the gap that produced the
v2026.05.17.1 `vdev · unknown` banner has reopened.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASING_MD = REPO_ROOT / "docs" / "releasing.md"


def test_releasing_doc_passes_vw_build_version_buildarg_twice():
    """Both `docker buildx build --push` commands (backend + UI) must pass
    `--build-arg VW_BUILD_VERSION=…`. Two occurrences = both commands covered."""
    text = RELEASING_MD.read_text()
    occurrences = text.count("--build-arg VW_BUILD_VERSION=")
    assert occurrences >= 2, (
        f"Expected --build-arg VW_BUILD_VERSION= at least twice in "
        f"{RELEASING_MD.relative_to(REPO_ROOT)} (once per buildx command), "
        f"found {occurrences}. Regressing #45 ships images with the "
        f"Dockerfile default `dev` baked in — the version banner reads "
        f"`vdev · unknown` instead of the release tag."
    )


def test_releasing_doc_passes_vw_build_sha_buildarg_twice():
    """Companion to VW_BUILD_VERSION — both buildx commands must also bake the
    commit SHA so the banner's second segment isn't `unknown`."""
    text = RELEASING_MD.read_text()
    occurrences = text.count("--build-arg VW_BUILD_SHA=")
    assert occurrences >= 2, (
        f"Expected --build-arg VW_BUILD_SHA= at least twice in "
        f"{RELEASING_MD.relative_to(REPO_ROOT)} (once per buildx command), "
        f"found {occurrences}."
    )
