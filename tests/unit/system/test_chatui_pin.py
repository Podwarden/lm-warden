"""The chat-ui source pin in frontend/Dockerfile cannot drift from package.json.

frontend/Dockerfile compiles ``@podwarden/chat-ui`` from its public GitHub
source and overlays the result on what ``npm ci`` installed from the lockfile.
That only makes sense while the two describe the SAME release: the lockfile is
what pins chat-ui's ~20 runtime dependencies, and installing 0.1.23's
dependency closure under 0.1.24's code is exactly the silent breakage this
guards. The Dockerfile asserts it at build time too, but a build takes minutes
and only CI runs it; this catches the drift in the unit suite instead.

Same discipline, and the same reasoning, as
test_llamacpp_version.py::test_the_dockerfile_pins_the_same_tag_the_fixtures_were_captured_from:
the pin lives in exactly one greppable place, and a test fails when a second
place disagrees with it.

A Python test for a Dockerfile ARG is deliberate. The frontend vitest suite
needs ``npm ci`` before it can run; the unit suite needs nothing, so this
survives the case where the frontend toolchain is what is broken.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DOCKERFILE = REPO / "frontend" / "Dockerfile"
PACKAGE_JSON = REPO / "frontend" / "package.json"


def _arg(name: str) -> str:
    m = re.search(rf"^ARG {name}=(\S+)", DOCKERFILE.read_text(), re.M)
    assert m, f"frontend/Dockerfile must declare ARG {name} as a greppable line"
    return m.group(1)


def test_the_pinned_version_is_the_version_package_json_depends_on():
    declared = json.loads(PACKAGE_JSON.read_text())["dependencies"]["@podwarden/chat-ui"]
    assert _arg("CHATUI_VERSION") == declared, (
        f"frontend/Dockerfile pins CHATUI_VERSION={_arg('CHATUI_VERSION')} but "
        f"frontend/package.json depends on @podwarden/chat-ui@{declared}. Bump "
        f"both, and point CHATUI_REF at the release commit carrying that version."
    )


def test_the_ref_is_a_full_commit_sha():
    """Not a branch and not a tag.

    The public mirror receives one squashed "Release vX" commit per release and
    carries NO tags at all, so a SHA is the only pin available there. It is also
    the only pin that cannot be moved after the fact, which is the property that
    matters for a build input.
    """
    ref = _arg("CHATUI_REF")
    assert re.fullmatch(r"[0-9a-f]{40}", ref), (
        f"CHATUI_REF={ref!r} is not a full 40-character commit SHA. A branch or "
        f"a short SHA makes the image build non-reproducible."
    )


def test_the_source_repo_is_public():
    """A private clone URL here would make the published tree unbuildable.

    The whole point of this stage is that an outsider with nothing but Docker
    can build the image. Pointing CHATUI_REPO at an authenticated host would
    leave the public snapshot with a build that only insiders can run — and the
    failure would appear at `docker build`, long after the change was merged.
    """
    repo = _arg("CHATUI_REPO")
    assert repo == "https://github.com/Podwarden/chat-ui.git", (
        f"CHATUI_REPO={repo!r} is not the public chat-ui mirror. The default "
        f"must stay credential-free; build a fork with --build-arg instead."
    )
