"""Regression net for #233 — the app must not boot with a well-known cookie
secret.

PodWarden's compose interpolation turns ``${VW_COOKIE_SECRET:?<message>}``
into the literal ``<message>`` when the variable is unset. That message is
~75 printable characters, passed the old length-only guard, and the app then
signed every session cookie with a constant published in the catalogue.

The guard in ``load_settings`` must therefore refuse anything that could be
a rendered message: in addition to the existing empty/too-short rule, any
value containing whitespace. The ``:?`` form itself may only live in files
generated at deploy time (install.sh); no compose/yaml file in the repo may
carry it.
"""

import base64
import os
import re
from pathlib import Path

import pytest

from app.config import load_settings

REPO_ROOT = Path(__file__).resolve().parents[2]

# The exact literal PodWarden's compose interpolation produced for
# ${VW_COOKIE_SECRET:?VW_COOKIE_SECRET required (>=32 chars); generate with:
# openssl rand -hex 24} (GitLab #233): ~75 characters, all printable, and
# long enough to pass the old len >= 32 check.
RENDERED_PLACEHOLDER = "VW_COOKIE_SECRET required (>=32 chars); generate with: openssl rand -hex 24"


@pytest.fixture
def _base_env(monkeypatch):
    # Minimum env for load_settings() to succeed, minus the secret itself:
    # every test sets VW_COOKIE_SECRET explicitly.
    monkeypatch.setenv("VW_CONTAINER_GPU_COUNT", "0")
    return monkeypatch


def test_rendered_compose_placeholder_is_refused(_base_env):
    """The literal rendered message from #233 is public catalogue text, not
    a secret: the app must refuse to boot with it."""
    _base_env.setenv("VW_COOKIE_SECRET", RENDERED_PLACEHOLDER)
    with pytest.raises(RuntimeError) as exc:
        load_settings()
    assert "VW_COOKIE_SECRET" in str(exc.value)
    assert "openssl rand -hex 24" in str(exc.value)


@pytest.mark.parametrize(
    "bad",
    [
        "a" * 16 + " " + "b" * 16,
        "a" * 31 + "\t",
        "a" * 10 + "\n" + "b" * 20,
        "  " + "a" * 40,
    ],
    ids=["middle-space", "trailing-tab", "embedded-newline", "leading-spaces"],
)
def test_any_whitespace_in_secret_is_refused(_base_env, bad):
    """A secret with whitespace in it cannot be a secret: every rendered
    ${VAR:?message} carries spaces, so whitespace is the discriminator."""
    _base_env.setenv("VW_COOKIE_SECRET", bad)
    with pytest.raises(RuntimeError) as exc:
        load_settings()
    assert "VW_COOKIE_SECRET" in str(exc.value)
    assert "openssl rand -hex 24" in str(exc.value)


@pytest.mark.parametrize(
    "good",
    [
        "0123456789abcdef" * 3,
        "conformance-secret-32-bytes-min-pad!",
        base64.b64encode(b"cookie-secret-bytes-32-long!!").decode(),
    ],
    ids=["hex-48", "conformance-fixture", "base64"],
)
def test_real_secrets_still_boot(_base_env, good):
    """Controls: the values every supported flow actually produces must keep
    booting."""
    _base_env.setenv("VW_COOKIE_SECRET", good)
    assert load_settings().cookie_secret == good


def test_unset_and_short_secret_keep_existing_error(_base_env):
    """The pre-existing guard (unset or <32 chars) is unchanged: same
    exception, same message the docs quote."""
    _base_env.delenv("VW_COOKIE_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="VW_COOKIE_SECRET must be set and >=32 chars"):
        load_settings()
    _base_env.setenv("VW_COOKIE_SECRET", "too-short")
    with pytest.raises(RuntimeError, match="VW_COOKIE_SECRET must be set and >=32 chars"):
        load_settings()


_FORBIDDEN_INTERP = re.compile(r"\$\{[^}]*:\?[^}]*\}")


# Working-tree directories that are not part of the product: version
# control internals, dependency checkouts, virtualenvs and the caches the
# docker lanes leave in the bind mount on every run (R1).
_PRUNED_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".next",
    ".coverage",
    "htmlcov",
    "build",
    "dist",
}


def _candidate_files() -> list[str]:
    """Every ``*.yml`` / ``*.yaml`` / ``*compose*`` file in the working tree.

    The deps image (python:3.11-slim) has no git binary, so `git ls-files`
    is not an option inside the lane. The original workaround parsed
    ``.git/index`` by hand, but in a linked worktree ``.git`` is a pointer
    to the main checkout's ``.git/worktrees/<name>`` — a host path outside
    the bind mount — so the read raised FileNotFoundError there (R6).
    Walking the working tree works in every checkout and also catches
    untracked compose files before they are committed.
    """
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in _PRUNED_DIRS]
        for name in filenames:
            if name.endswith((".yml", ".yaml")) or "compose" in name:
                rel = os.path.relpath(os.path.join(dirpath, name), REPO_ROOT)
                found.append(rel.replace(os.sep, "/"))
    return sorted(found)


def test_no_compose_file_uses_colon_question_interpolation():
    """No *.yml / *.yaml / compose* file in the working tree may use
    `${...:?...}`.

    The `:?` form is only safe against an interpolation engine that errors
    on an unset variable; an engine that renders the message instead hands
    the app a well-known cookie secret (#233). install.sh may generate such
    a guard into a deploy-time file — the operator's `docker compose config`
    then fails loudly — but a file in the repo must never carry one.
    """
    files = _candidate_files()
    assert "docker-compose.yml" in files and ".gitlab-ci.yml" in files, (
        "expected the repo's own compose/CI files among the working-tree "
        f"*.yml/*.yaml/compose files; got: {sorted(files)}"
    )
    offenders = [
        name
        for name in files
        if _FORBIDDEN_INTERP.search((REPO_ROOT / name).read_text(errors="replace"))
    ]
    assert not offenders, (
        "compose files use ${...:?...} interpolation, which renders "
        "the message as the value when the variable is unset and boots the "
        "app with a well-known cookie secret (#233): " + ", ".join(offenders)
    )
