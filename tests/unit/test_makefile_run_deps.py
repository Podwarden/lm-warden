"""Regression net for R1 — `make lint/typecheck/test*` running docker as root.

The `RUN_DEPS` variable in the Makefile drives every Python lane. Its
`docker run` carried no `--user`, so everything the tools write inside the
bind-mounted repo (`.pytest_cache/`, `.mypy_cache/`, `__pycache__/`,
coverage files) is created root-owned on the host. The invoking user cannot
delete a root-owned *directory*'s contents, so a later `git clean` fails and
the loop boundary cannot reset the tree.

CI already passes `--user "$(id -u):$(id -g)"` (.gitlab-ci.yml, the
unit-tests and integration-tests jobs); the Makefile must pass the
make-equivalent, `$(shell id -u):$(shell id -g)` — the same expression the
Makefile's own `generate-api-types` and `sync-shared-docs` targets use.
`HOME=/tmp` must stay: a non-root container user needs a writable home for
the `pip install --user` fallback and its tool caches.

File-content asserts, same shape as tests/unit/test_release_buildargs.py:
pytest is just the harness, and a red test here means the root-owned-tree
gap has reopened.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"


def _run_deps_recipe() -> str:
    """Return the Makefile's RUN_DEPS variable joined into one logical line.

    RUN_DEPS is a multi-line variable: each line but the last ends in a
    backslash. Stop at the first line that does not, so a recipe change that
    adds a line is still captured.
    """
    lines = MAKEFILE.read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("RUN_DEPS")]
    assert len(starts) == 1, (
        f"expected exactly one RUN_DEPS definition in {MAKEFILE.name}, " f"found {len(starts)}"
    )
    block = [lines[starts[0]]]
    for line in lines[starts[0] + 1 :]:
        if block[-1].rstrip().endswith("\\"):
            block.append(line)
        else:
            break
    joined = []
    for line in block:
        stripped = line.rstrip()
        joined.append(stripped[:-1] if stripped.endswith("\\") else stripped)
    return " ".join(part.strip() for part in joined)


def _docker_run_part(recipe: str) -> str:
    """Everything from `docker run` to the end of the recipe (the flags and
    the single-quoted payload)."""
    assert "docker run" in recipe, (
        f"RUN_DEPS no longer contains a `docker run` invocation; this test "
        f"parses that line:\n{recipe}"
    )
    return recipe[recipe.index("docker run") :]


def test_run_deps_docker_run_passes_invoking_user():
    """`docker run` must carry `--user $(shell id -u):$(shell id -g)` so files
    written into the bind-mounted repo belong to the invoking user."""
    run = _docker_run_part(_run_deps_recipe())
    assert "--user $(shell id -u):$(shell id -g)" in run, (
        "RUN_DEPS' docker run has no `--user $(shell id -u):$(shell id -g)`, "
        "so the container runs as root and everything it writes into the "
        "bind-mounted repo (`.pytest_cache/`, `.mypy_cache/`, `__pycache__/`) "
        "is root-owned on the host; the invoking user cannot delete "
        "root-owned directories, and a later `git clean` fails. CI already "
        "passes --user (see .gitlab-ci.yml)."
    )


def test_run_deps_user_flag_is_a_docker_flag_not_payload_text():
    """The `--user` flag must sit on the `docker run` invocation, before the
    `sh -c` payload — a `--user` string inside the payload would satisfy the
    previous test while changing nothing."""
    run = _docker_run_part(_run_deps_recipe())
    assert "--user" in run and "sh -c" in run
    assert run.index("--user") < run.index("sh -c"), (
        "`--user` appears only inside the sh -c payload of RUN_DEPS' docker "
        "run, where it is inert text, not a flag on the invocation."
    )


def test_run_deps_keeps_writable_home():
    """`HOME=/tmp` must stay: a non-root container user needs a writable home
    (the `pip install --user` fallback and its tool caches)."""
    run = _docker_run_part(_run_deps_recipe())
    assert "-e HOME=/tmp" in run, (
        "RUN_DEPS' docker run no longer passes -e HOME=/tmp. Running as the "
        "invoking uid, the container user has no home it can write to "
        "otherwise, and the pip-install fallback breaks."
    )
