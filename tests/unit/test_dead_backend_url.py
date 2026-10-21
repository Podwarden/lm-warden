"""The dead BACKEND_URL is gone from the shipped container definitions.

#155 unified-port deleted the ``rewrites()`` branches that read BACKEND_URL
(Caddy now proxies /api+ and /v1 directly to FastAPI), so nothing in
next.config.ts or frontend/src consumes it any more. The ``ENV`` line left in
frontend/Dockerfile and the key left in docker-compose.yml's ui service were
a dead trap: they look load-bearing and rot the moment the service name
changes. This test pins their absence structurally:

- the Dockerfile is scanned line by line, with comments stripped, so a
  resurrected ``ENV BACKEND_URL=...`` or ``ARG BACKEND_URL`` is caught but a
  prose mention is not;
- docker-compose.yml is parsed with ``yaml.safe_load`` and asserted on the
  ``ui`` service's environment mapping, not grepped;
- frontend/src and next.config.ts are scanned for any remaining reader of
  ``process.env.BACKEND_URL`` (frontend/tests is excluded on purpose:
  next-config-rewrites.test.ts sets the env var deliberately to prove the
  absence of those rewrites);
- when docker is available, ``docker compose config -q`` still parses the
  edited file (skipped when docker is absent — the yaml.safe_load leg above
  always runs).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "frontend" / "Dockerfile"
COMPOSE = ROOT / "docker-compose.yml"
NEXT_CONFIG = ROOT / "frontend" / "next.config.ts"
FRONTEND_SRC = ROOT / "frontend" / "src"


def _non_comment_lines(path: Path) -> list[str]:
    return [
        line
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_dockerfile_sets_no_backend_url() -> None:
    # Comments may still mention the history; the instruction lines must not.
    bad = [line for line in _non_comment_lines(DOCKERFILE) if "BACKEND_URL" in line]
    assert bad == [], f"frontend/Dockerfile still sets BACKEND_URL: {bad}"


def test_dockerfile_has_no_stale_backend_url_comment() -> None:
    # The old block justified the ENV line by claiming next.config.ts:rewrites()
    # reads BACKEND_URL; that is no longer true, so the comment is dead weight.
    text = DOCKERFILE.read_text()
    assert "rewrites() reads BACKEND_URL" not in text


def test_compose_ui_service_has_no_backend_url() -> None:
    doc = yaml.safe_load(COMPOSE.read_text())
    env = doc["services"]["ui"].get("environment", {})
    # compose accepts mapping or list-of-"KEY=value" forms; neither may carry it.
    if isinstance(env, list):
        keys = [e.split("=", 1)[0] for e in env]
    else:
        keys = list(env)
    assert "BACKEND_URL" not in keys, f"ui service environment: {keys}"


def test_no_frontend_source_reads_backend_url() -> None:
    files = [NEXT_CONFIG, *sorted(FRONTEND_SRC.rglob("*"))]
    readers = [
        p.relative_to(ROOT).as_posix()
        for p in files
        if p.is_file() and "process.env.BACKEND_URL" in p.read_text(errors="ignore")
    ]
    assert readers == [], f"still reads process.env.BACKEND_URL: {readers}"


def test_compose_still_parses_if_docker_available() -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker not available")
    out = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE), "config", "-q"],
        cwd=ROOT,
        capture_output=True,
    )
    assert out.returncode == 0, out.stderr.decode(errors="replace")
