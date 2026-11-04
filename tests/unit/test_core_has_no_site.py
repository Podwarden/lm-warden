"""Core does not contain the website (spec 2026-11-04, site split).

The website is the `site/` package, installed next to core
only in the vendor build and registered through the `lm_warden.extensions`
entry point. Core never names it, never imports it, and does not serve its
routes on its own.
"""

import importlib.util
import re
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
# Fragment-assembled (the house idiom, see test_publish_mac_patterns.py):
# this file ships in the public tree, whose block list names the package.
_NAMES = re.compile(r"app\.landing|lmwarden" + "_site")
_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".json", ".html", ".md", ".txt", ".sql"}


def _hits(root: Path) -> list[str]:
    out = []
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.suffix not in _SUFFIXES or "node_modules" in f.parts:
            continue
        for n, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if _NAMES.search(line):
                out.append(f"{f.relative_to(REPO_ROOT)}:{n}: {line.strip()}")
    return out


def test_core_never_names_the_website() -> None:
    hits = _hits(REPO_ROOT / "app") + _hits(REPO_ROOT / "frontend" / "src")
    assert hits == [], "\n".join(hits)


def test_app_landing_is_gone() -> None:
    assert importlib.util.find_spec("app.landing") is None
    assert not (REPO_ROOT / "app" / "landing").exists()


def test_core_alone_serves_none_of_the_websites_routes(
    ext_registry: object, client: TestClient
) -> None:
    for path in ("/download/mac", "/sitemap.xml", "/llms.txt", "/faq", "/coding-agents/aider"):
        r = client.get(path, follow_redirects=False)
        assert r.status_code == 404, (path, r.status_code)
        assert r.headers["content-type"].startswith("application/json"), path
