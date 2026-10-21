"""``publish/github-beta-release.sh`` against a fake GitHub API.

The script publishes a sealed beta DMG (and its .sha256) as a GitHub
PRE-RELEASE of the public repo, tagged ``<tag>-macos-beta`` on the commit the
public ``<tag>`` points at. It never creates a normal release, never touches
the stable release of ``<tag>``, never marks the beta "latest", never prints
the token, and refuses a DMG whose checksum file does not match.

Not marked ``macos``: bash + curl + jq + a stdlib HTTP server. Skipped where
curl or jq is missing (the publish job installs both in its container).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not (shutil.which("curl") and shutil.which("jq") and shutil.which("bash")),
    reason="needs bash, curl and jq",
)

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "publish" / "github-beta-release.sh"

TAG = "v2026.10.21.1"
BETA_TAG = f"{TAG}-macos-beta"
SLUG = "Podwarden/lm-warden"
PUBLIC_SHA = "c0ffee" * 6 + "abcd"
TOKEN = "ghp_test_token_never_printed_0123456789"


class FakeGitHub:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.public_tags: dict[str, str] = {TAG: PUBLIC_SHA}
        self.releases: dict[str, dict] = {}
        self._next_id = 100
        # GitHub's answer to a create; False simulates a create that did not
        # come back as a pre-release.
        self.create_as_prerelease = True
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a: object) -> None:
                pass

            def _send(self, code: int, body: object) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _record(self, body: bytes) -> None:
                fake.requests.append(
                    {
                        "method": self.command,
                        "path": self.path,
                        "auth": self.headers.get("Authorization"),
                        "ctype": self.headers.get("Content-Type"),
                        "body": body,
                    }
                )

            def do_GET(self) -> None:
                self._record(b"")
                base = f"/repos/{SLUG}"
                if self.path.startswith(f"{base}/commits/"):
                    ref = self.path.rsplit("/", 1)[1]
                    if ref in fake.public_tags:
                        return self._send(200, {"sha": fake.public_tags[ref]})
                    return self._send(422, {"message": "No commit found for SHA: " + ref})
                if self.path.startswith(f"{base}/releases/tags/"):
                    tag = self.path.rsplit("/", 1)[1]
                    if tag in fake.releases:
                        return self._send(200, fake.releases[tag])
                    return self._send(404, {"message": "Not Found"})
                return self._send(404, {"message": "Not Found"})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                self._record(body)
                base = f"/repos/{SLUG}"
                if self.path == f"{base}/releases":
                    payload = json.loads(body)
                    rid = fake._next_id
                    fake._next_id += 1
                    port = self.server.server_address[1]
                    rel = {
                        "id": rid,
                        "tag_name": payload["tag_name"],
                        "prerelease": payload["prerelease"] and fake.create_as_prerelease,
                        "html_url": f"https://github.com/{SLUG}/releases/tag/{payload['tag_name']}",
                        "upload_url": f"http://127.0.0.1:{port}/upload/{rid}/assets{{?name,label}}",
                        "assets": [],
                        "payload": payload,
                    }
                    fake.releases[payload["tag_name"]] = rel
                    return self._send(201, rel)
                if self.path.startswith("/upload/"):
                    rid = int(self.path.split("/")[2])
                    name = self.path.split("name=", 1)[1]
                    for rel in fake.releases.values():
                        if rel["id"] == rid:
                            rel["assets"].append(
                                {
                                    "name": name,
                                    "size": len(body),
                                    "sha256": hashlib.sha256(body).hexdigest(),
                                }
                            )
                            return self._send(201, {"name": name})
                    return self._send(404, {"message": "Not Found"})
                return self._send(404, {"message": "Not Found"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def posts(self) -> list[dict]:
        return [r for r in self.requests if r["method"] == "POST"]


@pytest.fixture
def gh():
    fake = FakeGitHub()
    yield fake
    fake.close()


def _artifacts(tmp_path: Path, *, good_sum: bool = True, name: str | None = None) -> Path:
    dist = tmp_path / "dist"
    dist.mkdir()
    dmg = dist / (name or f"LM-Warden-{TAG}-beta.dmg")
    dmg.write_bytes(b"not really a dmg\n" * 1000)
    digest = hashlib.sha256(dmg.read_bytes()).hexdigest() if good_sum else "0" * 64
    Path(f"{dmg}.sha256").write_text(f"{digest}  {dmg.name}\n")
    return dmg


def _run(gh: FakeGitHub, tag: str, dmg: Path, *, token: str | None = TOKEN, dry: bool = False):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GITHUB_")}
    env.pop("PUBLISH_DRY_RUN", None)
    env["GITHUB_API_URL"] = gh.url
    if token is not None:
        env["GITHUB_PUBLISH_TOKEN"] = token
    if dry:
        env["PUBLISH_DRY_RUN"] = "1"
    proc = subprocess.run(
        ["bash", str(SCRIPT), tag, str(dmg)], capture_output=True, text=True, env=env
    )
    assert TOKEN not in proc.stdout + proc.stderr, "the token was printed"
    return proc


def test_publishes_a_prerelease_with_the_dmg_and_its_checksum(
    gh: FakeGitHub, tmp_path: Path
) -> None:
    dmg = _artifacts(tmp_path)
    proc = _run(gh, TAG, dmg)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rel = gh.releases[BETA_TAG]
    p = rel["payload"]
    assert p["prerelease"] is True
    assert p["draft"] is False
    assert p["make_latest"] == "false"
    assert p["tag_name"] == BETA_TAG
    assert p["target_commitish"] == PUBLIC_SHA
    assert "beta" in p["name"].lower()
    body = p["body"]
    for needle in ("not notarized", "com.apple.quarantine", "Open", "shasum -a 256 -c", "update"):
        assert needle in body, f"release notes lack {needle!r}"
    names = sorted(a["name"] for a in rel["assets"])
    assert names == sorted([dmg.name, f"{dmg.name}.sha256"])
    dmg_asset = next(a for a in rel["assets"] if a["name"] == dmg.name)
    assert dmg_asset["sha256"] == hashlib.sha256(dmg.read_bytes()).hexdigest()
    # Every call is authenticated; the stable release tag is never created.
    assert all(r["auth"] == f"Bearer {TOKEN}" for r in gh.requests)
    assert TAG not in gh.releases
    assert rel["html_url"] in proc.stdout


def test_rerun_is_idempotent(gh: FakeGitHub, tmp_path: Path) -> None:
    dmg = _artifacts(tmp_path)
    assert _run(gh, TAG, dmg).returncode == 0
    before = len(gh.posts())
    proc = _run(gh, TAG, dmg)
    assert proc.returncode == 0, proc.stderr
    assert len(gh.posts()) == before, "a re-run created or uploaded again"


def test_rerun_uploads_only_a_missing_asset(gh: FakeGitHub, tmp_path: Path) -> None:
    dmg = _artifacts(tmp_path)
    assert _run(gh, TAG, dmg).returncode == 0
    rel = gh.releases[BETA_TAG]
    rel["assets"] = [a for a in rel["assets"] if a["name"] == dmg.name]
    proc = _run(gh, TAG, dmg)
    assert proc.returncode == 0, proc.stderr
    assert sorted(a["name"] for a in rel["assets"]) == sorted([dmg.name, f"{dmg.name}.sha256"])
    assert [a["name"] for a in rel["assets"]].count(dmg.name) == 1


def test_refuses_to_touch_a_non_prerelease(gh: FakeGitHub, tmp_path: Path) -> None:
    # A real upload_url: without the refusal the script WOULD upload to it.
    gh.releases[BETA_TAG] = {
        "id": 7,
        "tag_name": BETA_TAG,
        "prerelease": False,
        "assets": [],
        "upload_url": f"{gh.url}/upload/7/assets{{?name,label}}",
    }
    proc = _run(gh, TAG, _artifacts(tmp_path))
    assert proc.returncode == 1
    assert "not a pre-release" in proc.stderr
    assert gh.posts() == []
    assert gh.releases[BETA_TAG]["assets"] == []


def test_stops_when_github_did_not_create_a_prerelease(gh: FakeGitHub, tmp_path: Path) -> None:
    gh.create_as_prerelease = False
    proc = _run(gh, TAG, _artifacts(tmp_path))
    assert proc.returncode == 1
    assert "not as a pre-release" in proc.stderr
    assert gh.releases[BETA_TAG]["assets"] == [], "uploaded to a release that is not a pre-release"


@pytest.mark.parametrize("via", ["bash -x", "SHELLOPTS"])
@pytest.mark.parametrize("source", ["env", "file"])
def test_token_is_not_printed_under_xtrace(
    gh: FakeGitHub, tmp_path: Path, via: str, source: str
) -> None:
    dmg = _artifacts(tmp_path)
    env = {k: v for k, v in os.environ.items() if not k.startswith("GITHUB_")}
    env.pop("PUBLISH_DRY_RUN", None)
    env["GITHUB_API_URL"] = gh.url
    if source == "env":
        env["GITHUB_PUBLISH_TOKEN"] = TOKEN
    else:
        tok = tmp_path / "token"
        tok.write_text(TOKEN)
        env["GITHUB_PUBLISH_TOKEN_FILE"] = str(tok)
    cmd = ["bash", str(SCRIPT), TAG, str(dmg)]
    if via == "bash -x":
        cmd.insert(1, "-x")
    else:
        env["SHELLOPTS"] = "xtrace"
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert TOKEN not in proc.stdout + proc.stderr, "the token was printed under xtrace"
    assert all(r["auth"] == f"Bearer {TOKEN}" for r in gh.requests)


def test_refuses_a_checksum_mismatch_before_any_call(gh: FakeGitHub, tmp_path: Path) -> None:
    proc = _run(gh, TAG, _artifacts(tmp_path, good_sum=False))
    assert proc.returncode == 1
    assert gh.requests == []


def test_refuses_a_missing_checksum(gh: FakeGitHub, tmp_path: Path) -> None:
    dmg = _artifacts(tmp_path)
    Path(f"{dmg}.sha256").unlink()
    proc = _run(gh, TAG, dmg)
    assert proc.returncode == 1
    assert gh.requests == []


@pytest.mark.parametrize("tag", ["dev", "v2026.10.21", "2026.10.21.1", "v2026.10.21.1-rc1"])
def test_refuses_a_non_release_tag(gh: FakeGitHub, tmp_path: Path, tag: str) -> None:
    proc = _run(gh, tag, _artifacts(tmp_path))
    assert proc.returncode == 2
    assert gh.requests == []


def test_refuses_a_dmg_not_named_as_the_beta_of_the_tag(gh: FakeGitHub, tmp_path: Path) -> None:
    proc = _run(gh, TAG, _artifacts(tmp_path, name=f"LM-Warden-{TAG}.dmg"))
    assert proc.returncode == 2
    assert gh.requests == []


def test_refuses_when_the_tag_is_not_public_yet(gh: FakeGitHub, tmp_path: Path) -> None:
    gh.public_tags.clear()
    proc = _run(gh, TAG, _artifacts(tmp_path))
    assert proc.returncode == 1
    assert "publish:github" in proc.stderr
    assert gh.posts() == []


def test_needs_a_token_to_publish(gh: FakeGitHub, tmp_path: Path) -> None:
    proc = _run(gh, TAG, _artifacts(tmp_path), token=None)
    assert proc.returncode == 1
    assert gh.requests == []


def test_dry_run_writes_nothing(gh: FakeGitHub, tmp_path: Path) -> None:
    proc = _run(gh, TAG, _artifacts(tmp_path), token=None, dry=True)
    assert proc.returncode == 0, proc.stderr
    assert gh.posts() == []
    assert "DRY RUN" in proc.stdout
