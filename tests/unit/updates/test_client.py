"""The daily update check (spec 2026-11-20): what it sends, when it is off,
and that a bad answer never costs the install anything."""

import asyncio
import json
import shutil
import sqlite3
from pathlib import Path

import httpx
import pytest

from app.updates import client as uc

GOOD = {
    "latest": {"version": "v2026.12.01.1", "released_at": "2026-12-01", "url": "https://x/notes"},
    "advisories": [],
}


@pytest.fixture
def db_path(tmp_data_dir: Path, migrated_db_template: Path) -> Path:
    p = tmp_data_dir / "vllm-warden.db"
    shutil.copyfile(migrated_db_template, p)
    return p


def _setting(db_path: Path, key: str) -> str | None:
    with sqlite3.connect(db_path) as db:
        row = db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _transport(seen: list[httpx.Request], *, status: int = 200, body: object = GOOD):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if isinstance(body, Exception):
            raise body
        content = body if isinstance(body, bytes) else json.dumps(body).encode()
        return httpx.Response(status, content=content)

    return httpx.MockTransport(handler)


def test_check_sends_exactly_four_fields_and_stores_the_answer(db_path: Path) -> None:
    seen: list[httpx.Request] = []
    out = asyncio.run(
        uc.check_once(db_path, "community", "v2026.11.20.1", transport=_transport(seen))
    )
    assert out == GOOD
    (req,) = seen
    assert req.method == "POST"
    assert str(req.url) == uc.DEFAULT_URL
    assert req.headers["user-agent"] == "lm-warden/v2026.11.20.1"
    sent = json.loads(req.content)
    assert set(sent) == {"install_id", "version", "edition", "platform"}
    assert sent["version"] == "v2026.11.20.1"
    assert sent["edition"] == "community"
    assert sent["platform"] in ("linux", "macos")
    assert sent["install_id"] == _setting(db_path, uc.INSTALL_ID_KEY)
    assert json.loads(_setting(db_path, uc.RESULT_KEY) or "") == GOOD
    assert _setting(db_path, uc.CHECKED_KEY)


def test_install_id_is_stable_across_checks(db_path: Path) -> None:
    seen: list[httpx.Request] = []
    for _ in range(2):
        asyncio.run(
            uc.check_once(db_path, "community", "v2026.11.20.1", transport=_transport(seen))
        )
    ids = {json.loads(r.content)["install_id"] for r in seen}
    assert len(ids) == 1


@pytest.mark.parametrize(
    "kw",
    [
        {"status": 500},
        {"body": b"<html>nope</html>"},
        {"body": {"latest": "not-an-object"}},
        {"body": httpx.ConnectTimeout("slow")},
        {"body": [1, 2]},
    ],
)
def test_a_bad_answer_keeps_the_previous_result(db_path: Path, kw: dict) -> None:
    asyncio.run(uc.check_once(db_path, "community", "v2026.11.20.1", transport=_transport([])))
    out = asyncio.run(
        uc.check_once(db_path, "community", "v2026.11.20.1", transport=_transport([], **kw))
    )
    assert out is None
    assert json.loads(_setting(db_path, uc.RESULT_KEY) or "") == GOOD


def test_a_dev_build_never_checks(db_path: Path) -> None:
    seen: list[httpx.Request] = []
    assert (
        asyncio.run(uc.check_once(db_path, "community", "dev", transport=_transport(seen))) is None
    )
    assert seen == []
    assert _setting(db_path, uc.INSTALL_ID_KEY) is None


def test_url_override(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VW_UPDATE_CHECK_URL", "http://127.0.0.1:9/api/update-check")
    seen: list[httpx.Request] = []
    asyncio.run(uc.check_once(db_path, "community", "v2026.11.20.1", transport=_transport(seen)))
    assert str(seen[0].url) == "http://127.0.0.1:9/api/update-check"


@pytest.mark.parametrize(
    ("env", "edition", "row", "expected"),
    [
        ({}, "community", None, True),
        ({}, "enterprise", None, True),
        ({"VW_UPDATE_CHECK": "0"}, "community", None, False),
        ({"VW_UPDATE_CHECK": "false"}, "community", None, False),
        ({"HF_HUB_OFFLINE": "1"}, "community", None, False),
        ({}, "vendor", None, False),
        ({}, "community", "false", False),
        ({}, "community", "true", True),
        ({"VW_UPDATE_CHECK": "0"}, "community", "true", False),
    ],
)
def test_enabled(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env: dict,
    edition: str,
    row: str | None,
    expected: bool,
) -> None:
    monkeypatch.delenv("VW_UPDATE_CHECK", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    if row is not None:
        with sqlite3.connect(db_path) as db:
            db.execute("INSERT INTO settings(key, value) VALUES (?, ?)", (uc.ENABLED_KEY, row))
    assert asyncio.run(uc.enabled(db_path, edition)) is expected
