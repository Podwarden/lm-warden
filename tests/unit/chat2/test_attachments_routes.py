import io
import sqlite3
from types import SimpleNamespace

from PIL import Image

from app.chat2 import quota
from app.chat2 import routes_attachments as routes_attachments_module
from tests.conftest import csrf_header, jwt_login, seed_admin_user


def _png_bytes(w: int = 2, h: int = 2) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (0, 0, 255)).save(buf, format="PNG")
    return buf.getvalue()


def _cmyk_jpeg_bytes(w: int = 2, h: int = 2) -> bytes:
    buf = io.BytesIO()
    Image.new("CMYK", (w, h), (0, 0, 0, 0)).save(buf, format="JPEG")
    return buf.getvalue()


def _setup(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    headers = {**jwt_login(client), **csrf_header(client)}
    r = client.post("/api/chat2/chats", headers=headers, json={})
    assert r.status_code == 201, r.text
    return headers, r.json()["id"]


def test_upload_serve_delete_draft(tmp_data_dir, client) -> None:
    headers, chat_id = _setup(tmp_data_dir, client)
    r = client.post(
        "/api/chat2/attachments",
        headers=headers,
        files={"file": ("a.png", _png_bytes(), "image/png")},
        data={"chat_id": chat_id},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["width"] == 2 and body["url"].startswith("/api/chat2/attachments/")
    # served with signed url, no auth header
    g = client.get(body["url"])
    assert g.status_code == 200 and g.headers["content-type"] == "image/png"
    assert g.headers["x-content-type-options"] == "nosniff"
    # wrong token
    assert client.get(body["url"][:-3] + "xxx").status_code == 401
    # file exists under sha256 path
    files = list((tmp_data_dir / "chat2").rglob("*.png"))
    assert len(files) == 1 and files[0].stem == body["sha256"]
    # delete draft removes row and file
    d = client.delete(f"/api/chat2/attachments/{body['id']}", headers=headers)
    assert d.status_code == 204
    assert not files[0].exists()


def test_upload_rejects_type_size_and_quota(tmp_data_dir, client, monkeypatch) -> None:
    """Every rejection carries the shared chat2 `{code, message}` envelope."""
    headers, chat_id = _setup(tmp_data_dir, client)
    r = client.post(
        "/api/chat2/attachments",
        headers=headers,
        files={"file": ("a.svg", b"<svg/>", "image/svg+xml")},
        data={"chat_id": chat_id},
    )
    assert r.status_code == 415 and r.json()["detail"]["code"] == "unsupported_media"
    big = b"\x89PNG" + b"0" * (10 * 1024**2 + 1)
    r = client.post(
        "/api/chat2/attachments",
        headers=headers,
        files={"file": ("a.png", big, "image/png")},
        data={"chat_id": chat_id},
    )
    assert r.status_code == 413 and r.json()["detail"]["code"] == "too_large"
    # quota: set user quota to 1 byte. Settings is a frozen dataclass, so a
    # plain setattr()/monkeypatch.setattr() raises FrozenInstanceError; go
    # through the instance's own __dict__ (still auto-restored on teardown).
    s = client.app.state.settings
    monkeypatch.setitem(s.__dict__, "chat_quota_user_bytes", 1)
    r = client.post(
        "/api/chat2/attachments",
        headers=headers,
        files={"file": ("a.png", _png_bytes(), "image/png")},
        data={"chat_id": chat_id},
    )
    assert r.status_code == 413 and r.json()["detail"]["code"] == "quota_exceeded"
    assert "quota" in r.json()["detail"]["message"]


def test_upload_rejects_when_free_space_floor_is_reached(tmp_data_dir, client, monkeypatch) -> None:
    """The free-space floor is enforced independently of the per-user quota.

    The suite pins `chat_quota_free_floor_bytes` to 0 (tests/conftest.py) so a
    disk-constrained runner can't fail every upload test; this is the one place
    that raises it again and fakes `shutil.disk_usage`, so the behaviour stays
    covered without depending on the runner's real free space. BELOW THE FLOOR
    EVERY UPLOAD 413s BY DESIGN -- the floor is a hard stop, not a soft one.
    """
    headers, chat_id = _setup(tmp_data_dir, client)
    s = client.app.state.settings
    monkeypatch.setitem(s.__dict__, "chat_quota_free_floor_bytes", 5 * 1024**3)
    monkeypatch.setattr(
        quota.shutil,
        "disk_usage",
        lambda _p: SimpleNamespace(total=10 * 1024**3, used=9 * 1024**3, free=1024**3),
    )
    r = client.post(
        "/api/chat2/attachments",
        headers=headers,
        files={"file": ("a.png", _png_bytes(), "image/png")},
        data={"chat_id": chat_id},
    )
    assert r.status_code == 413 and r.json()["detail"]["code"] == "quota_exceeded"
    assert "floor" in r.json()["detail"]["message"]


def test_serve_and_delete_errors_use_the_shared_envelope(tmp_data_dir, client) -> None:
    headers, chat_id = _setup(tmp_data_dir, client)
    r = client.post(
        "/api/chat2/attachments",
        headers=headers,
        files={"file": ("a.png", _png_bytes(), "image/png")},
        data={"chat_id": chat_id},
    )
    url = r.json()["url"]
    bad = client.get(url[:-3] + "xxx")
    assert bad.status_code == 401 and bad.json()["detail"]["code"] == "bad_token"
    malformed = client.get(url.split("?")[0] + "?t=not-a-token")
    assert malformed.status_code == 401 and malformed.json()["detail"]["code"] == "bad_token"
    gone = client.delete("/api/chat2/attachments/nope", headers=headers)
    assert gone.status_code == 404 and gone.json()["detail"]["code"] == "not_found"
    orphan = client.post(
        "/api/chat2/attachments",
        headers=headers,
        files={"file": ("a.png", _png_bytes(3, 3), "image/png")},
        data={"chat_id": "no-such-chat"},
    )
    assert orphan.status_code == 404 and orphan.json()["detail"]["code"] == "not_found"


def test_upload_cmyk_jpeg_reencodes_successfully(tmp_data_dir, client) -> None:
    """CMYK source images (e.g. print-workflow JPEGs) have no alpha channel
    and can't be pasted onto an RGBA canvas headed for a JPEG save -- the
    storage layer must pick RGB for JPEG output and succeed end to end."""
    headers, chat_id = _setup(tmp_data_dir, client)
    r = client.post(
        "/api/chat2/attachments",
        headers=headers,
        files={"file": ("a.jpg", _cmyk_jpeg_bytes(), "image/jpeg")},
        data={"chat_id": chat_id},
    )
    assert r.status_code == 201, r.text
    assert r.json()["mime"] == "image/jpeg"


def test_upload_maps_image_too_large_to_413(tmp_data_dir, client, monkeypatch) -> None:
    """storage.ImageTooLarge -- raised by reencode_image for either the raw
    OR the re-encoded size (see test_storage.py for both cases individually)
    -- must map to 413 at the route layer, distinctly from
    storage.UnsupportedImage's 415. Exercised here by making reencode_image
    itself raise, so this test verifies the route's exception-to-status
    mapping only, independent of storage.py's internal size-check details."""
    headers, chat_id = _setup(tmp_data_dir, client)

    def _raise_too_large(raw: bytes) -> None:
        raise routes_attachments_module.storage.ImageTooLarge(
            "re-encoded image larger than 10485760 bytes"
        )

    monkeypatch.setattr(routes_attachments_module.storage, "reencode_image", _raise_too_large)
    r = client.post(
        "/api/chat2/attachments",
        headers=headers,
        files={"file": ("a.png", _png_bytes(), "image/png")},
        data={"chat_id": chat_id},
    )
    assert r.status_code == 413 and r.json()["detail"]["code"] == "too_large"
    assert "re-encoded" in r.json()["detail"]["message"], "the message must say WHICH size blew"


def test_upload_rejects_oversized_declared_content_length_before_reencoding(
    tmp_data_dir, client
) -> None:
    """A client honest about a huge Content-Length gets a fast 413 -- the
    upload is rejected on the declared header, not by actually reading and
    re-encoding a large body. Enforced by Chat2BodyLimitMiddleware (see the
    dedicated test below for proof the route itself is never reached);
    this test is the end-to-end response-shape check."""
    headers, chat_id = _setup(tmp_data_dir, client)
    declared = str(10 * 1024**2 + 4096 + 1)  # just over MAX_IMAGE_BYTES + overhead
    r = client.post(
        "/api/chat2/attachments",
        headers={**headers, "Content-Length": declared},
        files={"file": ("a.png", _png_bytes(), "image/png")},
        data={"chat_id": chat_id},
    )
    assert r.status_code == 413
    assert r.json()["detail"]["code"] == "too_large"


def test_content_length_guard_rejects_before_reaching_the_route(
    tmp_data_dir, client, monkeypatch
) -> None:
    """The declared-Content-Length guard runs in Chat2BodyLimitMiddleware, a
    raw-ASGI middleware registered ahead of FastAPI's routing -- NOT in the
    route function. Proof: even if reencode_image would blow up if called,
    an oversized declared Content-Length must 413 without ever calling it,
    because FastAPI would otherwise have already spooled the whole body via
    Request.form() before the route body (and this patched function) runs."""
    headers, chat_id = _setup(tmp_data_dir, client)

    def _must_not_be_called(raw: bytes) -> None:
        raise AssertionError("route/reencode_image must not run: middleware should reject first")

    monkeypatch.setattr(routes_attachments_module.storage, "reencode_image", _must_not_be_called)
    declared = str(10 * 1024**2 + 4096 + 1)  # just over MAX_IMAGE_BYTES + overhead
    r = client.post(
        "/api/chat2/attachments",
        headers={**headers, "Content-Length": declared},
        files={"file": ("a.png", _png_bytes(), "image/png")},
        data={"chat_id": chat_id},
    )
    assert r.status_code == 413
    assert r.json()["detail"]["code"] == "too_large"


def test_dedupe_by_sha256_shares_one_file(tmp_data_dir, client) -> None:
    headers, chat_id = _setup(tmp_data_dir, client)
    for _ in range(2):
        r = client.post(
            "/api/chat2/attachments",
            headers=headers,
            files={"file": ("a.png", _png_bytes(), "image/png")},
            data={"chat_id": chat_id},
        )
        assert r.status_code == 201
    c = sqlite3.connect(tmp_data_dir / "vllm-warden.db")
    (rows,) = c.execute("SELECT COUNT(*) FROM attachments").fetchone()
    assert rows == 2 and len(list((tmp_data_dir / "chat2").rglob("*.png"))) == 1
