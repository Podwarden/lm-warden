"""GodModeMediaStore — bounded in-memory blob store for god-mode images.

Same contract family as GodModeHub: sync, non-raising, heap-bounded,
oldest-first eviction, nothing persisted.
"""

from app.proxy.godmode import GodModeMediaStore


def test_put_get_roundtrip():
    s = GodModeMediaStore(store_chars=1000, max_item_chars=500)
    assert s.put("a" * 16, "image/png", "payload123") is True
    assert s.get("a" * 16) == ("image/png", "payload123")


def test_get_unknown_returns_none():
    s = GodModeMediaStore()
    assert s.get("f" * 16) is None


def test_per_item_cap_rejects_and_stores_nothing():
    s = GodModeMediaStore(store_chars=1000, max_item_chars=10)
    assert s.put("a" * 16, "image/png", "x" * 11) is False
    assert s.get("a" * 16) is None
    assert s.size_chars() == 0


def test_total_budget_evicts_oldest_first():
    s = GodModeMediaStore(store_chars=25, max_item_chars=20)
    s.put("a" * 16, "image/png", "x" * 10)
    s.put("b" * 16, "image/png", "y" * 10)
    s.put("c" * 16, "image/png", "z" * 10)  # 30 chars total -> evict oldest
    assert s.get("a" * 16) is None
    assert s.get("b" * 16) == ("image/png", "y" * 10)
    assert s.get("c" * 16) == ("image/png", "z" * 10)
    assert s.size_chars() == 20


def test_single_item_larger_than_budget_still_kept_if_under_item_cap():
    # Mirrors the hub's "always keep the newest" rule: an item within the
    # per-item cap but over the total budget evicts everything else and stays.
    s = GodModeMediaStore(store_chars=15, max_item_chars=20)
    s.put("a" * 16, "image/png", "x" * 5)
    s.put("b" * 16, "image/png", "y" * 18)
    assert s.get("a" * 16) is None
    assert s.get("b" * 16) == ("image/png", "y" * 18)


def test_mime_allowlist_rejects_non_raster():
    s = GodModeMediaStore()
    assert s.put("a" * 16, "text/html", "payload") is False
    assert s.put("b" * 16, "image/svg+xml; charset=utf-8", "p") is False  # params not allowed
    assert s.put("c" * 16, "image/png", "p") is True
    # SVG is scripted, active content. Stored images are rendered via blob:
    # URLs, which inherit the warden page's own origin — a same-origin SVG
    # document could act against the admin API (see spec §5). Excluded even
    # bare, unlike a generic image/* allowlist.
    assert s.put("d" * 16, "image/svg+xml", "p") is False
    assert s.get("a" * 16) is None
    assert s.get("d" * 16) is None


def test_mime_allowlist_accepts_explicit_raster_set():
    s = GodModeMediaStore()
    raster_mimes = [
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
        "image/avif",
        "image/bmp",
    ]
    for i, mime in enumerate(raster_mimes):
        media_id = f"{i:016d}"
        assert s.put(media_id, mime, "p") is True
        assert s.get(media_id) == (mime, "p")


def test_put_never_raises_on_garbage():
    s = GodModeMediaStore()
    assert s.put(None, None, None) is False  # type: ignore[arg-type]
    assert s.put("x", "image/png", 123) is False  # type: ignore[arg-type]


def test_duplicate_id_overwrites_without_double_count():
    s = GodModeMediaStore(store_chars=100, max_item_chars=50)
    s.put("a" * 16, "image/png", "x" * 10)
    s.put("a" * 16, "image/jpeg", "y" * 20)
    assert s.get("a" * 16) == ("image/jpeg", "y" * 20)
    assert s.size_chars() == 20


def test_settings_defaults():
    # Same construction as tests/unit/test_app_state.py — Settings is a
    # frozen dataclass whose required fields are data_dir, hf_cache_dir,
    # cookie_secret, container_gpu_count.
    from pathlib import Path

    from app.config import Settings

    s = Settings(
        data_dir=Path("/data"),
        hf_cache_dir=Path("/root/.cache/huggingface"),
        cookie_secret="x" * 32,
        container_gpu_count=0,
    )
    assert s.godmode_media_store_chars == 64_000_000
    assert s.godmode_max_image_chars == 14_000_000
    assert s.godmode_max_images_per_req == 16
