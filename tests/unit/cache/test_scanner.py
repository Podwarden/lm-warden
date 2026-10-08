"""Unit tests for the pure-sync HF cache walker.

The walker is decoupled from FastAPI so these tests just create a few
directories in a tmpdir and assert on the dataclasses that come back —
no app, no DB, no event loop. Every assertion here pins a behaviour
described in the spec § Testing / scanner.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.cache.scanner import CachedRepo, scan_hf_cache


def test_missing_cache_root_returns_empty(tmp_path: Path) -> None:
    """``scan_hf_cache`` must tolerate the configured root not existing.

    On a fresh container the HF cache PVC is mounted but empty; on a
    misconfigured one the path may not exist at all. Either way we
    must NOT raise — the API has to surface an empty list, not 500.
    """
    out = scan_hf_cache(tmp_path / "does-not-exist")
    assert out == []


def test_empty_cache_dir_returns_empty(tmp_path: Path) -> None:
    """Existing but empty cache root → ``[]`` (not None, not raise)."""
    assert scan_hf_cache(tmp_path) == []


def test_decodes_repo_names_and_skips_non_matching(tmp_path: Path) -> None:
    """``models--A--B/`` becomes ``A/B``; everything else is skipped.

    Three sibling directories:
      - ``models--Qwen--Qwen3.6-27B-GGUF`` → ``Qwen/Qwen3.6-27B-GGUF``
      - ``models--TinyLlama--TinyLlama-1.1B`` → ``TinyLlama/TinyLlama-1.1B``
      - ``some-other-dir`` → skipped silently
    """
    (tmp_path / "models--Qwen--Qwen3.6-27B-GGUF" / "blobs").mkdir(parents=True)
    (tmp_path / "models--TinyLlama--TinyLlama-1.1B" / "blobs").mkdir(parents=True)
    (tmp_path / "some-other-dir").mkdir()
    # Sprinkle some content so size_bytes is nonzero on at least one.
    (tmp_path / "models--Qwen--Qwen3.6-27B-GGUF" / "blobs" / "abcdef").write_bytes(b"x" * 1024)

    out = scan_hf_cache(tmp_path)
    repos = sorted(r.repo for r in out)
    assert repos == ["Qwen/Qwen3.6-27B-GGUF", "TinyLlama/TinyLlama-1.1B"]
    qwen = next(r for r in out if r.repo == "Qwen/Qwen3.6-27B-GGUF")
    assert qwen.size_bytes >= 1024
    assert qwen.path == (tmp_path / "models--Qwen--Qwen3.6-27B-GGUF").resolve()


def test_unions_root_and_hub_layouts(tmp_path: Path) -> None:
    """HF library writes either under root or under ``root/hub/`` depending on
    HF_HOME vs HUGGINGFACE_HUB_CACHE. Scanner must return the union."""
    (tmp_path / "models--A--top").mkdir()
    (tmp_path / "hub").mkdir()
    (tmp_path / "hub" / "models--B--under-hub").mkdir()
    out = scan_hf_cache(tmp_path)
    assert sorted(r.repo for r in out) == ["A/top", "B/under-hub"]


def test_dedups_when_hub_aliases_root(tmp_path: Path) -> None:
    """If a single physical dir appears under both ``<root>/`` and
    ``<root>/hub/`` via a symlink, scanner returns one entry, not two.

    (This is exactly what happens when a container exports BOTH HF_HOME
    and HUGGINGFACE_HUB_CACHE to compatible-but-different paths and an
    operator manually symlinks one to the other.)
    """
    (tmp_path / "models--A--name").mkdir()
    (tmp_path / "hub").mkdir()
    os.symlink(tmp_path / "models--A--name", tmp_path / "hub" / "models--A--name")
    out = scan_hf_cache(tmp_path)
    # The dedupe key is ``Path.resolve()``, so the symlink and the
    # real dir collapse to a single entry.
    assert [r.repo for r in out] == ["A/name"]


def test_permission_denied_is_tolerated(tmp_path: Path) -> None:
    """An unreadable subtree is logged + skipped; the rest of the
    walk still produces output. Never raises.

    On Linux/Docker this would normally manifest as a 0o000 dir under
    blobs/ that we can't recurse into. We exercise the tolerant path
    by chmodding a real subdir; root would bypass the chmod, so the
    test skips when running as root."""
    if os.geteuid() == 0:
        pytest.skip("running as root — cannot exercise permission-denied path")
    target = tmp_path / "models--A--locked"
    (target / "blobs").mkdir(parents=True)
    (target / "blobs" / "x").write_bytes(b"y" * 16)
    try:
        os.chmod(target / "blobs", 0o000)
        out = scan_hf_cache(tmp_path)
        # The repo entry IS returned (the dir-itself stat succeeded);
        # size may be lower than the actual blob bytes because the
        # walker couldn't recurse into ``blobs/`` — that's acceptable
        # graceful degradation, the alternative is raising and 500ing
        # the whole list endpoint over one bad subtree.
        assert [r.repo for r in out] == ["A/locked"]
    finally:
        os.chmod(target / "blobs", 0o755)  # let pytest clean up


def test_empty_repo_name_is_silently_skipped(tmp_path: Path) -> None:
    """A directory named exactly ``models--`` (the prefix with no body)
    must be skipped, not raise. Spec § Testing / scanner pins this as
    an "exotic encoding (skipped, logged)" case — HF would never write
    this layout, but an operator's manual mkdir or a half-aborted
    download can land it on disk and we must not 500 the list endpoint
    over it.

    Same applies to anything that doesn't decode cleanly via the
    ``models--<org>--<name>`` template; pair the empty body with a
    sibling that DOES decode to prove the walker keeps going after
    the skip rather than short-circuiting.
    """
    (tmp_path / "models--").mkdir()
    (tmp_path / "models--Good--Repo").mkdir()
    out = scan_hf_cache(tmp_path)
    # Only the well-formed sibling appears; the empty-body entry is
    # silently dropped (with a debug log) by ``_decode_repo_name``.
    assert [r.repo for r in out] == ["Good/Repo"]


def test_merges_root_and_hub_layouts_for_same_repo(tmp_path: Path) -> None:
    """vllm-warden#238: a repo split across ``<root>/`` (real weights) and
    ``<root>/hub/`` (HF's own hub layout, metadata only) must be reported
    as ONE row whose size is the real on-disk total — not two rows
    summing to roughly double it.

    Fixture mirrors the bug report exactly: ``<root>/models--org--name``
    holds the actual weight file directly (no symlinks — this is
    vllm-warden's own download layout), while
    ``<root>/hub/models--org--name`` has a proper HF ``blobs/`` +
    ``snapshots/`` layout where the snapshot entry is a REAL symlink
    into ``blobs/``. Before the fix, ``os.stat`` on that symlink summed
    the blob's bytes on top of the blob itself (double-counting the hub
    side) AND the two layouts were never merged (duplicate row).
    """
    weights_dir = tmp_path / "models--nvidia--Qwen3.6-35B-A3B-NVFP4"
    weights_dir.mkdir(parents=True)
    weight_file = weights_dir / "model.safetensors"
    weight_file.write_bytes(b"w" * 7_000_000)

    hub_repo = tmp_path / "hub" / "models--nvidia--Qwen3.6-35B-A3B-NVFP4"
    blobs = hub_repo / "blobs"
    snapshot = hub_repo / "snapshots" / "deadbeef"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    blob = blobs / "cafef00d"
    blob.write_bytes(b"c" * 26_000)
    # Real symlink into blobs/, exactly like the HF library writes.
    os.symlink(os.path.relpath(blob, snapshot), snapshot / "config.json")

    real_total = weight_file.stat().st_size + blob.stat().st_size

    out = scan_hf_cache(tmp_path)

    matching = [r for r in out if r.repo == "nvidia/Qwen3.6-35B-A3B-NVFP4"]
    assert len(matching) == 1, "repo must appear exactly once, not once per layout"
    assert matching[0].size_bytes == real_total, (
        "reported size must equal the real byte total counted once "
        "(blob referenced by a symlink must not be added twice)"
    )


def test_cached_repo_is_frozen() -> None:
    """``@dataclass(frozen=True)`` so route handlers can stash these in
    a dict keyed by path without worrying about mutation."""
    c = CachedRepo(repo="a/b", path=Path("/tmp"), size_bytes=0, last_modified=0.0)
    # ``frozen=True`` raises ``FrozenInstanceError`` (subclass of ``AttributeError``)
    # on any attribute assignment.
    with pytest.raises(AttributeError):
        c.repo = "x/y"  # type: ignore[misc]
