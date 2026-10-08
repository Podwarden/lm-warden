from pathlib import Path

import pytest

from app.models.pull_task import (
    DiskShortage,
    insufficient_disk,
)

GI = 1024 * 1024 * 1024


def _make_blobs(cache_dir: Path, repo: str, total_bytes: int) -> None:
    """Create sparse blob files under the HF cache layout summing to ``total_bytes``.

    huggingface_hub writes downloads to
    ``{cache_dir}/models--{org}--{name}/blobs/``; ``stat().st_size`` is what
    ``_snapshot_dir_size`` sums, so a sparse truncate is a faithful stand-in
    for a fully cached repo without allocating the bytes.
    """
    blobs = cache_dir / ("models--" + repo.replace("/", "--")) / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    half = total_bytes // 2
    for name, size in (("blob-a", half), ("blob-b", total_bytes - half)):
        with (blobs / name).open("wb") as f:
            f.truncate(size)


async def test_insufficient_disk_raises_when_estimate_exceeds_free(monkeypatch, tmp_path):
    from app.models import pull_task as pt

    monkeypatch.setattr(pt, "disk_free_bytes", lambda p: 10 * 1024 * 1024 * 1024)

    async def fake_estimate(*args, **kwargs):
        return 8 * 1024 * 1024 * 1024

    monkeypatch.setattr(pt, "estimate_repo_bytes", fake_estimate)

    with pytest.raises(DiskShortage) as exc:
        await insufficient_disk("Qwen/Qwen3.5-9B", "main", tmp_path, hf_token=None)
    assert "needs" in str(exc.value).lower()


async def test_insufficient_disk_passes_when_enough(monkeypatch, tmp_path):
    from app.models import pull_task as pt

    monkeypatch.setattr(pt, "disk_free_bytes", lambda p: 100 * 1024 * 1024 * 1024)

    async def fake_estimate(*args, **kwargs):
        return 8 * 1024 * 1024 * 1024

    monkeypatch.setattr(pt, "estimate_repo_bytes", fake_estimate)

    await insufficient_disk("o/r", "main", tmp_path, hf_token=None)


async def test_fully_cached_repo_does_not_trip_disk_shortage(monkeypatch, tmp_path):
    """A re-pull whose blobs are all cached needs ~0 new bytes, not the fresh
    estimate x safety factor (#272)."""
    from app.models import pull_task as pt

    monkeypatch.setattr(pt, "disk_free_bytes", lambda p: 5 * GI)
    _make_blobs(tmp_path, "o/r", 40 * GI)

    # 40 GiB estimate is already cached: incremental need ~0, so the safety
    # factor applied to it must not reject 5 GiB of headroom.
    await insufficient_disk("o/r", "main", tmp_path, hf_token=None, estimate=40 * GI)


async def test_empty_cache_still_raises_disk_shortage(monkeypatch, tmp_path):
    """The genuinely-fresh-download path keeps the full safety margin."""
    from app.models import pull_task as pt

    monkeypatch.setattr(pt, "disk_free_bytes", lambda p: 5 * GI)

    with pytest.raises(DiskShortage) as exc:
        await insufficient_disk("o/r", "main", tmp_path, hf_token=None, estimate=40 * GI)
    assert "needs" in str(exc.value).lower()


async def test_cache_larger_than_estimate_floors_incremental_at_zero(monkeypatch, tmp_path):
    """A cache bigger than the estimate (e.g. a prior bigger revision) means
    nothing to download: the incremental is floored at 0, never negative."""
    from app.models import pull_task as pt

    monkeypatch.setattr(pt, "disk_free_bytes", lambda p: 1 * GI)
    _make_blobs(tmp_path, "o/r", 48 * GI)

    await insufficient_disk("o/r", "main", tmp_path, hf_token=None, estimate=40 * GI)


async def test_partial_cache_is_judged_on_the_incremental(monkeypatch, tmp_path):
    """With 35 GiB of the 40 GiB estimate cached, the check must size the
    missing 5 GiB (x 1.5 safety), not the full estimate or nothing at all."""
    from app.models import pull_task as pt

    _make_blobs(tmp_path, "o/r", 35 * GI)

    # 5 GiB missing x 1.5 = 7.5 GiB > 6 GiB free: still refused.
    monkeypatch.setattr(pt, "disk_free_bytes", lambda p: 6 * GI)
    with pytest.raises(DiskShortage):
        await insufficient_disk("o/r", "main", tmp_path, hf_token=None, estimate=40 * GI)

    # ...but 10 GiB free comfortably covers 7.5 GiB: admitted. Pre-fix this
    # branch refused too, because the full 40 GiB x 1.5 = 60 GiB was compared.
    monkeypatch.setattr(pt, "disk_free_bytes", lambda p: 10 * GI)
    await insufficient_disk("o/r", "main", tmp_path, hf_token=None, estimate=40 * GI)
