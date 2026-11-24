"""Task 12 guards (Rulings R16, R17): the RAM pool must fit, the NVMe folder must be a disk.

Both fail open: a refused pool or folder means no store (recompute), never an error to a
client. The pool tests fake ``os.statvfs``. The NVMe tests fake ``/proc/self/mountinfo``
and the root device. The real clamp (``nvme_cap``) runs only in the tests that fake the
NVMe filesystem's statvfs; elsewhere ``_no_clamp`` keeps the cap as asked, so no test
depends on how full the host's disk is.
"""

from __future__ import annotations

import argparse
import logging
import mmap
import os
import secrets
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lmw_kv import daemon as daemon_mod
from lmw_kv import ram
from lmw_kv import store as store_mod
from lmw_kv.daemon import store_config, try_take_nvme
from lmw_kv.store import Store, StoreConfig

GIB = 1 << 30
REAL_NVME_CAP = daemon_mod.nvme_cap
PAGE = mmap.PAGESIZE
GB = 1_000_000_000


def _vfs(free: int, size: int | None = None, frsize: int = 1) -> SimpleNamespace:
    size = free if size is None else size
    return SimpleNamespace(
        f_bavail=free // frsize,
        f_bfree=free // frsize,
        f_frsize=frsize,
        f_blocks=size // frsize,
    )


@pytest.fixture(autouse=True)
def _no_clamp(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cap as asked: the real clamp would read the host disk's free space."""
    monkeypatch.setattr(daemon_mod, "nvme_cap", lambda _dir, cap: (cap, None, None))


@pytest.fixture
def ample_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store_mod, "fit_memory", lambda wanted, **kw: wanted)


# ---- guard 1: the RAM pool must fit --------------------------------------------------


def test_gib_figures_have_at_most_one_decimal() -> None:
    assert ram.fmt_gib(8 * GIB) == "8"
    assert ram.fmt_gib(2 * GIB) == "2"
    assert ram.fmt_gib(3 * GIB // 2) == "1.5"
    assert ram.fmt_gib(int(0.99 * GIB)) == "1"
    assert ram.fmt_gib(int(1.26 * GIB)) == "1.3"


def test_a_pool_that_fits_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ample_memory: None
) -> None:
    monkeypatch.setattr(ram, "SHM_RESERVE_BYTES", 0)
    monkeypatch.setattr(ram.os, "statvfs", lambda _p: _vfs(64 * PAGE, frsize=PAGE))
    s = Store(
        StoreConfig(
            ram_path=tmp_path / "ram.pool", ram_bytes=8 * PAGE, nvme_dir=None, nvme_cap_bytes=0
        )
    )
    try:
        assert s.ram_bytes() == 8 * PAGE
        assert (tmp_path / "ram.pool").exists()
        assert s.stats()["pool_error"] is None
        assert s.pool_error() is None
        d = daemon_mod.Daemon(s, str(tmp_path / "kv.sock"))
        hello = d._dispatch(None, "hello", {})  # type: ignore[arg-type]
        assert "pool_error" not in hello  # step 1's hello reply, unchanged
    finally:
        s.close()


def test_a_pool_larger_than_the_free_space_is_refused_with_one_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, ample_memory: None
) -> None:
    """Production: 8 GiB asked against a 2 GiB /dev/shm. No pool, no shrink, one ERROR."""
    pool = Path("/dev/shm") / f"lmw-kv-guard-{secrets.token_hex(4)}.pool"
    real = os.statvfs

    def statvfs(p: Any) -> Any:
        return _vfs(2 * GIB) if str(p).startswith("/dev/shm") else real(p)

    monkeypatch.setattr(ram.os, "statvfs", statvfs)
    created: list[Path] = []
    monkeypatch.setattr(store_mod, "create_pool_file", lambda p, n: created.append(p))
    caplog.set_level(logging.WARNING, logger="lmw_kv")
    s = Store(StoreConfig(ram_path=pool, ram_bytes=8 * GIB, nvme_dir=None, nvme_cap_bytes=0))
    try:
        st = s.stats()
        assert st["pool_error"] == "RAM pool 8 GiB does not fit /dev/shm 2 GiB"
        assert s.pool_error() == st["pool_error"]
        assert st["ram_bytes"] == 0 and s.ram_bytes() == 0
        assert created == [] and not pool.exists()
    finally:
        s.close()
    records = [r for r in caplog.records if r.name.startswith("lmw_kv")]
    assert [(r.levelno, r.getMessage()) for r in records] == [
        (logging.ERROR, "RAM pool 8 GiB does not fit /dev/shm 2 GiB")
    ]


def test_an_existing_pool_file_of_ours_counts_as_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ample_memory: None
) -> None:
    """A restarted daemon replaces the old pool: its bytes are free once it is unlinked."""
    monkeypatch.setattr(ram, "SHM_RESERVE_BYTES", 0)
    path = tmp_path / "ram.pool"
    path.write_bytes(b"\0" * (4 * PAGE))
    total = 8 * PAGE

    def statvfs(_p: Any) -> Any:  # a tmpfs of 8 pages: the old file takes 4 while it exists
        used = path.stat().st_size if path.exists() else 0
        return _vfs(total - used, total, frsize=PAGE)

    monkeypatch.setattr(ram.os, "statvfs", statvfs)
    s = Store(StoreConfig(ram_path=path, ram_bytes=8 * PAGE, nvme_dir=None, nvme_cap_bytes=0))
    try:
        assert s.stats()["pool_error"] is None
        assert s.ram_bytes() == 8 * PAGE
    finally:
        s.close()


def test_a_pool_still_held_by_an_earlier_daemon_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ample_memory: None
) -> None:
    """Live finding: an engine still maps the earlier generation's (deleted) pool."""
    shm = tmp_path / "shm"
    shm.mkdir()
    (shm / "other").write_bytes(b"\0" * PAGE)  # a file that is there: not counted as held
    used_by_files = (shm / "other").stat().st_blocks * 512
    held = GIB  # deleted but still mapped

    def statvfs(_p: Any) -> Any:
        total = 4 * GIB
        used = used_by_files + held
        return SimpleNamespace(
            f_bavail=total - used, f_bfree=total - used, f_frsize=1, f_blocks=total
        )

    monkeypatch.setattr(ram.os, "statvfs", statvfs)
    monkeypatch.setattr(ram.os.path, "ismount", lambda p: Path(p) == shm)
    want = 4 * GIB
    s = Store(
        StoreConfig(ram_path=shm / "ram.pool", ram_bytes=want, nvme_dir=None, nvme_cap_bytes=0)
    )
    try:
        err = s.stats()["pool_error"]
    finally:
        s.close()
    assert err == (
        f"RAM pool 4 GiB does not fit {shm} 3 GiB; 1 GiB there is held by deleted files, "
        "likely the pool of an earlier KV store daemon that a running engine still maps "
        "(reload the models that use the store)"
    )


def test_a_pool_shrunk_to_nothing_by_the_engine_reserve_has_a_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ample_memory: None
) -> None:
    """Never a silent capacity 0: the 2 GiB kept for the engines is named too."""
    monkeypatch.setattr(ram.os, "statvfs", lambda _p: _vfs(5 * GIB // 2))
    s = Store(
        StoreConfig(ram_path=tmp_path / "ram.pool", ram_bytes=GIB, nvme_dir=None, nvme_cap_bytes=0)
    )
    try:
        assert s.ram_bytes() == 0
        assert s.stats()["pool_error"] == (
            f"RAM pool 1 GiB does not fit {tmp_path} 2.5 GiB less the 2 GiB kept free for the "
            "engines"
        )
    finally:
        s.close()


def test_the_daemon_hello_carries_the_pool_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ram.os, "statvfs", lambda _p: _vfs(2 * GIB))
    s = Store(
        StoreConfig(
            ram_path=tmp_path / "ram.pool", ram_bytes=8 * GIB, nvme_dir=None, nvme_cap_bytes=0
        )
    )
    try:
        d = daemon_mod.Daemon(s, str(tmp_path / "kv.sock"))
        hello = d._dispatch(None, "hello", {})  # type: ignore[arg-type]
        assert hello["ram_bytes"] == 0
        assert hello["pool_error"] == f"RAM pool 8 GiB does not fit {tmp_path} 2 GiB"
    finally:
        s.close()


def test_the_pool_reason_is_the_same_for_the_supervisor(tmp_path: Path, monkeypatch) -> None:
    """The api's own refusal (no daemon started) shows the daemon's text."""
    monkeypatch.setattr(ram.os, "statvfs", lambda _p: _vfs(2 * GIB))
    got = ram.pool_error(tmp_path / "p", 8 * GIB, free=2 * GIB, shm_fit=0, mem_fit=8 * GIB)
    assert got == f"RAM pool 8 GiB does not fit {tmp_path} 2 GiB"
    assert ram.pool_error(tmp_path / "p", GIB, free=8 * GIB, shm_fit=GIB, mem_fit=GIB) is None
    mem = ram.pool_error(tmp_path / "p", 8 * GIB, free=20 * GIB, shm_fit=8 * GIB, mem_fit=GIB // 2)
    assert mem == (
        "RAM pool 8 GiB does not fit free memory: 0.5 GiB is left after the safety margin "
        "and the engine reserve"
    )


# ---- guard 2: the NVMe folder must be a separate filesystem --------------------------


def _args(tmp_path: Path, nvme: Path, *, allow: bool = False, cap_gb: float = 1.0) -> Any:
    return argparse.Namespace(
        ram_path=str(tmp_path / "ram.pool"),
        ram_gib=0.01,
        nvme_dir=str(nvme),
        nvme_cap_gb=cap_gb,
        nvme_allow_root_disk=allow,
    )


@pytest.fixture
def fake_mounts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A mountinfo with ``tmp_path`` on ext4 and the root device somewhere else."""
    info = tmp_path / "mountinfo"

    def mount(fstype: str, at: Path | None = None) -> None:
        point = str((at or tmp_path).resolve()).replace(" ", "\\040")
        info.write_text(
            "21 1 0:20 / / rw - overlay overlay rw\n"
            f"30 21 8:1 / {point} rw,relatime - {fstype} /dev/sdb1 rw\n"
        )

    mount("ext4")
    monkeypatch.setattr(daemon_mod, "MOUNTINFO", info)
    monkeypatch.setattr(daemon_mod, "_root_dev", lambda: -1)
    return mount


def test_fs_type_takes_the_longest_mount_prefix(tmp_path: Path, fake_mounts: Any) -> None:
    assert daemon_mod.fs_type(tmp_path / "a" / "b") == "ext4"
    assert daemon_mod.fs_type(Path("/")) == "overlay"


def test_fs_type_is_unknown_without_mountinfo(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(daemon_mod, "MOUNTINFO", tmp_path / "missing")
    assert daemon_mod.fs_type(tmp_path) is None


def test_a_folder_on_the_root_filesystem_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_mounts: Any, caplog: Any
) -> None:
    nvme = tmp_path / "nvme"
    fake_mounts("overlay")
    monkeypatch.setattr(daemon_mod, "_root_dev", lambda: os.stat(tmp_path).st_dev)
    caplog.set_level(logging.WARNING, logger="lmw_kv.daemon")
    cfg, fd = store_config(_args(tmp_path, nvme))
    assert cfg.nvme_dir is None and fd is None
    want = (
        f"NVMe tier off: {nvme} is on the root filesystem (overlay); mount a dedicated disk "
        "there, or set kv_store_nvme_allow_root_disk"
    )
    assert cfg.nvme_error == want
    assert [r.getMessage() for r in caplog.records if r.name == "lmw_kv.daemon"] == [want]
    assert not (nvme / daemon_mod.NVME_LOCK_NAME).exists()  # nothing written there
    s = Store(cfg)
    try:
        st = s.stats()
        assert st["nvme_enabled"] is False and st["nvme_error"] == want
        assert st["nvme_cap_effective_bytes"] == 0
    finally:
        s.close()


def test_the_root_filesystem_rule_works_without_mountinfo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """macOS has no /proc/self/mountinfo: only the st_dev rule applies."""
    monkeypatch.setattr(daemon_mod, "MOUNTINFO", tmp_path / "missing")
    monkeypatch.setattr(daemon_mod, "_root_dev", lambda: os.stat(tmp_path).st_dev)
    nvme = tmp_path / "nvme"
    assert daemon_mod.root_disk_reason(nvme) == (
        f"NVMe tier off: {nvme} is on the root filesystem; mount a dedicated disk there, "
        "or set kv_store_nvme_allow_root_disk"
    )
    monkeypatch.setattr(daemon_mod, "_root_dev", lambda: -1)
    assert daemon_mod.root_disk_reason(nvme) is None


def test_the_override_allows_the_root_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_mounts: Any
) -> None:
    nvme = tmp_path / "nvme"
    fake_mounts("overlay")
    monkeypatch.setattr(daemon_mod, "_root_dev", lambda: os.stat(tmp_path).st_dev)
    cfg, fd = store_config(_args(tmp_path, nvme, allow=True))
    try:
        assert cfg.nvme_dir == nvme and fd is not None and cfg.nvme_error is None
    finally:
        if fd is not None:
            os.close(fd)


@pytest.mark.parametrize("fstype", ["overlay", "tmpfs", "ramfs"])
def test_memory_and_overlay_filesystems_are_refused(
    tmp_path: Path, fake_mounts: Any, fstype: str
) -> None:
    nvme = tmp_path / "nvme"
    fake_mounts(fstype)  # another device than "/", but not a disk
    cfg, fd = store_config(_args(tmp_path, nvme))
    assert cfg.nvme_dir is None and fd is None
    assert cfg.nvme_error == (
        f"NVMe tier off: {nvme} is on a {fstype} filesystem, not a dedicated disk; mount a "
        "dedicated disk there, or set kv_store_nvme_allow_root_disk"
    )


def test_a_dedicated_disk_is_taken(tmp_path: Path, fake_mounts: Any) -> None:
    nvme = tmp_path / "nvme"
    cfg, fd = store_config(_args(tmp_path, nvme))
    try:
        assert cfg.nvme_dir == nvme and fd is not None and cfg.nvme_error is None
    finally:
        if fd is not None:
            os.close(fd)


def test_a_missing_folder_is_judged_by_the_folder_it_would_be_made_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(daemon_mod, "MOUNTINFO", tmp_path / "missing")
    monkeypatch.setattr(daemon_mod, "_root_dev", lambda: os.stat(tmp_path).st_dev)
    deep = tmp_path / "a" / "b" / "c"
    assert daemon_mod.root_disk_reason(deep) is not None
    assert not (tmp_path / "a").exists()


# ---- the effective NVMe cap ----------------------------------------------------------


def _fake_nvme_fs(monkeypatch: pytest.MonkeyPatch, nvme: Path, free: int, size: int) -> None:
    real = os.statvfs

    def statvfs(p: Any) -> Any:
        return _vfs(free, size) if str(p).startswith(str(nvme)) else real(p)

    monkeypatch.setattr(daemon_mod.os, "statvfs", statvfs)
    monkeypatch.setattr(daemon_mod, "nvme_cap", REAL_NVME_CAP)  # the clamp, on the fake fs


def test_the_cap_is_clamped_to_the_free_space_less_the_margin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_mounts: Any
) -> None:
    nvme = tmp_path / "nvme"
    nvme.mkdir()
    assert daemon_mod.NVME_FS_MARGIN == 0.15
    _fake_nvme_fs(monkeypatch, nvme, free=int(49.5 * GB), size=100 * GB)
    cfg, fd = store_config(_args(tmp_path, nvme, cap_gb=250))
    note = "cap 250 GB clamped to 34.5 GB: the filesystem has 49.5 GB free and 15% is kept free"
    try:
        assert fd is not None and cfg.nvme_dir == nvme
        assert cfg.nvme_cap_bytes == int(34.5 * GB)
        assert cfg.nvme_cap_note == note
        s = Store(cfg)
        try:
            st = s.stats()
            assert st["nvme_enabled"] is True
            assert st["nvme_cap_effective_bytes"] == int(34.5 * GB)
            assert st["nvme_cap_bytes"] == 250 * GB  # the operator's cap (minor 1)
            assert st["nvme_cap_note"] == note and st["nvme_error"] is None
        finally:
            s.close()
    finally:
        if fd is not None:
            os.close(fd)


def test_the_store_bytes_on_disk_count_as_room(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nvme = tmp_path / "nvme"
    (nvme / "ab").mkdir(parents=True)
    (nvme / "ab" / "000001.seg").write_bytes(b"\0" * 1000)
    _fake_nvme_fs(monkeypatch, nvme, free=1000, size=10_000)  # margin 1500
    cap, note, err = daemon_mod.nvme_cap(nvme, 10 * GB)
    assert (cap, err) == (500, None) and note is not None
    assert daemon_mod.nvme_cap(nvme, 400) == (400, None, None)  # under the room: as asked


def test_a_full_disk_turns_the_tier_off_with_a_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_mounts: Any, caplog: Any
) -> None:
    nvme = tmp_path / "nvme"
    nvme.mkdir()
    _fake_nvme_fs(monkeypatch, nvme, free=10 * GB, size=100 * GB)
    caplog.set_level(logging.WARNING, logger="lmw_kv.daemon")
    cfg, fd = store_config(_args(tmp_path, nvme, cap_gb=250))
    want = (
        f"NVMe tier off: {nvme} is too full: the filesystem has 10 GB free and 15% is kept " "free"
    )
    assert cfg.nvme_dir is None and fd is None and cfg.nvme_error == want
    assert [r.getMessage() for r in caplog.records if r.name == "lmw_kv.daemon"] == [want]
    # The lock was given back: another daemon (or the retry) can take it.
    again = daemon_mod.lock_nvme_dir(nvme)
    assert again is not None
    os.close(again)


def test_the_retry_applies_the_same_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_mounts: Any
) -> None:
    nvme = tmp_path / "nvme"
    nvme.mkdir()
    fake_mounts("overlay")
    root = os.stat(tmp_path).st_dev
    monkeypatch.setattr(daemon_mod, "_root_dev", lambda: root)
    cfg, fd = store_config(_args(tmp_path, nvme, cap_gb=250))
    assert fd is None
    s = Store(cfg)
    try:
        assert try_take_nvme(s, nvme) is None  # still the root disk
        assert s.stats()["nvme_error"].startswith(f"NVMe tier off: {nvme} is on the root")
        assert not (nvme / daemon_mod.NVME_LOCK_NAME).exists()
        # A disk is mounted there now, but it is nearly full.
        fake_mounts("ext4")
        monkeypatch.setattr(daemon_mod, "_root_dev", lambda: -1)
        _fake_nvme_fs(monkeypatch, nvme, free=10 * GB, size=100 * GB)
        assert try_take_nvme(s, nvme) is None
        assert s.stats()["nvme_error"] == (
            f"NVMe tier off: {nvme} is too full: the filesystem has 10 GB free and 15% is "
            "kept free"
        )
        # Room freed: the tier comes on with the clamped cap.
        _fake_nvme_fs(monkeypatch, nvme, free=int(49.5 * GB), size=100 * GB)
        fd2 = try_take_nvme(s, nvme)
        assert fd2 is not None
        st = s.stats()
        assert st["nvme_enabled"] is True and st["nvme_error"] is None
        assert st["nvme_cap_effective_bytes"] == int(34.5 * GB)
        assert st["nvme_cap_bytes"] == 250 * GB  # minor 1's field keeps the operator's cap
        assert st["nvme_cap_note"] == (
            "cap 250 GB clamped to 34.5 GB: the filesystem has 49.5 GB free and 15% is kept free"
        )
        os.close(fd2)
    finally:
        s.close()


def test_the_retry_honours_the_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_mounts: Any
) -> None:
    nvme = tmp_path / "nvme"
    fake_mounts("overlay")
    monkeypatch.setattr(daemon_mod, "_root_dev", lambda: os.stat(tmp_path).st_dev)
    s = Store(
        StoreConfig(
            ram_path=tmp_path / "ram.pool", ram_bytes=8 * PAGE, nvme_dir=None, nvme_cap_bytes=GB
        )
    )
    try:
        fd = try_take_nvme(s, nvme, allow_root_disk=True)
        assert fd is not None and s.stats()["nvme_enabled"] is True
        os.close(fd)
    finally:
        s.close()


def test_the_override_flag_parses() -> None:
    base = ["--socket", "s", "--ram-path", "p", "--ram-gib", "1"]
    assert daemon_mod.parse_args(base).nvme_allow_root_disk is False
    assert daemon_mod.parse_args([*base, "--nvme-allow-root-disk"]).nvme_allow_root_disk is True
