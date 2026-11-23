from __future__ import annotations

import errno
import mmap
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from lmw_kv import ram
from lmw_kv.ram import RamPool, fit_shm

GIB = 1 << 30
SLOT = 4096


def test_alloc_free_reuse(tmp_path: Path) -> None:
    pool = RamPool(tmp_path / "pool", 4 * SLOT, SLOT)
    assert pool.slots_total == 4
    assert pool.slots_free() == 4
    a = pool.alloc()
    b = pool.alloc()
    assert a is not None and b is not None and a != b
    assert pool.slots_free() == 2
    pool.free(a)
    assert pool.slots_free() == 3
    assert pool.alloc() == a
    assert pool.offset(b) == b * SLOT
    pool.close()


def test_view_writes_are_visible_through_a_second_mmap(tmp_path: Path) -> None:
    path = tmp_path / "pool"
    pool = RamPool(path, 4 * SLOT, SLOT)
    slot = pool.alloc()
    assert slot is not None
    v = pool.view(slot)
    assert len(v) == SLOT
    v[:5] = b"hello"
    fd = os.open(path, os.O_RDONLY)
    try:
        with mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ) as other:
            off = pool.offset(slot)
            assert other[off : off + 5] == b"hello"
    finally:
        os.close(fd)
    v.release()
    pool.close()


def test_full_pool_returns_none(tmp_path: Path) -> None:
    pool = RamPool(tmp_path / "pool", 2 * SLOT, SLOT)
    assert pool.alloc() is not None
    assert pool.alloc() is not None
    assert pool.alloc() is None
    pool.close()


def test_fit_shm_clamps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        os, "statvfs", lambda _p: SimpleNamespace(f_bavail=5 * GIB // 4096, f_frsize=4096)
    )
    assert fit_shm(tmp_path / "pool", 16 * GIB) == 3 * GIB
    assert fit_shm(tmp_path / "pool", GIB) == GIB


def test_fit_shm_keeps_two_gib_for_tensor_parallel_engines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pool is sized before the engines start; their 2 GiB of /dev/shm stays free."""
    assert ram.SHM_RESERVE_BYTES == 2 * GIB
    monkeypatch.setattr(
        os, "statvfs", lambda _p: SimpleNamespace(f_bavail=2 * GIB // 4096, f_frsize=4096)
    )
    assert fit_shm(tmp_path / "pool", 16 * GIB) == 0
    monkeypatch.setattr(
        os, "statvfs", lambda _p: SimpleNamespace(f_bavail=(2 * GIB + 4096) // 4096, f_frsize=4096)
    )
    assert fit_shm(tmp_path / "pool", 16 * GIB) == 4096


def test_pool_reserves_every_page(tmp_path: Path) -> None:
    path = tmp_path / "pool"
    pool = RamPool(path, 8 * mmap.PAGESIZE, mmap.PAGESIZE)
    assert os.stat(path).st_blocks * 512 >= pool.size_bytes
    pool.close()


@pytest.mark.parametrize("with_fallocate", [True, False])
def test_enospc_at_start_raises_and_removes_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_fallocate: bool
) -> None:
    def boom(*_a: object) -> int:
        raise OSError(errno.ENOSPC, "No space left on device")

    if with_fallocate:
        monkeypatch.setattr(ram.os, "posix_fallocate", boom, raising=False)
    else:
        monkeypatch.delattr(ram.os, "posix_fallocate", raising=False)
        monkeypatch.setattr(ram.os, "pwrite", boom)
    path = tmp_path / "pool"
    with pytest.raises(OSError) as exc:
        RamPool(path, 3 * mmap.PAGESIZE, mmap.PAGESIZE)
    assert exc.value.errno == errno.ENOSPC
    assert not path.exists()


def test_failed_start_keeps_a_preexisting_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: object) -> int:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.delattr(ram.os, "posix_fallocate", raising=False)
    monkeypatch.setattr(ram.os, "pwrite", boom)
    path = tmp_path / "pool"
    path.write_bytes(b"x")
    with pytest.raises(OSError):
        RamPool(path, mmap.PAGESIZE, mmap.PAGESIZE)
    assert path.exists()


def test_fixed_size_keeps_the_file_size_across_slot_sizes(tmp_path: Path) -> None:
    path = tmp_path / "pool"
    size = 16 * mmap.PAGESIZE
    pool = RamPool(path, size, 3 * mmap.PAGESIZE, fixed_size=True)
    assert pool.slots_total == 5
    assert pool.size_bytes == size
    assert os.path.getsize(path) == size
    pool.close()
    pool = RamPool(path, size, 4 * mmap.PAGESIZE, fixed_size=True)
    assert pool.slots_total == 4
    assert os.path.getsize(path) == size
    pool.close()


# ---- sizing against free memory (production gate, window 2) -----------------------


def _meminfo(tmp_path: Path, total_gib: float, avail_gib: float) -> Path:
    p = tmp_path / "meminfo"
    p.write_text(
        f"MemTotal:       {int(total_gib * GIB) // 1024} kB\n"
        f"MemFree:        {int(avail_gib * GIB) // 2048} kB\n"
        f"MemAvailable:   {int(avail_gib * GIB) // 1024} kB\n"
    )
    return p


def _cgroup(tmp_path: Path, max_: str, current: int, inactive_file: int) -> Path:
    d = tmp_path / "cgroup"
    d.mkdir()
    (d / "memory.max").write_text(max_ + "\n")
    (d / "memory.current").write_text(f"{current}\n")
    (d / "memory.stat").write_text(f"anon 1\ninactive_file {inactive_file}\nactive_file 5\n")
    return d


def test_the_memory_margin_is_four_gib_or_a_tenth_of_total() -> None:
    assert ram.MEMORY_MARGIN_MIN_BYTES == 4 * GIB
    assert ram.MIN_POOL_BYTES == GIB
    assert ram.memory_margin(16 * GIB) == 4 * GIB
    assert ram.memory_margin(100 * GIB) == 10 * GIB


def test_fit_memory_keeps_the_margin_free(tmp_path: Path) -> None:
    """Window 2: a 16 GiB pool on a VM with ~9 GiB available."""
    mi = _meminfo(tmp_path, total_gib=64, avail_gib=9)
    nowhere = tmp_path / "no-cgroup"
    # margin = max(4 GiB, 6.4 GiB) = 6.4 GiB: 2.6 GiB fits
    got = ram.fit_memory(16 * GIB, meminfo=mi, cgroup=nowhere)
    assert got == 9 * GIB - int(64 * GIB * 0.10)
    assert ram.fit_memory(GIB, meminfo=mi, cgroup=nowhere) == GIB  # fits: untouched


def test_fit_memory_honours_a_cgroup_limit(tmp_path: Path) -> None:
    mi = _meminfo(tmp_path, total_gib=64, avail_gib=40)
    cg = _cgroup(tmp_path, str(20 * GIB), current=15 * GIB, inactive_file=GIB)
    # cgroup: total 20 GiB, available 20 - 15 + 1 = 6 GiB, margin max(4, 2) = 4 GiB
    assert ram.fit_memory(16 * GIB, meminfo=mi, cgroup=cg) == 2 * GIB


def test_fit_memory_ignores_an_unlimited_cgroup(tmp_path: Path) -> None:
    mi = _meminfo(tmp_path, total_gib=64, avail_gib=40)
    cg = _cgroup(tmp_path, "max", current=15 * GIB, inactive_file=GIB)
    assert ram.fit_memory(16 * GIB, meminfo=mi, cgroup=cg) == 16 * GIB


def test_fit_memory_is_a_no_op_when_memory_cannot_be_read(tmp_path: Path) -> None:
    assert ram.fit_memory(16 * GIB, meminfo=tmp_path / "x", cgroup=tmp_path / "y") == 16 * GIB


def test_grant_takes_the_smallest_limit_and_turns_off_under_the_minimum() -> None:
    page = mmap.PAGESIZE
    assert ram.grant(16 * GIB, shm_fit=16 * GIB, mem_fit=16 * GIB) == 16 * GIB
    assert ram.grant(16 * GIB, shm_fit=10 * GIB, mem_fit=3 * GIB + 5) == 3 * GIB
    assert ram.grant(16 * GIB, shm_fit=3 * GIB, mem_fit=10 * GIB) == 3 * GIB
    assert ram.grant(16 * GIB, shm_fit=16 * GIB, mem_fit=GIB - page) == 0  # under 1 GiB: off
    # a small pool that fits is never turned off by the minimum
    assert ram.grant(8 * page, shm_fit=16 * GIB, mem_fit=16 * GIB) == 8 * page
    assert ram.grant(8 * page, shm_fit=16 * GIB, mem_fit=4 * page) == 0


def test_create_pool_file_commits_every_page_in_a_new_file(tmp_path: Path) -> None:
    path = tmp_path / "pool"
    ram.create_pool_file(path, 8 * mmap.PAGESIZE)
    st = os.stat(path)
    assert st.st_size == 8 * mmap.PAGESIZE
    assert st.st_blocks * 512 >= st.st_size
    assert st.st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):  # never reuses a file someone else made
        ram.create_pool_file(path, 8 * mmap.PAGESIZE)


def test_a_precommitted_pool_does_not_allocate_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pool"
    size = 16 * mmap.PAGESIZE
    ram.create_pool_file(path, size)

    def boom(*_a: object) -> int:
        raise AssertionError("allocated again")

    monkeypatch.setattr(ram.os, "posix_fallocate", boom, raising=False)
    monkeypatch.setattr(ram.os, "pwrite", boom)
    pool = RamPool(path, size, 4 * mmap.PAGESIZE, fixed_size=True, precommitted=True)
    assert pool.slots_total == 4
    pool.close()


# ---- review of !683: engine reserve, cgroup path from /proc/self/cgroup -------------


def test_fit_memory_keeps_a_reserve_per_engine_worker(tmp_path: Path) -> None:
    """The pool is sized before any engine exists: its workers' host memory is kept free."""
    assert ram.ENGINE_WORKER_RESERVE_BYTES == 4 * GIB
    mi = _meminfo(tmp_path, total_gib=64, avail_gib=40)
    nowhere = tmp_path / "no-cgroup"
    margin = int(64 * GIB * 0.10)
    assert ram.fit_memory(64 * GIB, meminfo=mi, cgroup=nowhere) == 40 * GIB - margin
    got = ram.fit_memory(64 * GIB, engine_workers=4, meminfo=mi, cgroup=nowhere)
    assert got == 40 * GIB - margin - 16 * GIB


def _proc_cgroup(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "proc-self-cgroup"
    p.write_text(text)
    return p


def test_the_cgroup_v2_path_comes_from_proc_self_cgroup(tmp_path: Path) -> None:
    """Without a cgroup namespace the process's cgroup is a subdirectory of the root."""
    mi = _meminfo(tmp_path, total_gib=64, avail_gib=40)
    root = tmp_path / "cg"
    sub = root / "kubepods" / "pod1"
    sub.mkdir(parents=True)
    (sub / "memory.max").write_text(f"{20 * GIB}\n")
    (sub / "memory.current").write_text(f"{15 * GIB}\n")
    (sub / "memory.stat").write_text(f"inactive_file {GIB}\n")
    proc = _proc_cgroup(tmp_path, "0::/kubepods/pod1\n")
    got = ram.fit_memory(16 * GIB, meminfo=mi, cgroup=root, proc_cgroup=proc)
    assert got == 2 * GIB  # 20 - 15 + 1 - 4


def test_the_cgroup_root_is_used_when_the_process_sees_itself_at_slash(tmp_path: Path) -> None:
    """With a cgroup namespace (a container) /proc/self/cgroup says ``0::/``."""
    mi = _meminfo(tmp_path, total_gib=64, avail_gib=40)
    root = _cgroup(tmp_path, str(20 * GIB), current=15 * GIB, inactive_file=GIB)
    proc = _proc_cgroup(tmp_path, "0::/\n")
    assert ram.fit_memory(16 * GIB, meminfo=mi, cgroup=root, proc_cgroup=proc) == 2 * GIB


def test_a_cgroup_v1_memory_limit_is_honoured(tmp_path: Path) -> None:
    mi = _meminfo(tmp_path, total_gib=64, avail_gib=40)
    root = tmp_path / "cg1"
    d = root / "memory" / "kubepods" / "pod2"
    d.mkdir(parents=True)
    (d / "memory.limit_in_bytes").write_text(f"{20 * GIB}\n")
    (d / "memory.usage_in_bytes").write_text(f"{15 * GIB}\n")
    (d / "memory.stat").write_text(f"cache 1\ntotal_inactive_file {GIB}\n")
    proc = _proc_cgroup(tmp_path, "11:cpu,cpuacct:/kubepods/pod2\n4:memory:/kubepods/pod2\n")
    assert ram.fit_memory(16 * GIB, meminfo=mi, cgroup=root, proc_cgroup=proc) == 2 * GIB
    # an unlimited v1 group writes a huge number
    (d / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    assert ram.fit_memory(16 * GIB, meminfo=mi, cgroup=root, proc_cgroup=proc) == 16 * GIB
