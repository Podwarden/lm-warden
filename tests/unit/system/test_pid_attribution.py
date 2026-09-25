# NOTE: these tests assume PIDs reported by nvidia-smi and PIDs tracked by
# the supervisor live in the same PID namespace. In production that
# invariant holds because the api container is started with
# `pid: host` (compose) / `hostPID: true` (K8s); see docker-compose.yml,
# the Hub catalogue compose and the Phase 1 entry in changelog.md. Without
# that flag, nvidia-smi returns host PIDs while the supervisor stores
# in-container PIDs and `attribute_pid_to_model` correctly falls back to
# returning None (callers then label the holder kind: external). The
# `test_attribute_pid_orphan_falls_back_to_external` case below pins that
# crash-recovery / pid-namespace-mismatch fallback behaviour.

from pathlib import Path

from app.system.pid_attribution import attribute_pid_to_model, read_pgid


def _write_stat(proc_root: Path, pid: int, *, comm: str, pgrp: int) -> None:
    """Write a fake /proc/<pid>/stat with the same layout the kernel emits.

    Layout: ``pid (comm) state ppid pgrp …`` — comm may contain parens, so we
    deliberately wrap it in parens to exercise the rfind(')') path.
    """
    d = proc_root / str(pid)
    d.mkdir(parents=True, exist_ok=True)
    # Use ppid=1 (init); pgrp is the field under test.
    (d / "stat").write_text(f"{pid} ({comm}) S 1 {pgrp} 1 1 1 1\n")


def test_read_pgid_parses_kernel_stat(tmp_path: Path) -> None:
    _write_stat(tmp_path, 4242, comm="vllm-worker", pgrp=4200)
    assert read_pgid(4242, proc_root=tmp_path) == 4200


def test_read_pgid_handles_comm_with_parens(tmp_path: Path) -> None:
    # ``foo (bar) baz`` — the inner ')' would trip a naïve split('(')[1].
    _write_stat(tmp_path, 99, comm="foo (bar) baz", pgrp=77)
    assert read_pgid(99, proc_root=tmp_path) == 77


def test_read_pgid_missing_returns_none(tmp_path: Path) -> None:
    assert read_pgid(123456, proc_root=tmp_path) is None


def test_attribute_pid_direct_parent_hit(tmp_path: Path) -> None:
    # The PID itself is the parent we spawned.
    parent_to_model = {1000: "model-a"}
    assert attribute_pid_to_model(1000, parent_to_model, proc_root=tmp_path) == "model-a"


def test_attribute_pid_via_pgrp(tmp_path: Path) -> None:
    # vLLM tensor-parallel worker: pgrp == parent's pid.
    _write_stat(tmp_path, 1234, comm="vllm-worker", pgrp=1000)
    parent_to_model = {1000: "model-a"}
    assert attribute_pid_to_model(1234, parent_to_model, proc_root=tmp_path) == "model-a"


def test_attribute_pid_unknown_returns_none(tmp_path: Path) -> None:
    # /proc entry exists but pgrp doesn't match any tracked parent — external.
    _write_stat(tmp_path, 4321, comm="Xorg", pgrp=4000)
    parent_to_model = {1000: "model-a"}
    assert attribute_pid_to_model(4321, parent_to_model, proc_root=tmp_path) is None


def test_attribute_pid_no_proc_returns_none(tmp_path: Path) -> None:
    # Process already exited; treat as external.
    parent_to_model = {1000: "model-a"}
    assert attribute_pid_to_model(9999, parent_to_model, proc_root=tmp_path) is None


def test_attribute_pid_orphan_falls_back_to_external(tmp_path: Path) -> None:
    """Defence test for crash-recovery and pid-namespace mismatch.

    Scenarios this covers:

    * Supervisor restarted while a vLLM subprocess survived: nvidia-smi still
      reports the running PID, but the in-memory parent_pid_to_model map has
      been rebuilt from scratch and no longer contains it.
    * Container started without ``pid: host`` / ``hostPID: true``: nvidia-smi
      returns the *host* PID while the supervisor only knows the *in-container*
      PID. The host PID's /proc entry inside the container is missing.

    In both cases the holder should degrade gracefully to ``None`` (caller
    labels it ``kind: external``) — never crash, never silently mis-attribute
    to the wrong model.
    """
    # /proc entry simulating "doesn't exist inside this PID namespace" — we
    # write nothing for pid 555555.
    parent_to_model = {1000: "model-a"}
    assert attribute_pid_to_model(555555, parent_to_model, proc_root=tmp_path) is None

    # Same orphan condition but /proc DOES list the pid with a pgrp that
    # doesn't match anything we track (e.g. tensor-parallel worker whose
    # parent already died and was re-parented to init). Still external.
    _write_stat(tmp_path, 555556, comm="vllm-worker", pgrp=42)
    assert attribute_pid_to_model(555556, parent_to_model, proc_root=tmp_path) is None
