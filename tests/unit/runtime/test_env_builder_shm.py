"""Engine shared-memory handling in env_builder (#210).

TP>1 vLLM engines mmap their shm_broadcast ring out of /dev/shm. tmpfs backs
those pages lazily, so an undersized /dev/shm does not fail at mmap time — the
engine takes an uncatchable SIGBUS in shm_broadcast.enqueue minutes into
serving. The docker driver enlarges the engine container's /dev/shm; the
local-subprocess driver cannot, so it warns at startup when the pod's is too
small. The real fix is a `medium: Memory` emptyDir on the pod (core #2417).
"""

import logging
import os

from app.runtime.backends.vllm.env import (
    ENGINE_SHM_MIN_BYTES,
    build_subprocess_env,
    dev_shm_bytes,
    warn_if_shm_undersized,
)


def _model(extra_env: dict | None = None) -> object:
    class M:
        id = "tp4"
        gpu_indices = [0, 1, 2, 3]
        tensor_parallel_size = 4

    M.extra_env = extra_env or {}
    return M()


class _FakeStatvfs:
    def __init__(self, total_bytes: int) -> None:
        self.f_frsize = 4096
        self.f_blocks = total_bytes // 4096


def _fake_shm(monkeypatch, total_bytes: int | None) -> None:
    """Make os.statvfs report a /dev/shm of ``total_bytes`` (None => missing)."""

    def _statvfs(path):
        if total_bytes is None:
            raise FileNotFoundError(path)
        return _FakeStatvfs(total_bytes)

    monkeypatch.setattr(os, "statvfs", _statvfs)


# ---------------------------------------------------------------------------
# No chunk-size cap by default
# ---------------------------------------------------------------------------


def test_no_mq_chunk_cap_by_default_under_either_driver():
    """The 4 MB cap was a stop-gap for a 64 MiB pod. With a real /dev/shm on
    the pod (core #2417) it only costs throughput — multimodal payloads above
    the chunk limit take vLLM's slower zmq overflow path — so the builder must
    not reintroduce it silently."""
    for driver in ("local", "docker"):
        env = build_subprocess_env(
            _model(), hf_token="tok", hf_cache_dir="/d", engine_driver=driver
        )
        assert "VLLM_MQ_MAX_CHUNK_BYTES_MB" not in env, driver


def test_extra_env_can_still_set_the_mq_chunk_cap():
    """A deployment that cannot get real shared memory keeps the per-model
    escape hatch: the key is allowed, not hard-locked."""
    env = build_subprocess_env(
        _model({"VLLM_MQ_MAX_CHUNK_BYTES_MB": "4"}),
        hf_token="tok",
        hf_cache_dir="/d",
    )
    assert env["VLLM_MQ_MAX_CHUNK_BYTES_MB"] == "4"


# ---------------------------------------------------------------------------
# Reading /dev/shm
# ---------------------------------------------------------------------------


def test_dev_shm_bytes_reads_the_filesystem_size(monkeypatch):
    _fake_shm(monkeypatch, 8 * 1024**3)
    assert dev_shm_bytes() == 8 * 1024**3


def test_dev_shm_bytes_returns_none_when_the_path_is_missing(tmp_path):
    """Degrade gracefully: a missing path is "unknown", never "too small"."""
    assert dev_shm_bytes(str(tmp_path / "nope")) is None


# ---------------------------------------------------------------------------
# The startup warning (DoD 3)
# ---------------------------------------------------------------------------


def test_warns_when_local_driver_has_an_undersized_shm(monkeypatch, caplog):
    _fake_shm(monkeypatch, 64 * 1024**2)  # the Kubernetes pod default
    with caplog.at_level(logging.WARNING, logger="app.runtime.backends.vllm.env"):
        warn_if_shm_undersized("local")
    assert len(caplog.records) == 1
    msg = caplog.records[0].getMessage()
    assert "/dev/shm" in msg
    assert "64 MiB" in msg
    assert "SIGBUS" in msg
    assert "#210" in msg


def test_no_warning_when_shm_is_large_enough(monkeypatch, caplog):
    _fake_shm(monkeypatch, ENGINE_SHM_MIN_BYTES)
    with caplog.at_level(logging.WARNING, logger="app.runtime.backends.vllm.env"):
        warn_if_shm_undersized("local")
    assert caplog.records == []


def test_no_warning_for_the_docker_driver(monkeypatch, caplog):
    """The warden's own /dev/shm says nothing about a sibling engine
    container's, so checking it there would only produce false alarms."""
    _fake_shm(monkeypatch, 64 * 1024**2)
    with caplog.at_level(logging.WARNING, logger="app.runtime.backends.vllm.env"):
        warn_if_shm_undersized("docker")
    assert caplog.records == []


def test_warns_when_shm_size_is_unreadable(monkeypatch, caplog):
    """Unknown is still worth saying out loud — but the message must not claim
    a size it never read."""
    _fake_shm(monkeypatch, None)
    with caplog.at_level(logging.WARNING, logger="app.runtime.backends.vllm.env"):
        warn_if_shm_undersized("local")
    assert len(caplog.records) == 1
    assert "could not read" in caplog.records[0].getMessage()
