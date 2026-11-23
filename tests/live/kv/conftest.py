"""tests/live/kv: the env contract for the KV-cache tier GPU tests.

Everything comes from ``VW_KV_LIVE_*`` env vars, sourced from a private env file
outside the repo. A missing required var skips the whole directory with the list
of what is missing. ``tests/live/conftest.py`` adds the ``live`` marker, so a bare
``pytest`` never collects these.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

#: Required. VW_KV_LIVE_ACK_RELOADS must be exactly "1": the run unloads and
#: reloads the model several times and changes its kv_store and revision.
REQUIRED = (
    "VW_KV_LIVE_BASE_URL",
    "VW_KV_LIVE_ADMIN_TOKEN",
    "VW_KV_LIVE_MODEL",
)


def missing_vars(env: dict[str, str] | None = None) -> list[str]:
    e = os.environ if env is None else env
    out = [v for v in REQUIRED if not e.get(v)]
    if e.get("VW_KV_LIVE_ACK_RELOADS") != "1":
        out.append("VW_KV_LIVE_ACK_RELOADS=1 (the run reloads the model)")
    return out


@dataclass
class KvLiveConfig:
    base_url: str
    admin_token: str = field(repr=False)
    model: str
    other_revision: str = ""
    kill_daemon_cmd: str = field(default="", repr=False)
    kill_delay_s: float = 0.1
    daemon_wait_s: float = 15.0
    log_cmd: str = field(default="", repr=False)
    break_nvme_cmd: str = field(default="", repr=False)
    sizes: tuple[int, ...] = (2048, 20000, 50000)
    max_tokens: int = 64
    min_speedup: float = 5.0
    tie_margin: float = 0.01
    load_timeout_s: float = 1200.0
    request_timeout_s: float = 900.0
    verify_tls: bool = True
    report_dir: Path = Path("~/.local/state/vw-kv-live").expanduser()
    # Plan 2 (test_kv_isolation_live.py)
    daemon_alive_cmd: str = field(default="", repr=False)
    api_log_cmd: str = field(default="", repr=False)
    segments_cmd: str = field(default="", repr=False)
    network_key_file: str = field(default="", repr=False)
    db_path: str = field(default="", repr=False)
    capture_dir: str = field(default="", repr=False)
    log_dir: str = field(default="", repr=False)
    g12_hf_repo: str = ""
    g12_revision: str = ""
    g12_gpu_util: float = 0.4
    g12_gpus: tuple[int, ...] = ()
    g12_dtype: str = "bfloat16"
    nvme_flush_s: float = 15.0
    gpu_case_size: int = 8192
    flood_factor: float = 10.0
    flood_max_tokens: int = 0


def load_config(env: dict[str, str] | None = None) -> KvLiveConfig:
    e = os.environ if env is None else env
    sizes = tuple(int(x) for x in e.get("VW_KV_LIVE_SIZES", "2048,20000,50000").split(",") if x)
    return KvLiveConfig(
        base_url=e["VW_KV_LIVE_BASE_URL"].rstrip("/"),
        admin_token=e["VW_KV_LIVE_ADMIN_TOKEN"],
        model=e["VW_KV_LIVE_MODEL"],
        other_revision=e.get("VW_KV_LIVE_OTHER_REVISION", ""),
        kill_daemon_cmd=e.get("VW_KV_LIVE_KILL_DAEMON_CMD", ""),
        kill_delay_s=float(e.get("VW_KV_LIVE_KILL_DELAY_S") or 0.1),
        daemon_wait_s=float(e.get("VW_KV_LIVE_DAEMON_WAIT_S") or 15.0),
        log_cmd=e.get("VW_KV_LIVE_LOG_CMD", ""),
        break_nvme_cmd=e.get("VW_KV_LIVE_BREAK_NVME_CMD", ""),
        sizes=sizes,
        min_speedup=float(e.get("VW_KV_LIVE_MIN_SPEEDUP") or 5.0),
        tie_margin=float(e.get("VW_KV_LIVE_TIE_MARGIN") or 0.01),
        load_timeout_s=float(e.get("VW_KV_LIVE_LOAD_TIMEOUT_S") or 1200.0),
        verify_tls=e.get("VW_KV_LIVE_VERIFY_TLS", "1") != "0",
        report_dir=Path(e.get("VW_KV_LIVE_REPORT_DIR") or "~/.local/state/vw-kv-live").expanduser(),
        daemon_alive_cmd=e.get("VW_KV_LIVE_DAEMON_ALIVE_CMD", ""),
        api_log_cmd=e.get("VW_KV_LIVE_API_LOG_CMD", ""),
        segments_cmd=e.get("VW_KV_LIVE_SEGMENTS_CMD", ""),
        network_key_file=e.get("VW_KV_LIVE_NETWORK_KEY_FILE", ""),
        db_path=e.get("VW_KV_LIVE_DB_PATH", ""),
        capture_dir=e.get("VW_KV_LIVE_CAPTURE_DIR", ""),
        log_dir=e.get("VW_KV_LIVE_LOG_DIR", ""),
        g12_hf_repo=e.get("VW_KV_LIVE_G12_HF_REPO", ""),
        g12_revision=e.get("VW_KV_LIVE_G12_REVISION", ""),
        g12_gpu_util=float(e.get("VW_KV_LIVE_G12_GPU_UTIL") or 0.4),
        g12_gpus=tuple(int(x) for x in e.get("VW_KV_LIVE_G12_GPUS", "").split(",") if x.strip()),
        g12_dtype=e.get("VW_KV_LIVE_G12_DTYPE") or "bfloat16",
        nvme_flush_s=float(e.get("VW_KV_LIVE_NVME_FLUSH_S") or 15.0),
        gpu_case_size=int(e.get("VW_KV_LIVE_GPU_CASE_SIZE") or 8192),
        flood_factor=float(e.get("VW_KV_LIVE_FLOOD_FACTOR") or 10.0),
        flood_max_tokens=int(e.get("VW_KV_LIVE_FLOOD_MAX_TOKENS") or 0),
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    missing = missing_vars()
    if not missing:
        return
    skip = pytest.mark.skip(reason=f"tests/live/kv: missing env: {', '.join(missing)}")
    for item in items:
        if "/tests/live/kv/" in "/" + str(item.path).replace("\\", "/"):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def kv_cfg() -> KvLiveConfig:
    missing = missing_vars()
    if missing:
        pytest.skip(f"tests/live/kv: missing env: {', '.join(missing)}")
    return load_config()
