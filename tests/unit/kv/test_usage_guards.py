"""``GET /api/kv/usage`` carries the store's pool and disk errors (Rulings R16, R17)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.kv.usage import kv_usage, nvme_capacity, store_errors
from tests.unit.kv.test_usage import _Client, _routes, _stats, _Sup

POOL = "RAM pool 8 GiB does not fit /dev/shm 2 GiB"
DISK = (
    "NVMe tier off: /tmp is on the root filesystem (overlay); mount a dedicated disk there, "
    "or set kv_store_nvme_allow_root_disk"
)


class _RefusedSup(_Sup):
    """A supervisor that started no daemon: the pool did not fit and there is no folder."""

    def pool_error(self) -> str | None:
        return POOL


def test_store_errors_read_the_stats_reply() -> None:
    assert store_errors({"pool_error": POOL, "nvme_error": DISK}, None) == {
        "pool_error": POOL,
        "nvme_error": DISK,
    }
    assert store_errors({}, None) == {"pool_error": None, "nvme_error": None}  # an older daemon
    assert store_errors({"pool_error": 3, "nvme_error": ["x"]}, None) == {
        "pool_error": None,
        "nvme_error": None,
    }


def test_store_errors_fall_back_to_the_supervisor() -> None:
    sup = SimpleNamespace(pool_error=lambda: POOL)
    assert store_errors(None, sup) == {"pool_error": POOL, "nvme_error": None}
    assert store_errors({"pool_error": None}, sup)["pool_error"] is None  # the daemon's word
    assert store_errors(None, SimpleNamespace()) == {"pool_error": None, "nvme_error": None}


def test_usage_route_carries_the_daemons_errors(tmp_data_dir: Any, client: Any) -> None:
    _plain, admin = _routes(client, tmp_data_dir)
    nk = client.app.state.cobs2_key.key
    st = {**_stats(nk), "pool_error": POOL, "nvme_error": DISK}
    client.app.state.kv_store = _Sup(_Client(st))
    client.app.state.kv_stats_cache = None
    u = client.get("/api/kv/usage", headers=admin).json()
    assert u["available"] is True
    assert u["pool_error"] == POOL and u["nvme_error"] == DISK


def test_usage_route_has_null_errors_when_fine(tmp_data_dir: Any, client: Any) -> None:
    _plain, admin = _routes(client, tmp_data_dir)
    u = client.get("/api/kv/usage", headers=admin).json()
    assert u["pool_error"] is None and u["nvme_error"] is None


def test_usage_route_shows_a_refused_start(tmp_data_dir: Any, client: Any) -> None:
    _plain, admin = _routes(client, tmp_data_dir, None)
    client.app.state.kv_store = _RefusedSup(None)
    client.app.state.kv_stats_cache = None
    u = client.get("/api/kv/usage", headers=admin).json()
    assert u["available"] is False and u["pool_error"] == POOL and u["nvme_error"] is None


def test_the_nvme_capacity_is_the_cap_in_force() -> None:
    """R17: ``nvme_cap_bytes`` is the operator's cap; the store enforces the effective one."""
    assert nvme_capacity({"nvme_cap_bytes": 250, "nvme_cap_effective_bytes": 34}) == 34
    assert nvme_capacity({"nvme_cap_bytes": 250}) == 250  # an older daemon
    st = {**_stats(b"k" * 32), "nvme_cap_bytes": 250, "nvme_cap_effective_bytes": 34}
    assert kv_usage(st, {})["tiers"]["nvme"]["capacity"] == 34
