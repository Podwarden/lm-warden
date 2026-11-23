"""Response models of the KV cache reports (``app.kv.usage``), for typed api-types.

A holder's route is declared with ``KvKeyReport``, so FastAPI's response filtering also
drops any admin-only field that reached the body by mistake.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class KvWindow(BaseModel):
    """One window of a key's turns whose GPU tokens are known. ``ram`` and ``nvme`` are
    null when the store reported on none of them."""

    gpu: int
    ram: int | None
    nvme: int | None
    recomputed: int
    hit_rate: float | None


class KvWindows(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    h1: KvWindow = Field(alias="1h")
    h24: KvWindow = Field(alias="24h")
    d7: KvWindow = Field(alias="7d")


class KvEvictedBy(BaseModel):
    token_id: str
    name: str
    count: int


class KvKeyStore(BaseModel):
    """The key's own store figures, since the store started (``since``, UTC)."""

    ram_bytes: int
    nvme_bytes: int
    #: The key's fair share of the tier, a fraction of its capacity.
    ram_share: float
    nvme_share: float
    ram_evictions: int
    nvme_evictions: int
    since: str | None


class KvKeyStoreAdmin(KvKeyStore):
    ram_share_bytes: int
    nvme_share_bytes: int
    evicted_by: list[KvEvictedBy]


class KvIsolationCost(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    h24: int = Field(alias="24h")


class KvKeyReport(BaseModel):
    """``GET /api/me/key/cache``: what the key holder sees."""

    isolation: Literal["sharing", "isolated"]
    windows: KvWindows
    store: KvKeyStore | None


class TokenCacheOut(BaseModel):
    """``GET /api/tokens/{id}/cache``: the prompt-cache observation plus the key's admin
    KV report."""

    token_id: str
    range: str
    since_epoch: float
    models: list[dict[str, Any]]
    isolation: Literal["sharing", "isolated"]
    windows: KvWindows
    store: KvKeyStoreAdmin | None
    isolation_cost_tokens: KvIsolationCost


class KvTier(BaseModel):
    capacity: int
    used: int
    dead: int


class KvTiers(BaseModel):
    ram: KvTier
    nvme: KvTier


class KvByTier(BaseModel):
    ram: int
    nvme: int


class KvUsed(KvByTier):
    total: int


class KvTenantUsage(BaseModel):
    token_id: str
    name: str
    used: KvUsed
    #: Fair share in bytes.
    share: KvByTier
    evictions: KvByTier
    evicted_by: list[KvEvictedBy]


class KvModelUsage(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    model_ids: list[str]
    weight: int
    part: KvByTier
    used: KvUsed
    tenants: list[KvTenantUsage]


class KvUsage(BaseModel):
    """``GET /api/kv/usage``: figures since the store started (``since``, UTC)."""

    available: bool
    policy: str | None
    since: str | None
    tiers: KvTiers | None
    models: list[KvModelUsage]
