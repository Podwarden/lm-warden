"""Response models of the KV cache reports (``app.kv.usage``), for typed api-types.

A holder's route is declared with ``KvKeyReport``, so FastAPI's response filtering also
drops any admin-only field that reached the body by mistake. The reuse models follow the
same split: ``KvReuseHolder`` and ``KvTopRow`` have no field for ``tenants_seen``, the
charged key, total reloads or regret (counters design 6.2 and 7.4).
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


class KvByTier(BaseModel):
    ram: int
    nvme: int


class KvHistBucket(BaseModel):
    positions: int
    bytes: int


class KvReuseHolder(BaseModel):
    """A key's reuse of what it saved, from the saver view only: "reused" means reloaded
    by the key itself, never by another key. ``positions``, ``positions_reused`` and
    ``reloads`` are of the positions stored now; the disk bytes are since the store
    started."""

    positions: int
    positions_reused: int
    #: ``positions_reused`` as a percent of ``positions``; null with none stored.
    reused_pct: float | None
    #: Reloads by the key itself of positions it saved.
    reloads: int
    nvme_bytes_written: int
    #: Disk bytes written for positions that left the disk never reloaded by their saver.
    never_reused_nvme_bytes: int
    half_life_s: float


class KvReuseAdmin(KvReuseHolder):
    """Adds the charged view (reloads by every key of the positions a key pays for now),
    the reads back from disk, the histogram of reloads at eviction (charged view: by
    every key) and regret."""

    charged_positions: KvByTier
    charged_positions_reused: KvByTier
    all_reloads: int
    ram_reloads: int
    nvme_reloads: int
    nvme_bytes_read: int
    #: tier -> bucket of reloads ("0", "1", "2-3", "4-7", "8+") -> what left that tier.
    evicted_hist: dict[str, dict[str, KvHistBucket]]
    regret: int


class KvTopRow(BaseModel):
    """One of the most reloaded cached parts, by its label (never a key or a tag)."""

    label: str
    #: The part's token offset; null when its depth or the block size is unknown.
    depth_tokens: int | None
    reloads: int


class KvTopRowAdmin(KvTopRow):
    decayed: float
    last_reload: str | None
    first_saved: str | None
    tiers: list[str]
    #: How many keys reloaded it: "1" to "3", or "4 or more".
    keys_seen: str
    charged_name: str


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
    #: Null: the store does not report reuse (an older daemon).
    reuse: KvReuseHolder | None = None


class KvKeyStoreAdmin(KvKeyStore):
    ram_share_bytes: int
    nvme_share_bytes: int
    evicted_by: list[KvEvictedBy]
    reuse: KvReuseAdmin | None = None


class KvIsolationCost(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    h24: int = Field(alias="24h")


class KvKeyReport(BaseModel):
    """``GET /api/me/key/cache``: what the key holder sees."""

    isolation: Literal["sharing", "isolated"]
    windows: KvWindows
    store: KvKeyStore | None
    #: The key's own most reloaded parts; null when the store cannot say.
    top_reused: list[KvTopRow] | None = None


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
    reuse: KvReuseAdmin | None = None


class KvModelUsage(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    model_ids: list[str]
    weight: int
    part: KvByTier
    used: KvUsed
    tenants: list[KvTenantUsage]
    #: The keys' reuse summed; null when the store does not report reuse.
    reuse: KvReuseAdmin | None = None
    #: The 10 most reloaded parts; null when the store cannot say (an older daemon).
    top_reused: list[KvTopRowAdmin] | None = None


class KvUsage(BaseModel):
    """``GET /api/kv/usage``: figures since the store started (``since``, UTC)."""

    available: bool
    policy: str | None
    since: str | None
    tiers: KvTiers | None
    models: list[KvModelUsage]
    #: Why the store has no RAM pool, e.g. "RAM pool 8 GiB does not fit /dev/shm 2 GiB";
    #: null when fine.
    pool_error: str | None = None
    #: Why the disk tier is off, e.g. a folder on the root disk; null when fine.
    nvme_error: str | None = None
