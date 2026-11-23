"""Weighted soft shares for the ``fair`` eviction policy (pure; no store, no I/O).

Two levels per tier. The tier's capacity is split between models by weight (a model's
``part``). A model's part is split between the tenants holding blocks in it, by weight
``priority + 1`` (a tenant's ``share``). Shares are soft: a tenant uses any free room,
and only the choice of a victim reads them (``over_order``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

#: A key's weight is its scheduler priority + 1; the default priority is 5.
DEFAULT_TENANT_WEIGHT = 6
#: A model's weight when none of its owners has one.
DEFAULT_OWNER_WEIGHT = 1

#: The keys ``Policy.from_wire`` reads (others are ignored): the document's, and each
#: tenant's. For a policy builder to test against.
WIRE_KEYS = frozenset({"mode", "idle_ram_s", "idle_nvme_s", "owners", "tenants"})
WIRE_TENANT_KEYS = frozenset({"weight", "max_ram_pct", "max_nvme_pct"})
#: Weights and caps on the wire are integers in this range.
WIRE_MIN, WIRE_MAX = 1, 100
#: Longest owner name, in UTF-8 bytes (as ``register_ns``).
OWNER_MAX = 255
TENANT_LEN = 16

_Pair = tuple[bytes, bytes]  # (model, tenant)


@dataclass(frozen=True)
class TenantPolicy:
    weight: int = DEFAULT_TENANT_WEIGHT
    #: Hard caps, a percentage of that tier's capacity; None is no cap.
    max_ram_pct: int | None = None
    max_nvme_pct: int | None = None


@dataclass(frozen=True)
class Policy:
    """The eviction policy. The mapping defaults are read-only ``MappingProxyType``s,
    made per instance: Python 3.11 rejects a ``mappingproxy`` as a plain default."""

    mode: Literal["lru", "fair"] = "lru"
    #: Seconds unused before a block loses its protection (class idle).
    idle_ram_s: float = 7200.0
    idle_nvme_s: float = 604800.0
    #: Owner -> model weight; a model takes the largest among its live owners.
    owner_weights: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    #: Tenant -> its policy; a tenant missing here gets ``TenantPolicy()``.
    tenants: Mapping[bytes, TenantPolicy] = field(default_factory=lambda: MappingProxyType({}))

    def tenant(self, tenant: bytes) -> TenantPolicy:
        return self.tenants.get(tenant, _DEFAULT_TENANT)

    @classmethod
    def from_wire(cls, doc: Mapping[str, Any]) -> Policy:
        """The policy of a ``set_policy`` request (PROTOCOL.md); raises ``ValueError``.

        Every key is optional (missing means the default) and unknown keys are ignored.
        The whole document is checked before a ``Policy`` exists, so a bad one changes
        nothing. Messages never contain a tenant tag or an owner name.
        """
        if not isinstance(doc, Mapping):
            raise ValueError("policy: object required")
        mode = doc.get("mode", "lru")
        if mode not in ("lru", "fair"):
            raise ValueError('policy.mode: "lru" or "fair" required')
        idle_ram = _seconds(doc, "idle_ram_s", 7200.0)
        idle_nvme = _seconds(doc, "idle_nvme_s", 604800.0)
        owners: dict[str, int] = {}
        for owner, w in _object(doc, "owners").items():
            if not isinstance(owner, str) or len(owner.encode("utf-8")) > OWNER_MAX:
                raise ValueError(f"policy.owners: a name is at most {OWNER_MAX} bytes")
            owners[owner] = _wire_int(w, "policy.owners: a weight")
        tenants: dict[bytes, TenantPolicy] = {}
        for tag, t in _object(doc, "tenants").items():
            try:
                raw = bytes.fromhex(tag) if isinstance(tag, str) else b""
            except ValueError:
                raw = b""
            if len(raw) != TENANT_LEN:
                raise ValueError(f"policy.tenants: a tenant is {TENANT_LEN} bytes of hex")
            if not isinstance(t, Mapping):
                raise ValueError("policy.tenants: each tenant is an object")
            tenants[raw] = TenantPolicy(
                weight=_wire_int(t.get("weight", DEFAULT_TENANT_WEIGHT), "policy.tenants: weight"),
                max_ram_pct=_cap(t, "max_ram_pct"),
                max_nvme_pct=_cap(t, "max_nvme_pct"),
            )
        return cls(
            mode=mode,
            idle_ram_s=idle_ram,
            idle_nvme_s=idle_nvme,
            owner_weights=MappingProxyType(owners),
            tenants=MappingProxyType(tenants),
        )


def _wire_int(v: Any, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or not WIRE_MIN <= v <= WIRE_MAX:
        raise ValueError(f"{what}: an integer {WIRE_MIN} to {WIRE_MAX} required")
    return int(v)


def _cap(t: Mapping[str, Any], key: str) -> int | None:
    v = t.get(key)
    return None if v is None else _wire_int(v, f"policy.tenants: {key}")


def _seconds(doc: Mapping[str, Any], key: str, default: float) -> float:
    v = doc.get(key, default)
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise ValueError(f"policy.{key}: a number of seconds required")
    f = float(v)
    if not math.isfinite(f) or f <= 0:
        raise ValueError(f"policy.{key}: a finite number above 0 required")
    return f


def _object(doc: Mapping[str, Any], key: str) -> Mapping[Any, Any]:
    v = doc.get(key, {})
    if not isinstance(v, Mapping):
        raise ValueError(f"policy.{key}: object required")
    return v


_DEFAULT_TENANT = TenantPolicy()


def parts(capacity: int, model_weights: Mapping[bytes, int]) -> dict[bytes, float]:
    """Each live model's part of ``capacity``: ``capacity * w / sum(w)``."""
    total = sum(model_weights.values())
    if total <= 0:
        return {m: 0.0 for m in model_weights}
    return {m: capacity * w / total for m, w in model_weights.items()}


def shares(
    usage: Mapping[_Pair, int],
    parts: Mapping[bytes, float],
    weight: Callable[[bytes], int],
) -> dict[_Pair, float]:
    """Each ``(model, tenant)`` pair's share of its model's part.

    Within a model the part is split among the tenants holding blocks in it (usage
    above zero), by ``weight``. A model with no part gives every tenant a share of 0.
    """
    tenants: dict[bytes, list[bytes]] = {}
    for (model, tenant), n in usage.items():
        if n > 0:
            tenants.setdefault(model, []).append(tenant)
    out: dict[_Pair, float] = {}
    for model, ts in tenants.items():
        ws = {t: weight(t) for t in ts}
        total = sum(ws.values())
        part = parts.get(model, 0.0)
        for t, w in ws.items():
            out[(model, t)] = part * w / total if total > 0 else 0.0
    return out


def _ratio(used: float, quota: float) -> float:
    if quota > 0:
        return used / quota
    return float("inf") if used > 0 else 0.0


def over_order(
    usage: Mapping[_Pair, int],
    parts: Mapping[bytes, float],
    shares: Mapping[_Pair, float],
) -> list[_Pair]:
    """The pairs to take victims from, in order, each at most once.

    Phase 1: models over their part, most over first, and inside each its tenants over
    their share, most over first. Phase 2: every other tenant over its share in any
    model, most over first. A pair at or under its share never appears. Ties go by
    the model, then the tenant bytes, so the order is deterministic.
    """
    model_use: dict[bytes, int] = {}
    for (model, _), n in usage.items():
        model_use[model] = model_use.get(model, 0) + n
    over: dict[_Pair, float] = {}
    for pair, n in usage.items():
        r = _ratio(n, shares.get(pair, 0.0))
        if r > 1.0:
            over[pair] = r
    over_models = sorted(
        (m for m, n in model_use.items() if _ratio(n, parts.get(m, 0.0)) > 1.0),
        key=lambda m: (-_ratio(model_use[m], parts.get(m, 0.0)), m),
    )
    order: list[_Pair] = []
    taken: set[_Pair] = set()
    for model in over_models:
        inside = sorted((p for p in over if p[0] == model), key=lambda p: (-over[p], p))
        order.extend(inside)
        taken.update(inside)
    order.extend(sorted((p for p in over if p not in taken), key=lambda p: (-over[p], p)))
    return order
