"""Weighted soft shares and the over-share order (``lmw_kv.fair``, pure)."""

from __future__ import annotations

import dataclasses

import pytest

from lmw_kv.fair import (
    DEFAULT_OWNER_WEIGHT,
    DEFAULT_TENANT_WEIGHT,
    Policy,
    TenantPolicy,
    over_order,
    parts,
    shares,
)

A = b"A" * 16
B = b"B" * 16
t1, t2, t3, t4, t5 = (bytes([i]) * 16 for i in range(1, 6))
hi = b"h" * 16
lo = b"l" * 16


def test_policy_defaults_are_immutable() -> None:
    p = Policy()
    assert p.mode == "lru" and dict(p.owner_weights) == {} and dict(p.tenants) == {}
    assert p.idle_ram_s == 7200.0 and p.idle_nvme_s == 604800.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.mode = "fair"  # type: ignore[misc]
    with pytest.raises(TypeError):
        p.owner_weights["x"] = 2  # type: ignore[index]
    assert DEFAULT_TENANT_WEIGHT == 6 and DEFAULT_OWNER_WEIGHT == 1
    assert TenantPolicy() == TenantPolicy(weight=6, max_ram_pct=None, max_nvme_pct=None)
    assert p.tenant(t1) == TenantPolicy()


def test_two_level_shares() -> None:
    p = parts(100, {A: 1, B: 1})
    assert p == {A: 50.0, B: 50.0}
    u = {(A, t1): 60, (A, t2): 10, (B, t3): 30}
    s = shares(u, p, lambda t: 6)
    assert s[(A, t1)] == s[(A, t2)] == 25.0 and s[(B, t3)] == 50.0
    assert over_order(u, p, s) == [(A, t1)]


def test_priority_weights() -> None:
    s = shares({(A, hi): 1, (A, lo): 1}, {A: 110.0}, {hi: 10, lo: 1}.__getitem__)
    assert s == {(A, hi): 100.0, (A, lo): 10.0}


def test_nobody_over_gives_empty_order() -> None:
    u, p = {(A, t1): 50, (B, t3): 50}, parts(100, {A: 1, B: 1})
    assert over_order(u, p, shares(u, p, lambda t: 6)) == []


def test_phase_two_after_over_models() -> None:
    # A over its part (t1 over); B at its part, but t4 over its share inside B
    u, p = {(A, t1): 60, (B, t4): 35, (B, t5): 5}, parts(100, {A: 1, B: 1})
    assert over_order(u, p, shares(u, p, lambda t: 6)) == [(A, t1), (B, t4)]


def test_model_weights_split_the_tier() -> None:
    assert parts(120, {A: 1, B: 2}) == {A: 40.0, B: 80.0}
    assert parts(0, {A: 1}) == {A: 0.0}


def test_most_over_first_in_each_phase() -> None:
    # C holds nothing, so A and B are both over their parts; A by more. Inside A both
    # t1 and t2 are over; t5 is under its share in B, t4 over it.
    c = b"C" * 16
    u = {(A, t1): 30, (A, t2): 20, (A, t3): 5, (B, t4): 30, (B, t5): 10}
    p = parts(99, {A: 1, B: 1, c: 1})
    s = shares(u, p, lambda t: 6)
    assert over_order(u, p, s) == [(A, t1), (A, t2), (B, t4)]
    # phase 2 is most over first too
    u = {(A, t1): 60, (B, t4): 40, (B, t5): 5, (c, t3): 24, (c, t2): 1}
    p = parts(100, {A: 1, B: 2, c: 1})
    s = shares(u, p, lambda t: 6)
    assert over_order(u, p, s) == [(A, t1), (c, t3), (B, t4)]


def test_each_pair_at_most_once_and_never_at_share() -> None:
    u = {(A, t1): 25, (A, t2): 25, (B, t3): 50}
    p = parts(100, {A: 1, B: 1})
    assert over_order(u, p, shares(u, p, lambda t: 6)) == []
    u = {(A, t1): 80, (B, t3): 20}
    order = over_order(u, p, shares(u, p, lambda t: 6))
    assert order == [(A, t1)] and len(set(order)) == len(order)


def test_a_model_without_a_part_is_over() -> None:
    u = {(A, t1): 10, (B, t3): 10}
    p = {A: 100.0}  # B has no part
    assert over_order(u, p, shares(u, p, lambda t: 6)) == [(B, t3)]
