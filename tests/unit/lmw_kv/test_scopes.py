"""Tenant, isolation scope, scope tag, GPU salt and model tag derivation."""

import pytest

from lmw_kv.scopes import (
    GPU_SALT_PREFIX,
    INTERNAL_TENANT_ID,
    MODEL_TAG_BYTES,
    SCOPE_BYTES,
    SCOPE_TAG_BYTES,
    TENANT_BYTES,
    gpu_cache_salt,
    isolation_scope,
    model_tag,
    scope_tag,
    tenant_tag,
)

NK = bytes(range(32))

# Computed independently with openssl (see lmw_kv/PROTOCOL.md, "Tenants and isolation
# scopes"), never with the module under test.
KAT_TENANT = "cc1426b65e852de2c716fa6e209b70e5"
KAT_SCOPE = "16077d3e8febec67e6006b804025ebe8bd8dc505c7187edfd737362d9e06f3ce"
KAT_SCOPE_TAG = "e59b05605013afe4e2b61c139735fd4d"
KAT_GPU_SALT = "lmw1:e0438dafb1f71eefab002a6908359e0484c4d494676d89650f025b1b8ed1cf13"
KAT_GPU_SALT_SHARING = "lmw1:a49883a97abbb59ce52fbbb41b8afb5ea9e2576b3c35cf5233aab7efc474191f"
KAT_MODEL = "6f466aae377f940c46bc6ad6566d9b46"


def test_constants() -> None:
    assert (TENANT_BYTES, SCOPE_BYTES, SCOPE_TAG_BYTES, MODEL_TAG_BYTES) == (16, 32, 16, 16)
    assert GPU_SALT_PREFIX == "lmw1:"
    assert INTERNAL_TENANT_ID == "internal"


def test_lengths_and_domains() -> None:
    assert len(tenant_tag(NK, "t1")) == 16
    assert len(isolation_scope(NK, "t1", 0)) == 32
    assert scope_tag(b"") == b"" and len(scope_tag(isolation_scope(NK, "t1", 0))) == 16
    assert isolation_scope(NK, "t1", 0) != isolation_scope(NK, "t1", 1)
    assert isolation_scope(NK, "t1", 0) != isolation_scope(NK, "t2", 0)
    assert tenant_tag(NK, "t1") != isolation_scope(NK, "t1", 0)[:16]


def test_keyed_by_network_key() -> None:
    other = bytes(32)
    assert tenant_tag(NK, "t1") != tenant_tag(other, "t1")
    assert isolation_scope(NK, "t1", 0) != isolation_scope(other, "t1", 0)
    assert gpu_cache_salt(NK, b"", "x") != gpu_cache_salt(other, b"", "x")


def test_gpu_salt_separates_keys_and_forms() -> None:
    a, c = isolation_scope(NK, "a", 0), isolation_scope(NK, "c", 0)
    assert gpu_cache_salt(NK, a, "x").startswith("lmw1:")
    assert gpu_cache_salt(NK, a, "x") == gpu_cache_salt(NK, a, "x")
    assert (
        len(
            {
                gpu_cache_salt(NK, a, "x"),
                gpu_cache_salt(NK, c, "x"),
                gpu_cache_salt(NK, b"", "x"),
                gpu_cache_salt(NK, a, ""),
            }
        )
        == 4
    )
    a_salt = gpu_cache_salt(NK, a, "")
    assert gpu_cache_salt(NK, b"", a_salt) != a_salt  # a sharing client replaying A's salt


def test_model_tag_separates_owner_and_revision() -> None:
    assert len(model_tag("m1", "r1")) == 16
    assert model_tag("m1", "r1") != model_tag("m1", "r2")
    assert model_tag("m1", "r1") != model_tag("m2", "r1")


def test_bad_inputs_raise() -> None:
    with pytest.raises(ValueError):
        isolation_scope(NK, "t1", -1)
    with pytest.raises(ValueError):
        tenant_tag(bytes(31), "t1")
    with pytest.raises(ValueError):
        isolation_scope(bytes(33), "t1", 0)
    with pytest.raises(ValueError):
        gpu_cache_salt(b"", b"", "x")
    with pytest.raises(ValueError):
        gpu_cache_salt(NK, bytes(256), "x")


def test_known_answers() -> None:
    assert tenant_tag(NK, "t1").hex() == KAT_TENANT
    assert isolation_scope(NK, "t1", 0).hex() == KAT_SCOPE
    assert scope_tag(bytes.fromhex(KAT_SCOPE)).hex() == KAT_SCOPE_TAG
    assert gpu_cache_salt(NK, bytes.fromhex(KAT_SCOPE), "abc") == KAT_GPU_SALT
    assert gpu_cache_salt(NK, b"", "lmw1:x") == KAT_GPU_SALT_SHARING
    assert model_tag("m1", "851bf6e8").hex() == KAT_MODEL


def test_empty_tenant_id_fails_closed() -> None:
    with pytest.raises(ValueError):
        tenant_tag(NK, "")
    with pytest.raises(ValueError):
        isolation_scope(NK, "", 0)


def test_epoch_must_be_an_int() -> None:
    with pytest.raises(ValueError):
        isolation_scope(NK, "t1", True)
    with pytest.raises(ValueError):
        isolation_scope(NK, "t1", 1.0)  # type: ignore[arg-type]


def test_model_tag_rejects_nul() -> None:
    with pytest.raises(ValueError):
        model_tag("a\x00b", "c")
    with pytest.raises(ValueError):
        model_tag("a", "b\x00c")


def test_bad_client_salt_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        gpu_cache_salt(NK, b"", "\ud800")
