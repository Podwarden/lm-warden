"""Keyed tenant, isolation scope, scope tag, GPU cache salt and model tag.

Pure functions, no I/O. The encodings are defined in PROTOCOL.md ("Tenants and
isolation scopes"); the known-answer test in tests/unit/lmw_kv/test_scopes.py checks
them. Tenant tags and scopes are HMACs under the network key, so they reveal nothing
about the key id and nobody without the network key can compute another key's scope.
Callers must never log the values these functions return.
"""

from __future__ import annotations

import hashlib
import hmac
import struct

TENANT_BYTES = 16
SCOPE_BYTES = 32
SCOPE_TAG_BYTES = 16
MODEL_TAG_BYTES = 16
GPU_SALT_PREFIX = "lmw1:"
INTERNAL_TENANT_ID = "internal"

_NETWORK_KEY_BYTES = 32


def _check_key(network_key: bytes) -> None:
    if len(network_key) != _NETWORK_KEY_BYTES:
        raise ValueError("network key must be 32 bytes")


def _check_tenant_id(kv_tenant_id: str) -> None:
    # An empty id would give every such key the same scope and GPU salt: fail closed.
    if not kv_tenant_id:
        raise ValueError("kv_tenant_id must not be empty")


def _hmac(network_key: bytes, msg: bytes) -> bytes:
    _check_key(network_key)
    return hmac.digest(network_key, msg, "sha256")


def tenant_tag(network_key: bytes, kv_tenant_id: str) -> bytes:
    """16-byte accounting name of a key. Never part of a block key."""
    _check_tenant_id(kv_tenant_id)
    msg = b"lmw-tenant\x00v1\x00" + kv_tenant_id.encode("utf-8")
    return _hmac(network_key, msg)[:TENANT_BYTES]


def isolation_scope(network_key: bytes, kv_tenant_id: str, epoch: int) -> bytes:
    """32-byte scope of an isolated key at one epoch. A sharing key's scope is b""."""
    _check_tenant_id(kv_tenant_id)
    if type(epoch) is not int:
        raise ValueError("epoch must be an int")
    if epoch < 0:
        raise ValueError("epoch must be >= 0")
    msg = (
        b"lmw-scope\x00v1\x00" + kv_tenant_id.encode("utf-8") + b"\x00" + str(epoch).encode("ascii")
    )
    return _hmac(network_key, msg)


def scope_tag(scope: bytes) -> bytes:
    """16-byte public handle of a scope, so the daemon can retire it without seeing it."""
    if not scope:
        return b""
    return hashlib.sha256(b"lmw-scope-tag\x00" + scope).digest()[:SCOPE_TAG_BYTES]


def gpu_cache_salt(network_key: bytes, scope: bytes, client_salt: str) -> str:
    """Engine `cache_salt` for a request: `"lmw1:"` plus 64 hex chars.

    The scope length byte keeps the sharing form (empty scope) and the isolated form
    unambiguous; the client salt is length-prefixed as a u32 little-endian.

    A client salt that is not valid UTF-8 (a lone surrogate) raises UnicodeEncodeError,
    a ValueError. Callers must map ValueError to a client error, never a 500.
    """
    if len(scope) > 255:
        raise ValueError("scope longer than 255 bytes")
    cs = client_salt.encode("utf-8")
    msg = b"lmw-gpu\x00" + bytes([len(scope)]) + scope + struct.pack("<I", len(cs)) + cs
    return GPU_SALT_PREFIX + _hmac(network_key, msg).hex()


def model_tag(owner: str, revision: str) -> bytes:
    """16-byte name of a model (owner and revision) for per-model accounting."""
    if "\x00" in owner or "\x00" in revision:
        raise ValueError("owner and revision must not contain NUL")
    msg = b"lmw-model\x00" + owner.encode("utf-8") + b"\x00" + revision.encode("utf-8")
    return hashlib.sha256(msg).digest()[:MODEL_TAG_BYTES]
