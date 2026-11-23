"""Namespace ids and chained, keyed block keys.

Block keys are HMAC-SHA-256 over (salt, namespace, parent key, token ids), so a
key commits to the whole prefix and cannot be forged without the secret.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import struct
from collections.abc import Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Namespace:
    """Everything that makes KV blocks interchangeable between engines."""

    model_repo: str
    revision_sha: str
    tokenizer_sha: str
    kv_dtype: str
    block_size: int
    attention_layout: str
    kv_groups: tuple[str, ...]
    tp: int
    pp: int
    engine_family: str
    layout_version: int


def namespace_id(ns: Namespace) -> bytes:
    """32-byte id: SHA-256 of a tagged canonical JSON of the namespace."""
    doc = json.dumps(asdict(ns), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(b"ns\x00" + doc.encode("utf-8")).digest()


def _salt_field(salt: bytes) -> bytes:
    if len(salt) > 255:
        raise ValueError("salt longer than 255 bytes")
    return bytes([len(salt)]) + salt


def root_key(secret: bytes, salt: bytes, ns_id: bytes) -> bytes:
    """Parent key for the first block of a prefix chain."""
    msg = b"root\x00" + _salt_field(salt) + ns_id
    return hmac.new(secret, msg, hashlib.sha256).digest()


def block_key(
    secret: bytes, salt: bytes, ns_id: bytes, parent: bytes, token_ids: Sequence[int]
) -> bytes:
    """Key of one block, chained on its parent key."""
    tokens = b"".join(struct.pack("<I", t) for t in token_ids)
    msg = b"blk\x00" + _salt_field(salt) + ns_id + parent + tokens
    return hmac.new(secret, msg, hashlib.sha256).digest()
