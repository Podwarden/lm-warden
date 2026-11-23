"""Per-key KV isolation in the proxy: a request's KV identity and its engine cache salt.

Every key has a tenant tag (accounting only; never part of a block key). An isolated key
also has a scope, derived from its family's ``kv_tenant_id`` and ``kv_epoch`` under the
cobs2 network key (``lmw_kv.scopes``; the exact encodings are in ``lmw_kv/PROTOCOL.md``).

- GPU tier: an isolated request's ``cache_salt`` becomes ``gpu_cache_salt(nk, scope,
  client_salt)``, so the key hits its own prefix blocks and nobody else's. A sharing key's
  salt passes through, except a salt with the reserved ``lmw1:`` prefix, which is
  rewritten under the empty scope, so no client can present an isolated key's salt.
- RAM and NVMe tiers: ``lmw_tenant`` and ``lmw_scope`` (hex) in ``kv_transfer_params``
  (``app.proxy.kv.set_kv_params``).

The refusals (``check_cache_salt``, a missing network key, an isolated key whose scope
cannot be derived) are decided before admission; the rewrite runs just before the
upstream send, after the cobs2 ids and the cache-obs chain were computed over the
client's own salt.

Every value here is a secret of the key holder: never log, capture or store one.
"""

from __future__ import annotations

import functools
import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from app.db.repos.tokens import TokenRow

_scopes: ModuleType | None
try:
    _scopes = importlib.import_module("lmw_kv.scopes")
except ImportError:  # an install without the KV tier package (no vLLM engine to isolate)
    _scopes = None

#: ``lmw_kv.scopes.GPU_SALT_PREFIX`` and ``INTERNAL_TENANT_ID`` (checked equal in the tests),
#: here too so a sharing request works where ``lmw_kv`` is not installed.
GPU_SALT_PREFIX = "lmw1:"
INTERNAL_TENANT_ID = "internal"

CACHE_SALT = "cache_salt"
SALT_NOT_A_STRING = "cache_salt must be a string"
SALT_NOT_TEXT = "cache_salt must be valid UTF-8 text"
ISOLATION_UNAVAILABLE = "cache isolation is unavailable right now"
#: The relay's refusal of an isolated key's body that cannot be parsed and stripped.
RELAY_BODY_UNCHECKED = "this key is isolated and the request body cannot be checked"
CACHE_CONTROL = "cache_control"

#: How a model treats an isolated key: ``isolated`` (its engine salts the prefix cache per
#: key) or ``unsupported`` (the key is refused there, O4). A relayed model is
#: ``not_isolated_at_provider`` (O6): the provider's cache is shared by the account.
CacheIsolation = Literal["isolated", "unsupported"]
NOT_ISOLATED_AT_PROVIDER: Literal["not_isolated_at_provider"] = "not_isolated_at_provider"


@dataclass(frozen=True, repr=False)
class KvIdentity:
    """A request's KV identity. ``scope`` and ``scope_tag`` are ``b""`` for a sharing key;
    ``tenant`` is ``b""`` only when no network key was available (a sharing request)."""

    isolated: bool
    tenant: bytes
    scope: bytes
    scope_tag: bytes

    def __repr__(self) -> str:  # the bytes are secrets of the key holder
        return f"KvIdentity(isolated={self.isolated}, <redacted>)"

    __str__ = __repr__


#: A sharing request when the network key is unavailable: unattributed, as in step 1.
UNKEYED_SHARING = KvIdentity(False, b"", b"", b"")


class CacheSaltRefused(ValueError):
    """The request's ``cache_salt`` cannot be used; the client gets a 400 with this text."""


def is_isolated(token: Any) -> bool:
    """Whether ``token`` is an isolated key: any ``kv_isolation`` other than ``sharing``."""
    return getattr(token, "kv_isolation", "sharing") != "sharing"


def can_isolate_backend(name: Any) -> bool:
    """Whether the backend named ``name`` (None: the default) isolates its prefix cache by
    ``cache_salt`` (vLLM). False on any registry error (an unknown backend): an isolated
    key is refused there, unlike cache observation's default of True."""
    try:
        from app.runtime.backends import registry

        return registry.get(name).capabilities.honours_cache_salt is True
    except Exception:  # noqa: BLE001 -- fail closed
        return False


def can_isolate(model: Any) -> bool:
    """``can_isolate_backend`` of the row's backend: the models lists' answer."""
    return can_isolate_backend(getattr(model, "backend", None))


def running_backend(state: Any, model: Any, variant: Any = None) -> str | None:
    """The backend the engine serving ``model`` was launched with (the supervisor's
    variant; for an engine it did not launch, the row's). None when it cannot be told."""
    try:
        if variant is None:
            from app.runtime.variants import running_variant

            variant = running_variant(state, model)
        name = variant.descriptor.get("backend")
    except Exception:  # noqa: BLE001 -- unknown: the caller fails closed
        return None
    return name if isinstance(name, str) and name else None


def can_isolate_running(state: Any, model: Any, variant: Any = None) -> bool:
    """Whether the RUNNING engine of ``model`` can isolate: decided by the launched backend,
    so a row edited while the engine runs cannot mislabel it. False when unknown."""
    name = running_backend(state, model, variant)
    return name is not None and can_isolate_backend(name)


def cache_isolation(model: Any) -> CacheIsolation:
    """``can_isolate`` as the models lists show it."""
    return "isolated" if can_isolate(model) else "unsupported"


def cannot_isolate_message(served_name: Any) -> str:
    """The 403 of an isolated key on a model that cannot isolate its cache (O4)."""
    return f"this key is isolated and model {served_name} cannot isolate its cache"


def _drop_marks(blocks: Any) -> bool:
    """Remove ``cache_control`` from each dict of a list of blocks, from the blocks of a
    nested content list (``tool_result``) and of a document's ``source.content``; True
    when it removed any."""
    if not isinstance(blocks, list):
        return False
    removed = False
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if CACHE_CONTROL in block:
            del block[CACHE_CONTROL]
            removed = True
        if _drop_marks(block.get("content")):
            removed = True
        source = block.get("source")
        if isinstance(source, dict) and _drop_marks(source.get("content")):
            removed = True
    return removed


def strip_cache_control(body: dict[str, Any]) -> bool:
    """Remove every Anthropic ``cache_control`` mark from a Messages body, in place.

    The top level, ``system`` blocks, each message and its ``content`` blocks (and the
    blocks of a ``tool_result``'s content or a document's ``source.content``) and ``tools``. A Message Batches body (``requests[*]
    .params``) is stripped request by request. True when it removed any.
    """
    removed = False
    if CACHE_CONTROL in body:
        del body[CACHE_CONTROL]
        removed = True
    if _drop_marks(body.get("system")):
        removed = True
    if _drop_marks(body.get("tools")):
        removed = True
    messages = body.get("messages")
    if isinstance(messages, list):
        if _drop_marks(messages):  # a mark on the message itself, and its content blocks
            removed = True
    requests = body.get("requests")
    if isinstance(requests, list):
        for r in requests:
            params = r.get("params") if isinstance(r, dict) else None
            if isinstance(params, dict) and strip_cache_control(params):
                removed = True
    return removed


def _lib() -> ModuleType:
    if _scopes is None:
        raise ValueError("the lmw_kv package is not installed")
    return _scopes


@functools.lru_cache(maxsize=4096)
def _derive(network_key: bytes, tenant_id: str, epoch: int, isolated: bool) -> KvIdentity:
    lib = _lib()
    tenant: bytes = lib.tenant_tag(network_key, tenant_id)
    if not isolated:
        return KvIdentity(False, tenant, b"", b"")
    scope: bytes = lib.isolation_scope(network_key, tenant_id, epoch)
    return KvIdentity(True, tenant, scope, lib.scope_tag(scope))


def gpu_salt(network_key: bytes, scope: bytes, client_salt: str) -> str:
    """``lmw_kv.scopes.gpu_cache_salt``: the engine's ``cache_salt`` for a scope."""
    salt: str = _lib().gpu_cache_salt(network_key, scope, client_salt)
    return salt


def kv_identity(row: TokenRow | None, network_key: bytes) -> KvIdentity:
    """The KV identity of a request made with ``row`` (None: an internal sharing request).

    Memoised on (network key, tenant id, epoch, isolated). Raises ValueError for an
    isolated key whose scope cannot be derived (no tenant id, a bad epoch, a bad network
    key, no ``lmw_kv`` package): the caller refuses the request (fail closed). An isolated key never falls back
    to the shared ``internal`` tenant, which would pool every such key into one scope.
    Any ``kv_isolation`` value other than ``"sharing"`` counts as isolated.
    """
    if row is None:
        return _derive(network_key, INTERNAL_TENANT_ID, 0, False)
    isolated = getattr(row, "kv_isolation", "sharing") != "sharing"
    tenant_id = getattr(row, "kv_tenant_id", "") or ""
    if not isolated:
        return _derive(network_key, tenant_id or INTERNAL_TENANT_ID, 0, False)
    epoch = getattr(row, "kv_epoch", None)
    if not tenant_id or not isinstance(tenant_id, str):
        raise ValueError("isolated key without a kv_tenant_id")
    if type(epoch) is not int:  # bool is an int; the cache would mix True with 1
        raise ValueError("kv_epoch must be an int")
    return _derive(network_key, tenant_id, epoch, True)


def _encodable(s: str) -> bool:
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def reserved_salt(body_json: Any) -> bool:
    """Whether the body's ``cache_salt`` is a string with the reserved ``lmw1:`` prefix."""
    if not isinstance(body_json, dict):
        return False
    salt = body_json.get(CACHE_SALT)
    return isinstance(salt, str) and salt.startswith(GPU_SALT_PREFIX)


def check_cache_salt(body_json: dict[str, Any], ident: KvIdentity) -> None:
    """Refuse a ``cache_salt`` the rewrite could not handle (CacheSaltRefused).

    Isolated: a salt that is present and neither a string nor null, or a string that is
    not valid UTF-8 (a lone surrogate). A null salt is no salt, as vLLM reads it.
    Sharing: only a reserved ``lmw1:`` salt that is not valid UTF-8; any other value is
    forwarded as today and the engine decides.
    """
    if not isinstance(body_json, dict) or CACHE_SALT not in body_json:
        return
    salt = body_json[CACHE_SALT]
    if ident.isolated:
        if salt is None:
            return
        if not isinstance(salt, str):
            raise CacheSaltRefused(SALT_NOT_A_STRING)
        if not _encodable(salt):
            raise CacheSaltRefused(SALT_NOT_TEXT)
        return
    if isinstance(salt, str) and salt.startswith(GPU_SALT_PREFIX) and not _encodable(salt):
        raise CacheSaltRefused(SALT_NOT_TEXT)


def rewrite_cache_salt(body_json: dict[str, Any], ident: KvIdentity, network_key: bytes) -> bool:
    """Set the engine's ``cache_salt`` for ``ident``; True when the body changed.

    Isolated: always (a missing or null client salt is ``""``). Sharing: only a string
    with the reserved prefix. The caller ran ``check_cache_salt`` before admission.
    """
    salt = body_json.get(CACHE_SALT)
    if ident.isolated:
        body_json[CACHE_SALT] = gpu_salt(
            network_key, ident.scope, salt if isinstance(salt, str) else ""
        )
        return True
    if isinstance(salt, str) and salt.startswith(GPU_SALT_PREFIX):
        body_json[CACHE_SALT] = gpu_salt(network_key, b"", salt)
        return True
    return False


def isolation_cost(
    star_frac: float | None,
    prompt_tokens: int,
    gpu: int | None,
    ram: int | None,
    nvme: int | None,
) -> int | None:
    """Tokens an isolated request recomputed that sharing would have served.

    ``star_frac`` is the operator lens's matched share of the prompt, taken before the
    forward (None when cache observation did not run). Recomputed is the prompt minus
    every tier's hits, 0 when any of them is unknown.
    """
    if star_frac is None:
        return None
    if gpu is None or ram is None or nvme is None:
        recomputed = 0
    else:
        recomputed = max(prompt_tokens - (gpu + ram + nvme), 0)
    return min(recomputed, round(star_frac * prompt_tokens))
