"""The vLLM connector's logic, free of vLLM and torch so it is unit-testable anywhere.

``lmw_kv.connector`` is a thin adapter: it turns vLLM objects into the plain values
used here (token ids, ``kv_transfer_params``, per-group block id lists with ``None``
for vLLM's null block) and does the GPU copies this module asks for.

Store layout chosen by the adapter (the daemon only sees namespaces, keys, groups):

- **Keys.** One key chain per request, over the prompt's whole blocks, derived as in
  ``PROTOCOL.md`` with the *model* namespace id (every KV-cache group listed).
- **One store namespace per (KV-cache group, TP rank).** Its id commits to the whole
  model namespace plus that group and rank, and its single group holds that rank's
  bytes of that group for one block. vLLM's hybrid manager gives every group the same
  page size, so these namespaces share the daemon's slot size without waste.
- **Attention groups** are saved for every whole prompt block (a prefix chain).
  **State groups** (GDN / Mamba) are saved only where vLLM materialised a state: the
  last block of a step that ended exactly on a block boundary. A state block is
  stored under the same key as the attention block it ends, so it is sparse.
- **A match** is the longest stored attention prefix shortened to the deepest block
  that also has every state group's snapshot (probed one key at a time, capped).
- **Attribution** (protocol minor 1). Every store namespace is registered with the
  config's ``owner``, one ``model`` tag for the whole model, its group's role and its TP
  rank. Saves carry the request's ``lmw_tenant`` and the scope tag of its ``lmw_scope``;
  loads and hits carry the tenant. No ``pos`` is sent: a block's key is already the same
  in every group and rank, so the daemon's default (``pos = key``) groups a position.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import sys
import threading
import time
from array import array
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, TypeGuard

from lmw_kv.keys import Namespace, namespace_id, root_key
from lmw_kv.scopes import TENANT_BYTES, scope_tag

log = logging.getLogger("lmw_kv.connector")

KEY_ENV = "LMW_KV_KEY_FILE"
DEFAULT_MIN_TOKENS = 2048
TIERS = ("ram", "ram+nvme")
LAYOUT_VERSION = 1
STATE_PROBES = 16  # single-key state lookups per match, at most

# ---- configuration ------------------------------------------------------------


@dataclass(frozen=True)
class ConnectorConfig:
    socket: str
    tiers: str
    min_tokens: int
    #: The commits LM Warden resolved the model and tokenizer revisions to (40 hex), or
    #: None. Without them the connector stays off: a branch name such as ``main`` moves.
    revision: str | None = None
    tokenizer_revision: str | None = None
    #: Who holds the store namespaces (LM Warden's model row id); ``""`` is the legacy owner.
    owner: str = ""


def parse_config(extra: Mapping[str, Any] | None) -> ConnectorConfig | None:
    """``kv_connector_extra_config`` -> config, or None (the connector stays off)."""
    if not isinstance(extra, Mapping):
        return None
    sock, tiers, mt = extra.get("socket"), extra.get("tiers"), extra.get("min_tokens")
    if not isinstance(sock, str) or not sock or tiers not in TIERS:
        return None
    if mt is None:
        mt = DEFAULT_MIN_TOKENS
    if isinstance(mt, bool):
        return None
    try:
        min_tokens = int(mt)
    except (TypeError, ValueError):
        return None
    if min_tokens < 0:
        return None
    owner = extra.get("owner", "")
    if not isinstance(owner, str):
        return None
    rev, tok = extra.get("revision"), extra.get("tokenizer_revision")
    return ConnectorConfig(
        sock,
        str(tiers),
        min_tokens,
        rev if isinstance(rev, str) else None,
        tok if isinstance(tok, str) else None,
        owner,
    )


def is_commit(rev: object) -> bool:
    """A full 40-hex commit sha (lower case, as the HF cache writes it)."""
    return isinstance(rev, str) and len(rev) == 40 and all(c in "0123456789abcdef" for c in rev)


def namespace_revisions(
    cfg: ConnectorConfig, engine_revision: object, engine_tokenizer_revision: object
) -> tuple[str, str]:
    """The (model, tokenizer) commits the namespace is keyed by; ValueError keeps the tier off.

    They come from LM Warden, which resolves the row's revision to the commit in the HF
    cache that the engine loads. Never a branch name: ``main`` moves, and a re-pull would
    then serve KV computed with older weights. A commit the engine was started with must
    agree with the resolved one.
    """
    rev, tok = cfg.revision, cfg.tokenizer_revision
    if not is_commit(rev):
        raise ValueError("the model revision is not resolved to a commit")
    if not is_commit(tok):
        raise ValueError("the tokenizer revision is not resolved to a commit")
    assert rev is not None and tok is not None
    if is_commit(engine_revision) and engine_revision != rev:
        raise ValueError(f"engine revision {engine_revision} != resolved commit {rev}")
    if is_commit(engine_tokenizer_revision) and engine_tokenizer_revision != tok:
        raise ValueError(
            f"engine tokenizer revision {engine_tokenizer_revision} != resolved commit {tok}"
        )
    return rev, tok


def uses_nvme(tiers: str) -> bool:
    """``ram+nvme`` writes blocks through to NVMe and loads them back; ``ram`` never does."""
    return tiers == "ram+nvme"


def read_secret(path: str | None) -> bytes | None:
    """The 32-byte key from a ``v1:<64 hex>`` file, or None if missing or malformed."""
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            first = f.readline().strip()
    except (OSError, UnicodeDecodeError):
        return None
    version, sep, hexkey = first.partition(":")
    if version != "v1" or not sep or len(hexkey) != 64:
        return None
    try:
        return bytes.fromhex(hexkey)
    except ValueError:
        return None


def parse_scope(value: Any) -> bytes | None:
    """``lmw_scope`` (hex) -> salt bytes; ``b""`` when absent; None when malformed."""
    if value is None or value == "":
        return b""
    if not isinstance(value, str):
        return None
    try:
        salt = bytes.fromhex(value)
    except ValueError:
        return None
    return salt if len(salt) <= 255 else None


_HEX = frozenset("0123456789abcdefABCDEF")


def parse_tenant(value: Any) -> bytes | None:
    """``lmw_tenant`` (hex) -> the 16-byte tenant tag; ``b""`` when absent or empty (shared,
    unattributed); None when malformed (the request is not cached)."""
    if value is None or value == "":
        return b""
    if not isinstance(value, str) or len(value) != 2 * TENANT_BYTES or not _HEX.issuperset(value):
        return None
    return bytes.fromhex(value)


def fold_cache_salt(scope: bytes, cache_salt: Any) -> bytes | None:
    """The block-key salt for a request: its scope, with vLLM's ``cache_salt`` folded in.

    No ``cache_salt`` (None or empty) leaves the scope as it is, so the keys do not
    change. Otherwise the salt is a SHA-256 over the scope and the cache salt: two
    cache salts on one prefix never share a block. None (never cached) when the cache
    salt is not a string.
    """
    if cache_salt is None or cache_salt == "":
        return scope
    if not isinstance(cache_salt, str):
        return None
    return hashlib.sha256(
        b"lmw-cache-salt\x00" + bytes([len(scope)]) + scope + cache_salt.encode("utf-8")
    ).digest()


@dataclass(frozen=True, init=False, repr=False)
class BlockScope:
    """A request's block-key salt and the scope tag of its raw scope, made together.

    The only constructor takes the raw ``lmw_scope`` bytes, so a save under an isolation
    scope cannot be expressed without that scope's tag (the daemon cannot tell: a folded
    client ``cache_salt`` is legitimately untagged). The raw scope itself is not kept.
    ValueError when the cache salt cannot be folded (not a string, not UTF-8).
    """

    salt: bytes  # sent as ``begin_save`` ``scope``; the block keys derive from it
    tag: bytes  # ``scopes.scope_tag(raw)``: empty for a sharing request

    def __init__(self, raw: bytes, cache_salt: Any = None) -> None:
        salt = fold_cache_salt(raw, cache_salt)
        if salt is None:
            raise ValueError("cache_salt is not a string")
        object.__setattr__(self, "salt", salt)
        object.__setattr__(self, "tag", scope_tag(raw))

    def __repr__(self) -> str:  # the salt and the tag are secrets of the key holder
        return f"BlockScope(isolated={bool(self.tag)})"


# ---- namespaces ---------------------------------------------------------------


@dataclass(frozen=True)
class GroupDesc:
    """One vLLM KV-cache group, as far as identity goes."""

    gid: int
    kind: str  # the spec class name, e.g. FullAttentionSpec, MambaSpec
    n_layers: int
    page_bytes: int  # the spec's page_size_bytes (identity only)
    is_state: bool  # Mamba / GDN state: saved sparsely
    eagle: bool = False

    @property
    def name(self) -> str:
        return f"g{self.gid}.{self.kind}"


def role_of(group: GroupDesc) -> str:
    """The ``register_ns`` role of a group: ``mtp``, ``gdn_state`` or ``attention``."""
    if group.eagle:
        return "mtp"
    if group.is_state:
        return "gdn_state"
    return "attention"


def model_namespace(
    *,
    model_repo: str,
    revision: str,
    tokenizer: str,
    kv_dtype: str,
    block_size: int,
    groups: Sequence[GroupDesc],
    tp: int,
    pp: int,
    engine_family: str,
    layout_extra: str = "",
) -> Namespace:
    """The namespace every key of this engine derives from."""
    if block_size <= 0:
        raise ValueError(f"bad block size {block_size}")
    gs = sorted(groups, key=lambda g: g.gid)
    layout = "|".join(
        f"{g.name}:x{g.n_layers}:p{g.page_bytes}" + (":eagle" if g.eagle else "") for g in gs
    )
    if layout_extra:
        layout += ";" + layout_extra
    return Namespace(
        model_repo=model_repo,
        revision_sha=revision,
        tokenizer_sha=tokenizer,
        kv_dtype=kv_dtype,
        block_size=block_size,
        attention_layout=layout,
        kv_groups=tuple(g.name for g in gs),
        tp=tp,
        pp=pp,
        engine_family=engine_family,
        layout_version=LAYOUT_VERSION,
    )


def store_group_name(group: GroupDesc, rank: int) -> str:
    return f"{group.name}@tp{rank}"


def store_namespace(model_ns: Namespace, group: GroupDesc, rank: int) -> Namespace:
    """Where one rank's bytes of one group live: the model namespace narrowed to them."""
    return replace(model_ns, kv_groups=(store_group_name(group, rank),))


# ---- keys and matching --------------------------------------------------------


def block_keys_for(
    secret: bytes, salt: bytes, ns_id: bytes, token_ids: Iterable[int], block_size: int
) -> list[bytes]:
    """Chained keys of the whole blocks of ``token_ids`` (``keys.block_key``, batched)."""
    toks = array("I", token_ids)
    if toks.itemsize != 4:  # pragma: no cover - no mainstream platform
        raise RuntimeError("array('I') is not 32-bit here")
    if sys.byteorder != "little":  # pragma: no cover
        toks.byteswap()
    n = len(toks) // block_size
    raw = toks.tobytes()
    step = 4 * block_size
    prefix = b"blk\x00" + bytes([len(salt)]) + salt + ns_id
    parent = root_key(secret, salt, ns_id)
    keys = []
    for i in range(n):
        parent = hmac.digest(secret, prefix + parent + raw[i * step : (i + 1) * step], "sha256")
        keys.append(parent)
    return keys


def matched_beyond_gpu(num_computed_tokens: int, ext_blocks: int, block_size: int) -> int:
    """Tokens the store adds beyond what the GPU prefix cache already has."""
    return max(0, ext_blocks * block_size - num_computed_tokens)


def max_load_blocks(num_prompt_tokens: int, block_size: int) -> int:
    """Whole blocks loadable: vLLM must still compute at least the last prompt token."""
    return max(0, num_prompt_tokens - 1) // block_size


def save_admitted(num_prompt_tokens: int, min_tokens: int) -> bool:
    return num_prompt_tokens >= min_tokens


def in_bounds(offset: object, nbytes: int, limit: int) -> TypeGuard[int]:
    """A slot offset from the daemon is usable only if it lies wholly inside the mapping."""
    return (
        isinstance(offset, int)
        and not isinstance(offset, bool)
        and offset >= 0
        and offset + nbytes <= limit
    )


def matched_blocks(
    lookup: Callable[[bytes, list[bytes]], int],
    attn_ns: Sequence[bytes],
    state_ns: Sequence[bytes],
    keys: Sequence[bytes],
    probes: int = STATE_PROBES,
) -> int:
    """Leading blocks loadable: stored in every attention group, and with every state
    group's snapshot at the last one. Any failure is a miss (0)."""
    try:
        n = len(keys)
        for ns in attn_ns:
            if n == 0:
                return 0
            n = min(n, int(lookup(ns, list(keys[:n]))))
        if n <= 0 or not state_ns:
            return max(n, 0)
        for i in range(n - 1, max(n - 1 - probes, -1), -1):
            if all(int(lookup(ns, [keys[i]])) >= 1 for ns in state_ns):
                return i + 1
        return 0
    except Exception as exc:  # fail open
        log.debug("lookup failed, treating as a miss: %s", exc)
        return 0


# ---- what crosses from the scheduler to the workers ---------------------------


@dataclass
class LoadEntry:
    gid: int
    keys: list[bytes]
    block_ids: list[int]
    state: bool = False  # a GDN / Mamba snapshot, not prefix tokens


@dataclass
class LoadJob:
    req_id: str
    tag: str
    tokens: int
    entries: list[LoadEntry]
    doomed: bool = False  # a block it must fill is missing: fail it, never half-load
    tenant: bytes = field(default=b"", repr=False)  # takes over the loaded blocks' charge


@dataclass
class SaveEntry:
    key: bytes
    parent: bytes
    block_scope: BlockScope
    gid: int
    block_id: int
    tenant: bytes = field(default=b"", repr=False)  # pays for the block

    @property
    def scope(self) -> bytes:
        return self.block_scope.salt

    @property
    def scope_tag(self) -> bytes:
        return self.block_scope.tag


@dataclass
class StepPlan:
    loads: list[LoadJob] = field(default_factory=list)
    saves: list[SaveEntry] = field(default_factory=list)


# ---- scheduler side -----------------------------------------------------------


class _LookupClient(Protocol):
    def lookup(self, ns_id: bytes, keys: list[bytes]) -> int: ...


@dataclass
class _Req:
    keys: list[bytes]
    root: bytes
    scope: BlockScope
    tag: str
    prompt_len: int
    tenant: bytes = field(default=b"", repr=False)
    saved_upto: int = 0  # blocks [0, saved_upto) are stored or already sent to save
    matched: int = 0  # blocks the last match found
    computed: int = 0  # GPU-computed tokens at the last match

    @property
    def salt(self) -> bytes:
        return self.scope.salt

    @property
    def scope_tag(self) -> bytes:
        return self.scope.tag


BlockLists = Sequence[Sequence[int | None]]  # per group, per block position; None = null


#: Bound on the refused-request set, should ``finished`` never come for some.
_MAX_REFUSED = 65536


class SchedulerLogic:
    """Scheduler-side decisions. Only ``lookup`` touches the daemon (short deadline)."""

    def __init__(
        self,
        client: _LookupClient,
        secret: bytes,
        cfg: ConnectorConfig,
        model_ns_id: bytes,
        groups: Sequence[GroupDesc],
        store_ns_ids: Mapping[int, bytes],
        block_size: int,
        probes: int = STATE_PROBES,
    ) -> None:
        self._client = client
        self._secret = secret
        self._cfg = cfg
        self._ns_id = model_ns_id
        self._groups = list(groups)
        self._attn_ns = [store_ns_ids[g.gid] for g in groups if not g.is_state]
        self._state_ns = [store_ns_ids[g.gid] for g in groups if g.is_state]
        self._bs = block_size
        self._probes = probes
        self._reqs: dict[str, _Req] = {}
        self._load_tried: set[str] = set()  # one external load attempt per request (R19)
        self._pending_loads: list[LoadJob] = []
        self._tenant_warned = False
        # Requests refused at admission (malformed params or cache_salt): vLLM may ask
        # again while one waits; it is refused once and warned about once.
        self._refused: set[str] = set()

    def match(
        self,
        req_id: str,
        token_ids: Sequence[int],
        params: Mapping[str, Any] | None,
        num_computed_tokens: int,
        cache_salt: Any = None,
    ) -> tuple[int, bool]:
        """``get_num_new_matched_tokens``: (tokens to load, load async).

        ``cache_salt`` is the request's vLLM ``cache_salt``; it isolates the RAM and
        NVMe tiers the way it isolates vLLM's own prefix cache.
        """
        if req_id in self._load_tried or req_id in self._refused:
            # A failed load comes back here with num_computed_tokens = 0 (recompute);
            # matching again would retry a deterministic failure forever.
            return 0, False
        try:
            req = self._reqs.get(req_id) or self._admit(req_id, token_ids, params, cache_salt)
            if req is None:
                return 0, False
            max_b = min(max_load_blocks(req.prompt_len, self._bs), len(req.keys))
            if max_b * self._bs <= num_computed_tokens:
                return 0, False
            n = matched_blocks(
                self._client.lookup, self._attn_ns, self._state_ns, req.keys[:max_b], self._probes
            )
            req.matched, req.computed = n, num_computed_tokens
            req.saved_upto = max(req.saved_upto, n)
            ext = matched_beyond_gpu(num_computed_tokens, n, self._bs)
            return ext, ext > 0
        except Exception as exc:  # fail open
            log.warning("KV lookup failed, continuing without it: %s", exc)
            return 0, False

    def _admit(
        self,
        req_id: str,
        token_ids: Sequence[int],
        params: Mapping[str, Any] | None,
        cache_salt: Any = None,
    ) -> _Req | None:
        p = params if isinstance(params, Mapping) else {}
        if not save_admitted(len(token_ids), self._cfg.min_tokens):
            return None
        scope = parse_scope(p.get("lmw_scope"))
        if scope is None:
            self._refuse(req_id)
            log.warning("request %s: malformed lmw_scope; not cached", req_id)
            return None
        tenant = parse_tenant(p.get("lmw_tenant"))
        if tenant is None:
            self._refuse(req_id)
            if not self._tenant_warned:  # once, and never the value
                self._tenant_warned = True
                log.warning("request %s: malformed lmw_tenant; not cached", req_id)
            return None
        try:
            bscope = BlockScope(scope, cache_salt)
        except ValueError:  # never cached, never an untagged isolated save
            self._refuse(req_id)  # once per request, and never the value
            log.warning("request %s: malformed cache_salt; not cached", req_id)
            return None
        tag = p.get("lmw_req")
        req = _Req(
            keys=block_keys_for(self._secret, bscope.salt, self._ns_id, token_ids, self._bs),
            root=root_key(self._secret, bscope.salt, self._ns_id),
            scope=bscope,
            tag=tag if isinstance(tag, str) and tag else req_id,
            prompt_len=len(token_ids),
            tenant=tenant,
        )
        self._reqs[req_id] = req
        return req

    def _refuse(self, req_id: str) -> None:
        if len(self._refused) >= _MAX_REFUSED:  # finished() normally drains it
            self._refused.clear()
        self._refused.add(req_id)

    def after_alloc(self, req_id: str, blocks: BlockLists, num_external_tokens: int) -> None:
        """``update_state_after_alloc``: plan the load into the blocks vLLM allocated."""
        if num_external_tokens <= 0:
            return
        self._load_tried.add(req_id)
        req = self._reqs.get(req_id)
        doomed = req is None
        start = (req.computed if req else 0) // self._bs
        end = ((req.computed if req else 0) + num_external_tokens) // self._bs
        if req is not None and (end > req.matched or end > len(req.keys) or end <= start):
            doomed = True
        entries: list[LoadEntry] = []
        for g in self._groups:
            gb = blocks[g.gid] if g.gid < len(blocks) else []
            positions = [end - 1] if g.is_state else list(range(start, end))
            e = LoadEntry(g.gid, [], [], g.is_state)
            for bi in positions:
                bid = gb[bi] if 0 <= bi < len(gb) else None
                if bid is None:
                    doomed = True
                    continue
                if req is None or bi >= len(req.keys):
                    doomed = True
                    continue
                e.keys.append(req.keys[bi])
                e.block_ids.append(bid)
            if e.block_ids:
                entries.append(e)
        # A job that fails is reported as a failed request (vLLM recomputes it all),
        # whatever blocks it lists; an empty job can only fail.
        doomed = doomed or not entries
        tag = req.tag if req else req_id
        tenant = req.tenant if req else b""
        self._pending_loads.append(
            LoadJob(req_id, tag, num_external_tokens, entries, doomed, tenant)
        )
        if req is not None:
            req.saved_upto = max(req.saved_upto, end)

    def build(
        self,
        scheduled: Iterable[tuple[str, int, int]],
        get_blocks: Callable[[str], BlockLists | None],
    ) -> StepPlan:
        """``build_connector_meta``: this step's loads and saves.

        ``scheduled`` is (request id, computed tokens before the step, tokens scheduled).
        """
        plan = StepPlan(self._pending_loads, [])
        self._pending_loads = []
        for rid, nct, nsched in scheduled:
            try:
                self._plan_saves(plan, rid, nct, nsched, get_blocks)
            except Exception as exc:  # fail open: drop this request's saves
                log.warning("KV save planning failed for %s: %s", rid, exc)
        return plan

    def _plan_saves(
        self,
        plan: StepPlan,
        rid: str,
        nct: int,
        nsched: int,
        get_blocks: Callable[[str], BlockLists | None],
    ) -> None:
        req = self._reqs.get(rid)
        if req is None or not req.keys:
            return
        end_tok = min(nct + nsched, req.prompt_len)  # prompt blocks only
        full = min(end_tok // self._bs, len(req.keys))
        done = req.saved_upto
        if full <= done:
            return
        blocks = get_blocks(rid)
        if blocks is None:
            return
        state_at = full - 1 if end_tok % self._bs == 0 else None  # materialised this step
        for g in self._groups:
            gb = blocks[g.gid] if g.gid < len(blocks) else []
            if g.is_state:
                positions = [state_at] if state_at is not None else []
            else:
                positions = list(range(done, full))
            for bi in positions:
                bid = gb[bi] if bi < len(gb) else None
                if bid is None:
                    continue
                parent = req.keys[bi - 1] if bi > 0 else req.root
                plan.saves.append(
                    SaveEntry(req.keys[bi], parent, req.scope, g.gid, bid, req.tenant)
                )
        req.saved_upto = full

    def finished(self, req_id: str) -> None:
        """``request_finished``: forget the request. Blocks are never held for us: vLLM
        itself keeps a request's blocks while it waits for a load (``WAITING_FOR_REMOTE_KVS``)
        and frees them when the load is reported in ``finished_recving``."""
        self._reqs.pop(req_id, None)
        self._load_tried.discard(req_id)
        self._refused.discard(req_id)


# ---- worker side --------------------------------------------------------------


class LoadTracker:
    """Loads in flight and finished; thread-safe.

    ``poll`` gives (finished_recving, failed_recving). Every load is reported exactly
    once in ``finished_recving``, failed or not, aborted or not (never as sending); a
    failed one is also in ``failed_recving``, so vLLM recomputes the whole request.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inflight: set[str] = set()
        self._recving: set[str] = set()
        self._failed: set[str] = set()

    def start(self, req_id: str) -> None:
        with self._lock:
            self._inflight.add(req_id)

    def finish(self, req_id: str, ok: bool) -> None:
        with self._lock:
            self._inflight.discard(req_id)
            self._recving.add(req_id)
            if not ok:
                self._failed.add(req_id)

    def poll(self) -> tuple[set[str], set[str]]:
        with self._lock:
            recving, failed = self._recving, self._failed
            self._recving, self._failed = set(), set()
        return recving, failed


class _JobQueue(Protocol):
    def get(self) -> LoadJob | None: ...


def submit_load(
    tracker: LoadTracker,
    job: LoadJob,
    ready: bool,
    loader_alive: bool,
    put: Callable[[LoadJob], None],
) -> bool:
    """``start_load_kv`` for one job: hand it to the loader thread, or fail it at once.

    A worker that is off or a loader thread that is not alive fails the load (vLLM
    recomputes the request) instead of leaving the request waiting for a load nobody
    will do. True when the job was queued.
    """
    tracker.start(job.req_id)
    if ready and loader_alive:
        try:
            put(job)
            return True
        except Exception as exc:  # fail open
            log.warning("LM Warden KV: cannot queue the load for %s: %r", job.req_id, exc)
    tracker.finish(job.req_id, False)
    return False


def serve_loads(
    jobs: _JobQueue,
    tracker: LoadTracker,
    setup: Callable[[], Any],
    run: Callable[[LoadJob, Any], list[int] | None],
    record: Callable[[LoadJob, list[int], float], None],
    on_setup_failed: Callable[[BaseException], None],
) -> None:
    """The loader thread's body; returns on a ``None`` job.

    ``setup()`` makes the thread's context (the CUDA device and stream). If it raises,
    ``on_setup_failed`` turns the worker off and every job is failed, so no request
    waits forever. ``run(job, ctx)`` does one load: per-entry NVMe block counts, or
    None for a failure (raising counts as one). ``record`` gets a good load's hits
    before the load is reported finished (the api reads them once the turn ends).
    Every job is reported to ``tracker`` exactly once, whatever raises.
    """
    ctx: Any = None
    ok_setup = True
    try:
        ctx = setup()
    except Exception as exc:
        ok_setup = False
        log.warning("LM Warden KV loader thread cannot start: %r", exc)
        try:
            on_setup_failed(exc)
        except Exception:  # pragma: no cover - defensive
            pass
    while True:
        job = jobs.get()
        if job is None:
            return
        t0 = time.perf_counter()
        nvme: list[int] | None = None
        try:
            if ok_setup:
                try:
                    nvme = run(job, ctx)
                except Exception as exc:  # fail open
                    log.warning("LM Warden KV load failed for %s: %r", job.req_id, exc)
                    nvme = None
            if nvme is not None:
                try:
                    record(job, nvme, time.perf_counter() - t0)
                except Exception as exc:  # the hit count is an optimisation
                    log.debug("LM Warden KV: record_hits failed: %r", exc)
        finally:
            tracker.finish(job.req_id, nvme is not None)


class _WorkerClient(Protocol):
    def begin_load(
        self, ns_id: bytes, keys: list[bytes], *, tenant: bytes = b""
    ) -> tuple[int, list[dict[str, int]], int] | None: ...

    def end_load(self, handle: int) -> bool | None: ...

    def begin_save(
        self,
        ns_id: bytes,
        key: bytes,
        parent: bytes,
        scope: bytes = b"",
        *,
        tenant: bytes = b"",
        scope_tag: bytes = b"",
    ) -> dict[str, int] | None: ...

    def commit_save(self, ns_id: bytes, key: bytes) -> bool | None: ...

    def abort_save(self, ns_id: bytes, key: bytes) -> bool | None: ...


Copy = Callable[[int, int, int], None]  # (group id, GPU block id, byte offset in the RAM file)


def run_load_job(
    client: _WorkerClient,
    job: LoadJob,
    ns_ids: Mapping[int, bytes],
    names: Mapping[int, str],
    sizes: Mapping[int, int],
    limit: int,
    copy: Copy,
    sync: Callable[[], None],
    same_file: Callable[[], bool],
) -> list[int] | None:
    """Pin every slot, check every offset, copy RAM -> GPU, wait, release the pins.

    Returns, per entry, how many of its blocks the daemon read from NVMe; None (the
    request is reported failed, vLLM recomputes it) when the job is doomed or empty,
    the daemon fails, fewer blocks come back than planned, an offset is out of range,
    or the RAM file was replaced meanwhile. Nothing is copied unless every slot checks.
    Every failure logs one warning naming the request, so the recompute is visible.
    """
    try:
        nvme = _load(client, job, ns_ids, names, sizes, limit, copy, sync, same_file)
        why = "incomplete"
    except Exception as exc:  # fail open
        nvme, why = None, repr(exc)
    if nvme is None:
        log.warning("LM Warden KV load failed for %s (%s); vLLM recomputes it", job.req_id, why)
    return nvme


def _load(
    client: _WorkerClient,
    job: LoadJob,
    ns_ids: Mapping[int, bytes],
    names: Mapping[int, str],
    sizes: Mapping[int, int],
    limit: int,
    copy: Copy,
    sync: Callable[[], None],
    same_file: Callable[[], bool],
) -> list[int] | None:
    if job.doomed or not job.entries:
        return None
    handles: list[int] = []
    try:
        todo: list[tuple[int, int, int]] = []
        nvme: list[int] = []
        for e in job.entries:
            got = client.begin_load(ns_ids[e.gid], e.keys, tenant=job.tenant)
            if got is None:
                return None
            handle, slots, from_nvme = got
            handles.append(handle)
            if len(slots) < len(e.keys):
                return None
            nvme.append(min(from_nvme, len(e.keys)))
            for slot, bid in zip(slots, e.block_ids, strict=False):
                off = slot.get(names[e.gid]) if isinstance(slot, dict) else None
                if not in_bounds(off, sizes[e.gid], limit):
                    return None
                todo.append((e.gid, bid, off))
        for gid, bid, off in todo:
            copy(gid, bid, off)
        sync()
        return nvme if same_file() else None
    finally:
        if handles:
            try:
                sync()  # never release a pin while a copy may still read it
            except Exception:
                pass
            for h in handles:
                client.end_load(h)


class SetupError(Exception):
    """The worker cannot attach to the daemon; the connector turns itself off."""


def attach_store(
    client: Any,
    namespaces: Sequence[tuple[int, Namespace, Mapping[str, int]]],
    map_ram: Callable[[], None],
    same_file: Callable[[], bool],
    nvme: bool = True,
    *,
    owner: str,
    model: bytes,
    rank: int,
    groups: Sequence[GroupDesc],
) -> dict[int, bytes]:
    """Register the worker's store namespaces and map the daemon's RAM file, in that order.

    The daemon creates its RAM file at start, before it answers; the first
    ``register_ns`` sets the slot size. Registering again after the mapping
    (``register_ns`` is idempotent) covers a daemon that restarted in between: it has
    the file we mapped but not our namespaces. ``nvme`` is registered with every
    namespace (``uses_nvme`` of the tiers), and so are ``owner``, ``model`` (the model
    tag), the group's ``role_of`` and the TP ``rank``. Returns group id -> store
    namespace id.
    """
    roles = {g.gid: role_of(g) for g in groups}
    who = (owner, model, rank, roles)
    _register_all(client, namespaces, nvme, who)
    map_ram()
    ids = _register_all(client, namespaces, nvme, who)
    if not same_file():
        raise SetupError("the daemon restarted during setup")
    return ids


def _register_all(
    client: Any,
    namespaces: Sequence[tuple[int, Namespace, Mapping[str, int]]],
    nvme: bool,
    who: tuple[str, bytes, int, Mapping[int, str]],
) -> dict[int, bytes]:
    owner, model, rank, roles = who
    ids: dict[int, bytes] = {}
    for gid, ns, groups in namespaces:
        want = namespace_id(ns)
        role = roles.get(gid)
        if role is None:
            raise SetupError(f"group {gid} is not a known KV-cache group")
        got = client.register_ns(
            ns, dict(groups), nvme=nvme, owner=owner, model=model, role=role, rank=rank
        )
        if got != want:
            raise SetupError(f"register_ns for group {gid} failed or disagreed")
        ids[gid] = want
    return ids


def hit_split(job: LoadJob, nvme: Sequence[int], block_size: int) -> tuple[int, int]:
    """(ram_tokens, nvme_tokens) of a finished load, for ``record_hits``.

    Tokens are counted once per prefix block, not once per group: a block counts as
    NVMe if a prefix (attention) group read it from NVMe; the most NVMe blocks of any
    attention entry are taken. With no attention entry (a state-only model) the whole
    load counts as NVMe if its snapshot came from NVMe.
    """
    prefix = [n for e, n in zip(job.entries, nvme, strict=False) if not e.state]
    if prefix:
        nvme_tokens = min(job.tokens, max(prefix) * block_size)
    else:
        nvme_tokens = job.tokens if any(nvme) else 0
    return job.tokens - nvme_tokens, nvme_tokens


def record_load_hits(
    client: Any, job: LoadJob, nvme: Sequence[int], elapsed: float, block_size: int, rank: int
) -> None:
    """Rank 0 records a good load's hits for its turn, attributed to the job's tenant."""
    if rank != 0:
        return
    ram_tokens, nvme_tokens = hit_split(job, nvme, block_size)
    client.record_hits(job.tag, ram_tokens, nvme_tokens, elapsed, tenant=job.tenant)


def run_saves(
    client: _WorkerClient,
    saves: Sequence[SaveEntry],
    ns_ids: Mapping[int, bytes],
    names: Mapping[int, str],
    sizes: Mapping[int, int],
    limit: int,
    copy: Copy,
    sync: Callable[[], None],
    same_file: Callable[[], bool],
) -> int:
    """Begin each save, copy GPU -> RAM, wait, then commit; returns blocks committed.

    A block already stored (or no slot) is skipped; an out-of-range offset is aborted;
    a failed copy or a replaced RAM file aborts everything begun.
    """
    pending: list[tuple[bytes, bytes]] = []
    try:
        for s in saves:
            ns = ns_ids[s.gid]
            offs = client.begin_save(
                ns, s.key, s.parent, s.scope, tenant=s.tenant, scope_tag=s.scope_tag
            )
            if offs is None:
                continue
            off = offs.get(names[s.gid]) if isinstance(offs, dict) else None
            if not in_bounds(off, sizes[s.gid], limit):
                client.abort_save(ns, s.key)
                continue
            pending.append((ns, s.key))
            copy(s.gid, s.block_id, off)
        sync()
        if not same_file():
            raise RuntimeError("the daemon's RAM file was replaced")
    except Exception as exc:  # fail open
        log.warning("KV save failed, dropped: %s", exc)
        for ns, key in pending:
            client.abort_save(ns, key)
        return 0
    return sum(1 for ns, key in pending if client.commit_save(ns, key))
