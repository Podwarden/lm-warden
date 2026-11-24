"""LmwKVConnector: LM Warden's KV-cache tiers (RAM + NVMe store daemon) for vLLM.

Written against vLLM 0.32.0 (V2 model runner), the API the October spike connector
proved on a hybrid Qwen3.5 model. Loaded only by argv:

  --kv-transfer-config '{"kv_connector":"LmwKVConnector","kv_role":"kv_both",
     "kv_connector_module_path":"lmw_kv.connector",
     "kv_connector_extra_config":{"socket":"<path>","tiers":"ram","min_tokens":2048,
       "revision":"<40 hex>","tokenizer_revision":"<40 hex>","owner":"<model row id>"}}'

and the env var ``LMW_KV_KEY_FILE`` (the block-key secret, ``v1:<64 hex>``).

How it works (decisions live in ``connector_logic``, which needs no vLLM):

- Scheduler side: ``get_num_new_matched_tokens`` asks the daemon (``lookup`` only, a
  short deadline) for the longest stored prefix and returns ``(n, True)``: hybrid
  restores are async only (a sync restore gave silently wrong tokens on the V2
  runner). ``update_state_after_alloc`` plans the load into the allocated blocks;
  ``build_connector_meta`` ships loads and the new whole prompt blocks to save.
- Worker side: ``register_kv_caches`` builds per-group byte views of every group's
  tensors (all hybrid groups; MTP sits in the full-attention group), maps the
  daemon's whole RAM file once and pins it once with ``cudaHostRegister``.
  ``start_load_kv`` hands loads to a background thread (``begin_load`` may wait on an
  NVMe promotion), which copies RAM -> GPU on its own stream and reports through
  ``get_transfer_results``. ``wait_for_save`` copies the step's new blocks and GDN states
  (only where vLLM materialised them, in this same step) GPU -> RAM, then commits.
  Per-layer hooks are no-ops: they never fire for GDN layers.
- Fail open everywhere: a missing key file, a bad config, a revision not resolved to a
  commit (``revision`` / ``tokenizer_revision`` in the extra config, set by LM Warden), a
  daemon that is down, slow or restarted means no loads and no saves, never an
  exception into vLLM. A load that cannot be completed exactly reports its request
  in ``failed_recving`` (request level: vLLM refuses block-level errors on hybrid
  models) so vLLM recomputes the whole request.
"""

from __future__ import annotations

import mmap
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import vllm
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
    group_kernel_blocks,
)

from lmw_kv import connector_logic as cl
from lmw_kv.client import Client
from lmw_kv.keys import Namespace, namespace_id
from lmw_kv.scopes import model_tag

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

# Under "vllm": vLLM's logging config gives a handler only to that logger tree, so an
# INFO line from "lmw_kv.connector" (the "store on" and "worker ready" lines operators
# check after a load) never reached the engine log.
logger = init_logger("vllm.lmw_kv.connector")

# Env vars that change the bytes of a KV block without changing the config we read.
_LAYOUT_ENV = ("VLLM_KV_CACHE_LAYOUT", "VLLM_ATTENTION_BACKEND")


class _Off(Exception):
    """The connector stays off (fail open); the message says why."""


@dataclass
class LmwKVMeta(KVConnectorMetadata):
    plan: cl.StepPlan = field(default_factory=cl.StepPlan)


def _block_lists(blocks: KVCacheBlocks) -> list[list[int | None]]:
    """vLLM ``KVCacheBlocks`` -> per group, per position: block id, None for null."""
    return [[None if b.is_null else b.block_id for b in gb] for gb in blocks.blocks]


def _page_round(n: int) -> int:
    """A group's bytes as the daemon lays them out in a slot: rounded up to a page."""
    return -(-n // mmap.PAGESIZE) * mmap.PAGESIZE


def _cuda_ok(rc: Any) -> bool:
    return rc == 0 or getattr(rc, "value", None) == 0 or str(rc).endswith("success")


class LmwKVConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        self._off_logged = False
        self._cfg: cl.ConnectorConfig | None = None
        self._client: Client | None = None
        self._model_ns: Namespace | None = None
        self._model = b""  # model tag: owner and resolved revision
        self._groups: list[cl.GroupDesc] = []
        self._block_size = 0
        self._sched: cl.SchedulerLogic | None = None
        # worker side
        self._ready = False
        self._rank = 0
        self._views: dict[int, list[torch.Tensor]] = {}
        self._sizes: dict[int, int] = {}
        self._names: dict[int, str] = {}
        self._ns_ids: dict[int, bytes] = {}
        self._mm: mmap.mmap | None = None
        self._host: torch.Tensor | None = None
        self._pinned_addr: int | None = None
        self._ram_path = ""
        self._ram_id: tuple[int, int] | None = None
        self._limit = 0
        self._hello: dict[str, Any] = {}
        self._window = 1  # blocks per load window (connector_logic.window_blocks)
        self._device: torch.device | None = None
        self._tracker = cl.LoadTracker()
        self._jobs: queue.Queue[cl.LoadJob | None] = queue.Queue()
        self._loader: threading.Thread | None = None
        # The next load window's begin_load runs here, on its own daemon connection,
        # while the loader thread copies the current one.
        self._prefetch: ThreadPoolExecutor | None = None
        try:
            self._setup(vllm_config, kv_cache_config)
        except _Off as exc:
            self._off(str(exc))
        except Exception as exc:  # fail open
            self._off(f"setup failed: {exc!r}")

    # ------------------------------------------------------------------ setup
    def _off(self, why: str) -> None:
        self._sched = None
        self._ready = False
        if not self._off_logged:
            self._off_logged = True
            logger.warning("LM Warden KV store off (%s): %s", self._role.name, why)

    def _setup(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> None:
        cfg = cl.parse_config(self._kv_transfer_config.kv_connector_extra_config)
        if cfg is None:
            raise _Off(
                "kv_connector_extra_config needs socket, tiers ram|ram+nvme|nvme, min_tokens"
            )
        secret = cl.read_secret(os.environ.get(cl.KEY_ENV))
        if secret is None:
            raise _Off(f"{cl.KEY_ENV} missing, unreadable or not v1:<64 hex>")
        self._block_size, self._groups, self._model_ns = self._namespace(
            vllm_config, kv_cache_config, cfg
        )
        try:
            self._model = model_tag(cfg.owner, self._model_ns.revision_sha)
        except ValueError as exc:
            raise _Off(f"bad owner: {exc}") from None
        self._cfg = cfg
        self._client = Client(cfg.socket)
        if self._role == KVConnectorRole.SCHEDULER:
            store_ids = {
                g.gid: namespace_id(cl.store_namespace(self._model_ns, g, 0)) for g in self._groups
            }
            self._sched = cl.SchedulerLogic(
                self._client,
                secret,
                cfg,
                namespace_id(self._model_ns),
                self._groups,
                store_ids,
                self._block_size,
            )
            logger.info(
                "LM Warden KV store on: block size %d, %d groups, tiers %s",
                self._block_size,
                len(self._groups),
                cfg.tiers,
            )

    @staticmethod
    def _namespace(
        vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, cfg: cl.ConnectorConfig
    ) -> tuple[int, list[cl.GroupDesc], Namespace]:
        mc = vllm_config.model_config
        cc = vllm_config.cache_config
        pc = vllm_config.parallel_config
        bs = int(cc.block_size)
        groups = []
        for gid, g in enumerate(kv_cache_config.kv_cache_groups):
            spec = g.kv_cache_spec
            sbs = getattr(spec, "block_size", None)
            if sbs is not None and int(sbs) != bs:
                raise _Off(f"group {gid} block size {sbs} != cache_config.block_size {bs}")
            groups.append(
                cl.GroupDesc(
                    gid=gid,
                    kind=type(spec).__name__,
                    n_layers=len(g.layer_names),
                    page_bytes=int(getattr(spec, "page_size_bytes", 0) or 0),
                    is_state=isinstance(spec, MambaSpec),
                    eagle=bool(getattr(g, "is_eagle_group", False)),
                )
            )
        if not groups:
            raise _Off("no KV cache groups")
        pp = int(pc.pipeline_parallel_size)
        if pp != 1:
            raise _Off(f"pipeline parallel {pp} is not supported")
        dcp = int(getattr(pc, "decode_context_parallel_size", 1) or 1)
        if dcp != 1:
            raise _Off(f"decode context parallel {dcp} is not supported")
        try:  # commits LM Warden resolved, never a branch name that can move
            revision, tok_revision = cl.namespace_revisions(
                cfg, mc.revision, getattr(mc, "tokenizer_revision", None)
            )
        except ValueError as exc:
            raise _Off(str(exc)) from None
        extra = [f"{k}={os.environ.get(k, '')}" for k in _LAYOUT_ENV]
        extra.append(f"mamba_cache_mode={getattr(cc, 'mamba_cache_mode', None)}")
        ns = cl.model_namespace(
            model_repo=str(mc.model),
            revision=revision,
            tokenizer=f"{mc.tokenizer}@{tok_revision}",
            kv_dtype=f"{cc.cache_dtype}/{mc.dtype}",
            block_size=bs,
            groups=groups,
            tp=int(pc.tensor_parallel_size),
            pp=pp,
            engine_family=f"vllm-{vllm.__version__}",
            layout_extra=",".join(extra),
        )
        return bs, groups, ns

    # ------------------------------------------------------- scheduler side
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        if self._sched is None:
            return 0, False
        try:
            token_ids = request.prompt_token_ids
            if token_ids is None:
                return 0, False
            return self._sched.match(
                request.request_id,
                token_ids,
                request.kv_transfer_params,
                num_computed_tokens,
                getattr(request, "cache_salt", None),
            )
        except Exception as exc:  # fail open
            logger.warning("LM Warden KV lookup failed: %r", exc)
            return 0, False

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        if self._sched is None or num_external_tokens == 0:
            return
        try:
            lists = _block_lists(blocks)
        except Exception as exc:
            logger.error("LM Warden KV: cannot read allocated blocks: %r", exc)
            lists = []  # the job is doomed: every planned block is missing
        self._sched.after_alloc(request.request_id, lists, num_external_tokens)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = LmwKVMeta()
        if self._sched is None:
            return meta
        try:
            nst = scheduler_output.num_scheduled_tokens
            sched = [
                (nr.req_id, nr.num_computed_tokens, nst.get(nr.req_id, 0))
                for nr in scheduler_output.scheduled_new_reqs
            ]
            cr = scheduler_output.scheduled_cached_reqs
            sched += [
                (rid, nct, nst.get(rid, 0))
                for rid, nct in zip(cr.req_ids, cr.num_computed_tokens, strict=False)
            ]
        except Exception as exc:  # fail open: no saves this step
            logger.warning("LM Warden KV: cannot read the scheduler output: %r", exc)
            sched = []
        meta.plan = self._sched.build(sched, self._blocks_of)
        return meta

    def _blocks_of(self, req_id: str) -> list[list[int | None]] | None:
        mgr = getattr(self, "_kv_cache_manager", None)
        if mgr is None:
            return None
        return _block_lists(mgr.get_blocks(req_id))

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished_all_groups(request, (block_ids,))

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        # Never delay: vLLM itself holds the blocks of a request aborted while waiting for
        # its load and frees them when the load is reported in finished_recving.
        if self._sched is not None:
            self._sched.finished(request.request_id)
        return False, None

    # ---------------------------------------------------------- worker side
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self._model_ns is None or self._client is None:
            return
        try:
            self._setup_worker(kv_caches)
        except _Off as exc:
            self._off(str(exc))
        except Exception as exc:  # fail open
            self._off(f"worker setup failed: {exc!r}")

    def _setup_worker(self, kv_caches: dict[str, torch.Tensor]) -> None:
        from vllm.distributed import get_tensor_model_parallel_rank

        assert self._model_ns is not None and self._client is not None
        try:
            self._rank = get_tensor_model_parallel_rank()
        except Exception:
            self._rank = 0
        nb = self._kv_cache_config.num_blocks
        for g, vg in zip(self._groups, self._kv_cache_config.kv_cache_groups, strict=True):
            spec = vg.kv_cache_spec
            per_layer = spec.kv_cache_specs if isinstance(spec, UniformTypeKVCacheSpecs) else {}
            views = []
            for ln in vg.layer_names:
                t = kv_caches[ln]
                if isinstance(t, list | tuple):
                    t = t[0]
                lspec = per_layer.get(ln, spec)
                page = lspec.page_size_bytes
                ref = group_kernel_blocks(t, nb)
                es = ref.element_size()
                stride = ref.stride(0) * es if isinstance(lspec, AttentionSpec) else page
                v = torch.tensor([], dtype=torch.int8, device=ref.device).set_(
                    ref.untyped_storage(), ref.storage_offset() * es, (nb, page), (stride, 1)
                )
                views.append(v)
            if not views:
                raise _Off(f"group {g.gid} has no layers")
            self._views[g.gid] = views
            self._sizes[g.gid] = sum(int(v.shape[1]) for v in views)
            self._names[g.gid] = cl.store_group_name(g, self._rank)
            self._device = views[0].device
        namespaces = [
            (
                g.gid,
                cl.store_namespace(self._model_ns, g, self._rank),
                {self._names[g.gid]: self._sizes[g.gid]},
            )
            for g in self._groups
        ]
        try:  # register, map, register again (attach_store: survives a daemon restart)
            assert self._cfg is not None
            self._ns_ids = cl.attach_store(
                self._client,
                namespaces,
                self._map_ram,
                self._same_file,
                nvme=cl.uses_nvme(self._cfg.tiers),
                staging=cl.uses_staging(self._cfg.tiers),
                owner=self._cfg.owner,
                model=self._model,
                rank=self._rank,
                groups=self._groups,
            )
        except cl.SetupError as exc:
            raise _Off(str(exc)) from None
        if not cl.store_supports(self._cfg.tiers, self._hello):
            raise _Off("the daemon is too old for disk-only")
        block_bytes = max(_page_round(s) for s in self._sizes.values())
        self._window = cl.window_blocks(
            self._limit, block_bytes, self._model_ns.tp * self._model_ns.pp
        )
        self._prefetch = ThreadPoolExecutor(1, thread_name_prefix="lmw-kv-prefetch")
        self._loader = threading.Thread(target=self._load_loop, name="lmw-kv-load", daemon=True)
        self._loader.start()
        self._ready = True
        logger.info(
            "LM Warden KV worker rank %d ready: %d groups, %d bytes/block, RAM file %d bytes, "
            "loads in windows of %d blocks",
            self._rank,
            len(self._groups),
            sum(self._sizes.values()),
            self._limit,
            self._window,
        )

    def _map_ram(self) -> None:
        """Map the daemon's whole RAM file once and pin it once."""
        assert self._client is not None
        h = self._client.hello()
        if not h or h.get("protocol") != 1:
            raise _Off("daemon unreachable or wrong protocol")
        self._hello = h
        path, ram_bytes = h.get("ram_path"), h.get("ram_bytes")
        if not isinstance(path, str) or not isinstance(ram_bytes, int) or ram_bytes <= 0:
            raise _Off("daemon has no RAM tier")
        fd = os.open(path, os.O_RDWR)
        try:
            st = os.fstat(fd)
            if st.st_size < ram_bytes:
                raise _Off("RAM file smaller than the daemon reports")
            mm = mmap.mmap(fd, st.st_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        host = torch.frombuffer(mm, dtype=torch.int8)
        addr = host.data_ptr()
        rc = torch.cuda.cudart().cudaHostRegister(addr, st.st_size, 0)  # cudaHostRegisterDefault
        if not _cuda_ok(rc):
            del host
            mm.close()
            raise _Off(f"cudaHostRegister failed: {rc}")
        self._mm, self._host, self._pinned_addr = mm, host, addr
        self._ram_path, self._ram_id = path, (st.st_dev, st.st_ino)
        self._limit = min(st.st_size, ram_bytes)

    def _same_file(self) -> bool:
        """Is the daemon's RAM file still the one we mapped? A restarted daemon has a new one."""
        try:
            st = os.stat(self._ram_path)
        except OSError:
            return False
        return (st.st_dev, st.st_ino) == self._ram_id

    def _copy_in(self, gid: int, block_id: int, off: int) -> None:
        assert self._host is not None
        for v in self._views[gid]:
            n = v.shape[1]
            v[block_id].copy_(self._host[off : off + n], non_blocking=True)
            off += n

    def _copy_out(self, gid: int, block_id: int, off: int) -> None:
        assert self._host is not None
        for v in self._views[gid]:
            n = v.shape[1]
            self._host[off : off + n].copy_(v[block_id], non_blocking=True)
            off += n

    def _load_loop(self) -> None:
        cl.serve_loads(
            self._jobs,
            self._tracker,
            self._loader_setup,
            self._run_load,
            self._record_hits,
            lambda exc: self._off(f"the loader thread cannot start: {exc!r}"),
        )

    def _loader_setup(self) -> torch.cuda.Stream:
        assert self._device is not None
        torch.cuda.set_device(self._device)
        return torch.cuda.Stream(device=self._device)

    def _run_load(self, job: cl.LoadJob, stream: torch.cuda.Stream) -> list[int] | None:
        assert self._client is not None
        nvme: list[int] | None = None
        if self._ready:
            with torch.cuda.stream(stream):
                nvme = cl.run_load_job(
                    self._client,
                    job,
                    self._ns_ids,
                    self._names,
                    self._sizes,
                    self._limit,
                    self._copy_in,
                    stream.synchronize,
                    self._same_file,
                    window=self._window,
                    prefetch=None if self._prefetch is None else self._prefetch.submit,
                )
        if nvme is None and self._ready and not self._same_file():
            self._off("the daemon restarted; restart the engine to use the KV store again")
        return nvme

    def _record_hits(self, job: cl.LoadJob, nvme: list[int], elapsed: float) -> None:
        """Rank 0 records the turn's hits; before the load is reported finished (M1)."""
        if self._client is not None:
            cl.record_load_hits(self._client, job, nvme, elapsed, self._block_size, self._rank)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        if not self.has_connector_metadata():
            return
        meta = self._get_connector_metadata()
        if not isinstance(meta, LmwKVMeta):
            return
        alive = self._loader is not None and self._loader.is_alive()
        for job in meta.plan.loads:
            # never leave a request waiting for a load nobody will do
            cl.submit_load(self._tracker, job, self._ready, alive, self._jobs.put)

    def get_transfer_results(self, finished_req_ids: set[str]) -> KVConnectorTransferResults:
        # Request-level results only: with more than one KV cache group, vLLM 0.32 refuses
        # block-level errors (get_block_ids_with_load_errors stays the base's empty set).
        # A failed load is in both finished_recving and failed_recving, so vLLM resets the
        # request to num_computed_tokens = 0 and recomputes it. get_finished is not
        # overridden: it would lose the failure flag (base returns nothing).
        recving, failed = self._tracker.poll()
        return KVConnectorTransferResults(
            finished_sending=set(), finished_recving=recving, failed_recving=failed
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any, **kwargs: Any
    ) -> None:
        return

    def wait_for_save(self) -> None:
        if not self._ready or not self.has_connector_metadata():
            return
        meta = self._get_connector_metadata()
        if not isinstance(meta, LmwKVMeta) or not meta.plan.saves:
            return
        assert self._client is not None
        try:
            torch.cuda.synchronize()  # this step's forward has written the blocks
            cl.run_saves(
                self._client,
                meta.plan.saves,
                self._ns_ids,
                self._names,
                self._sizes,
                self._limit,
                self._copy_out,
                torch.cuda.current_stream().synchronize,
                self._same_file,
            )
        except Exception as exc:  # fail open
            logger.warning("LM Warden KV save failed: %r", exc)
        if not self._same_file():
            self._off("the daemon restarted; restart the engine to use the KV store again")

    def shutdown(self) -> None:
        self._ready = False
        if self._loader is not None:
            self._jobs.put(None)
            self._loader.join(timeout=5)
            self._loader = None
        if self._prefetch is not None:
            self._prefetch.shutdown(wait=False, cancel_futures=True)
            self._prefetch = None
        if self._pinned_addr is not None:
            try:
                torch.cuda.cudart().cudaHostUnregister(self._pinned_addr)
            except Exception:
                pass
            self._pinned_addr = None
        self._host = None
        if self._mm is not None:
            try:
                self._mm.close()
            except (BufferError, ValueError):
                pass  # a view is still alive; the process is going away anyway
            self._mm = None
        if self._client is not None:
            self._client.close()
