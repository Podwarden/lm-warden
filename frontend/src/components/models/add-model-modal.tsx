"use client";

import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import useSWR, { useSWRConfig } from "swr";
import { Modal } from "@/components/ui/modal";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { authFetch, authFetchJSON } from "@/lib/auth-fetch";
import type { TemplateDTO } from "@/components/templates/template-list";
import {
  classifyFit,
  formatBytes,
  recommendMaxModelLen,
  verdictBadgeClass,
  verdictLabel,
  type FitVerdict,
} from "@/lib/fit";
import { groupShardFamilies, type ShardFamily } from "@/lib/sharding";
import { GpuChecklist, type GpuInfo } from "@/components/gpu/gpu-checklist";

// Parallelism strategy mirrors backend `ModelCreate.parallelism_strategy`
// (`app/models/schemas.py` Literal["tp","pp","auto"]). `auto` is the safe
// default — single GPU is no-parallelism, multi-GPU defaults to TP. Single-
// host PP is explicitly accepted (#87 CTO call); the backend's cmd_builder
// (#88) emits --pipeline-parallel-size in that case, and vLLM accepts it.
type ParallelismStrategy = "auto" | "tp" | "pp";

// Mirrors backend ModelCreate bounds (`Field(default=1, ge=1, le=64)`).
const MAX_BATCH_SIZE_MIN = 1;
const MAX_BATCH_SIZE_MAX = 64;

// The image channel an engine-version pin resolves against. `resolve_image`
// needs (channel, version) as a pair -- a version on its own is stored and
// never turned into an image. The wizard offers only the stable channel: the
// other two exist for the Try-stack panel's trial-and-error loop, which is
// where an operator goes when stable has already failed. Adding a channel
// dropdown here would ask a question nobody creating a model can answer yet.
const ENGINE_CHANNEL = "cuda-stable";

// Mirror app/models/schemas.py:SLUG_RE so the user gets immediate feedback
// without round-tripping a malformed request through the API. Backend is
// still the source of truth — these client checks are a UX nicety, not a
// security boundary.
const SLUG_RE = /^[a-zA-Z0-9._-]+$/;
const HF_REPO_RE = /^[\w.-]+\/[\w.-]+$/;

// ---- Discovery / fit-preview wire shapes ----------------------------------
// `app/models/routes_api.py:discover_model_repo` returns plain `dict[str, Any]`
// (the api-types generator can't infer this so we model it locally). Keep
// these mirrored against `app/models/discovery.py:DiscoveryResult.to_dict`.

type FileKind =
  | "safetensors_single"
  | "safetensors_sharded"
  | "gguf"
  // Sub-project C. A multimodal projector is a GGUF, but it is NOT weights:
  // llama.cpp takes it via --mmproj alongside -m. Deliberately absent from
  // isWeightsFile, so it never appears in the weights radio -- picking it there
  // produces a load failure that reads like a corrupt download.
  | "mmproj"
  | "pytorch_bin"
  | "config"
  | "tokenizer"
  | "other";

/** Which engine serves a model. Mirrors ModelCreate.backend's Literal. */
type BackendName = "vllm" | "llamacpp";

/**
 * Decision D6's pre-selection rule, stated once.
 *
 * A `.gguf` weights file PRE-SELECTS llama.cpp -- in this wizard only. It is
 * never persisted logic and never a server-side inference, and the caller
 * applies it only while the operator has not chosen: a GGUF that vLLM can also
 * serve must stay available to vLLM for throughput.
 *
 * BOTH selection paths go through this, and that is the point. The wizard picks
 * a first weights file automatically on discovery, so a rule applied only in
 * the click handler would leave a GGUF repo showing "vLLM" until the operator
 * happened to click a DIFFERENT file -- the pre-selection would be invisible in
 * exactly the common case it exists for.
 */
function backendForFilename(filename: string): BackendName {
  return filename.toLowerCase().endsWith(".gguf") ? "llamacpp" : "vllm";
}

interface DiscoveredFile {
  filename: string;
  size: number;
  kind: FileKind;
  quant: string | null;
  params: number | null;
}

/**
 * Per-file soft signal from `app/models/discovery.py:DiscoveryWarning` (#101).
 * Distinct from `errors` which is repo-level. Surfaced inline next to the file
 * row so the operator sees arch mismatches BEFORE picking — most GGUF load
 * failures are silent vLLM-side crashes that are very hard to diagnose
 * post-pull.
 */
interface DiscoveryWarning {
  type: "gguf_arch_unsupported" | "gguf_arch_unknown";
  filename: string;
  arch: string | null;
}

/** One entry of GET /api/system/backends. Only the fields this dialog reads. */
interface BackendCapability {
  name: string;
  display_name: string;
  /** The engine version baked into THIS warden image; null outside a build. */
  version: string | null;
  /** The ENGINE fact. Deliberately unused here — see the useSWR comment. */
  supports_version_pin: boolean;
  /** The DEPLOYMENT fact. This is the one a control is gated on. */
  version_pin_available: boolean;
  /** Non-null exactly when `version_pin_available` is false. */
  version_pin_reason: string | null;
  /**
   * null = this backend honours the operator's gpu_memory_utilization.
   * A number = it has no such flag and uses that fraction of the card.
   * Mirrors BackendCapabilities.vram_cap_fraction; must match what
   * /api/models/fit-preview used, or the client-side budget recompute
   * disagrees with the badge the server produced.
   */
  vram_cap_fraction: number | null;
  /**
   * Can this engine run N identical replicas (vLLM --data-parallel-size)?
   * Optional: an older server omits it, and the dialog then treats vLLM as
   * true and everything else as false (#286).
   */
  supports_data_parallel?: boolean;
}

interface BackendsResponse {
  default: string;
  driver: string;
  engine_version: string | null;
  backends: BackendCapability[];
}

interface DiscoveryResultDict {
  files: DiscoveredFile[];
  config: Record<string, unknown> | null;
  repo: Record<string, unknown>;
  errors: string[];
  /** Optional for backward compat with older backends; treat absent as []. */
  warnings?: DiscoveryWarning[];
}

/** Typed envelope for the `auth_required` 401 from `GET /api/models/discover`. */
interface DiscoverErrorDetail {
  error_code: "auth_required" | "repo_not_found" | "discovery_failed";
  message: string;
  repo_id: string;
  revision: string;
}

interface FitPreviewBreakdown {
  total_vram: number;
  weights_budget: number;
  kv_reserve: number;
  file_size: number;
  ratio: number;
  dtype_bytes: number;
  max_model_len_used: number;
}

interface FitPreviewResponse {
  verdict: FitVerdict;
  breakdown: FitPreviewBreakdown;
  recommended_max_model_len: number | null;
  warnings: string[];
}

// ---- Helpers --------------------------------------------------------------

/** Best-guess display bucket for the file-kind badge. */
function kindBadgeLabel(kind: FileKind): string {
  switch (kind) {
    case "safetensors_single":
    case "safetensors_sharded":
      return "safetensors";
    case "gguf":
      return "gguf";
    case "pytorch_bin":
      return "bin";
    default:
      return kind;
  }
}

/** Is this a row a candidate weights file (something the operator can pick)? */
function isWeightsFile(kind: FileKind): boolean {
  return (
    kind === "safetensors_single" ||
    kind === "safetensors_sharded" ||
    kind === "gguf" ||
    kind === "pytorch_bin"
  );
}

const MIB = 1024 * 1024;

/** Total VRAM (bytes) across selected GPU indices. */
function vramBudget(gpus: GpuInfo[], selected: Set<number>, gmu: number): number {
  let total = 0;
  for (const g of gpus) {
    if (selected.has(g.index)) total += g.memory_total_mib * MIB;
  }
  return Math.floor(total * gmu);
}

/** Best-effort derivation of a default served_model_name from `owner/repo[#filename]`. */
function deriveServedName(hfRepo: string, filename: string | null): string {
  const repoTail = hfRepo.split("/").pop() ?? "";
  const base = (filename ?? "").split("/").pop() ?? "";
  // Strip extension and known weights suffixes for a cleaner default.
  const stripped = base
    .replace(/\.(safetensors|gguf|bin)$/i, "")
    .replace(/-\d{5}-of-\d{5}$/i, "");
  if (stripped && stripped !== repoTail) return `${repoTail}-${stripped}`.toLowerCase();
  return repoTail.toLowerCase();
}

// ---- Component ------------------------------------------------------------

type Stage = "enter-repo" | "discovering" | "select-file" | "submitting";

interface AddModelModalProps {
  open: boolean;
  onClose: () => void;
}

export function AddModelModal({ open, onClose }: AddModelModalProps) {
  const { mutate } = useSWRConfig();
  const [stage, setStage] = useState<Stage>("enter-repo");
  const [hfRepo, setHfRepo] = useState("");
  const [hfRevision, setHfRevision] = useState("main");
  const [name, setName] = useState("");
  // #162 — chosen engine template. Empty string == "Custom" (no template),
  // which leaves every field user-editable and omits template_id from the
  // create POST so the backend takes the legacy direct path. A non-empty
  // value prefills the wizard's repo/revision/max-len and is forwarded so the
  // backend resolves the remaining knobs (dtype, TP size, GMU, engine combo)
  // from the stored template.
  const [templateId, setTemplateId] = useState("");
  // Sub-project C: WHICH ENGINE serves this model. `vllm` is the persisted
  // default (decision D6) -- a client that never chooses behaves exactly as
  // before. `backendTouched` is what makes the .gguf rule a SUGGESTION rather
  // than a policy: once the operator has answered, changing the weights file
  // must not silently undo "serve this GGUF on vLLM for the throughput".
  const [backend, setBackend] = useState<BackendName>("vllm");
  const [backendTouched, setBackendTouched] = useState(false);
  // llama.cpp only. The multimodal projector GGUF that sits beside the weights
  // in the same repo; llama-server takes it as a separate --mmproj file.
  const [mmprojFilename, setMmprojFilename] = useState("");
  const [authRequired, setAuthRequired] = useState<DiscoverErrorDetail | null>(null);
  const [discovery, setDiscovery] = useState<DiscoveryResultDict | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [gpus, setGpus] = useState<GpuInfo[]>([]);
  // The setup wizard's `allowed_gpu_indices`, from GET /api/system/gpus. null
  // means "none recorded" -- no restriction -- which is deliberately NOT the
  // same as an empty list.
  const [allowedGpuIndices, setAllowedGpuIndices] = useState<number[] | null>(null);
  const [selectedGpus, setSelectedGpus] = useState<Set<number>>(new Set());
  // Engine version pin. Only ever sent when the ACTIVE backend's
  // `version_pin_available` is true on this deployment; see the Engine version
  // block in the render for why gating on `supports_version_pin` instead would
  // reintroduce #177.
  const [engineVersion, setEngineVersion] = useState("");
  const [selectedFilename, setSelectedFilename] = useState<string | null>(null);
  const [fitByFilename, setFitByFilename] = useState<Record<string, FitPreviewResponse>>({});
  const [fitInFlight, setFitInFlight] = useState<Set<string>>(new Set());
  // Advanced-section state (#87). The backend accepts all three optional;
  // we keep empty-string for the numeric inputs so a blank field means
  // "let the backend decide" rather than coercing to 0 (which the backend's
  // `gt=0` validator would reject).
  const [parallelismStrategy, setParallelismStrategy] =
    useState<ParallelismStrategy>("auto");
  const [maxBatchSize, setMaxBatchSize] = useState<string>("1");
  // Data-parallel replicas (#286). A string like the other Advanced inputs so
  // a half-typed value never round-trips through Number().
  const [dataParallelSize, setDataParallelSize] = useState<string>("1");
  const [maxModelLen, setMaxModelLen] = useState<string>("");
  // #106 — optional GGUF plumbing for repos that omit config.json (common for
  // unsloth republishes). ``baseRepo`` defaults both ``--hf-config-path`` and
  // ``--tokenizer`` to the same upstream repo; ``tokenizerRepo`` is an
  // override for the rare case where the tokenizer lives elsewhere. Both
  // empty means "do not pass the flags" (legacy behaviour, server stores
  // null). Validated server-side against the same owner/name slug regex as
  // ``hf_repo``.
  const [baseRepo, setBaseRepo] = useState<string>("");
  const [tokenizerRepo, setTokenizerRepo] = useState<string>("");
  // Default gpu_memory_utilization matches backend (FitPreviewRequest field default).
  const GMU = 0.9;

  // Open-generation ref guards against stale in-flight responses landing on
  // a later session. Every full close (resetForm) bumps the counter; every
  // request captures the counter at issue time and refuses to apply its
  // result if the captured value no longer matches. This covers:
  //   1. Modal closed mid-discovery → reopened → late response from prior
  //      open would otherwise yank the new open into a stale select-file.
  //   2. fit-preview in flight when modal closes → resolves on unmounted
  //      tree → warning + stale `fitByFilename` entry for the next open.
  //   3. Reopened with a different `hf_repo` before original resolves →
  //      late response would surface a wrong-repo fit verdict.
  // Cheap, local, and doesn't require threading AbortControllers through
  // authFetch (the wrapper does support `signal` via RequestInit, but the
  // cancel paths here are limited to "ignore late results", not "free the
  // socket", so a guard is sufficient and keeps the diff scoped).
  const openGenRef = useRef(0);

  // Per-fetch sequence counter for fit-preview, separate from openGenRef.
  // openGenRef is bumped on modal close/reopen — it can't disambiguate two
  // concurrent fit-preview requests within the SAME open session. When the
  // operator edits `max_batch_size` while the first-paint preview is still
  // in flight (#87 fix-up), a slower first-paint response can land AFTER
  // the debounced refetch and clobber the fresh answer with stale data.
  // Every fetch captures the seq value at issue time and refuses to write
  // its result if the live seq has moved on. The seq is local to the
  // session so resetForm() doesn't need to touch it.
  const fitSeqRef = useRef(0);

  // Mirror `stage` into a ref so async callbacks (notably startDiscovery's
  // post-response branches) can sample the LATEST value without re-running
  // when stage changes. Reading `stage` from closure would always see the
  // value captured when startDiscovery was invoked ("discovering"), defeating
  // the Cancel race check.
  const stageRef = useRef<Stage>(stage);
  useEffect(() => {
    stageRef.current = stage;
  }, [stage]);

  // Stable ids so each Input can be linked to its hint <span> via
  // aria-describedby. The parent always keeps the modal mounted (it just
  // toggles `open`), so useId() values are stable across open cycles too.
  const nameId = useId();
  const nameHintId = useId();
  const repoId = useId();
  const revisionId = useId();

  const resetForm = useCallback(() => {
    // Bump first — any in-flight discovery/fit-preview captured the prior
    // value and will short-circuit when it resolves into this open session.
    openGenRef.current += 1;
    setStage("enter-repo");
    setHfRepo("");
    setHfRevision("main");
    setTemplateId("");
    setBackend("vllm");
    setBackendTouched(false);
    setMmprojFilename("");
    setName("");
    setAuthRequired(null);
    setDiscovery(null);
    setError(null);
    setSelectedGpus(new Set());
    setAllowedGpuIndices(null);
    setEngineVersion("");
    setSelectedFilename(null);
    setFitByFilename({});
    setFitInFlight(new Set());
    setParallelismStrategy("auto");
    setMaxBatchSize("1");
    setDataParallelSize("1");
    setMaxModelLen("");
    setBaseRepo("");
    setTokenizerRepo("");
  }, []);

  // #162 — engine templates for the "Template" dropdown. Fetched only while
  // the modal is open (null key when closed) so the closed-but-mounted modal
  // doesn't poll. authFetchJSON returns the merged builtin+user list.
  const { data: templates } = useSWR<TemplateDTO[]>(
    open ? "/api/models/templates" : null,
    authFetchJSON,
  );

  // Which engines this build has, and what the ACTIVE driver lets each of them
  // do. Two version-pin fields ship and they disagree on the deployment this
  // was reported from:
  //
  //   supports_version_pin   the ENGINE fact  -- can this engine be pinned?
  //   version_pin_available  the DEPLOYMENT fact -- can this driver honour it?
  //
  // Gate on the SECOND. Gating on the first lights up a selector on a box that
  // cannot swap the image, which is #177 verbatim. `version_pin_reason` is the
  // server's sentence for the disabled state -- the frontend cannot write it,
  // because it cannot tell whether the obstacle is the driver or the backend.
  const { data: backendCaps } = useSWR<BackendsResponse>(
    open ? "/api/system/backends" : null,
    authFetchJSON,
  );
  const activeBackendCaps = backendCaps?.backends.find((b) => b.name === backend) ?? null;
  // The VRAM cap fraction the SELECTED backend actually imposes. Falls back to
  // GMU when the capability list has not loaded yet, which matches the server
  // default (vLLM) rather than inventing a more generous number.
  const effectiveCapFraction =
    activeBackendCaps?.vram_cap_fraction ?? GMU;
  // #286: the replica control is offered only to an engine that can run
  // replicas. A missing key (older server) reads true for vLLM, false otherwise.
  const supportsDataParallel =
    activeBackendCaps?.supports_data_parallel ?? backend === "vllm";
  // While loading (undefined) do NOT disable: the same no-flash rule
  // try-stack-panel.tsx follows. Only a known `false` locks the control.
  const versionPinDisabled = activeBackendCaps
    ? activeBackendCaps.version_pin_available === false
    : false;
  // #215 (slice b): the wizard offers no engine-version pin the RUNNING DRIVER
  // cannot honour -- on the in-container driver a typed pin is persisted and
  // then refused at load with EnginePinUnsupported, and no form action would
  // unlock one (the docker engine driver is a deployment choice). The fact is
  // /api/system/engine's `supports_version_select`; the dialog already reads
  // the superset /api/system/backends, whose `driver` field is the same
  // machine-readable answer (both derive from the driver's
  // `supports_engine_image` with the same defensive default), so there is no
  // second request. While the response is still loading the selector stays up:
  // the same no-flash rule as `versionPinDisabled`.
  const versionSelectHidden = backendCaps?.driver === "subprocess";

  // Apply (or clear) a template selection. Prefills the wizard fields the
  // enter-repo stage manages directly (repo, revision, max_model_len) and
  // stashes the id for the create POST; the backend resolves dtype / TP size /
  // GMU / engine from the stored template. "Custom" (empty id) clears the
  // stash and leaves the current field values untouched so the operator can
  // hand-tune without losing what they typed.
  const applyTemplate = useCallback(
    (id: string) => {
      setTemplateId(id);
      if (!id) return;
      const tpl = (templates ?? []).find((t) => t.id === id);
      if (!tpl) return;
      setHfRepo(tpl.hf_repo);
      setHfRevision(tpl.hf_revision || "main");
      setMaxModelLen(
        tpl.max_model_len != null ? String(tpl.max_model_len) : "",
      );
      // #286: a template saved from a data-parallel row restores its replicas.
      setDataParallelSize(String(tpl.data_parallel_size ?? 1));
    },
    [templates],
  );

  // The parent always renders <AddModelModal> (mount-stable) and just flips
  // `open`, so without an explicit reset the user's previous keystrokes
  // would resurface the next time the modal opens. Wrap onClose so Cancel,
  // backdrop click, Escape, and the Modal X button all go through the same
  // reset+close path.
  const handleClose = useCallback(() => {
    resetForm();
    onClose();
  }, [resetForm, onClose]);

  // Stage 1 → discovering: hit `GET /api/models/discover` and route the
  // result into either `auth_required` (gated repo) or `select-file`.
  //
  // Cancel race (I1, !86 CR): the operator can click Cancel on the
  // discovering spinner before the response lands. Pre-fix, the late
  // response unconditionally setStage("select-file") and populated form
  // state — yanking the operator into a screen they explicitly backed
  // out of. authFetch doesn't yet thread an AbortController through, so
  // we cancel cooperatively: capture `stage` at issue time (Cancel flips
  // it back to "enter-repo") and short-circuit every terminal setter
  // when the captured stage no longer holds. The openGen capture catches
  // the harder case where the operator hits Cancel → closes the modal →
  // reopens before the response resolves: stage IS "discovering" again,
  // but the gen number has moved on, so we still bail.
  async function startDiscovery(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setAuthRequired(null);
    // #283 — a repo id or revision pasted from the HuggingFace Hub UI
    // carries incidental leading/trailing whitespace. Normalize at this
    // gate and persist it, so the discover query, the auth-required
    // detail, the fit-preview body, the create POST and the spinner line
    // all read the same clean value. Interior whitespace is untouched —
    // the owner/name regex below still refuses it.
    const repo = hfRepo.trim();
    const rev = hfRevision.trim();
    if (repo !== hfRepo) setHfRepo(repo);
    if (rev !== hfRevision) setHfRevision(rev);
    if (!HF_REPO_RE.test(repo)) {
      setError("HF repo must be in owner/name format (e.g., meta-llama/Llama-3-8B)");
      return;
    }
    setStage("discovering");
    const gen = openGenRef.current;
    // Predicate captured by closure — every terminal branch consults it
    // before mutating state. Using a function (not just a bool) so each
    // check re-reads the live React state, not a stale snapshot.
    const stillDiscovering = () =>
      openGenRef.current === gen && stageRef.current === "discovering";
    try {
      const qs = new URLSearchParams({ repo_id: repo, revision: rev || "main" });
      const r = await authFetch(`/api/models/discover?${qs.toString()}`);
      if (!stillDiscovering()) return;
      if (r.status === 401) {
        // Gated/private repo — surface the CTA, do not flip back to enter-repo.
        const detail = await r
          .json()
          .then((d) => (d?.detail as DiscoverErrorDetail) ?? null)
          .catch(() => null);
        if (!stillDiscovering()) return;
        setAuthRequired(
          detail ?? {
            error_code: "auth_required",
            message:
              "HuggingFace Hub requires authentication for this repo. Update the HF token in Settings.",
            repo_id: repo,
            revision: rev || "main",
          },
        );
        setStage("select-file");
        return;
      }
      if (!r.ok) {
        const detail = await r
          .json()
          .then((d) => (d?.detail as DiscoverErrorDetail | string | null) ?? null)
          .catch(() => null);
        if (!stillDiscovering()) return;
        const msg =
          (detail && typeof detail === "object" && "message" in detail
            ? detail.message
            : typeof detail === "string"
              ? detail
              : null) ?? `Discovery failed (HTTP ${r.status})`;
        setError(msg);
        setStage("enter-repo");
        return;
      }
      const data = (await r.json()) as DiscoveryResultDict;
      if (!stillDiscovering()) return;
      setDiscovery(data);
      // Pick a sensible first weights-file as the default selection so the
      // operator sees a fit row immediately rather than an empty table.
      const firstWeights = data.files.find((f) => isWeightsFile(f.kind));
      if (firstWeights) {
        setSelectedFilename(firstWeights.filename);
        setName(deriveServedName(repo, firstWeights.filename));
        // D6's pre-selection has to run here too, not only in the click
        // handler: this IS the selection for a repo the operator does not
        // re-click, which is the common case.
        setBackend(backendForFilename(firstWeights.filename));
      }
      // Best-effort load GPUs in parallel with the discovery transition.
      // `/api/system/gpus` already has a 2 s TTL cache server-side. The
      // GPU fetch is independent — even if the operator transitions to
      // a later screen we still want gpus seeded for the open session,
      // so we only gate on openGen (a Cancel-then-reopen would have
      // bumped it).
      authFetchJSON<{
        gpus: GpuInfo[];
        probed_at: string;
        probe_error: string | null;
        // The setup wizard's allowlist. null = none recorded = no restriction;
        // NOT the same as [].
        allowed_indices?: number[] | null;
      }>("/api/system/gpus")
        .then((resp) => {
          if (openGenRef.current !== gen) return;
          const list = resp.gpus ?? [];
          const allowed = resp.allowed_indices ?? null;
          setGpus(list);
          setAllowedGpuIndices(allowed);
          // Default selection = the first SELECTABLE GPU, so the fit row has a
          // non-zero budget on first paint. It used to be `list[0].index`
          // unconditionally, which on a deployment whose allowlist excludes
          // GPU 0 pre-filled a form the server was always going to 400.
          const selectable =
            allowed === null ? list : list.filter((g) => allowed.includes(g.index));
          if (selectable.length > 0) {
            setSelectedGpus(new Set([selectable[0].index]));
          }
        })
        .catch(() => {
          if (openGenRef.current !== gen) return;
          // Empty GPU list still lets the operator fill in the form; the
          // backend POST /api/models will surface the real validation
          // error if their pick isn't in `allowed_gpu_indices`.
          setGpus([]);
          setAllowedGpuIndices(null);
        });
      setStage("select-file");
    } catch (err) {
      if (!stillDiscovering()) return;
      setError(err instanceof Error ? err.message : "Network error during discovery");
      setStage("enter-repo");
    }
  }

  // Fetch a fit-preview for a (filename, gpu set, advanced-overrides) tuple
  // and cache the result keyed by filename.
  //
  // Advanced numeric inputs (#87 fix-up): `max_batch_size` and `max_model_len`
  // both feed the backend's KV-reserve math (kv_reserve = bytes_per_token *
  // max_model_len * max_batch_size). Without threading them through, the
  // tooltip + recommended-L hint would describe a different submission than
  // the one Add actually POSTs. Blank inputs are omitted so the backend's
  // `gt=0` validators don't reject — the FastAPI handler treats absent
  // fields as "use the request default" (`max_batch_size=1`,
  // `max_model_len=None` falls back to `config.max_position_embeddings`).
  //
  // Dedup key (#87): includes the override values so a debounced refetch
  // (operator typing in the Advanced section) isn't blocked by an earlier
  // first-paint that's still in flight for the same filename.
  //
  // Stale-response guards:
  //   - `openGenRef` covers cross-session staleness (modal closed/reopened);
  //     mirrors the I3 fix from !86 CR.
  //   - `fitSeqRef` covers same-session staleness (rapid edits → a slow
  //     first-paint clobbering the latest answer). Each request captures
  //     the seq at issue time and refuses to write if it's been superseded.
  const fetchFitPreview = useCallback(
    async (filename: string, gpuIndices: number[]) => {
      if (gpuIndices.length === 0) return;
      // Parse the Advanced numeric overrides into the wire types. We mirror
      // the same validation the submit() path uses so an in-progress invalid
      // value (e.g. mid-typing "1") doesn't fire a request the backend will
      // 422 — just omit until the value parses cleanly.
      const batchN = Number(maxBatchSize);
      const includeBatch =
        maxBatchSize.trim() !== "" &&
        Number.isInteger(batchN) &&
        batchN >= MAX_BATCH_SIZE_MIN &&
        batchN <= MAX_BATCH_SIZE_MAX;
      const lenN = Number(maxModelLen);
      const includeLen =
        maxModelLen.trim() !== "" && Number.isInteger(lenN) && lenN > 0;
      const dpN = Number(dataParallelSize);
      const includeDp =
        supportsDataParallel &&
        Number.isInteger(dpN) &&
        dpN > 1 &&
        gpuIndices.length % dpN === 0;
      const key = JSON.stringify({
        filename,
        gpu_indices: gpuIndices,
        max_batch_size: includeBatch ? batchN : null,
        max_model_len: includeLen ? lenN : null,
        data_parallel_size: includeDp ? dpN : null,
      });
      if (fitInFlight.has(key)) return;
      const gen = openGenRef.current;
      fitSeqRef.current += 1;
      const mySeq = fitSeqRef.current;
      setFitInFlight((prev) => {
        const next = new Set(prev);
        next.add(key);
        return next;
      });
      try {
        const body: Record<string, unknown> = {
          repo_id: hfRepo,
          revision: hfRevision || "main",
          filename,
          gpu_indices: gpuIndices,
          gpu_memory_utilization: GMU,
          // The verdict is about a specific engine. Without this the server
          // defaults to vLLM and applies GMU to a llama.cpp row, which is
          // what made an 11.28 GiB GGUF read "won't fit" on a 16 GiB card.
          backend,
        };
        if (includeBatch) body.max_batch_size = batchN;
        if (includeLen) body.max_model_len = lenN;
        if (includeDp) body.data_parallel_size = dpN;
        const r = await authFetch("/api/models/fit-preview", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!r.ok) return;
        const data = (await r.json()) as FitPreviewResponse;
        if (openGenRef.current !== gen) return;
        // A later fit-preview already wrote — don't overwrite the fresher
        // answer with a stale one. Crucial when the operator changes
        // batch/len while the first-paint is still in flight.
        if (fitSeqRef.current !== mySeq) return;
        setFitByFilename((prev) => ({ ...prev, [filename]: data }));
      } catch {
        // Network blip — leave the row's badge in the optimistic
        // client-classified state; the operator can re-tick a GPU box
        // to retry, and the backend will validate at submit time.
      } finally {
        // Always clear in-flight on the SAME open session. If the gen
        // moved on, the in-flight Set was cleared by resetForm already
        // and a write here would resurrect a phantom entry.
        if (openGenRef.current === gen) {
          setFitInFlight((prev) => {
            const next = new Set(prev);
            next.delete(key);
            return next;
          });
        }
      }
    },
    // `fitInFlight` reads only — the setter inside is functional, so
    // omitting it from deps doesn't cause stale-state bugs. Pinning it
    // *out* avoids an extra render that would queue duplicate fetches.
    // `maxBatchSize` / `maxModelLen` ARE in deps so the closure reads the
    // operator's latest overrides — debounced refetch on either change is
    // wired in the effect below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [hfRepo, hfRevision, maxBatchSize, maxModelLen, dataParallelSize, supportsDataParallel, backend],
  );

  // First-paint fit-preview for the auto-selected weights file. Runs when
  // the operator transitions into `select-file` AND has at least one GPU
  // picked. Subsequent recomputes are client-side via `classifyFit()`,
  // except for changes to the Advanced numeric overrides — those are
  // wired through the debounced refetch effect below.
  useEffect(() => {
    if (stage !== "select-file") return;
    if (!selectedFilename) return;
    if (selectedGpus.size === 0) return;
    if (fitByFilename[selectedFilename]) return;
    void fetchFitPreview(selectedFilename, Array.from(selectedGpus).sort((a, b) => a - b));
  }, [stage, selectedFilename, selectedGpus, fitByFilename, fetchFitPreview]);

  // Debounced refetch when the operator edits `max_batch_size` or
  // `max_model_len` in the Advanced section (#87 fix-up). Both fields flow
  // into the backend's KV-reserve term, so the cached preview becomes
  // stale the instant either changes — without this refetch the tooltip
  // primitives (`bytes_per_token`, `kv_reserve`, `weights_budget`, `ratio`)
  // and the "Recommended max_model_len" hint would describe the
  // first-paint defaults, not the values the modal would actually POST.
  //
  // 300ms is the conventional debounce window for a number field; an
  // operator typing "16384" produces five keystrokes in well under that
  // budget, so the request fires once with the settled value rather than
  // five times during typing. The first render of this effect is skipped
  // via `advancedRefetchInitialRender` so it doesn't double-fire with the
  // first-paint effect above on stage transition.
  // Switching engine invalidates every cached verdict. `fitByFilename` is
  // keyed by filename alone, so without this the badge keeps showing the
  // answer for the PREVIOUS backend -- the wrong number, rendered with full
  // confidence, which is worse than the dash on an unevaluated row. Dropping
  // the cache lets the first-paint effect above refetch for the new engine.
  const backendRefetchInitialRender = useRef(true);
  useEffect(() => {
    if (backendRefetchInitialRender.current) {
      backendRefetchInitialRender.current = false;
      return;
    }
    setFitByFilename({});
  }, [backend]);

  const advancedRefetchInitialRender = useRef(true);
  useEffect(() => {
    if (advancedRefetchInitialRender.current) {
      advancedRefetchInitialRender.current = false;
      return;
    }
    if (stage !== "select-file") return;
    if (!selectedFilename) return;
    if (selectedGpus.size === 0) return;
    const filename = selectedFilename;
    const indices = Array.from(selectedGpus).sort((a, b) => a - b);
    const t = setTimeout(() => {
      void fetchFitPreview(filename, indices);
    }, 300);
    return () => clearTimeout(t);
    // `stage`, `selectedFilename`, `selectedGpus` are intentionally
    // omitted: first-paint owns those triggers. We only want this effect
    // to fire when the Advanced overrides themselves change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [maxBatchSize, maxModelLen, dataParallelSize, fetchFitPreview]);

  // Derive per-file budget from the cached fit-preview response. Budget is
  // GPU-dependent — when the operator unchecks a GPU we want to reflect
  // that immediately client-side without re-fetching for every row. The
  // backend math is `total_vram * gpu_util - kv_reserve`; we recompute
  // `total_vram` locally from `gpus` and reuse the server-supplied
  // `kv_reserve` (which only depends on the model config + max_model_len,
  // not on which GPUs are selected).
  const liveBudget = useMemo(() => {
    if (!selectedFilename) return 0;
    const cached = fitByFilename[selectedFilename];
    const totalVram = (Array.isArray(gpus) ? gpus : [])
      .filter((g) => selectedGpus.has(g.index))
      .reduce((acc, g) => acc + g.memory_total_mib * MIB, 0);
    const cap = Math.floor(totalVram * effectiveCapFraction);
    if (!cached) return cap; // pre-server case: no KV reserve known yet
    return cap - cached.breakdown.kv_reserve;
  }, [selectedFilename, fitByFilename, gpus, selectedGpus, effectiveCapFraction]);

  function pickFilename(filename: string) {
    setSelectedFilename(filename);
    setName(deriveServedName(hfRepo, filename));
    // D6: a .gguf PRE-SELECTS llama.cpp -- in this wizard only. It is never
    // persisted logic and never a server-side inference, and it is always
    // overridable, because a GGUF that vLLM can also serve must stay available
    // to vLLM for throughput. Sub-project E adds the compatibility verdicts
    // that make this pre-selection intelligent; C stops at the suggestion.
    if (!backendTouched) {
      setBackend(backendForFilename(filename));
    }
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    if (!selectedFilename) {
      setError("Pick a weights file from the list");
      return;
    }
    if (!name) {
      setError("Served model name is required");
      return;
    }
    if (!SLUG_RE.test(name)) {
      setError("Served model name must be alphanumeric, dot, dash, or underscore only");
      return;
    }
    if (name.length > 100) {
      setError("Served model name must be at most 100 characters");
      return;
    }
    if (selectedGpus.size === 0) {
      setError("Select at least one GPU");
      return;
    }

    // Data-parallel replicas (#286): N identical replicas, so the GPU count
    // must split evenly. Checked here, before any POST, with the number the
    // operator can act on. Only meaningful for an engine that supports it.
    let parsedDp = 1;
    if (supportsDataParallel && dataParallelSize.trim() !== "") {
      const n = Number(dataParallelSize);
      if (!Number.isInteger(n) || n < 1 || selectedGpus.size % n !== 0) {
        setError(
          `Data-parallel replicas must divide the number of selected GPUs (${selectedGpus.size})`,
        );
        return;
      }
      parsedDp = n;
    }

    // Parse + validate the advanced numeric inputs. Empty = "let backend
    // decide" (omit from payload). Anything non-empty must parse as a
    // positive integer within the backend's accepted range. Note: single-
    // host PP is NOT blocked (CTO call on #87) — the backend accepts it
    // and vLLM's pipeline-parallel mode runs fine on one host.
    let parsedBatch: number | null = null;
    if (maxBatchSize.trim() !== "") {
      const n = Number(maxBatchSize);
      if (!Number.isInteger(n) || n < MAX_BATCH_SIZE_MIN || n > MAX_BATCH_SIZE_MAX) {
        setError(
          `max_batch_size must be an integer between ${MAX_BATCH_SIZE_MIN} and ${MAX_BATCH_SIZE_MAX}`,
        );
        return;
      }
      parsedBatch = n;
    }
    let parsedMaxModelLen: number | null = null;
    if (maxModelLen.trim() !== "") {
      const n = Number(maxModelLen);
      if (!Number.isInteger(n) || n <= 0) {
        setError("max_model_len override must be a positive integer (or leave blank)");
        return;
      }
      parsedMaxModelLen = n;
    }

    const body: Record<string, unknown> = {
      served_model_name: name,
      hf_repo: hfRepo,
      hf_revision: hfRevision || "main",
      gpu_indices: Array.from(selectedGpus).sort((a, b) => a - b),
      filename: selectedFilename,
      parallelism_strategy: parallelismStrategy,
      backend,
    };
    // llama.cpp only, and only when the operator picked one. Absent stays
    // absent so the column keeps its NULL rather than an empty string.
    if (backend === "llamacpp" && mmprojFilename) {
      body.mmproj_filename = mmprojFilename;
    }
    if (parsedBatch !== null) body.max_batch_size = parsedBatch;
    // tensor_parallel_size is deliberately not sent: the server derives
    // len(gpu_indices) / data_parallel_size, which is the one place the
    // invariant is computed.
    if (parsedDp > 1) body.data_parallel_size = parsedDp;
    if (parsedMaxModelLen !== null) body.max_model_len = parsedMaxModelLen;
    // Engine version pin. Sent ONLY when this deployment can honour it --
    // `versionPinDisabled` is derived from `version_pin_available`, the
    // deployment fact. Sending a pin the driver will discard is precisely the
    // silent no-op the two-field capability split exists to prevent, and the
    // backend refuses it anyway. The channel travels with it because
    // `resolve_image` needs both to produce an image; a version alone is
    // stored and never resolved.
    const trimmedEngineVersion = engineVersion.trim();
    if (!versionPinDisabled && trimmedEngineVersion !== "") {
      body.engine_vllm_version = trimmedEngineVersion;
      body.engine_channel = ENGINE_CHANNEL;
    }
    // #162 — forward the chosen template so the backend resolves dtype / TP
    // size / GMU / engine combo from the stored template. Empty == Custom,
    // which omits the field and keeps the legacy direct-create path.
    if (templateId) body.template_id = templateId;
    // #106 — Base repo defaults both --hf-config-path AND --tokenizer to the
    // same upstream repo (the common case for unsloth GGUF republishes). A
    // separate tokenizerRepo Input wins for --tokenizer when the operator
    // explicitly overrides it. Empty fields stay absent from the payload so
    // the backend's `_empty_string_to_none` validator stores NULL.
    const trimmedBase = baseRepo.trim();
    const trimmedTokenizer = tokenizerRepo.trim();
    if (trimmedBase !== "") {
      body.hf_config_repo = trimmedBase;
      body.tokenizer_repo = trimmedTokenizer !== "" ? trimmedTokenizer : trimmedBase;
    } else if (trimmedTokenizer !== "") {
      // Operator filled only the override — pass it on its own so they don't
      // silently lose the value just because Base was blank.
      body.tokenizer_repo = trimmedTokenizer;
    }

    setStage("submitting");
    try {
      const r = await authFetch("/api/models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const detail = await r.json().catch(() => null);
        const msg =
          (detail && typeof detail === "object" && "detail" in detail
            ? String((detail as { detail: unknown }).detail)
            : null) ?? `Failed to add model (HTTP ${r.status})`;
        setError(msg);
        setStage("select-file");
        return;
      }
      // Auto-trigger pull on register. The detail page has no Pull button —
      // a newly-registered row would sit at status="registered" forever
      // without this kick. Fire-and-forget: any failure surfaces on the
      // detail page via the row's last_error / status badge.
      try {
        const created = (await r.json()) as { id?: string };
        if (created.id) {
          authFetch(`/api/models/${encodeURIComponent(created.id)}/pull`, {
            method: "POST",
          }).catch(() => {});
        }
      } catch {
        /* ignore — row will still be visible, operator can retry */
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Network error");
      setStage("select-file");
      return;
    }

    // Success path — outside try/catch so a revalidation failure on the
    // already-created row doesn't reopen the error state for a model that
    // does exist server-side.
    resetForm();
    onClose();
    mutate("/api/models").catch(() => {});
  }

  // ---- Render ------------------------------------------------------------

  const busy = stage === "discovering" || stage === "submitting";
  const showGguf = selectedFilename?.toLowerCase().endsWith(".gguf") ?? false;

  return (
    // Disable the close path while a POST or discovery is in flight so the
    // X button doesn't unmount mid-fetch and detach the modal's state.
    <Modal open={open} onClose={busy ? () => {} : handleClose} title="Add model" size="lg">
      {stage === "enter-repo" && (
        <form onSubmit={startDiscovery} noValidate className="space-y-4">
          {/* #162 — engine template picker. "Custom" leaves every field
              user-editable; choosing a template prefills repo/revision/max-len
              and forwards template_id so the backend resolves the rest. */}
          <label className="block space-y-1">
            <span className="text-sm">Template</span>
            <select
              data-testid="template-select"
              className="mt-1 w-full rounded-md border border-slate-700 bg-slate-900 p-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500"
              value={templateId}
              onChange={(e) => applyTemplate(e.target.value)}
            >
              <option value="">Custom</option>
              {(templates ?? []).map((t) => (
                <option key={t.id} value={t.id}>
                  {t.label}
                </option>
              ))}
            </select>
          </label>

          <label htmlFor={repoId} className="block space-y-1">
            <span className="text-sm">HF repo</span>
            <Input
              id={repoId}
              name="hf_repo"
              aria-required="true"
              required
              value={hfRepo}
              onChange={(e) => setHfRepo(e.target.value)}
              placeholder="meta-llama/Llama-3-8B"
              autoComplete="off"
              autoFocus
            />
          </label>

          <label htmlFor={revisionId} className="block space-y-1">
            <span className="text-sm">HF revision</span>
            <Input
              id={revisionId}
              name="hf_revision"
              value={hfRevision}
              onChange={(e) => setHfRevision(e.target.value)}
              placeholder="main"
              autoComplete="off"
            />
          </label>

          {error && <p className="text-sm text-red-500">{error}</p>}

          <div className="flex justify-end gap-2 pt-2">
            <Button type="button" variant="outline" onClick={handleClose}>
              Cancel
            </Button>
            <Button type="submit">Discover</Button>
          </div>
        </form>
      )}

      {stage === "discovering" && (
        <div
          role="status"
          aria-live="polite"
          className="flex flex-col items-center justify-center gap-3 py-10"
        >
          <div
            className="h-8 w-8 animate-spin rounded-full border-2 border-slate-600 border-t-emerald-500"
            aria-hidden
          />
          <p className="text-sm text-slate-400">
            Discovering files for {hfRepo}@{hfRevision || "main"}…
          </p>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => {
              // No AbortController on authFetch yet — flip state back so
              // the in-flight discovery's resolution lands on a screen the
              // operator left, and gate that landing on `stage !== ...`
              // via the natural React effect dependency below.
              setStage("enter-repo");
            }}
          >
            Cancel
          </Button>
        </div>
      )}

      {stage === "select-file" && authRequired && (
        <AuthRequiredCta
          detail={authRequired}
          onBack={() => {
            setAuthRequired(null);
            setStage("enter-repo");
          }}
          onClose={handleClose}
        />
      )}

      {stage === "select-file" && !authRequired && discovery && (
        <SelectFileStage
          discovery={discovery}
          nameId={nameId}
          nameHintId={nameHintId}
          name={name}
          onNameChange={setName}
          gpus={gpus}
          selectedGpus={selectedGpus}
          onGpusChange={setSelectedGpus}
          allowedGpuIndices={allowedGpuIndices}
          selectedFilename={selectedFilename}
          onPickFilename={pickFilename}
          backend={backend}
          onBackendChange={(b) => {
            setBackend(b);
            setBackendTouched(true);
          }}
          engineVersion={engineVersion}
          onEngineVersionChange={setEngineVersion}
          engineVersionDisabled={versionPinDisabled}
          engineVersionReason={activeBackendCaps?.version_pin_reason ?? null}
          engineVersionRunning={activeBackendCaps?.version ?? null}
          engineVersionHidden={versionSelectHidden}
          mmprojFilename={mmprojFilename}
          onMmprojFilenameChange={setMmprojFilename}
          fitByFilename={fitByFilename}
          liveBudget={liveBudget}
          showGgufWarn={showGguf}
          error={error}
          onCancel={handleClose}
          onSubmit={submit}
          parallelismStrategy={parallelismStrategy}
          onParallelismStrategyChange={setParallelismStrategy}
          maxBatchSize={maxBatchSize}
          onMaxBatchSizeChange={setMaxBatchSize}
          dataParallelSize={dataParallelSize}
          onDataParallelSizeChange={setDataParallelSize}
          supportsDataParallel={supportsDataParallel}
          maxModelLen={maxModelLen}
          onMaxModelLenChange={setMaxModelLen}
          baseRepo={baseRepo}
          onBaseRepoChange={setBaseRepo}
          tokenizerRepo={tokenizerRepo}
          onTokenizerRepoChange={setTokenizerRepo}
        />
      )}

      {stage === "submitting" && (
        <div
          role="status"
          aria-live="polite"
          className="flex flex-col items-center justify-center gap-3 py-10"
        >
          <div
            className="h-8 w-8 animate-spin rounded-full border-2 border-slate-600 border-t-emerald-500"
            aria-hidden
          />
          <p className="text-sm text-slate-400">Registering model…</p>
        </div>
      )}
    </Modal>
  );
}

// ---- Sub-components -------------------------------------------------------

function AuthRequiredCta({
  detail,
  onBack,
  onClose,
}: {
  detail: DiscoverErrorDetail;
  onBack: () => void;
  onClose: () => void;
}) {
  return (
    <div className="space-y-4">
      <div
        role="alert"
        className="rounded-md border border-amber-700 bg-amber-900/20 p-3 text-sm text-amber-200"
      >
        <p className="font-medium">HuggingFace token required</p>
        <p className="mt-1 text-amber-300/90">
          {detail.repo_id}@{detail.revision} is gated or private. Add an HF token in Settings
          and try again.
        </p>
      </div>
      <p className="text-sm text-slate-400">{detail.message}</p>
      <div className="flex justify-end gap-2 pt-2">
        <Button type="button" variant="outline" onClick={onBack}>
          Back
        </Button>
        {/* Plain anchor — the operator may already be on /setup/hf-token,
            and Next.js's router can be flaky inside a portal-mounted modal. */}
        <a
          href="/setup/hf-token"
          onClick={onClose}
          className="inline-flex h-9 items-center justify-center rounded-md bg-emerald-600 px-4 text-sm font-medium text-white hover:bg-emerald-500"
        >
          Open token settings
        </a>
      </div>
    </div>
  );
}

interface SelectFileStageProps {
  discovery: DiscoveryResultDict;
  nameId: string;
  nameHintId: string;
  name: string;
  onNameChange: (v: string) => void;
  gpus: GpuInfo[];
  selectedGpus: Set<number>;
  onGpusChange: (s: Set<number>) => void;
  /** Setup's `allowed_gpu_indices`; null = none recorded = no restriction. */
  allowedGpuIndices: number[] | null;
  selectedFilename: string | null;
  onPickFilename: (f: string) => void;
  backend: BackendName;
  onBackendChange: (b: BackendName) => void;
  // Engine version pin, for the SELECTED backend. `engineVersionDisabled`
  // comes from `version_pin_available` (the deployment fact) and
  // `engineVersionReason` is the server's explanation for it.
  engineVersion: string;
  onEngineVersionChange: (v: string) => void;
  engineVersionDisabled: boolean;
  engineVersionReason: string | null;
  /** The version this warden image actually runs, for the placeholder. */
  engineVersionRunning: string | null;
  /**
   * #215 (slice b): true when the RUNNING DRIVER cannot honour a pin
   * (/api/system/engine's `supports_version_select`, read from the same
   * backends response the dialog already fetches). The whole block is hidden
   * and one line of help is shown instead.
   */
  engineVersionHidden: boolean;
  mmprojFilename: string;
  onMmprojFilenameChange: (v: string) => void;
  fitByFilename: Record<string, FitPreviewResponse>;
  liveBudget: number;
  showGgufWarn: boolean;
  error: string | null;
  onCancel: () => void;
  onSubmit: (e: React.FormEvent) => void | Promise<void>;
  // Advanced section (#87)
  parallelismStrategy: ParallelismStrategy;
  onParallelismStrategyChange: (v: ParallelismStrategy) => void;
  maxBatchSize: string;
  onMaxBatchSizeChange: (v: string) => void;
  dataParallelSize: string;
  onDataParallelSizeChange: (v: string) => void;
  supportsDataParallel: boolean;
  maxModelLen: string;
  onMaxModelLenChange: (v: string) => void;
  // #106 GGUF base-repo + tokenizer override
  baseRepo: string;
  onBaseRepoChange: (v: string) => void;
  tokenizerRepo: string;
  onTokenizerRepoChange: (v: string) => void;
}

function SelectFileStage({
  discovery,
  nameId,
  nameHintId,
  name,
  onNameChange,
  gpus,
  selectedGpus,
  onGpusChange,
  allowedGpuIndices,
  selectedFilename,
  onPickFilename,
  backend,
  onBackendChange,
  engineVersion,
  onEngineVersionChange,
  engineVersionDisabled,
  engineVersionReason,
  engineVersionRunning,
  engineVersionHidden,
  mmprojFilename,
  onMmprojFilenameChange,
  fitByFilename,
  liveBudget,
  showGgufWarn,
  error,
  onCancel,
  onSubmit,
  parallelismStrategy,
  onParallelismStrategyChange,
  maxBatchSize,
  onMaxBatchSizeChange,
  dataParallelSize,
  onDataParallelSizeChange,
  supportsDataParallel,
  maxModelLen,
  onMaxModelLenChange,
  baseRepo,
  onBaseRepoChange,
  tokenizerRepo,
  onTokenizerRepoChange,
}: SelectFileStageProps) {
  // Only the rows that represent a usable weights file are clickable; we
  // still render config.json / tokenizer rows so the operator gets a
  // complete picture of the repo, just non-selectable + dimmed.
  const rows = discovery.files;

  // Only `mmproj`-kind files (app/models/discovery.py classifies them). A
  // projector is a GGUF but it is NOT weights, so it is excluded from the
  // weights radio by isWeightsFile and offered here instead.
  const mmprojCandidates = useMemo(
    () => rows.filter((f) => f.kind === "mmproj"),
    [rows],
  );

  // #112: group sharded safetensors / GGUF into a single "family" row with a
  // disclosure-triangle that expands to individual shards. The fit-preview is
  // computed against the family's representative shard (shard-00001) — the
  // backend's `_effective_file_size` aggregates across the whole family
  // server-side, so the verdict we surface here already reflects total size.
  const { families, loose } = useMemo(
    () => groupShardFamilies(rows, (f) => f.size),
    [rows],
  );

  // Build a quick lookup from filename → discovery warning so each file row
  // can surface its specific GGUF-arch advisory inline (replacing the old
  // catch-all GGUF banner). Map preserves insertion order; an array is fine.
  const warningsByFilename = useMemo(() => {
    const m = new Map<string, DiscoveryWarning>();
    // The other half of the gate 200 lines below. `gguf_arch_unsupported` is
    // measured against KNOWN_GGUF_ARCHES, which is vLLM's GGUF-loader
    // allowlist -- so on llama.cpp it is not a weaker warning, it is a
    // statement about a loader that is not being used. GGUF is llama.cpp's
    // native format; ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF serves on it while
    // sitting outside that allowlist.
    //
    // Gating here rather than at the two render sites keeps `backend` out of
    // ShardFamilyRows and FileRow: this map is the only way a warning reaches
    // either of them.
    if (backend !== "vllm") return m;
    for (const w of discovery.warnings ?? []) {
      m.set(w.filename, w);
    }
    return m;
  }, [discovery.warnings, backend]);

  // VRAM cap (`total_vram * GMU`) for the currently-selected GPUs, mirrored
  // from the modal's `vramBudget()` helper. The cap differs from `liveBudget`
  // by exactly `kv_reserve` — we need the cap (not budget) to feed
  // `recommendMaxModelLen()` so the FE solver mirrors `app/models/fit.py`.
  // GMU=0.9 matches the backend FitPreviewRequest default (we don't yet
  // expose gpu_memory_utilization in the wizard).
  const liveCapBytes = (() => {
    const total = gpus
      .filter((g) => selectedGpus.has(g.index))
      .reduce((acc, g) => acc + g.memory_total_mib * MIB, 0);
    return Math.floor(total * 0.9);
  })();

  // Focus management on stage transition (I4, !86 CR). The enter-repo
  // stage autoFocuses the HF repo Input, but when it unmounts on transition
  // into select-file, browser default behaviour falls back to <body> —
  // operators relying on keyboard nav lose their place. Move focus to the
  // first enabled weights radio so Tab from there reaches the GPU
  // checkboxes and onward in DOM order.
  //
  // We scope the query to this form via ref instead of document.* so a
  // stray radio elsewhere on the page can't accidentally claim focus.
  // The effect has no deps — it runs once on mount, which coincides with
  // the stage transition because the parent unmounts/remounts this
  // component on every stage flip into select-file.
  const formRef = useRef<HTMLFormElement>(null);
  useEffect(() => {
    const form = formRef.current;
    if (!form) return;
    const firstRadio = form.querySelector<HTMLInputElement>(
      'input[type="radio"][name="weights-file"]:not(:disabled)',
    );
    if (firstRadio) firstRadio.focus();
  }, []);

  return (
    <form ref={formRef} onSubmit={onSubmit} noValidate className="space-y-4">
      <div>
        <p className="text-xs uppercase tracking-wide text-slate-500">Files</p>
        <div className="mt-1 max-h-72 overflow-y-auto rounded-md border border-slate-700">
          <table className="w-full text-xs" data-testid="file-table">
            <thead className="bg-slate-800 text-slate-400">
              <tr>
                <th className="w-8" />
                <th className="px-2 py-1.5 text-left font-medium">filename</th>
                <th className="px-2 py-1.5 text-right font-medium">size</th>
                <th className="px-2 py-1.5 text-left font-medium">kind</th>
                <th className="px-2 py-1.5 text-left font-medium">quant</th>
                <th className="px-2 py-1.5 text-left font-medium">fit</th>
              </tr>
            </thead>
            <tbody>
              {families.map((fam) => (
                <ShardFamilyRows
                  key={fam.key}
                  family={fam}
                  selectedFilename={selectedFilename}
                  onPick={onPickFilename}
                  fitByFilename={fitByFilename}
                  liveBudget={liveBudget}
                  liveCapBytes={liveCapBytes}
                  hasGpuSelection={selectedGpus.size > 0}
                  warningsByFilename={warningsByFilename}
                />
              ))}
              {loose.map((f) => (
                <FileRow
                  key={f.filename}
                  file={f}
                  selected={f.filename === selectedFilename}
                  selectable={isWeightsFile(f.kind)}
                  onPick={onPickFilename}
                  fit={fitByFilename[f.filename] ?? null}
                  liveBudget={liveBudget}
                  liveCapBytes={liveCapBytes}
                  hasGpuSelection={selectedGpus.size > 0}
                  warning={warningsByFilename.get(f.filename) ?? null}
                />
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Backend ------------------------------------------------------
          Rendered BELOW the file table, not beside the template picker: the
          backend is a CONSEQUENCE of the model (design spec §9.1), so it has to
          appear after the file is chosen rather than as a question asked first. */}
      <div>
        <label
          htmlFor="backend-select"
          className="text-xs uppercase tracking-wide text-slate-500"
        >
          Engine
        </label>
        <select
          id="backend-select"
          data-testid="backend-select"
          value={backend}
          onChange={(e) => onBackendChange(e.target.value as BackendName)}
          className="mt-1 w-full rounded-md border border-slate-700 bg-slate-900 px-2 py-1.5 text-sm text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
        >
          <option value="vllm">vLLM</option>
          <option value="llamacpp">llama.cpp</option>
        </select>
        <p className="mt-1 text-[11px] text-slate-500">
          {backend === "llamacpp"
            ? "llama.cpp serves a single .gguf file. It splits a multi-GPU model by layer rather than by tensor, so vLLM remains the choice when throughput across several cards matters."
            : "vLLM is the default. It serves safetensors and GGUF, and is the only backend here with tensor-parallel multi-GPU."}
        </p>
      </div>

      {/* Engine version ----------------------------------------------
          #215 (slice b): rendered only while this deployment can actually
          honour a pin. When the running driver cannot swap the engine image
          (`engineVersionHidden`, the deployment fact behind
          /api/system/engine's `supports_version_select`) the selector is
          hidden and one line says why: a pin chosen there would be persisted
          and then refused at load with EnginePinUnsupported, and nothing in
          this form would unlock it -- a greyed-out field would be a dead
          control with a redeploy instruction attached.

          While a capable driver is known, the per-backend obstacle
          (llama.cpp) keeps the old disabled-and-explained state, gated on
          `version_pin_available` and carrying the server's sentence, which is
          the only party that knows whether the obstacle is the driver or the
          engine. When the driver can honour a pin, nothing changes. */}
      {engineVersionHidden ? (
        <p data-testid="engine-version-fixed" className="text-[11px] text-slate-500">
          The engine version is fixed by this LM Warden build.
        </p>
      ) : (
        <div>
          <label
            htmlFor="engine-version"
            className="text-xs uppercase tracking-wide text-slate-500"
          >
            Engine version
          </label>
          <Input
            id="engine-version"
            data-testid="engine-version"
            value={engineVersion}
            onChange={(e) => onEngineVersionChange(e.target.value)}
            disabled={engineVersionDisabled}
            placeholder={engineVersionRunning ?? ""}
            aria-describedby="engine-version-note"
            className="mt-1"
          />
          <p
            id="engine-version-note"
            data-testid="engine-version-note"
            className={
              "mt-1 text-[11px] " +
              (engineVersionDisabled ? "text-amber-300" : "text-slate-500")
            }
          >
            {engineVersionDisabled
              ? engineVersionReason ??
                "This deployment cannot pin an engine version."
              : `Leave blank to use the version this warden runs${
                  engineVersionRunning ? ` (${engineVersionRunning})` : ""
                }.`}
          </p>
        </div>
      )}

      {/* Multimodal projector, llama.cpp only -------------------------
          Only `mmproj`-kind files are offered: a projector is a GGUF but it is
          not weights, and llama.cpp takes it via --mmproj alongside -m. */}
      {backend === "llamacpp" && mmprojCandidates.length > 0 && (
        <div>
          <label
            htmlFor="mmproj-select"
            className="text-xs uppercase tracking-wide text-slate-500"
          >
            Vision projector
          </label>
          <select
            id="mmproj-select"
            data-testid="mmproj-select"
            value={mmprojFilename}
            onChange={(e) => onMmprojFilenameChange(e.target.value)}
            className="mt-1 w-full rounded-md border border-slate-700 bg-slate-900 px-2 py-1.5 text-sm text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
          >
            <option value="">(none)</option>
            {mmprojCandidates.map((f) => (
              <option key={f.filename} value={f.filename}>
                {f.filename}
              </option>
            ))}
          </select>
          <p className="mt-1 text-[11px] text-slate-500">
            Needed for image input. Pulled alongside the weights; a vision model
            started without it loads, serves, and silently ignores every image.
          </p>
        </div>
      )}

      {/* Gated on the vLLM backend: both banners are statements about VLLM's
          loader ("vLLM-known GGUF allowlist", "verify vLLM supports this
          GGUF"), and both are false and confusing next to a llama.cpp
          selection. Sub-project E replaces them with a real per-backend
          verdict strip; C's job is only to stop showing the wrong one. */}
      {backend === "vllm" && showGgufWarn && (discovery.warnings?.length ?? 0) === 0 && (
        <div
          role="alert"
          data-testid="gguf-warn"
          className="rounded-md border border-amber-700 bg-amber-900/20 p-3 text-xs text-amber-200"
        >
          vLLM GGUF serving is supported but tightly coupled to the model
          architecture — verify the inferred arch above matches a vLLM-known
          family before loading.
        </div>
      )}

      <div>
        <p className="text-xs uppercase tracking-wide text-slate-500">GPUs</p>
        <div className="mt-1">
          <GpuChecklist
            gpus={gpus}
            selected={Array.from(selectedGpus).sort((a, b) => a - b)}
            onChange={(next) => onGpusChange(new Set(next))}
            allowedIndices={allowedGpuIndices}
          />
        </div>
      </div>

      <label htmlFor={nameId} className="block space-y-1">
        <span className="text-sm">Served model name</span>
        <Input
          id={nameId}
          name="served_model_name"
          aria-describedby={nameHintId}
          aria-required="true"
          required
          value={name}
          onChange={(e) => onNameChange(e.target.value)}
          placeholder="e.g. llama3-8b"
          autoComplete="off"
        />
        <span id={nameHintId} className="text-xs text-slate-500">
          Letters, digits, dot, dash, underscore. Max 100 chars.
        </span>
      </label>

      <AdvancedSection
        parallelismStrategy={parallelismStrategy}
        onParallelismStrategyChange={onParallelismStrategyChange}
        maxBatchSize={maxBatchSize}
        onMaxBatchSizeChange={onMaxBatchSizeChange}
        dataParallelSize={dataParallelSize}
        onDataParallelSizeChange={onDataParallelSizeChange}
        supportsDataParallel={supportsDataParallel}
        gpuCount={selectedGpus.size}
        maxModelLen={maxModelLen}
        onMaxModelLenChange={onMaxModelLenChange}
        baseRepo={baseRepo}
        onBaseRepoChange={onBaseRepoChange}
        tokenizerRepo={tokenizerRepo}
        onTokenizerRepoChange={onTokenizerRepoChange}
      />

      {error && <p className="text-sm text-red-500">{error}</p>}

      <div className="flex justify-end gap-2 pt-2">
        <Button type="button" variant="outline" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="submit">Add</Button>
      </div>
    </form>
  );
}

/**
 * Render a sharded weight family as a collapsible group of rows (#112).
 *
 * The top-level row is selectable and represents the WHOLE family — picking
 * it sends the representative shard's filename, which is what the backend
 * keys fit-preview / load against. Expanding the disclosure shows the
 * individual shards as non-selectable child rows so the operator can see
 * exactly what will be pulled.
 *
 * Single-shard "families" with only one member are flattened to a normal
 * FileRow because the disclosure would be pointless.
 */
function ShardFamilyRows({
  family,
  selectedFilename,
  onPick,
  fitByFilename,
  liveBudget,
  liveCapBytes,
  hasGpuSelection,
  warningsByFilename,
}: {
  family: ShardFamily<DiscoveredFile>;
  selectedFilename: string | null;
  onPick: (f: string) => void;
  fitByFilename: Record<string, FitPreviewResponse>;
  liveBudget: number;
  liveCapBytes: number;
  hasGpuSelection: boolean;
  warningsByFilename: Map<string, DiscoveryWarning>;
}) {
  const [expanded, setExpanded] = useState(false);
  // A "family" with only one member is just a single sharded shard — render
  // it as a plain row (no disclosure) so the disclosure indicator doesn't
  // mislead the operator into thinking there are more shards.
  if (family.members.length === 1) {
    const only = family.members[0];
    return (
      <FileRow
        file={only}
        selected={only.filename === selectedFilename}
        selectable={isWeightsFile(only.kind)}
        onPick={onPick}
        fit={fitByFilename[only.filename] ?? null}
        liveBudget={liveBudget}
        liveCapBytes={liveCapBytes}
        hasGpuSelection={hasGpuSelection}
        warning={warningsByFilename.get(only.filename) ?? null}
      />
    );
  }

  const rep = family.representative;
  const selected = rep.filename === selectedFilename;
  const familyLabel = `${family.prefix}-{00001..${String(family.total).padStart(5, "0")}}.${family.ext}`;
  // Surface the strongest warning across the family — typically all shards
  // share the same arch so the first hit is representative.
  const familyWarning = family.members
    .map((m) => warningsByFilename.get(m.filename))
    .find((w): w is DiscoveryWarning => w != null)
    ?? null;
  return (
    <>
      <tr
        data-testid="file-row"
        data-shard-family={family.key}
        data-shard-total={family.total}
        className={
          (selected ? "bg-slate-800/70" : "") + " hover:bg-slate-800/50"
        }
      >
        <td className="px-2 py-1">
          <input
            type="radio"
            name="weights-file"
            aria-label={`select shard family ${familyLabel}`}
            checked={selected}
            onChange={() => onPick(rep.filename)}
            className="h-3.5 w-3.5"
          />
        </td>
        <td className="px-2 py-1 font-mono text-slate-200">
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            aria-expanded={expanded}
            aria-controls={`shard-family-${family.key}`}
            data-testid="shard-family-toggle"
            className="inline-flex items-center gap-1 text-left hover:text-amber-300 focus:outline-none focus:ring-1 focus:ring-amber-600 rounded-sm"
          >
            <span
              aria-hidden
              className={
                "inline-block w-3 text-[10px] text-slate-500 transition-transform " +
                (expanded ? "rotate-90" : "")
              }
            >
              ▶
            </span>
            <span>{familyLabel}</span>
            <span className="ml-1 text-[10px] text-slate-500">
              ({family.members.length} shards)
            </span>
          </button>
        </td>
        <td className="px-2 py-1 text-right text-slate-300">
          {formatBytes(family.aggregateSize)}
        </td>
        <td className="px-2 py-1">
          <Badge variant="default">{kindBadgeLabel(rep.kind)}</Badge>
        </td>
        <td className="px-2 py-1 text-slate-400">{rep.quant ?? "—"}</td>
        <td className="px-2 py-1">
          <FamilyFitBadge
            family={family}
            fit={fitByFilename[rep.filename] ?? null}
            liveBudget={liveBudget}
            liveCapBytes={liveCapBytes}
            hasGpuSelection={hasGpuSelection}
            selected={selected}
          />
        </td>
      </tr>
      {familyWarning && (
        <tr data-testid="gguf-arch-warning-row" data-filename={rep.filename}>
          <td />
          <td colSpan={5} className="px-2 pb-2">
            <ArchWarningBanner warning={familyWarning} />
          </td>
        </tr>
      )}
      {expanded &&
        family.members.map((m) => (
          <tr
            key={m.filename}
            id={`shard-family-${family.key}`}
            data-testid="shard-member-row"
            data-filename={m.filename}
            className="text-slate-500"
          >
            <td />
            <td className="px-2 py-1 pl-7 font-mono text-[11px]">{m.filename}</td>
            <td className="px-2 py-1 text-right text-[11px]">{formatBytes(m.size)}</td>
            <td className="px-2 py-1" />
            <td className="px-2 py-1" />
            <td className="px-2 py-1" />
          </tr>
        ))}
    </>
  );
}

/** Lightweight fit verdict badge keyed off the representative shard's fit-preview. */
function FamilyFitBadge({
  family,
  fit,
  liveBudget,
  liveCapBytes,
  hasGpuSelection,
  selected,
}: {
  family: ShardFamily<DiscoveredFile>;
  fit: FitPreviewResponse | null;
  liveBudget: number;
  liveCapBytes: number;
  hasGpuSelection: boolean;
  selected: boolean;
}) {
  const showVerdict = selected || fit !== null;
  let verdict: FitVerdict | null = null;
  if (showVerdict && fit) {
    if (hasGpuSelection && liveBudget !== 0) {
      const sizeForFit =
        fit.breakdown.file_size > 0 ? fit.breakdown.file_size : family.aggregateSize;
      verdict = classifyFit(sizeForFit, liveBudget);
    } else {
      verdict = fit.verdict;
    }
  }
  if (!verdict) return <span className="text-slate-600">—</span>;
  // Tooltip text trimmed down vs the per-file FileRow to avoid duplicating
  // the same math twice; the operator can expand the family to inspect
  // each shard separately.
  const tooltip = fit
    ? [
        `family_size: ${formatBytes(family.aggregateSize)}`,
        `kv_reserve: ${formatBytes(fit.breakdown.kv_reserve)}`,
        `weights_budget: ${formatBytes(fit.breakdown.weights_budget)}`,
      ].join("\n")
    : undefined;
  return (
    <span
      data-testid={`fit-badge-${family.key}`}
      data-verdict={verdict}
      title={tooltip}
      className={
        "inline-flex w-fit items-center rounded-full px-2 py-0.5 text-[10px] font-medium " +
        verdictBadgeClass(verdict)
      }
    >
      {verdictLabel(verdict)}
    </span>
  );
  // unused params kept to keep call sites simple; tsc would whine if we
  // dropped them entirely — `liveCapBytes` is for future per-family
  // recommended_max_model_len; backend already computes against the
  // aggregate size today.
  void liveCapBytes;
}

/** Amber inline banner for per-file GGUF-arch warnings (#101). */
function ArchWarningBanner({ warning }: { warning: DiscoveryWarning }) {
  const message =
    warning.type === "gguf_arch_unsupported"
      ? `Architecture "${warning.arch}" is not in the vLLM-known GGUF allowlist — load is likely to fail.`
      : "Could not infer architecture from filename or config.json — verify vLLM supports this GGUF before loading.";
  return (
    <div
      role="alert"
      data-testid="gguf-arch-warning"
      data-warning-type={warning.type}
      data-arch={warning.arch ?? ""}
      className="rounded-md border border-amber-700 bg-amber-900/20 px-2 py-1.5 text-[11px] text-amber-200"
    >
      {message}{" "}
      <span data-testid="gguf-arch-warning-remedy">
        GGUF is llama.cpp&apos;s native format — switching this model&apos;s
        engine to llama.cpp skips the vLLM allowlist entirely.
      </span>
    </div>
  );
}

function FileRow({
  file,
  selected,
  selectable,
  onPick,
  fit,
  liveBudget,
  liveCapBytes,
  hasGpuSelection,
  warning,
}: {
  file: DiscoveredFile;
  selected: boolean;
  selectable: boolean;
  onPick: (f: string) => void;
  fit: FitPreviewResponse | null;
  liveBudget: number;
  liveCapBytes: number;
  hasGpuSelection: boolean;
  warning?: DiscoveryWarning | null;
}) {
  // Verdict source-of-truth ladder:
  //   1. If the row isn't selected yet, we can't compute a budget for it
  //      because the backend's KV-reserve term needs the per-file context.
  //      Skip the badge to avoid showing a stale verdict from a sibling.
  //   2. If we have a cached fit-preview AND a current GPU selection,
  //      classify client-side via `classifyFit(file.size, liveBudget)` so
  //      ticking GPU checkboxes recomputes without a server round-trip.
  //   3. Fall back to the verdict the server returned (covers the
  //      "no GPU selected" case where liveBudget would be misleading).
  const showVerdict = selectable && (selected || fit !== null);
  let verdict: FitVerdict | null = null;
  if (showVerdict) {
    if (fit && hasGpuSelection && liveBudget !== 0) {
      // `file.size` is the per-file size; for sharded safetensors the
      // backend aggregates across shards (see `_effective_file_size` in
      // routes_api.py). The client classifier uses `fit.breakdown.file_size`
      // for sharded rows so we don't undercount.
      const sizeForFit = fit.breakdown.file_size > 0 ? fit.breakdown.file_size : file.size;
      verdict = classifyFit(sizeForFit, liveBudget);
    } else if (fit) {
      verdict = fit.verdict;
    }
  }

  // Tooltip values follow the verdict ladder above:
  //   - `bytes_per_token` and `kv_reserve` are GPU-independent (only depend
  //     on the model config + max_model_len), so we surface the server's
  //     snapshot unchanged.
  //   - `weights_budget` and `ratio` ARE GPU-dependent (budget = cap - kv
  //     where cap scales with selected GPUs). The badge recomputes
  //     client-side when GPUs toggle; the tooltip MUST track that or it
  //     contradicts the verdict it's annotating (I2, !86 CR). When we have
  //     a live budget AND a current GPU selection, render those locally
  //     using the same `sizeForFit` the classifier consumed. When the
  //     operator has unchecked every GPU, `liveBudget` is meaningless —
  //     fall back to the server's snapshot to avoid a misleading 0-byte /
  //     ∞-ratio line.
  //
  // #87 — derive `bytes_per_token` once and reuse it for the
  // "Recommended max_model_len" hint on orange rows. The backend's
  // `recommended_max_model_len` field is preferred when present; we only
  // fall back to a client-side solve when the server didn't populate it
  // (e.g. degraded config or a future code path that yields a null rec).
  let tooltip: string | undefined;
  let bytesPerToken = 0;
  if (fit) {
    const sizeForFitTooltip = fit.breakdown.file_size > 0 ? fit.breakdown.file_size : file.size;
    const useLive = hasGpuSelection && liveBudget > 0;
    const budgetForTooltip = useLive ? liveBudget : fit.breakdown.weights_budget;
    const ratioForTooltip = useLive
      ? sizeForFitTooltip / liveBudget
      : fit.breakdown.ratio;
    bytesPerToken =
      fit.breakdown.max_model_len_used > 0
        ? Math.round(fit.breakdown.kv_reserve / fit.breakdown.max_model_len_used)
        : 0;
    tooltip = [
      `bytes_per_token: ${formatBytes(bytesPerToken)}`,
      `kv_reserve: ${formatBytes(fit.breakdown.kv_reserve)}`,
      `weights_budget: ${formatBytes(budgetForTooltip)}`,
      `ratio: ${ratioForTooltip.toFixed(3)}`,
    ].join("\n");
  }

  // Recommended max_model_len hint (#87). Only surface on `orange` rows
  // (the "tight" band) — green/yellow rows fit fine, and red rows can't
  // be salvaged by trimming context. The backend already computes this
  // for orange/red verdicts via `recommend_max_model_len()` and returns
  // it in the FitPreviewResponse; we prefer that value because the
  // server saw the live config.json (we don't). When the backend value
  // is null (degraded config), we attempt a client-side solve using the
  // tooltip-derived `bytes_per_token` and the live VRAM cap.
  let recommendedMaxModelLen: number | null = null;
  if (verdict === "orange" && fit) {
    if (fit.recommended_max_model_len != null && fit.recommended_max_model_len > 0) {
      recommendedMaxModelLen = fit.recommended_max_model_len;
    } else if (hasGpuSelection && bytesPerToken > 0 && liveCapBytes > 0) {
      const sizeForRec =
        fit.breakdown.file_size > 0 ? fit.breakdown.file_size : file.size;
      recommendedMaxModelLen = recommendMaxModelLen({
        fileSize: sizeForRec,
        capBytes: liveCapBytes,
        bytesPerToken,
      });
    }
  }

  return (
    <>
      <tr
        data-testid="file-row"
        data-filename={file.filename}
        className={
          (selected ? "bg-slate-800/70" : "") +
          (selectable ? " hover:bg-slate-800/50" : " opacity-60")
        }
      >
        <td className="px-2 py-1">
          <input
            type="radio"
            name="weights-file"
            aria-label={`select ${file.filename}`}
            checked={selected}
            onChange={() => onPick(file.filename)}
            disabled={!selectable}
            className="h-3.5 w-3.5"
          />
        </td>
        <td className="px-2 py-1 font-mono text-slate-200">{file.filename}</td>
        <td className="px-2 py-1 text-right text-slate-300">{formatBytes(file.size)}</td>
        <td className="px-2 py-1">
          <Badge variant="default">{kindBadgeLabel(file.kind)}</Badge>
        </td>
        <td className="px-2 py-1 text-slate-400">{file.quant ?? "—"}</td>
        <td className="px-2 py-1">
          {verdict ? (
            <div className="flex flex-col gap-0.5">
              <span
                data-testid={`fit-badge-${file.filename}`}
                data-verdict={verdict}
                title={tooltip}
                className={
                  "inline-flex w-fit items-center rounded-full px-2 py-0.5 text-[10px] font-medium " +
                  verdictBadgeClass(verdict)
                }
              >
                {verdictLabel(verdict)}
              </span>
              {recommendedMaxModelLen !== null && (
                <span
                  data-testid={`recommended-max-model-len-${file.filename}`}
                  data-recommended-max-model-len={recommendedMaxModelLen}
                  className="text-[10px] text-amber-300/90"
                >
                  Recommended max_model_len: {recommendedMaxModelLen.toLocaleString()}
                </span>
              )}
            </div>
          ) : (
            <span className="text-slate-600">—</span>
          )}
        </td>
      </tr>
      {warning && (
        <tr data-testid="gguf-arch-warning-row" data-filename={file.filename}>
          <td />
          <td colSpan={5} className="px-2 pb-2">
            <ArchWarningBanner warning={warning} />
          </td>
        </tr>
      )}
    </>
  );
}

// ---- Advanced section (#87) ----------------------------------------------
//
// Collapsible disclosure for the three "I know what I'm doing" knobs that
// most operators never need to touch:
//
//   - parallelism_strategy: auto/tp/pp. CTO call (#87): single-host PP is
//     explicitly allowed — vLLM accepts it and the backend's cmd_builder
//     emits `--pipeline-parallel-size` when set. Do NOT add a client-side
//     guard against pp on single-host configs.
//   - max_batch_size: feeds the KV-reserve math in the fit-preview. Bounded
//     1..64 by the backend; we mirror that here.
//   - max_model_len: explicit override of the discovered max_position_embeddings.
//     Empty = "use config default" (backend treats null the same way).
//
// Implemented as a native <details>/<summary> pair so it's keyboard-accessible
// out of the box and doesn't pull in an a11y library. Collapsed by default
// per the issue body — most operators won't expand it.
function AdvancedSection({
  parallelismStrategy,
  onParallelismStrategyChange,
  maxBatchSize,
  onMaxBatchSizeChange,
  maxModelLen,
  onMaxModelLenChange,
  baseRepo,
  onBaseRepoChange,
  tokenizerRepo,
  onTokenizerRepoChange,
  dataParallelSize,
  onDataParallelSizeChange,
  supportsDataParallel,
  gpuCount,
}: {
  dataParallelSize: string;
  onDataParallelSizeChange: (v: string) => void;
  supportsDataParallel: boolean;
  gpuCount: number;
  parallelismStrategy: ParallelismStrategy;
  onParallelismStrategyChange: (v: ParallelismStrategy) => void;
  maxBatchSize: string;
  onMaxBatchSizeChange: (v: string) => void;
  maxModelLen: string;
  onMaxModelLenChange: (v: string) => void;
  baseRepo: string;
  onBaseRepoChange: (v: string) => void;
  tokenizerRepo: string;
  onTokenizerRepoChange: (v: string) => void;
}) {
  const parallelismId = useId();
  const batchId = useId();
  const dpId = useId();
  const lenId = useId();
  const dpN = Number(dataParallelSize);
  const dpValid =
    dataParallelSize.trim() !== "" && Number.isInteger(dpN) && dpN >= 1;
  const dpDivides = dpValid && gpuCount > 0 && gpuCount % dpN === 0;
  const baseRepoId = useId();
  const tokenizerRepoId = useId();
  return (
    <details
      data-testid="advanced-section"
      className="rounded-md border border-slate-700 bg-slate-900/50"
    >
      <summary
        data-testid="advanced-toggle"
        className="cursor-pointer select-none px-3 py-2 text-xs font-medium uppercase tracking-wide text-slate-400 hover:text-slate-200"
      >
        Advanced
      </summary>
      <div
        data-testid="advanced-body"
        className="space-y-3 border-t border-slate-700 px-3 py-3"
      >
        <label htmlFor={parallelismId} className="block space-y-1">
          <span className="text-sm">Parallelism strategy</span>
          <select
            id={parallelismId}
            data-testid="parallelism-strategy"
            name="parallelism_strategy"
            value={parallelismStrategy}
            onChange={(e) =>
              onParallelismStrategyChange(e.target.value as ParallelismStrategy)
            }
            className="flex h-9 w-full rounded-md border border-slate-600 bg-slate-900 px-3 py-1 text-sm text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900"
          >
            <option value="auto">auto (TP on multi-GPU)</option>
            <option value="tp">tp (tensor parallel)</option>
            <option value="pp">pp (pipeline parallel)</option>
          </select>
          <span className="text-xs text-slate-500">
            Single-host pipeline parallel is allowed — vLLM accepts it.
          </span>
        </label>

        <label htmlFor={batchId} className="block space-y-1">
          <span className="text-sm">Max batch size</span>
          <Input
            id={batchId}
            data-testid="max-batch-size"
            name="max_batch_size"
            type="number"
            inputMode="numeric"
            min={MAX_BATCH_SIZE_MIN}
            max={MAX_BATCH_SIZE_MAX}
            step={1}
            value={maxBatchSize}
            onChange={(e) => onMaxBatchSizeChange(e.target.value)}
            placeholder="1"
            autoComplete="off"
          />
          <span className="text-xs text-slate-500">
            1–{MAX_BATCH_SIZE_MAX}. Feeds the KV-reserve math; larger values
            tighten the fit verdict.
          </span>
        </label>

        {supportsDataParallel && (
          <div className="space-y-1">
            <label htmlFor={dpId} className="block space-y-1">
              <span className="text-sm">Data-parallel replicas</span>
              <Input
                id={dpId}
                data-testid="data-parallel-size"
                name="data_parallel_size"
                type="number"
                inputMode="numeric"
                min={1}
                step={1}
                value={dataParallelSize}
                onChange={(e) => onDataParallelSizeChange(e.target.value)}
                placeholder="1"
                autoComplete="off"
              />
              <span className="text-xs text-slate-500">
                vLLM only. N identical replicas, each on (selected GPUs ÷ N)
                GPUs with tensor parallelism inside the replica. Conversations
                are pinned to a replica by the proxy.
              </span>
            </label>
            {dpValid && dpN > 1 && dpDivides && (
              <p
                data-testid="dp-derived"
                className="text-xs text-emerald-400"
              >
                {dpN} replicas × {gpuCount / dpN} GPU
                {gpuCount / dpN === 1 ? "" : "s"} each (tensor parallel size{" "}
                {gpuCount / dpN} inside a replica).
              </p>
            )}
            {dpValid && dpN > 1 && !dpDivides && (
              <p
                role="alert"
                data-testid="dp-invalid-error"
                className="text-xs text-red-400"
              >
                Data-parallel replicas must divide the number of selected GPUs (
                {gpuCount}).
              </p>
            )}
          </div>
        )}

        <label htmlFor={lenId} className="block space-y-1">
          <span className="text-sm">max_model_len override</span>
          <Input
            id={lenId}
            data-testid="max-model-len"
            name="max_model_len"
            type="number"
            inputMode="numeric"
            min={1}
            step={1}
            value={maxModelLen}
            onChange={(e) => onMaxModelLenChange(e.target.value)}
            placeholder="(use config default)"
            autoComplete="off"
          />
          <span className="text-xs text-slate-500">
            Leave blank to use the model&apos;s config.json
            max_position_embeddings. Set this to land an &quot;orange&quot; row
            back in the safe band.
          </span>
        </label>

        {/* #106 — GGUF repos that omit config.json (common for unsloth
            republishes) require --hf-config-path <original_repo>. We default
            --tokenizer to the same repo and let the operator override it
            separately if their tokenizer lives elsewhere. Leave both blank
            for self-contained GGUF or safetensors models. */}
        <label htmlFor={baseRepoId} className="block space-y-1">
          <span className="text-sm">Base repo (config + tokenizer)</span>
          <Input
            id={baseRepoId}
            data-testid="base-repo"
            name="base_repo"
            type="text"
            value={baseRepo}
            onChange={(e) => onBaseRepoChange(e.target.value)}
            placeholder="e.g. Qwen/Qwen3-30B-A3B"
            autoComplete="off"
          />
          <span className="text-xs text-slate-500">
            For GGUF repos that don&apos;t ship config.json. Sets
            --hf-config-path and --tokenizer. Leave blank for self-contained
            repos.
          </span>
        </label>

        <details
          data-testid="tokenizer-override-section"
          className="rounded-md border border-slate-700 bg-slate-900/40"
        >
          <summary
            data-testid="tokenizer-override-toggle"
            className="cursor-pointer select-none px-3 py-2 text-xs font-medium uppercase tracking-wide text-slate-400 hover:text-slate-200"
          >
            Override tokenizer separately
          </summary>
          <div className="space-y-3 border-t border-slate-700 px-3 py-3">
            <label htmlFor={tokenizerRepoId} className="block space-y-1">
              <span className="text-sm">Tokenizer repo</span>
              <Input
                id={tokenizerRepoId}
                data-testid="tokenizer-repo"
                name="tokenizer_repo"
                type="text"
                value={tokenizerRepo}
                onChange={(e) => onTokenizerRepoChange(e.target.value)}
                placeholder="(defaults to Base repo)"
                autoComplete="off"
              />
              <span className="text-xs text-slate-500">
                Overrides --tokenizer only. Leave blank to reuse the Base
                repo above.
              </span>
            </label>
          </div>
        </details>
      </div>
    </details>
  );
}
