export type RestartKind = 'none' | 'model-reload' | 'warden-restart';

export interface FieldHint {
  label: string;
  /** The full explanation. Always kept verbatim: with `chrome="token"` it sits
   *  behind the field's "More" toggle rather than being deleted. */
  hint: string;
  /** Optional operator-facing one-liner (about 90 characters or fewer). Only
   *  rendered when a SettingField opts in with `chrome="token"`; the global
   *  settings tabs never set it and keep showing `hint`. */
  short?: string;
  restart: RestartKind;
}

export const RUNTIME_HINTS: Record<string, FieldHint> = {
  admin_username: {
    label: 'Admin username',
    hint: 'The single operator account. Used for the login page.',
    restart: 'none',
  },
  admin_password: {
    label: 'Admin password',
    hint: 'Updates the bcrypt hash. All sessions invalidated on change. At least 12 characters (at most 72 bytes).',
    restart: 'none',
  },
  hf_token: {
    label: 'Hugging Face token',
    hint: 'Passed to vLLM via `HF_TOKEN` env. Required for gated repos (Llama, gpt-oss, Mistral, etc.). New value applies to next model load.',
    restart: 'model-reload',
  },
  default_gpu_indices: {
    label: 'Default GPU indices',
    hint: 'Pre-selected when adding a new model. Check the GPUs to include by default.',
    restart: 'none',
  },
  default_token_expiration_days: {
    label: 'Default token expiration',
    hint: 'Pre-fills the new-token dialog. Affects new tokens only.',
    restart: 'none',
  },
  rotation_grace_hours: {
    // #185 — this value is stored and validated but nothing reads it: the
    // rotate route's default is a Pydantic literal and the dialog's is its
    // own useState. The old hint described a behaviour the product does not
    // have. Not wired up here deliberately — a 0 saved in settings would
    // PRE-SELECT immediate revoke in the rotate dialog, which is the footgun
    // #185 exists to close, arriving through the settings door. Tracked
    // separately alongside `default_token_expiration_days`, which is inert
    // in exactly the same way.
    label: 'Rotation grace window',
    hint: 'Not yet wired up — rotation currently defaults to 24h and the rotate dialog is where you choose the grace window (or revoke the old token immediately).',
    restart: 'none',
  },
  session_access_ttl_minutes: {
    label: 'Session access TTL',
    hint: 'How long a login JWT stays valid before refresh. Short = safer. Existing tokens keep their original TTL.',
    restart: 'warden-restart',
  },
  session_refresh_ttl_days: {
    label: 'Session refresh TTL',
    hint: 'How long until forced re-login. Existing refresh cookies keep their original TTL.',
    restart: 'warden-restart',
  },
  sse_ticket_ttl_seconds: {
    label: 'SSE ticket TTL',
    hint: 'How long a single SSE connect ticket is valid. SSE streams stay open after auth; only the initial connect needs the ticket.',
    restart: 'none',
  },
  vllm_version: {
    label: 'vLLM version',
    hint: 'vLLM Python package version. Warden container rebuild required (image baked at build time).',
    restart: 'warden-restart',
  },
  log_retention_lines: {
    label: 'Log retention',
    hint: 'Per-model log buffer size kept in memory + on disk. Applied to new lines after change.',
    restart: 'none',
  },
  landing_page_enabled: {
    label: 'Public website',
    hint: 'Serves the LM Warden product website (home, coding agents, FAQ and the rest) at the root of this warden. Off by default: a private install opens the console at `/`. Turn it on only for a public instance that should promote LM Warden; set VW_LANDING_CANONICAL_URL as well to let search engines index it.',
    restart: 'none',
  },
  watchdog_enabled: {
    label: 'Engine watchdog',
    hint: 'Probes each loaded engine\'s own /health and restarts it when it stops answering. The warden\'s exit watcher only sees the `vllm serve` wrapper, which can outlive a dead EngineCore — when that happens the model still reads "loaded", /v1/models still returns 200, and every generation fails until someone notices. Leave on unless you are debugging a live crash and want the scene preserved.',
    restart: 'none',
  },
  watchdog_restore_on_boot: {
    label: 'Reload model after warden restart',
    hint: 'The engine is a child of the warden process, so restarting the warden takes the engine down with it. When enabled, any model that was serving at that moment is loaded again with its stored settings. Models that were mid-pull, deliberately unloaded, or failed on their own merits are left alone.',
    restart: 'none',
  },
  watchdog_interval_s: {
    label: 'Watchdog probe interval (seconds)',
    hint: 'How often each loaded engine is probed. 5–3600.',
    restart: 'none',
  },
  watchdog_failure_threshold: {
    label: 'Failures before restart',
    hint: 'Consecutive failed probes before the engine is declared dead. 2–20. With the default 30s interval, 3 means roughly 90 seconds — long enough that a GC pause or a momentary hang does not trigger a restart.',
    restart: 'none',
  },
  watchdog_max_restarts: {
    label: 'Max automatic restarts per hour',
    hint: 'Crash-loop guard. After this many automatic restarts within an hour the model is left failed so the evidence survives and a human can look. Set to 0 to detect and record crashes without ever restarting automatically.',
    restart: 'none',
  },
  public_url: {
    label: 'Public URL',
    hint: 'External base URL used in client-facing snippets (curl examples, OpenAI client configs). Leave unset to use the browser address bar — that\'s the right default unless this warden sits behind a reverse proxy whose external URL differs from what your browser sees.',
    restart: 'none',
  },
  analytics_ga4_id: {
    label: 'Google Analytics 4 measurement ID',
    hint: 'Starts with G-. Find it in Google Analytics under Admin → Data streams → your web stream. When set, every page of the public website loads Google’s tag and shows a cookie banner; analytics cookies are set only after a visitor accepts. Leave empty to load nothing from Google.',
    restart: 'none',
  },
  analytics_google_ads_id: {
    label: 'Google Ads tag ID',
    hint: 'Starts with AW-. Optional: needed for Google Ads conversion tracking and remarketing on the same tag. If you link Google Analytics to Google Ads and import conversions from it, you can leave this empty.',
    restart: 'none',
  },
  search_console_verification: {
    label: 'Search Console verification',
    hint: 'The content value from Google Search Console’s HTML-tag verification method (the part inside content="…"). Adds one meta tag to the home page; loads nothing.',
    restart: 'none',
  },
};

// ---------------------------------------------------------------------------
// Per-tab key sets for the #154 settings redesign.
//
// The /settings page is split into five tabs: General → Networking →
// Sessions & Tokens → Maintenance → Model. The first four are backed by
// these RUNTIME_HINTS keys; the fifth (Model) is a navigation pivot with
// no fields of its own.
//
// Each tab imports its own slice in declared render order. The
// `settings-tab-membership.test.ts` contract test pins that:
//   * every key in RUNTIME_HINTS appears in exactly one of these arrays
//     (no orphans, no duplicates);
//   * the four arrays together equal Object.keys(RUNTIME_HINTS).
// Adding a new RUNTIME_HINTS entry without placing it in one of these
// arrays fails CI — that's the point.
// ---------------------------------------------------------------------------

export const RUNTIME_GENERAL_KEYS = [
  // Identity
  'admin_username',
  'admin_password',
  // Hugging Face
  'hf_token',
  // Defaults for new models
  'default_gpu_indices',
] as const;

export const RUNTIME_NETWORKING_KEYS = [
  // Public access
  'public_url',
  'landing_page_enabled',
] as const;

// Website analytics (spec 2026-10-07 §8): rendered as a second section of
// the Networking tab, saved by the same Save button.
export const RUNTIME_ANALYTICS_KEYS = [
  'analytics_ga4_id',
  'analytics_google_ads_id',
  'search_console_verification',
] as const;

export const RUNTIME_SESSIONS_KEYS = [
  // Browser session
  'session_access_ttl_minutes',
  'session_refresh_ttl_days',
  // Token defaults
  'default_token_expiration_days',
  'rotation_grace_hours',
  // Streaming
  'sse_ticket_ttl_seconds',
] as const;

export const RUNTIME_MAINTENANCE_KEYS = [
  // vLLM runtime
  'vllm_version',
  // Logs
  'log_retention_lines',
  // Engine watchdog
  'watchdog_enabled',
  'watchdog_restore_on_boot',
  'watchdog_interval_s',
  'watchdog_failure_threshold',
  'watchdog_max_restarts',
] as const;

export const MODEL_HINTS: Record<string, FieldHint> = {
  served_model_name: {
    label: 'Served model name',
    hint: 'The name clients pass in `model:` for `/v1/completions`. Slug only — current backend regex allows alphanumeric + `.`, `_`, `-`. Dots permitted because vLLM model names like `Mixtral-8x7B-v0.1` are common. Frontend mirrors that regex client-side for instant validation.',
    short: 'What clients send as `model`. Letters, digits, `. _ -`.',
    restart: 'model-reload',
  },
  hf_repo: {
    label: 'Hugging Face repo',
    hint: 'Hugging Face repo path, e.g. `facebook/opt-125m` or `openai/gpt-oss-20b`.',
    short: 'e.g. `openai/gpt-oss-20b`',
    restart: 'model-reload',
  },
  hf_revision: {
    label: 'HF revision',
    hint: 'Branch, tag, or commit SHA. Default `main` — pin to a SHA for reproducibility.',
    short: 'Branch, tag or commit. Pin a SHA to reproduce.',
    restart: 'model-reload',
  },
  gpu_indices: {
    label: 'GPU indices',
    hint: 'Which GPUs this model loads on. At least one required. Determines `CUDA_VISIBLE_DEVICES` for the engine subprocess, whichever backend serves the model.',
    short: 'At least one. Sets `CUDA_VISIBLE_DEVICES` for the engine.',
    restart: 'model-reload',
  },
  tensor_parallel_size: {
    label: 'Tensor-parallel size',
    hint: 'GPUs per replica: how many GPUs shard each layer\'s weights inside one replica. Derived so that `tensor_parallel_size × data_parallel_size = len(gpu_indices)`; it follows the GPU selection and the replica count below.',
    short: 'GPUs per replica. Follows GPUs ÷ replicas.',
    restart: 'model-reload',
  },
  data_parallel_size: {
    label: 'Data-parallel replicas',
    hint: 'vLLM only. Number of identical replicas of the model, each on `len(gpu_indices) ÷ replicas` GPUs. Must divide the number of selected GPUs. Weights are replicated on every replica; the proxy pins each conversation to one replica so its prefix cache stays warm.',
    short: 'Identical copies of the model. Must divide the GPU count.',
    restart: 'model-reload',
  },
  dp_affinity_enabled: {
    label: 'Replica affinity',
    hint: 'Pin each conversation to one replica (keyed on the client\'s user id, session header or first prompt) so its prefix cache is reused. Off hands every request to vLLM\'s own balancer. Takes effect immediately, no reload.',
    short: 'Keep each conversation on one replica so its prefix cache is reused. Off lets vLLM balance every request.',
    restart: 'none',
  },
  dp_spill_threshold: {
    label: 'Spill threshold',
    hint: 'Requests in flight on a replica before new conversations spill to the least-loaded one instead of their pinned replica. Leave blank for auto: `max_num_seqs / 4` (64 with vLLM\'s default of 256). Lower favours balance, higher favours cache hits. Takes effect immediately, no reload.',
    short: 'In-flight requests before new conversations move to the least-loaded replica. Blank = auto (max_num_seqs ÷ 4).',
    restart: 'none',
  },
  gpu_memory_utilization: {
    label: 'GPU memory utilization',
    hint: 'vLLM only. Fraction of VRAM vLLM may consume per GPU (0.0–1.0, default 0.9). Lower if you hit OOM during paged-attention warmup. llama.cpp has no equivalent — it sizes its own allocation — so this field is hidden for that backend.',
    short: 'Share of each GPU vLLM may use. Lower it after an out-of-memory at startup.',
    restart: 'model-reload',
  },
  dtype: {
    label: 'dtype',
    hint: 'One of `auto`, `float16`, `bfloat16`, `float32`. `auto` follows the model\'s config.',
    short: "Default follows the model's config.",
    restart: 'model-reload',
  },
  quantization: {
    label: 'Quantization',
    hint: 'One of `none`, `awq`, `gptq`, `fp8`, `bitsandbytes`. Most models ship with weights pre-quantized — leave `none` unless you know otherwise.',
    restart: 'model-reload',
  },
  kv_cache_dtype: {
    label: 'KV cache dtype',
    hint: 'One of `auto`, `fp8`, `fp8_e5m2`. `fp8` halves KV cache memory; tiny accuracy impact.',
    restart: 'model-reload',
  },
  n_gpu_layers: {
    label: 'GPU layers',
    hint: 'llama.cpp only. Blank lets llama.cpp size the offload itself, which is the right answer almost always. An explicit number is a deliberate PARTIAL offload -- keeping some layers in CPU RAM so a model that does not fit the card still runs. It works, and it is a performance cliff.',
    short: 'Blank lets llama.cpp size the offload. A number keeps some layers in CPU RAM (slow).',
    restart: 'model-reload',
  },
  mmproj_filename: {
    label: 'Vision projector',
    hint: 'llama.cpp only. The multimodal projector GGUF beside the weights in the same repo, passed as --mmproj. A vision model started without it loads, serves, and silently ignores every image.',
    short: 'Projector GGUF from the same repo. Without it a vision model ignores images.',
    restart: 'model-reload',
  },
  // ---- llama.cpp dials stored inside extra_args -------------------------
  //
  // No column of their own, deliberately -- see @/lib/llamacpp-args. They are
  // controls all the same: "no column" was never a reason for "no control",
  // and hand-typing a flag into Extra args was the only way to reach the three
  // knobs an operator on a 16 GB card reaches for most.
  //
  // Each hint names the trade-off rather than restating the flag, because the
  // control IS the flag; what is not obvious from `--cache-type-k` is what it
  // buys and what it costs.
  flash_attn: {
    label: 'Flash attention',
    hint: "llama.cpp only. Unset leaves the engine on its own default ('auto', resolved at load time against the card); 'on' forces it, 'off' is the escape hatch when a model or quantisation trips over it. Written as --flash-attn in Extra args, so a value typed there by hand appears in this control instead of twice.",
    short: 'Unset follows the engine default (auto). Off is the escape hatch if a model trips on it.',
    restart: 'model-reload',
  },
  cache_type_k: {
    label: 'KV cache type (K)',
    hint: 'llama.cpp only. The main lever for fitting a longer context on a small card: q8_0 roughly halves the K cache against the f16 default, at some quality cost. Unset keeps f16. Written as --cache-type-k in Extra args.',
    short: 'q8_0 roughly halves the K cache against f16, at some quality cost. Unset keeps f16.',
    restart: 'model-reload',
  },
  cache_type_v: {
    label: 'KV cache type (V)',
    hint: 'llama.cpp only. The V half of the same lever; the two are independent flags and quantising only one is a legitimate, if unusual, choice. Unset keeps f16. Written as --cache-type-v in Extra args.',
    short: 'The V half of the same lever, set independently. Unset keeps f16.',
    restart: 'model-reload',
  },
  max_model_len: {
    label: 'Max model length',
    hint: 'Max sequence length (prompt + generation). Lower = less KV cache memory; higher = fits longer contexts. Capped by model\'s training context.',
    short: 'Prompt + output tokens. Lower frees KV-cache memory.',
    restart: 'model-reload',
  },
  block_size: {
    label: 'Block size',
    hint: 'Affects GPU memory granularity; 16 is correct for nearly all deployments. Changing it requires a model reload and invalidates the KV cache.',
    restart: 'model-reload',
  },
  swap_space: {
    label: 'Swap space (GiB)',
    hint: 'GiB of CPU RAM to spill KV cache to under pressure. Default 4. 0 = disable.',
    restart: 'model-reload',
  },
  max_num_seqs: {
    label: 'Max concurrent seqs',
    hint: 'Max concurrent sequences. Higher = more throughput, more memory.',
    restart: 'model-reload',
  },
  max_num_batched_tokens: {
    label: 'Max batched tokens',
    hint: 'Per-step token budget across all sequences. Default = `max_model_len`. Lower to bound step latency.',
    restart: 'model-reload',
  },
  enforce_eager: {
    label: 'Enforce eager',
    hint: 'If true, disable CUDA graphs. Useful for debugging and tiny models; ~5% slower.',
    restart: 'model-reload',
  },
  // Tri-state capability flags (app/settings/routes_api.py
  // _MODEL_TRISTATE_FIELDS). "Auto" is a real answer, not a blank: the chat
  // catalog sniffs the model's own HF config while nobody has stated one.
  // Read fresh out of the DB on every chat turn, so no reload is needed.
  supports_vision: {
    label: 'Vision',
    hint: 'Whether this model can read images. Auto detects it from the model’s HF config; answer explicitly to override what the config says.',
    short: 'Can read images.',
    restart: 'none',
  },
  supports_tools: {
    label: 'Tools',
    hint: 'Whether the chat may offer this model tool calls. Auto uses the built-in default for the architecture.',
    short: 'Chat may offer tool calls.',
    restart: 'none',
  },
  supports_reasoning: {
    label: 'Reasoning',
    hint: 'Whether this model emits a thinking preamble. Drives the "Enable thinking" toggle in chat. Auto uses the built-in default.',
    short: 'Shows “Enable thinking”.',
    restart: 'none',
  },
  trust_remote_code: {
    label: 'Trust remote code',
    hint: 'A stored grant only: it is not passed to the engine, and no longer to the tokenizer either. To let an engine run a repo\u2019s own Python, add --trust-remote-code to Extra args.',
    short: 'Stored grant only, not passed to the engine. Use `--trust-remote-code` in Extra args.',
    restart: 'none',
  },
  disable_log_requests: {
    label: 'Disable log requests',
    hint: 'Suppress per-request vLLM logs. Default off.',
    restart: 'model-reload',
  },
  extra_args: {
    label: 'Extra args (yours only)',
    // The operator's question, verbatim: "why is Extra args empty but
    // Effective argv has so many arguments?" The distinction was correct and
    // invisible. This box holds ONLY what they added; everything else in the
    // command is derived from the fields above and appears in Effective argv.
    hint: 'Only the args YOU add. The rest of the command is derived from the fields above — see Effective argv at the bottom of this page for the whole thing. Yours are appended last, so they win. One per row: `--worker-use-ray` for vLLM, `--threads 8` for llama.cpp. Flags that have their own control above (flash attention, KV cache types) are edited there and do not appear here.',
    short: 'One per line, appended last so they win. The full command is in Effective command.',
    restart: 'model-reload',
  },
  extra_env: {
    label: 'Extra env',
    hint: 'Free-form env vars passed to the engine subprocess. The allowlist is PER-BACKEND and narrowing: vLLM accepts `VLLM_`, `TRITON_`, `NCCL_`, `PYTORCH_`, `TORCH_`, `OMP_`; llama.cpp accepts `GGML_` only, because every llama.cpp flag has an `LLAMA_ARG_*` env twin and allowing that prefix would let a model row re-bind the unauthenticated engine API. `GET /api/system/backends` reports the exact list per backend.',
    short: 'KEY=value per line. Prefixes: vLLM `VLLM_ NCCL_ TORCH_ …`; llama.cpp `GGML_` only.',
    restart: 'model-reload',
  },
};
