// Mirrors of the sessions API JSON (app/sessions/model.py). Epoch seconds, snake_case.

export type Harness = "claude_code" | "codex" | "opencode" | "openai_sdk" | "other";
export type Where = "local" | "relayed" | "hybrid";

/** What a prompt was made of, in ≈ tokens; the parts sum to the prompt. */
export interface Anatomy {
  cached: number;
  dead: number;
  recomputed: number;
  new: number;
  unknown: number;
}

export type DeadWeightKind = "unused_tools" | "dup" | "boil" | "reason";

export type DeadWeightKinds = Record<
  DeadWeightKind,
  { tokens: number | null; share: number | null; per_turn: number | null }
>;

/** A session's or conversation's dead weight, over the turns where it was measured. */
export interface DeadWeight {
  /** Prompt tokens of the measured turns: the base of `share`. */
  prompt_tokens: number;
  total_tokens: number | null;
  share: number | null;
  per_turn: number | null;
  turns: number;
  turns_measured: number;
  estimated: true;
  kinds: DeadWeightKinds;
}

export interface TurnDeadWeight {
  unused_tools: number | null;
  dup: number | null;
  boil: number | null;
  reason: number | null;
  total: number;
  share: number | null;
}

export type PauseLabel = "1-5m" | "5-15m" | "15-60m" | "1-4h" | "4h+";

export interface PauseBucket {
  label: PauseLabel;
  min_s: number;
  max_s: number | null;
  resumes: number;
  evicted: number;
  lost_tokens: number;
  lost_s: number | null;
}

export interface Pauses {
  resumes: number;
  evicted: number;
  unassessed: number;
  lost_tokens: number;
  lost_s: number | null;
  buckets: PauseBucket[];
  disk_tier_hint: { keep_s: number; recovered_s: number } | null;
}

export type Engine = "lm_warden" | "vllm" | "llamacpp" | "mlx_lm" | "omlx";

export interface EngineReplay {
  estimated: true;
  scope: string;
  turns: number;
  skipped_time: number;
  slots: number;
  replicas: number;
  disk_load_excluded: boolean;
  engines: { engine: Engine; recompute_tokens: number | null; recompute_s: number | null; measured: boolean }[];
}

export interface Shape {
  chars?: number;
  sys?: number;
  tools?: number;
  tools_chars?: number;
  msgs?: { user: number; assistant: number; tool: number };
  last_user?: number | null;
  brk?: number | null;
  dw?: { dup: number; boil: number; reason: number };
}

export interface ToolRow {
  id: string;
  name: string | null;
  family: string;
  chars: number;
  tokens: number;
  called: number;
  offered_turns: number;
}

export interface Tools {
  offered: number;
  called: number;
  unused: number;
  unused_tokens_per_turn: number | null;
  unused_tokens_total: number | null;
  unused_share: number | null;
  estimated: true;
  truncated: boolean;
  rows: ToolRow[];
}

export type ConversationKind = "main" | "subagent" | "compaction" | "conversation" | "unlinked";
export type LinkKind = "known" | "inferred" | "compaction";

export interface Conversation {
  id: string | null;
  label: string;
  kind: ConversationKind;
  parent: { conversation_id: string | null; request_id: string; link: LinkKind } | null;
  first_at: number;
  last_at: number;
  turns: number;
  prefill_tokens: number;
  cached_tokens: number;
  generated_tokens: number;
  relayed: boolean;
  lineage_unknown: boolean;
  tools: Tools;
  dead_weight?: DeadWeight;
  anatomy?: Anatomy;
  saved_vs_vanilla_s?: number | null;
}

export interface Turn {
  request_id: string;
  conversation_id: string | null;
  turn: number;
  started_at: number;
  finished_at: number;
  edge: string | null;
  node: string | null;
  parent_node: string | null;
  parent_request_id: string | null;
  prompt_tokens: number;
  cached_tokens: number | null;
  generated_tokens: number;
  reusable_tokens: number | null;
  cache_outcome: string | null;
  cache_outcome_own: string | null;
  divergence_cause: string | null;
  divergence_cause_own: string | null;
  dp_rank: number | null;
  queued_s: number | null;
  ttft_s: number | null;
  prefill_s: number | null;
  gen_s: number | null;
  finish_reason: string | null;
  shape: Shape | null;
  tiers: { gpu: number | null; ram: number | null; nvme: number | null; remote: number | null; load_s: number | null };
  node_id: string | null;
  relayed: boolean;
  upstream: string | null;
  model: string;
  lineage_unknown: boolean;
  side_call: boolean;
  dead_weight?: TurnDeadWeight | null;
  anatomy?: Anatomy;
  cache_known?: boolean;
  pause_s?: number | null;
  evicted_on_resume?: boolean;
  lost_tokens?: number | null;
  lost_s?: number | null;
}

export interface SessionTotals {
  started_at: number | null;
  last_at: number | null;
  wall_s: number | null;
  requests: number;
  conversations: number;
  side_calls: number;
  compactions: number;
  prefill_tokens: number;
  cached_tokens: number;
  cached_share: number | null;
  generated_tokens: number;
  cache_served_s: number | null;
  saved_vs_vanilla_s: number | null;
  /** Cached over prompt tokens where the cache is known (relayed: the upstream's); null when unknown. */
  cache_share?: number | null;
  prompt_tokens?: number;
  recomputed_s?: number | null;
  replicas?: number | null;
  where?: Where;
  relayed: { requests: number; prefill_tokens: number; cached_tokens: number; upstream: string | null } | null;
}

export interface SessionResponse {
  session_id: string;
  token_id: string | null;
  token_name: string | null;
  models: string[];
  truncated: boolean;
  oldest_data_at: number | null;
  totals: SessionTotals;
  tools: Tools;
  conversations: Conversation[];
  side_calls: { request_id: string; started_at: number; model: string; prompt_tokens: number }[];
  turns: Turn[];
  harness?: Harness;
  dead_weight?: DeadWeight;
  anatomy?: Anatomy;
  engine_replay?: EngineReplay;
  pauses?: Pauses;
}

export interface SessionListItem {
  session_id: string;
  token_id: string | null;
  token_name: string | null;
  models: string[];
  started_at: number;
  last_at: number;
  duration_s: number;
  requests: number;
  conversations: number;
  prefill_tokens: number;
  cached_tokens: number;
  cached_share: number | null;
  relayed_requests: number;
  generated_tokens: number;
  compactions: number;
  active: boolean;
  harness: Harness;
  where: Where;
  prompt_tokens: number;
  anatomy: Anatomy;
  dead_weight_share: number | null;
  dead_weight_tokens: number | null;
  dead_weight_measured: boolean;
  saved_vs_vanilla_s: number | null;
  replicas: number | null;
  recomputed_s: number | null;
  cache_served_s: number | null;
  cache_known: boolean;
  cache_share: number | null;
  evicted_resumes: number;
  evicted_lost_s: number | null;
}

export type ListView = "all" | "running" | "biggest" | "dead_weight" | "cache_trouble" | "long_running" | "evicted";
export type ListGroupBy = "key" | "harness" | "day" | "none";
export type ListWhere = "all" | Where;
export type ListSort = "newest" | "prompt_tokens" | "dead_weight" | "duration" | "evicted_lost";

export interface ListGroup {
  key: string;
  label: string;
  harnesses: string[];
  where: Record<Where, number>;
  sessions: number;
  duration_s: number;
  prompt_tokens: number;
  cached_share: number | null;
  anatomy: Anatomy;
  dead_weight_share: number | null;
  saved_vs_vanilla_s: number | null;
  evicted_resumes: number;
  evicted_lost_s: number | null;
  items: SessionListItem[];
  more: number;
}

export interface Histogram {
  bin_s: number;
  start: number;
  bins: { t: number; sessions: number; over_dw: number }[];
}

export interface SessionsListResponse {
  sessions: SessionListItem[];
  next_cursor: string | null;
  truncated_before: number | null;
  oldest_data_at: number | null;
  total: number;
  views: Record<ListView, number>;
  histogram: Histogram;
  pause_summary: Pauses;
  groups?: ListGroup[] | null;
}
