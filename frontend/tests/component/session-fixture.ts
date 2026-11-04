// A synthetic Claude Code session for the session-page tests: main (4 turns, turn 2 dispatches), 3 inferred
// subagents, a compaction continuation, one side call, one relayed turn.
import type { Anatomy, Conversation, DeadWeight, EngineReplay, Pauses, SessionResponse, Tools, Turn } from '@/lib/sessions/types';

export const EMPTY_TOOLS: Tools = {
  offered: 0, called: 0, unused: 0, unused_tokens_per_turn: null, unused_tokens_total: null,
  unused_share: null, estimated: true, truncated: false, rows: [],
};

export const TURN_ANAT: Anatomy = { cached: 850, dead: 100, recomputed: 0, new: 50, unknown: 0 };

export const DEAD_WEIGHT: DeadWeight = {
  prompt_tokens: 11000, total_tokens: 2420, share: 0.22, per_turn: 186, turns: 13, turns_measured: 13, estimated: true,
  kinds: {
    unused_tools: { tokens: 1540, share: 0.14, per_turn: 118 },
    dup: { tokens: 550, share: 0.05, per_turn: 42 },
    boil: { tokens: 220, share: 0.02, per_turn: 17 },
    reason: { tokens: null, share: null, per_turn: null },
  },
};

export const REPLAY: EngineReplay = {
  estimated: true, scope: 'session', turns: 12, skipped_time: 0, slots: 4, replicas: 3, disk_load_excluded: true,
  engines: [
    { engine: 'lm_warden', recompute_tokens: 600, recompute_s: 360, measured: true },
    { engine: 'vllm', recompute_tokens: 2500, recompute_s: 1500, measured: false },
    { engine: 'llamacpp', recompute_tokens: 6300, recompute_s: 3780, measured: false },
    { engine: 'mlx_lm', recompute_tokens: 9900, recompute_s: 5940, measured: false },
    { engine: 'omlx', recompute_tokens: 3400, recompute_s: 2040, measured: false },
  ],
};

export const PAUSES: Pauses = {
  resumes: 14, evicted: 5, unassessed: 2, lost_tokens: 3000, lost_s: 180,
  buckets: [
    { label: '1-5m', min_s: 60, max_s: 300, resumes: 6, evicted: 0, lost_tokens: 0, lost_s: null },
    { label: '5-15m', min_s: 300, max_s: 900, resumes: 4, evicted: 1, lost_tokens: 600, lost_s: 30 },
    { label: '15-60m', min_s: 900, max_s: 3600, resumes: 2, evicted: 2, lost_tokens: 1200, lost_s: 60 },
    { label: '1-4h', min_s: 3600, max_s: 14400, resumes: 2, evicted: 2, lost_tokens: 1200, lost_s: 90 },
    { label: '4h+', min_s: 14400, max_s: null, resumes: 0, evicted: 0, lost_tokens: 0, lost_s: null },
  ],
  disk_tier_hint: { keep_s: 14400, recovered_s: 120 },
};

export function turn(id: string, conv: string | null, n: number, start: number, extra: Partial<Turn> = {}): Turn {
  return {
    request_id: id, conversation_id: conv, turn: n, started_at: start, finished_at: start + 2, edge: n === 1 ? 'new' : 'continue',
    node: 'n-' + id, parent_node: null, parent_request_id: null, prompt_tokens: 1000, cached_tokens: 990,
    generated_tokens: 50, reusable_tokens: 1000, cache_outcome: 'hit', cache_outcome_own: 'hit',
    divergence_cause: null, divergence_cause_own: null, dp_rank: 1, queued_s: 0, ttft_s: 0.4, prefill_s: 0.4,
    gen_s: 1.6, finish_reason: 'stop', shape: { chars: 4000, sys: 1200, tools: 31, tools_chars: 800, msgs: { user: 2, assistant: 1, tool: 0 }, last_user: 300, brk: null },
    tiers: { gpu: null, ram: null, nvme: null, remote: null, load_s: null }, node_id: null, relayed: false,
    upstream: null, model: 'qwen', lineage_unknown: false, side_call: false, dead_weight: null, anatomy: TURN_ANAT,
    cache_known: true, pause_s: null, evicted_on_resume: false, lost_tokens: null, lost_s: null, ...extra,
  };
}

function conv(id: string, label: string, kind: Conversation['kind'], first: number, parent: Conversation['parent'] = null, extra: Partial<Conversation> = {}): Conversation {
  return {
    id, label, kind, parent, first_at: first, last_at: first + 10, turns: 2, prefill_tokens: 2000,
    cached_tokens: 1980, generated_tokens: 100, relayed: false, lineage_unknown: false, tools: EMPTY_TOOLS,
    anatomy: { cached: 1700, dead: 200, recomputed: 0, new: 100, unknown: 0 }, saved_vs_vanilla_s: 60, ...extra,
  };
}

export function fixture(): SessionResponse {
  const main = conv('c-main', 'main', 'main', 0, null, {
    turns: 4,
    tools: {
      offered: 31, called: 6, unused: 25, unused_tokens_per_turn: 7900, unused_tokens_total: 31600, unused_share: 0.09,
      estimated: true, truncated: false,
      rows: [
        { id: 'ab12cd34', name: 'browser_navigate', family: 'web', chars: 3000, tokens: 750, called: 0, offered_turns: 4 },
        { id: 'ef56ab78', name: null, family: 'other', chars: 2000, tokens: 500, called: 0, offered_turns: 4 },
      ],
    },
  });
  const subs = [1, 2, 3].map((k) => conv(`c-sub${k}`, `subagent #${k}`, 'subagent', 12 + k, { conversation_id: 'c-main', request_id: 'm2', link: 'inferred' }));
  const comp = conv('c-comp', 'main after compaction', 'compaction', 40, { conversation_id: 'c-main', request_id: 'm4', link: 'compaction' });
  const turns: Turn[] = [
    turn('m1', 'c-main', 1, 0),
    turn('m2', 'c-main', 2, 10),
    turn('m3', 'c-main', 3, 20, { dead_weight: { unused_tools: 1200, dup: 600, boil: 200, reason: 100, total: 2100, share: 0.14 }, cached_tokens: 620, cache_outcome: 'partial', divergence_cause: 'harness_rewrite', shape: { chars: 4000, sys: 1200, tools: 31, tools_chars: 800, msgs: { user: 3, assistant: 2, tool: 0 }, last_user: 300, brk: 3 } }),
    turn('m4', 'c-main', 4, 30, { pause_s: 1380, evicted_on_resume: true, lost_tokens: 600, lost_s: 30 }),
    ...[1, 2, 3].flatMap((k) => [turn(`s${k}a`, `c-sub${k}`, 1, 12 + k), turn(`s${k}b`, `c-sub${k}`, 2, 14 + k)]),
    turn('p1', 'c-comp', 1, 40, { edge: 'compacted_from' }),
    turn('p2', 'c-comp', 2, 45, { relayed: true, upstream: 'anthropic.example', model: 'claude-sonnet' }),
  ];
  return {
    session_id: 's_' + 'f'.repeat(32), token_id: 'tok1', token_name: 'alice', models: ['claude-sonnet', 'qwen'],
    truncated: false, oldest_data_at: 0,
    totals: {
      started_at: 0, last_at: 47, wall_s: 47, requests: turns.length, conversations: 5, side_calls: 1, compactions: 1,
      prefill_tokens: 11000, cached_tokens: 10400, cached_share: 0.945, generated_tokens: 600, cache_served_s: 6480,
      saved_vs_vanilla_s: 1140, prompt_tokens: 11000, cache_share: 0.91, recomputed_s: 360, replicas: 3, where: 'hybrid',
      relayed: { requests: 1, prefill_tokens: 1000, cached_tokens: 990, upstream: 'anthropic.example' },
    },
    tools: main.tools,
    conversations: [main, ...subs, comp],
    side_calls: [{ request_id: 't1', started_at: 1, model: 'qwen', prompt_tokens: 200 }],
    turns,
    harness: 'opencode',
    dead_weight: DEAD_WEIGHT,
    anatomy: { cached: 9300, dead: 1000, recomputed: 300, new: 400, unknown: 0 },
    engine_replay: REPLAY,
    pauses: PAUSES,
  };
}
