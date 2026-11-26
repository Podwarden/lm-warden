// Synthetic scorecard and session list answers (obviously fake values: codes
// and numbers only, as the API sends them).

import type { Scorecard, SessionList, ScoreOut } from '@/lib/prompt-lens/scorecard';

export const SESSION_ID = 's_' + 'a'.repeat(32);

const saving = (tokens: number, prefill: number | null = null, money: number | null = null) => ({
  tokens,
  prefill_s: prefill,
  decode_s: null,
  money,
});

export function scored(layer: ScoreOut['layer'], value: number, o: Partial<ScoreOut> = {}): ScoreOut {
  return { layer, scored: true, value, lo: value - 7, hi: value + 6, confidence: 'high', turns: 24, inputs: {}, excluded_rules: [], ...o };
}

export function notScored(layer: ScoreOut['layer'], reason: string): ScoreOut {
  return { layer, scored: false, reason, turns: 2 };
}

export function scorecardFixture(over: Partial<Scorecard> = {}): Scorecard {
  const turns = Array.from({ length: 24 }, (_, i) => ({
    n: i + 1,
    req_id: `req-fake-${String(i + 1).padStart(4, '0')}`,
    reused: i === 0 ? 0 : 6000 + 400 * (i - 1),
    recomputed: i === 8 ? 8800 : 0,
    new: i === 0 ? 6000 : 400,
    markers: i === 8 ? ['E-01', 'R-04'] : [],
  }));
  return {
    session: { harness: 'other', model: 'fake-model', route: 'local', turns: 24, duration_s: 1080, version: null },
    scores: {
      overall: scored('overall', 62, { lo: 55, hi: 68, confidence: 'medium', inputs: { from: ['E', 'C', 'R'] } }),
      E: scored('E', 48, { lo: 40, hi: 57, inputs: { ideal: 185000, lost: 96200 } }),
      C: scored('C', 71, { lo: 66, hi: 75, inputs: { waste: 89000, prompt: 612000 } }),
      R: scored('R', 74, { lo: 61, hi: 83, confidence: 'medium', inputs: { turns: 24, bad: 5, completion: 4800, waste_out: 430 } }),
    },
    top: [
      { rule: 'E-01', layer: 'E', turns: [2, 3, 9], counts: { line: 2, class: 1, turns: 3, lost: 61400, sessions: 1 }, tool: null, cause: 'sys_changed', saving: saving(61400, 11.2), confidence: 'high', muted: false, part: 'sys' },
      { rule: 'C-01', layer: 'C', turns: [1, 2, 3], counts: { unused: 7, tools: 12, tokens: 5800 }, tool: null, cause: null, saving: saving(139000, 8), confidence: 'high', muted: false, part: 'tools' },
      { rule: 'R-04', layer: 'R', turns: [7, 9, 12, 14], counts: { bad: 4, calls: 6, code: 3 }, tool: 'fa000003', cause: 'schema_undocumented', saving: saving(2100), confidence: 'medium', muted: false, part: 'call:0' },
    ],
    notice: null,
    findings: [
      { rule: 'E-01', layer: 'E', turns: [2, 3, 9], counts: { line: 2, class: 1, turns: 3, lost: 61400, sessions: 1 }, tool: null, cause: 'sys_changed', saving: saving(61400, 11.2), confidence: 'high', muted: false, part: 'sys' },
      { rule: 'E-08', layer: 'E', turns: [17], counts: { turns: 2, lost: 800, pause_s: 840 }, tool: null, cause: 'server_evicted', saving: saving(800), confidence: 'high', muted: false, part: null },
      { rule: 'C-01', layer: 'C', turns: [1, 2, 3], counts: { unused: 7, tools: 12, tokens: 5800 }, tool: null, cause: null, saving: saving(139000, 8), confidence: 'high', muted: false, part: 'tools' },
      { rule: 'R-04', layer: 'R', turns: [7, 9, 12, 14], counts: { bad: 4, calls: 6, code: 3 }, tool: 'fa000003', cause: 'schema_undocumented', saving: saving(2100), confidence: 'medium', muted: false, part: 'call:0' },
    ],
    not_scored: {},
    turns,
    cost: { prompt_tokens: 612000, reused_tokens: 309000, recomputed_tokens: 96000, fixable_lost_tokens: 61400, card: null, actual_input: null, saving: null, saving_pct: null },
    score_version: 1,
    weights: { E: 0.45, C: 0.15, R: 0.4 },
    review: null,
    names: { fa000003: 'search_docs' },
    ...over,
  };
}

export function shortSession(): Scorecard {
  return scorecardFixture({
    session: { harness: 'other', model: 'fake-model', route: 'local', turns: 3, duration_s: 60, version: null },
    scores: {
      overall: notScored('overall', '1 of 2 layers scored'),
      E: notScored('E', '2 of 3 continuing turns'),
      C: scored('C', 90, { turns: 3 }),
      R: notScored('R', '3 of 5 assistant turns'),
    },
    top: [],
    findings: [],
    not_scored: { E: '2 of 3 continuing turns', R: '3 of 5 assistant turns' },
    turns: scorecardFixture().turns.slice(0, 3),
  });
}

export function listFixture(): SessionList {
  return {
    sessions: [
      { session_id: SESSION_ID, model: 'fake-model', route: 'local', harness: 'other', turns: 24, started: 1_700_000_000, last_at: 1_700_001_000, overall: 62, e: 48, c: 71, r: 74, confidence: 'medium', reasons: {}, score_version: 1, key_name: 'alpha-key' },
      { session_id: 's_' + 'b'.repeat(32), model: 'fake-model', route: 'local', harness: 'other', turns: 3, started: 1_700_002_000, last_at: 1_700_002_100, overall: null, e: null, c: 90, r: null, confidence: null, reasons: { E: '2 of 3 continuing turns', R: '3 of 5 assistant turns' }, score_version: 1, key_name: 'alpha-key' },
    ],
    next_cursor: null,
  };
}

/** The scorecard of a session of prompt version v4 with E-01 marked won't fix. */
export function mutedSession(by: 'key' | 'admin' = 'key'): Scorecard {
  const base = scorecardFixture();
  const mute = <T extends { rule: string }>(f: T): T => (f.rule === 'E-01' ? { ...f, muted: true, muted_by: by } : f);
  return {
    ...base,
    session: { ...base.session, version: 'v4', version_id: 'aaaa1111', version_since: 1_700_000_000, key_id: 'tok-fake-0001' },
    scores: {
      ...base.scores,
      E: scored('E', 100, { inputs: { ideal: 185000, lost: 0, with_marks: 48 }, excluded_rules: ['E-01'] }),
      overall: scored('overall', 79, { lo: 74, hi: 83, inputs: { from: ['E', 'C', 'R'], with_marks: 62 }, excluded_rules: ['E-01'] }),
    },
    top: base.top.filter((f) => f.rule !== 'E-01'),
    findings: base.findings.map(mute),
  };
}
