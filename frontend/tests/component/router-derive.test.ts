import { describe, expect, it } from 'vitest';
import {
  deriveRouterState,
  fmtAgo,
  fmtPct,
  globMatches,
  KNOWN_CLAUDE_IDS,
  partsText,
  relayFromTokens,
  ruleTraffic,
  showSetupChecklist,
  setupSteps,
  trafficSplit,
  type RelayInfo,
  type RouterStateInput,
} from '@/lib/router';
import {
  DECISIONS_FALLBACKS,
  DECISIONS_HEALTHY,
  NOW,
  RULE_A,
  RULE_B,
  SETTINGS,
  STATS_BREAKER_OPEN,
  STATS_EMPTY,
  STATS_HEALTHY,
  TOKENS_NONE,
  TOKENS_RELAY,
  TOKENS_TRUNCATED,
  decision,
} from './router-test-utils';

const ON = { ...SETTINGS, enabled: true };
const RELAY: RelayInfo = { relayKeys: 1, relayKnown: true, relayLoading: false };
const NO_RELAY: RelayInfo = { relayKeys: 0, relayKnown: true, relayLoading: false };
// RULE_B's target was deleted; most cases want only the healthy RULE_A.
const LOADED_B = { ...RULE_B, target_model_id: 'm2', target_served_name: 'qwen-big', target_status: 'loaded' };

function input(over: Partial<RouterStateInput> = {}): RouterStateInput {
  return {
    settings: ON,
    rules: [RULE_A, LOADED_B],
    stats: STATS_HEALTHY,
    statsError: false,
    decisions: DECISIONS_HEALTHY.decisions,
    relay: RELAY,
    now: NOW,
    ...over,
  } as RouterStateInput;
}

function strip(over: Partial<RouterStateInput> = {}) {
  const s = deriveRouterState(input(over));
  return {
    s,
    title: s.strip ? partsText(s.strip.title) : '',
    body: s.strip ? partsText(s.strip.body) : '',
    actions: s.strip?.actions.map((a) => a.label) ?? [],
  };
}

describe('deriveRouterState — §4.1 priority order', () => {
  it('loading: no strip until settings arrive', () => {
    const s = deriveRouterState(input({ settings: undefined }));
    expect(s.kind).toBe('loading');
    expect(s.strip).toBeNull();
  });

  it('1. a stats fetch error is the headline, with Retry, over everything else', () => {
    const { s, title, body, actions } = strip({ statsError: true, settings: SETTINGS });
    expect(s.kind).toBe('error');
    expect(s.strip?.tone).toBe('danger');
    expect(s.strip?.role).toBe('alert');
    expect(title).toBe("Can't load router status.");
    expect(body).toBe('Showing the last good data.');
    expect(actions).toEqual(['Retry']);
  });

  it('1. stats error with no data says counts are unavailable', () => {
    expect(strip({ statsError: true, stats: undefined }).body).toMatch(/unavailable/);
  });

  it('2. off with no enabled rule is setup (neutral status)', () => {
    const { s, title, body } = strip({ settings: SETTINGS, rules: [{ ...RULE_A, enabled: false }] });
    expect(s.kind).toBe('setup');
    expect(s.strip?.tone).toBe('neutral');
    expect(s.strip?.role).toBe('status');
    expect(title).toBe('Not set up yet.');
    expect(body).toMatch(/nothing is relayed/);
  });

  it('2. off with rules keeps them and offers Turn on', () => {
    const { s, title, body, actions } = strip({ settings: SETTINGS });
    expect(s.kind).toBe('off');
    expect(title).toBe('Routing is off.');
    expect(body).toBe(
      'Your 2 rules are kept; Claude model names get 404 unless a model is served under that name. Nothing is relayed.',
    );
    expect(actions).toEqual(['Turn on']);
  });

  it('2. off outranks a breaker incident', () => {
    expect(strip({ settings: SETTINGS, stats: STATS_BREAKER_OPEN }).s.kind).toBe('off');
  });

  it('3. an open breaker on a fall-back rule: danger, impact, probe, last failure, three actions', () => {
    const { s, title, body, actions } = strip({ stats: STATS_BREAKER_OPEN });
    expect(s.kind).toBe('danger');
    expect(s.strip?.role).toBe('alert');
    expect(title).toBe('qwen is failing — its breaker is open.');
    expect(body).toMatch(
      new RegExp(
        '^Requests for claude-haiku\\* are going to Anthropic instead \\(the rule falls back\\)\\. ' +
          'Next probe in 41 s\\. Last failure: first_byte_timeout, 3 in a row\\. ' +
          '214 fell back since \\d\\d:\\d\\d — Activity lists the last 100 requests\\.$',
      ),
    );
    expect(actions).toEqual(['Open qwen', 'See recent fallbacks', 'Pause this rule']);
    const pause = s.strip!.actions.find((a) => a.kind === 'pause_rule');
    expect(pause).toMatchObject({ ruleId: 'ra' });
    expect(s.strip!.actions.find((a) => a.kind === 'see_fallbacks')).toMatchObject({
      href: '/router/stats?route=fallback',
    });
    expect(s.strip!.actions.find((a) => a.kind === 'open_model')).toMatchObject({ href: '/models/m1' });
    // the pattern and the reason render as code
    expect(s.strip!.body).toContainEqual({ code: 'claude-haiku*' });
    expect(s.strip!.body).toContainEqual({ code: 'first_byte_timeout' });
  });

  it('3. refuse-rule copy says 529', () => {
    const { body } = strip({ stats: STATS_BREAKER_OPEN, rules: [{ ...RULE_A, fallback: false }] });
    expect(body).toMatch(/^Requests for claude-haiku\* are refused with 529 \(Claude Code retries\)\./);
  });

  it('3. half_open is amber: the next request will probe (no countdown, no "failing now")', () => {
    const half = {
      ...STATS_BREAKER_OPEN,
      targets: [{ ...STATS_BREAKER_OPEN.targets[0], breaker: 'half_open', open_until: null }],
    };
    const { s, title, body } = strip({ stats: half });
    expect(s.kind).toBe('amber');
    expect(s.strip?.tone).toBe('amber');
    expect(title).toBe("qwen's breaker is half-open — the next request will probe it.");
    expect(body).not.toMatch(/Next probe in/);
    expect(body).not.toMatch(/probing now/);
    expect(body).toMatch(/closes if it answers/);
  });

  it('3. a half-open breaker ranks below a deleted target (danger first)', () => {
    const half = {
      ...STATS_BREAKER_OPEN,
      targets: [{ ...STATS_BREAKER_OPEN.targets[0], breaker: 'half_open', open_until: null }],
    };
    const s = deriveRouterState(input({ rules: [RULE_A, RULE_B], stats: half }));
    expect(s.issues.map((i) => i.code)).toEqual(['target_missing', 'breaker']);
    expect(s.kind).toBe('danger');
  });

  it('3 before 4 per rule: a deleted target whose breaker is open reports the breaker', () => {
    const gone = { ...RULE_A, target_served_name: null, target_status: null };
    const s = deriveRouterState(input({ rules: [gone], stats: STATS_BREAKER_OPEN }));
    expect(s.issues.map((i) => i.code)).toEqual(['breaker']);
  });

  it('4. a deleted target: danger, target_missing, Edit and Pause', () => {
    const { s, title, body, actions } = strip({ rules: [RULE_A, RULE_B], stats: STATS_HEALTHY });
    expect(s.kind).toBe('danger');
    expect(title).toBe('The local model for claude-sonnet-4-5 was deleted.');
    expect(body).toMatch(/target_missing/);
    expect(body).toMatch(/529/); // RULE_B refuses
    expect(actions).toEqual(['Edit rule', 'Pause this rule']);
  });

  it('3 outranks 4: an open breaker is the headline and the deleted target is "more"', () => {
    const s = deriveRouterState(input({ rules: [RULE_A, RULE_B], stats: STATS_BREAKER_OPEN }));
    expect(s.strip?.more.map((i) => i.code)).toEqual(['target_missing']);
    expect(s.issues.map((i) => i.code)).toEqual(['breaker', 'target_missing']);
  });

  it('5. a target that is not loaded: amber, Open model', () => {
    const { s, title, body, actions } = strip({
      rules: [RULE_A, { ...LOADED_B, target_status: 'registered' }],
    });
    expect(s.kind).toBe('amber');
    expect(s.strip?.role).toBe('alert');
    expect(title).toBe('qwen-big is not loaded.');
    expect(body).toBe('claude-sonnet-4-5 requests are refused (529).');
    expect(actions).toEqual(['Open qwen-big']);
  });

  it('5. not loaded on a fall-back rule goes to Anthropic', () => {
    const { body } = strip({ rules: [{ ...RULE_A, target_status: 'failed' }] });
    expect(body).toBe('claude-haiku* requests are going to Anthropic.');
  });

  it('a disabled rule raises nothing', () => {
    const { s } = strip({ rules: [{ ...RULE_B, enabled: false }, RULE_A] });
    expect(s.kind).toBe('ok');
  });

  it('6. no relay key while pass-through is on: amber, create + API keys', () => {
    const { s, title, body, actions } = strip({ relay: NO_RELAY });
    expect(s.kind).toBe('amber');
    expect(title).toBe('No key may relay to Anthropic.');
    expect(body).toBe('Unmatched models and fallbacks are refused (403 relay_not_allowed).');
    expect(actions).toEqual(['Create a relay key', 'API Keys']);
  });

  it('6. skipped when neither pass-through nor any fallback rule needs relaying', () => {
    const { s } = strip({
      relay: NO_RELAY,
      settings: { ...ON, passthrough_unmatched: false },
      rules: [{ ...RULE_A, fallback: false }],
    });
    expect(s.kind).toBe('ok');
  });

  it('6. relayKnown=false skips the check (never claim what was not verified)', () => {
    const { s } = strip({ relay: { relayKeys: 0, relayKnown: false, relayLoading: false } });
    expect(s.kind).toBe('ok');
  });

  it('7. recent fallbacks: N of the last M, top reason, See them', () => {
    const { s, title, body, actions } = strip({ decisions: DECISIONS_FALLBACKS.decisions });
    expect(s.kind).toBe('amber');
    expect(title).toBe('3 of the last 4 requests fell back.');
    expect(body).toBe('Top reason: breaker_open.');
    expect(actions).toEqual(['See them']);
  });

  it('7. only the last 20 decisions count', () => {
    const decisions = [
      ...Array.from({ length: 20 }, () => decision()),
      decision({ route: 'fallback', reason: 'x' }),
    ];
    expect(strip({ decisions }).s.kind).toBe('ok');
  });

  it('7. refused and error count too', () => {
    const { title } = strip({
      decisions: [decision({ route: 'refused', reason: 'relay_not_allowed' }), decision({ route: 'error', reason: 'upstream' })],
    });
    expect(title).toBe('2 of the last 2 requests fell back or failed.');
  });

  it('8. on with no traffic yet: waiting, pulse, Connect', () => {
    const { s, title, body } = strip({ stats: { ...STATS_EMPTY, enabled: true }, decisions: [] });
    expect(s.kind).toBe('waiting');
    expect(s.strip?.tone).toBe('neutral');
    expect(s.strip?.role).toBe('status');
    expect(s.strip?.pulse).toBe(true);
    expect(title).toBe('Routing is on — waiting for the first request.');
    expect(body).toMatch(/Connect below/);
  });

  it('on with no enabled rule adds the polite note', () => {
    const s = deriveRouterState(input({ rules: [], stats: { ...STATS_EMPTY, enabled: true }, decisions: [] }));
    expect(s.strip?.note).toBe('On with no rules: every Claude model goes to Anthropic.');
    const s404 = deriveRouterState(
      input({ rules: [], settings: { ...ON, passthrough_unmatched: false }, stats: STATS_HEALTHY }),
    );
    expect(s404.strip?.note).toBe('On with no rules: every Claude model gets 404.');
  });

  it('9. ok: last request, healthy targets, local share, since', () => {
    const { s, title, body } = strip();
    expect(s.kind).toBe('ok');
    expect(s.strip?.tone).toBe('ok');
    expect(s.strip?.role).toBe('status');
    expect(title).toBe('Routing normally.');
    // 1580 of 1900 = 83 %
    expect(body).toMatch(/^Last request 4 s ago · 2 local targets healthy · 83 % answered locally since \d\d:\d\d\.$/);
  });

  it('ok with stats still loading does not claim health', () => {
    const s = deriveRouterState(input({ stats: undefined }));
    expect(s.kind).toBe('loading');
    expect(s.strip).toBeNull();
  });
});

describe('setupSteps', () => {
  const base = { settings: SETTINGS, rules: [] as typeof RULE_A[], stats: STATS_EMPTY, decisions: [], relay: NO_RELAY };

  it('first run: nothing done, step 1 current', () => {
    const steps = setupSteps(base);
    expect(steps.map((s) => s.id)).toEqual(['rule', 'relay', 'connect', 'enable', 'first_request']);
    expect(steps.filter((s) => s.done)).toHaveLength(0);
    expect(steps.find((s) => s.current)?.id).toBe('rule');
  });

  it('progression: a rule, then a relay key, then on', () => {
    let steps = setupSteps({ ...base, rules: [RULE_A] });
    expect(steps.find((s) => s.current)?.id).toBe('relay');
    steps = setupSteps({ ...base, rules: [RULE_A], relay: RELAY });
    expect(steps.find((s) => s.current)?.id).toBe('connect');
    steps = setupSteps({ ...base, rules: [RULE_A], relay: RELAY, settings: ON });
    expect(steps.filter((s) => s.done).map((s) => s.id)).toEqual(['rule', 'relay', 'enable']);
    expect(steps.find((s) => s.current)?.id).toBe('connect');
  });

  it('a seen request completes connect and first_request', () => {
    const steps = setupSteps({ ...base, rules: [RULE_A], settings: ON, stats: STATS_HEALTHY, decisions: DECISIONS_HEALTHY.decisions });
    expect(steps.find((s) => s.id === 'connect')?.done).toBe(true);
    expect(steps.find((s) => s.id === 'first_request')?.done).toBe(true);
    expect(steps.find((s) => s.id === 'relay')?.done).toBe(false);
  });

  it('a disabled rule does not count', () => {
    expect(setupSteps({ ...base, rules: [{ ...RULE_A, enabled: false }] })[0].done).toBe(false);
  });

  it('loading data gives a neutral mark, never a false tick', () => {
    const steps = setupSteps({
      settings: undefined, rules: undefined, stats: undefined, decisions: undefined,
      relay: { relayKeys: 0, relayKnown: false, relayLoading: true },
    });
    expect(steps.every((s) => s.loading && !s.done && !s.current)).toBe(true);
  });

  it('an unverifiable relay answer is marked unverified, not done', () => {
    const relay = { relayKeys: 0, relayKnown: false, relayLoading: false };
    const step = setupSteps({ ...base, relay }).find((s) => s.id === 'relay');
    expect(step).toMatchObject({ done: false, unverified: true, loading: false });
  });
});

describe('relayFromTokens', () => {
  it('counts only live relay keys', () => {
    expect(relayFromTokens(TOKENS_RELAY, false)).toEqual({ relayKeys: 1, relayKnown: true, relayLoading: false });
    expect(relayFromTokens(TOKENS_NONE, false)).toEqual({ relayKeys: 0, relayKnown: true, relayLoading: false });
  });
  it('a truncated list with none found is unknown', () => {
    expect(relayFromTokens(TOKENS_TRUNCATED, false)).toEqual({ relayKeys: 0, relayKnown: false, relayLoading: false });
  });
  it('loading and error states', () => {
    expect(relayFromTokens(undefined, false)).toEqual({ relayKeys: 0, relayKnown: false, relayLoading: true });
    expect(relayFromTokens(undefined, true)).toEqual({ relayKeys: 0, relayKnown: false, relayLoading: false });
    // SWR keeps stale data next to an error: do not trust it.
    expect(relayFromTokens(TOKENS_RELAY, true)).toEqual({ relayKeys: 0, relayKnown: false, relayLoading: false });
  });
});

describe('trafficSplit / fmtPct', () => {
  it('four segments in order with counts and percentages', () => {
    const t = trafficSplit(STATS_BREAKER_OPEN.totals);
    expect(t.total).toBe(1829);
    expect(t.segments.map((s) => [s.key, s.label, s.n])).toEqual([
      ['local', 'Local', 1302],
      ['passthrough', 'Anthropic (pass-through)', 311],
      ['fallback', 'Fell back to Anthropic', 214],
      ['refused', 'Refused or error', 2],
    ]);
    expect(t.segments[0].pct).toBeCloseTo(71.19, 1);
  });

  it('refused folds in errors; missing keys are 0; empty total gives 0 %', () => {
    expect(trafficSplit({ refused: 2, error: 3 }).segments[3].n).toBe(5);
    const z = trafficSplit({});
    expect(z.total).toBe(0);
    expect(z.segments.every((s) => s.pct === 0)).toBe(true);
  });

  it('fmtPct: integers from 10, one decimal below, 0 is 0', () => {
    expect(fmtPct(76.4)).toBe('76 %');
    expect(fmtPct(9.63)).toBe('9.6 %');
    expect(fmtPct(0.05)).toBe('0.1 %');
    expect(fmtPct(0)).toBe('0 %');
  });
});

describe('ruleTraffic', () => {
  it('names only non-zero parts', () => {
    expect(ruleTraffic(STATS_HEALTHY, 'ra')).toMatchObject({ local: 1284, fallback: 6, refused: 0, total: 1290, label: '1,284 local · 6 fell back' });
    expect(ruleTraffic(STATS_HEALTHY, 'rb')?.label).toBe('296 local · 2 refused');
  });
  it('null when there are no counters', () => {
    expect(ruleTraffic(undefined, 'ra')).toBeNull();
    expect(ruleTraffic(STATS_HEALTHY, 'zz')).toBeNull();
    expect(ruleTraffic({ ...STATS_HEALTHY, rules: [{ ...STATS_HEALTHY.rules[0], local: 0, fallback: 0 }] }, 'ra')).toBeNull();
  });
});

describe('fmtAgo', () => {
  it('seconds, minutes, hours, days', () => {
    expect(fmtAgo('2026-10-04T14:31:02Z', NOW)).toBe('4 s ago');
    expect(fmtAgo('2026-10-04T14:28:06Z', NOW)).toBe('3 min ago');
    expect(fmtAgo('2026-10-04T12:31:06Z', NOW)).toBe('2 h ago');
    expect(fmtAgo('2026-10-01T14:31:06Z', NOW)).toBe('3 d ago');
  });
  it('now, future (clock skew) and garbage', () => {
    expect(fmtAgo('2026-10-04T14:31:06Z', NOW)).toBe('just now');
    expect(fmtAgo('2026-10-04T14:31:30Z', NOW)).toBe('just now');
    expect(fmtAgo('nope', NOW)).toBe('—');
  });
});

describe('globMatches — parity with app/router/rules.py (fnmatchcase)', () => {
  it('* matches any run, including empty', () => {
    expect(globMatches('claude-haiku*', 'claude-haiku-4-5')).toBe(true);
    expect(globMatches('claude-haiku*', 'claude-haiku')).toBe(true);
    expect(globMatches('claude-*-x', 'claude-haiku-4')).toBe(false);
    expect(globMatches('*haiku*', 'claude-3-5-haiku-latest')).toBe(true);
  });
  it('? is exactly one character', () => {
    expect(globMatches('claude-?-sonnet', 'claude-3-sonnet')).toBe(true);
    expect(globMatches('claude-?-sonnet', 'claude-33-sonnet')).toBe(false);
  });
  it('exact patterns match only themselves', () => {
    expect(globMatches('claude-haiku-4', 'claude-haiku-4')).toBe(true);
    expect(globMatches('claude-haiku-4', 'claude-haiku-4-1')).toBe(false);
    expect(globMatches('claude-haiku-4', 'claude-haiku-')).toBe(false);
  });
  it('is case-sensitive and treats . literally', () => {
    expect(globMatches('claude-haiku*', 'Claude-Haiku-x')).toBe(false);
    expect(globMatches('a.b', 'axb')).toBe(false);
  });
  it('KNOWN_CLAUDE_IDS covers the presets', () => {
    for (const p of ['claude-haiku*', 'claude-sonnet*', 'claude-opus*']) {
      expect(KNOWN_CLAUDE_IDS.some((id) => globMatches(p, id))).toBe(true);
    }
  });
});

describe('showSetupChecklist — durable facts only (review #1)', () => {
  const seenInput = { stats: STATS_HEALTHY, decisions: DECISIONS_HEALTHY.decisions };
  const unseen = { stats: STATS_EMPTY, decisions: [] };

  it('no enabled rule: the checklist shows, on or off', () => {
    expect(showSetupChecklist({ settings: SETTINGS, rules: [], ...unseen })).toBe(true);
    expect(showSetupChecklist({ settings: ON, rules: [{ ...RULE_A, enabled: false }], ...seenInput })).toBe(true);
  });

  it('switched off mid-incident (traffic seen): no checklist — the evidence stays', () => {
    expect(showSetupChecklist({ settings: SETTINGS, rules: [RULE_A], ...seenInput })).toBe(false);
  });

  it('on with a rule and no traffic (after a restart or a reset): no checklist', () => {
    expect(showSetupChecklist({ settings: ON, rules: [RULE_A], ...unseen })).toBe(false);
  });

  it('off with a rule and never seen: still onboarding, the checklist shows', () => {
    expect(showSetupChecklist({ settings: SETTINGS, rules: [RULE_A], ...unseen })).toBe(true);
  });

  it('off with a rule while traffic is loading: no flash of the checklist', () => {
    expect(showSetupChecklist({ settings: SETTINGS, rules: [RULE_A], stats: undefined, decisions: undefined })).toBe(false);
  });

  it('nothing while settings or rules load', () => {
    expect(showSetupChecklist({ settings: undefined, rules: [], ...unseen })).toBe(false);
    expect(showSetupChecklist({ settings: SETTINGS, rules: undefined, ...unseen })).toBe(false);
  });
});
