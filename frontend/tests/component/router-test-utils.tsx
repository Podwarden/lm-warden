import { vi } from 'vitest';
import type { TokenListPage } from '@/lib/router';

type TokenItem = TokenListPage['items'][number];

export const RULE_A = {
  id: 'ra',
  position: 0,
  pattern: 'claude-haiku*',
  target_model_id: 'm1',
  target_served_name: 'qwen',
  target_status: 'loaded',
  enabled: true,
  fallback: true,
  strip_thinking: true,
  min_max_tokens: 0,
  created_at: '2026-10-04T00:00:00Z',
  updated_at: '2026-10-04T00:00:00Z',
};
export const RULE_B = {
  ...RULE_A,
  id: 'rb',
  position: 1,
  pattern: 'claude-sonnet-4-5',
  target_model_id: 'm2',
  target_served_name: null,
  target_status: null,
  fallback: false,
  strip_thinking: false,
  min_max_tokens: 64,
};
export const SETTINGS = {
  enabled: false,
  upstream_url: 'https://api.anthropic.com',
  passthrough_unmatched: true,
  local_header_timeout_s: 30,
  local_nonstream_timeout_s: 300,
  breaker_threshold: 3,
  breaker_open_s: 30,
  max_body_mb: 32,
  header_name: 'X-LMWarden-Key',
};
export const MODELS = {
  models: [
    { id: 'm1', served_model_name: 'qwen', status: 'loaded' },
    { id: 'm3', served_model_name: 'llama', status: 'registered' },
  ],
};

export function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status });
}

export type Handler = (init?: RequestInit) => Response;

/** Fetch stub keyed by "METHOD /path"; records every call. */
export function stubFetch(routes: Record<string, Handler>) {
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : (input as Request).url;
    const key = `${(init?.method ?? 'GET').toUpperCase()} ${url}`;
    const h = routes[key];
    return h ? h(init) : new Response(`unmocked: ${key}`, { status: 404 });
  });
  vi.stubGlobal('fetch', fn);
  return fn;
}

export function callsTo(fn: ReturnType<typeof vi.fn>, key: string) {
  return fn.mock.calls.filter(
    ([i, init]) =>
      `${((init as RequestInit | undefined)?.method ?? 'GET').toUpperCase()} ${
        typeof i === 'string' ? i : (i as Request).url
      }` === key,
  );
}

export function baseRoutes(over: Record<string, Handler> = {}): Record<string, Handler> {
  return {
    'GET /api/router/rules': () => json({ rules: [RULE_A, RULE_B] }),
    'GET /api/router/settings': () => json(SETTINGS),
    'GET /api/models': () => json(MODELS),
    'GET /api/settings/runtime': () => json({ public_url: 'https://warden.example' }),
    'GET /api/router/stats': () => json(STATS_EMPTY),
    'GET /api/router/decisions?limit=20': () => json(DECISIONS_EMPTY),
    'GET /api/router/decisions?limit=100': () => json(DECISIONS_EMPTY),
    [`GET ${TOKENS_URL}`]: () => json(TOKENS_NONE),
    ...over,
  };
}

// ── Router redesign fixtures (stats / decisions / tokens) ──────────────────
// A fixed clock for the derived-state tests: 2026-10-04 14:31:06Z.
export const NOW = Date.parse('2026-10-04T14:31:06Z');

const P = { p50: null, p95: null };

export const STATS_EMPTY = {
  enabled: false,
  since: null,
  totals: { local: 0, passthrough: 0, fallback: 0, refused: 0, error: 0 },
  by_reason: {},
  error_reasons: {},
  rules: [],
  targets: [],
  passthrough: { requests: 0, errors: 0, latency_ms: P, ttfb_ms: P, input_tokens: 0, output_tokens: 0 },
};

/** RULE_A (m1) healthy and busy; 79 % local. */
export const STATS_HEALTHY = {
  enabled: true,
  since: '2026-10-04T09:12:00Z',
  totals: { local: 1580, passthrough: 311, fallback: 6, refused: 2, error: 1 },
  by_reason: { first_byte_timeout: 6 },
  error_reasons: {},
  rules: [
    {
      rule_id: 'ra', pattern: 'claude-haiku*', target_model_id: 'm1', target_served_name: 'qwen',
      local: 1284, fallback: 6, refused: 0, latency_ms: { p50: 820, p95: 2100 }, ttfb_ms: { p50: 140, p95: 400 },
    },
    {
      rule_id: 'rb', pattern: 'claude-sonnet-4-5', target_model_id: 'm2', target_served_name: null,
      local: 296, fallback: 0, refused: 2, latency_ms: P, ttfb_ms: P,
    },
  ],
  targets: [
    { model_id: 'm1', served_name: 'qwen', breaker: 'closed', consecutive_failures: 0, open_until: null, last_reason: null, failures: 6 },
  ],
  passthrough: { requests: 311, errors: 1, latency_ms: P, ttfb_ms: P, input_tokens: 1000, output_tokens: 500 },
};

/** m1's breaker is open for another 41 s after 3 first-byte timeouts. */
export const STATS_BREAKER_OPEN = {
  ...STATS_HEALTHY,
  totals: { local: 1302, passthrough: 311, fallback: 214, refused: 2, error: 0 },
  by_reason: { first_byte_timeout: 180, breaker_open: 34 },
  rules: [{ ...STATS_HEALTHY.rules[0], local: 1302, fallback: 214 }, STATS_HEALTHY.rules[1]],
  targets: [
    {
      model_id: 'm1', served_name: 'qwen', breaker: 'open', consecutive_failures: 3,
      open_until: '2026-10-04T14:31:47Z', last_reason: 'first_byte_timeout', failures: 9,
    },
  ],
};

export function decision(over: Record<string, unknown> = {}) {
  return {
    ts: '2026-10-04T14:31:02Z', path: '/v1/messages', model_in: 'claude-haiku-4-5', model_out: 'qwen',
    rule_id: 'ra', route: 'local', reason: null, status: 200, latency_ms: 820, ttfb_ms: 140,
    stream: true, token_name: 'laptop', ...over,
  };
}

export const DECISIONS_EMPTY = { decisions: [] };
/** Newest first, as the server returns them. */
export const DECISIONS_HEALTHY = {
  decisions: [
    decision(),
    decision({ ts: '2026-10-04T14:30:59Z', model_in: 'claude-opus-4-1', model_out: null, rule_id: null, route: 'passthrough', token_name: 'ci-runner' }),
    decision({ ts: '2026-10-04T14:30:41Z' }),
  ],
};
export const DECISIONS_FALLBACKS = {
  decisions: [
    decision({ route: 'fallback', reason: 'breaker_open', model_out: 'claude-haiku-4-5' }),
    decision({ ts: '2026-10-04T14:30:59Z', route: 'fallback', reason: 'breaker_open' }),
    decision({ ts: '2026-10-04T14:30:20Z', route: 'fallback', reason: 'first_byte_timeout' }),
    decision({ ts: '2026-10-04T14:30:00Z' }),
  ],
};

export const TOKENS_URL = '/api/tokens?limit=500&sort=created';

export function token(over: Partial<TokenItem> = {}): TokenItem {
  return {
    id: 't1', name: 'laptop', prefix: 'vw_abcd', preview: 'vw_abcd…', created_at: '2026-10-01T00:00:00Z',
    last_used_at: null, expires_at: null, rotated_at: null, rotated_from: null, successor_id: null,
    successor_deleted: false, is_expired: false, is_near_expiry: false, revoked_at: null, is_revoked: false,
    priority: 5, anthropic_relay: false, usage_24h: { requests: 0, prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    paused_at: null, is_paused: false, ...over,
  };
}

function tokenPage(items: TokenItem[], total = items.length): TokenListPage {
  return { items, total, limit: 500, offset: 0, near_expiry: 0 };
}

/** Keys exist but none may relay (a revoked/expired/paused relay key does not count). */
export const TOKENS_NONE = tokenPage([
  token(),
  token({ id: 't2', name: 'old', anthropic_relay: true, is_revoked: true }),
  token({ id: 't3', name: 'stale', anthropic_relay: true, is_expired: true }),
  token({ id: 't4', name: 'held', anthropic_relay: true, is_paused: true }),
]);
export const TOKENS_RELAY = tokenPage([token(), token({ id: 't5', name: 'claude-code', anthropic_relay: true })]);
/** First page has no relay key but the list is longer than one page. */
export const TOKENS_TRUNCATED = tokenPage([token()], 812);
