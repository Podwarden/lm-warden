import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { useState } from 'react';

import type { LensFinding, RequestFindings } from '@/lib/prompt-lens/findings';
import type { RequestView } from '@/lib/prompt-lens/store';

import FIXTURES from '../unit/prompt-lens-findings-fixtures.json';
import { PROMPT, part, reqWith } from '../unit/prompt-lens-kvsig-fixtures';

const h = vi.hoisted(() => ({ admin: vi.fn(), key: vi.fn() }));

vi.mock('@/lib/auth-fetch', async (orig) => ({
  ...(await orig<typeof import('@/lib/auth-fetch')>()),
  authFetchJSON: (u: string) => h.admin(u),
  authFetch: vi.fn(async () => new Response(new Blob(['x']))),
}));
vi.mock('@/lib/sessions/client', async (orig) => ({
  ...(await orig<typeof import('@/lib/sessions/client')>()),
  sessionsFetcher: () => (u: string) => h.key(u),
}));
vi.mock('@podwarden/chat-ui', () => ({ copyToClipboard: vi.fn() }));

import { RequestScreen } from '@/components/prompt-lens/dock/request-screen';
import { useCollapseMemory } from '@/components/prompt-lens/dock/section-box';
import { FindingChip } from '@/components/prompt-lens/findings/finding-chip';
import { CATALOGUE, forScope, ruleCopy } from '@/lib/prompt-lens/findings';
import { resetServerClock } from '@/lib/prompt-lens/server-clock';

type Fixtures = Record<string, RequestFindings>;
const FX = Object.fromEntries(Object.entries(FIXTURES).filter(([k]) => !k.startsWith('_'))) as unknown as Fixtures;
const ALL: LensFinding[] = Object.values(FX).flatMap((a) => a.findings);
const TURN9 = FX.turn9;

/** A finding the seeds do not produce, built in the server's shape. */
function finding(rule: string, counts: Record<string, number>, o: Partial<LensFinding> = {}): LensFinding {
  return {
    rule,
    turns: [9],
    counts,
    tool: null,
    cause: null,
    saving: { tokens: 0, prefill_s: null, decode_s: null, money: null },
    confidence: 'high',
    muted: false,
    part: null,
    guard: null,
    ...o,
  };
}

const EXTRA: LensFinding[] = [
  finding('C-03', { share: 74, tokens: 9100, turns: 24, reused: 95, info: 1 }, { confidence: 'low', part: 'sys' }),
  finding('C-03', { share: 74, tokens: 9100, turns: 24 }, { part: 'sys' }),
  finding('C-04', { share: 78, largest: 9200, results: 3 }, { tool: 'fa000003', confidence: 'medium' }),
  finding('C-07', { share: 12, tokens: 900, turns: 24 }, { confidence: 'medium' }),
  finding('C-05', { growth: 1800, p90: 2400, fill_turn: 31, turns: 12 }, { confidence: 'medium' }),
  finding('C-05', { compact_at: 41, compactions: 2 }, { confidence: 'medium' }),
  finding('C-08', { tokens: 3100, x: 4, used: 2, turns: 24 }, { tool: 'fa000003', part: 'tools' }),
  finding('C-09', { tools: 140, limit: 128 }, { part: 'tools' }),
  finding('R-02', { cuts: 1, turn: 22, prompt: 32000 }, { cause: 'window_filled' }),
  finding('R-03', { turns: 1, think: 1, repeat: 0, hard: 0, info: 1 }, { cause: 'none_found', confidence: 'low' }),
  finding('R-06', { failed: 3, runs: 1, len: 32 }, { tool: 'fa000003', cause: 'error_too_short', confidence: 'medium', turns: [7, 8, 9] }),
  finding('R-08', { turns: 4, reasoning: 900, excess: 1550, info: 1 }, { cause: 'thinking_for_short_step', confidence: 'low' }),
  finding('R-09', { values: 2, rules: 1 }, { cause: 'secret_in_system', part: 'sys', guard: { rules: ['fake_secret_rule'], where: ['system'] } }),
  finding('R-10', { turns: 6, ttft_ms: 4100, recompute_ms: 3600, queue_ms: 100 }, { cause: 'recompute_dominates' }),
  finding('E-03', { turns: 2, lost: 9000 }, { cause: 'tools_changed', part: 'tools' }),
  finding('E-04', { turns: 2, lost: 9000, message: 6 }, { cause: 'history_rewritten', part: 'node:6' }),
  finding('E-05', { switches: 3, dropped: 2, kept: 1, lost: 4000 }, { cause: 'reasoning_dropped' }),
  finding('E-07', { compactions: 1, after: 4, lost: 9000 }, { cause: 'compacted', part: 'sys' }),
  finding('E-09', { turns: 2, lost: 3000, message: 3 }, { cause: 'media_changed', part: 'node:3' }),
];

const NAMES = { fa000003: 'search_docs' };

beforeEach(() => {
  h.admin.mockReset();
  h.key.mockReset();
  h.admin.mockImplementation(() => new Promise(() => {}));
  h.key.mockImplementation(() => new Promise(() => {}));
  resetServerClock();
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

// ---------------------------------------------------------------------------
// ruleCopy: the catalogue's copy, filled from the server's counts

describe('ruleCopy', () => {
  it('fills every rule the server answered with, leaving no placeholder, markdown or reference', () => {
    const rules = new Set<string>();
    for (const f of [...ALL, ...EXTRA]) {
      rules.add(f.rule);
      const { title, fix } = ruleCopy(f.rule, f, NAMES);
      expect(title, f.rule).not.toMatch(/[{}]/);
      expect(title, f.rule).not.toMatch(/undefined|NaN/);
      expect(fix, f.rule).not.toMatch(/[{}*]|\(A-\d+\)|^as [ECR]-|^see [ECR]-|per 8\.2/);
      expect(fix.length, f.rule).toBeGreaterThan(3);
    }
    // The seeds and the extras cover the whole catalogue.
    expect([...rules].sort()).toEqual(Object.keys(CATALOGUE).sort());
  });

  it('pins the turn 9 answer: E-01 with its line and class, R-04 with the tool name and cause', () => {
    const by = Object.fromEntries(TURN9.findings.map((f) => [f.rule, f]));
    expect(ruleCopy('E-01', by['E-01'], TURN9.names).title).toBe(
      'System prompt line 2 changes every turn (time); the lines after it are recomputed',
    );
    expect(ruleCopy('E-01', by['E-01']).fix).toBe(
      'make the value stable (a day, not a time; a fixed path) or move it to the newest user message; re-check answers',
    );
    expect(ruleCopy('R-04', by['R-04'], TURN9.names).title).toBe(
      'Tool search_docs was called with missing arg 4 times of 6; a parameter has no description',
    );
    // Without a name the hash's head stands in, never a guessed name.
    expect(ruleCopy('R-04', by['R-04']).title).toMatch(/^Tool #fa00 was called/);
  });

  it('maps the E-01 class index back to its name (probes.SPAN_CLASSES)', () => {
    const e01 = FX.path_line.findings.find((f) => f.rule === 'E-01')!;
    expect(ruleCopy('E-01', e01).title).toContain('line 4 changes every turn (path)');
  });

  it('resolves the cross-references: C-06 as E-01, R-02 see C-05', () => {
    const c06 = TURN9.findings.find((f) => f.rule === 'C-06')!;
    expect(ruleCopy('C-06', c06).fix).toBe(ruleCopy('E-01', c06).fix);
    const r02 = EXTRA.find((f) => f.rule === 'R-02')!;
    expect(ruleCopy('R-02', r02).fix).toBe('trim tool results, compact later or with a cache-aware summary');
  });

  it("strips R-08's markdown emphasis", () => {
    const r08 = EXTRA.find((f) => f.rule === 'R-08')!;
    expect(ruleCopy('R-08', r08).fix).toBe('a lower thinking budget for this step, with an answer check');
  });

  it('R-09 names the rule and the role from the guard data', () => {
    const r09 = EXTRA.find((f) => f.rule === 'R-09')!;
    expect(ruleCopy('R-09', r09).title).toBe(
      '2 values matched rule fake_secret_rule in the system prompt; the model saw placeholders',
    );
  });

  it("never changes a tool name's case (M8)", () => {
    const tool = { tool: 'fa000003' };
    const named: LensFinding[] = [
      finding('E-03', { turns: 2, lost: 9000 }, { ...tool, cause: 'tools_changed' }),
      finding('C-08', { tokens: 3100, x: 4, used: 0 }, tool),
      finding('R-04', { bad: 3, calls: 5, code: 3 }, { ...tool, cause: 'schema_untyped' }),
      finding('R-05', { calls: 3 }, { ...tool, cause: 'result_not_final' }),
      finding('R-06', { failed: 3, len: 32 }, { ...tool, cause: 'error_too_short', turns: [7, 8, 9] }),
    ];
    for (const f of named) {
      const { title } = ruleCopy(f.rule, f, NAMES);
      expect(title, f.rule).toMatch(/^Tool search_docs\b/);
      expect(title, f.rule).not.toContain('Search_docs');
      // A tool with no name for this caller: its hash's head, after "Tool".
      expect(ruleCopy(f.rule, f).title, f.rule).toMatch(/^Tool #fa00\b/);
      // No tool at all.
      expect(ruleCopy(f.rule, { ...f, tool: null }).title, f.rule).toMatch(/^A tool\b/);
    }
    // A camel-case name keeps its case too.
    const camel = finding('R-04', { bad: 3, calls: 5 }, { tool: 'fa000009' });
    expect(ruleCopy('R-04', camel, { fa000009: 'readFile' }).title).toMatch(/^Tool readFile was called/);
  });

  it("R-04's parser mismatch fix is admin-side, never a prompt edit (I2)", () => {
    const f = finding('R-04', { bad: 3, calls: 5, code: 5 }, { tool: 'fa000003', cause: 'parser_mismatch_admin' });
    const admin = ruleCopy('R-04', f, NAMES, 'admin');
    expect(admin.title).toContain("an engine setting, not the prompt");
    expect(admin.fix).toBe("set the engine's tool parser to match the model's tool-call format");
    const key = ruleCopy('R-04', f, NAMES, 'key');
    expect(key.fix).toBe("nothing to change in your prompt: the server's tool parser does not match this model's format");
    for (const fix of [admin.fix, key.fix]) expect(fix).not.toMatch(/description|required|parameter/);
    // Any other R-04 cause keeps the catalogue's prompt fix, in both scopes.
    const schema = { ...f, cause: 'schema_untyped' };
    expect(ruleCopy('R-04', schema, NAMES, 'key').fix).toMatch(/^add the format and an example to the description/);
    expect(ruleCopy('R-04', schema, NAMES).fix).toBe(ruleCopy('R-04', schema, NAMES, 'admin').fix);
  });

  it('the chip shows the scoped R-04 fix', () => {
    const f = finding('R-04', { bad: 3, calls: 5, code: 5 }, { tool: 'fa000003', cause: 'parser_mismatch_admin' });
    const { unmount } = render(<FindingChip finding={f} names={NAMES} scope="key" />);
    expect(screen.getByTestId('finding-chip').textContent).toContain(
      "Fix: nothing to change in your prompt: the server's tool parser does not match this model's format.",
    );
    unmount();
    render(<FindingChip finding={f} names={NAMES} scope="admin" />);
    expect(screen.getByTestId('finding-chip').textContent).toContain(
      "Fix: set the engine's tool parser to match the model's tool-call format.",
    );
  });

  it('key-holder copy never says cached', () => {
    const c03 = EXTRA[1];
    const { fix } = ruleCopy('C-03', c03);
    expect(fix).toContain('cached');
    expect(forScope(fix, 'key')).not.toMatch(/cached/);
    expect(forScope(fix, 'key')).toContain('reused from your earlier requests');
  });
});

// ---------------------------------------------------------------------------
// The request view

const PARTS = [
  part('sys', -1, -1, 'new', { chars: 4000 }),
  part('tools', -1, -1, 'new', { chars: 4800 }),
  part('nodes', 0, 15, 'new', { chars: 9800, roles: { user: 5, assistant: 6, tool: 5 } }),
  part('nodes', 16, 16, 'new', { chars: 900, roles: { tool: 1 } }),
];

function ended(o: Partial<RequestView> = {}): RequestView {
  const base = reqWith({ parts: PARTS, prompt: PROMPT, end: { prompt_tokens: 9400, completion_tokens: 40 } });
  return {
    ...base,
    reqId: 'a1b2c3d4e5f60718',
    sessionId: '3f9a0c21deadbeef',
    turn: 9,
    finish: 'tool_calls',
    toolCalls: [{ index: 0, callId: 'call_c41e', name: 'search_docs', args: '{"from":"last month"}', done: true, elided: false }],
    ...o,
  };
}

function Screen({ r, scope = 'admin', phone = false }: { r: RequestView; scope?: 'admin' | 'key'; phone?: boolean }) {
  const collapse = useCollapseMemory();
  return <RequestScreen request={r} scope={scope} phone={phone} collapse={collapse} />;
}

function Live({ first }: { first: RequestView }) {
  const [r, setR] = useState(first);
  return (
    <>
      <button type="button" onClick={() => setR(ended())}>
        end
      </button>
      <Screen r={r} />
    </>
  );
}

async function settle() {
  await act(async () => {
    await new Promise((res) => setTimeout(res, 0));
  });
}

function answer(body: RequestFindings) {
  return () => Promise.resolve(body);
}

describe('request findings in the request view', () => {
  it('shows the strip and an E-01 chip on the sys part with its line', async () => {
    h.admin.mockImplementation(answer(TURN9));
    render(<Screen r={ended()} />);
    await settle();
    expect(h.admin).toHaveBeenCalledWith('/api/admin/prompt-lens/requests/a1b2c3d4e5f60718/findings');
    const strip = screen.getByTestId('findings-strip');
    expect(within(strip).getByText('Findings on this request')).toBeTruthy();
    expect(within(strip).getAllByTestId('findings-pill').map((p) => p.dataset.rule)).toEqual(['E-01', 'C-06', 'R-04']);
    expect(within(strip).getByText('line 2 changes')).toBeTruthy();
    const sysRow = screen.getAllByRole('row').find((row) => within(row).queryByText('sys'))!;
    const chips = within(sysRow).getAllByTestId('finding-chip');
    expect(chips.map((c) => c.dataset.rule)).toEqual(['E-01', 'C-06']);
    expect(chips[0].textContent).toContain('System prompt line 2 changes every turn (time)');
    expect(chips[0].textContent).toContain('Fix: make the value stable');
    // The tools row carries none of them.
    const toolsRow = screen.getAllByRole('row').find((row) => within(row).queryByText('tools'))!;
    expect(within(toolsRow).queryAllByTestId('finding-chip')).toEqual([]);
  });

  it('puts R-04 under the tool call it judged, with the key-scoped name', async () => {
    h.key.mockImplementation(answer(TURN9));
    render(<Screen r={ended()} scope="key" />);
    await settle();
    expect(h.key).toHaveBeenCalledWith('/api/me/prompt-lens/requests/a1b2c3d4e5f60718/findings');
    expect(h.admin).not.toHaveBeenCalled();
    const calls = screen.getByTestId('call-findings');
    const chip = within(calls).getByTestId('finding-chip');
    expect(chip.dataset.rule).toBe('R-04');
    expect(chip.textContent).toContain('tool call 1 · search_docs');
    expect(chip.textContent).toContain('missing arg 4 times of 6');
  });

  it('asks only once the request ended', async () => {
    h.admin.mockImplementation(answer(TURN9));
    const running = { ...ended(), endedTs: null, end: null, finish: null };
    render(<Live first={running} />);
    await settle();
    expect(h.admin.mock.calls.filter(([u]) => String(u).endsWith('/findings'))).toEqual([]);
    expect(screen.queryByTestId('findings-strip')).toBeNull();
    fireEvent.click(screen.getByText('end'));
    await settle();
    expect(screen.getByTestId('findings-strip')).toBeTruthy();
  });

  it('shows no strip when the request has no findings or the read fails', async () => {
    h.admin.mockImplementation(answer({ turn: 1, findings: [], names: {} }));
    render(<Screen r={ended()} />);
    await settle();
    expect(screen.queryByTestId('findings-strip')).toBeNull();
    cleanup();
    h.admin.mockImplementation(() => Promise.reject(Object.assign(new Error('HTTP 500'), { status: 500 })));
    render(<Screen r={ended()} />);
    await settle();
    expect(screen.queryByTestId('findings-strip')).toBeNull();
  });

  it('a notice is muted and shows no fix', async () => {
    const c03 = EXTRA[0];
    h.admin.mockImplementation(answer({ turn: 9, findings: [c03], names: {} }));
    render(<Screen r={ended()} />);
    await settle();
    const chip = screen.getByTestId('finding-chip');
    expect(chip.dataset.notice).toBe('true');
    expect(chip.textContent).toContain('notice');
    expect(chip.textContent).not.toContain('Fix:');
  });

  it('never renders the word cobs2, god mode, or (for a key holder) cached', async () => {
    const all = { turn: 9, findings: [...TURN9.findings, ...EXTRA, ...ALL], names: NAMES };
    for (const scope of ['admin', 'key'] as const) {
      const fn = scope === 'admin' ? h.admin : h.key;
      fn.mockImplementation(answer(all));
      const { container } = render(<Screen r={ended()} scope={scope} />);
      await settle();
      const strip = screen.getByTestId('findings-strip');
      const text = container.textContent ?? '';
      const titles = Array.from(strip.querySelectorAll('[title]')).map((e) => e.getAttribute('title') ?? '');
      for (const t of [text, ...titles]) {
        expect(t).not.toMatch(/cobs2/i);
        expect(t).not.toMatch(/god ?mode/i);
      }
      if (scope === 'key') {
        const chips = screen.getAllByTestId('finding-chip').map((c) => c.textContent ?? '');
        for (const t of [...chips, ...titles]) expect(t).not.toMatch(/\bcached\b/);
      }
      cleanup();
    }
  });

  it('phone: chips wrap under the part inside its card', async () => {
    h.admin.mockImplementation(answer(TURN9));
    render(<Screen r={ended()} phone />);
    await settle();
    expect(screen.getByTestId('findings-strip').className).toContain('flex-wrap');
    fireEvent.click(screen.getByRole('button', { name: /Show the 4 parts/ }));
    const cards = screen.getAllByTestId('parts-card');
    expect(within(cards[0]).getAllByTestId('finding-chip').map((c) => c.dataset.rule)).toEqual(['E-01', 'C-06']);
  });
});
