/**
 * Sensitive data guard UI (spec phase 3): the platform default (Settings → Sensitive data), a key's
 * own choice (inherit / off / own settings) on the token page, the try-it box and the action log.
 * Values are never shown back: the log and the try-it findings carry masked forms and lengths only.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { SWRConfig } from 'swr';

import { GuardPlatformTab } from '@/components/settings/guard-tab';
import { GuardCard, type GuardCardProps } from '@/components/tokens/detail/guard-card';
import { G } from '@/lib/guard/strings';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

const PACKS = {
  default: ['cloud-api-keys', 'private-keys'],
  packs: [
    { id: 'cloud-api-keys', label: 'Cloud and API keys', default: true, rules: [{ id: 'aws-access-key', label: 'AWS access key', public_prefix: 'AKIA' }] },
    { id: 'private-keys', label: 'Private keys', default: true, rules: [{ id: 'pem-private-key', label: 'PEM private key', public_prefix: null }] },
    { id: 'pii-contact', label: 'E-mail and phone', default: false, rules: [{ id: 'email', label: 'E-mail', public_prefix: null }] },
  ],
};

type Handler = (init?: RequestInit) => Response;

function stub(routes: Record<string, Handler>) {
  const f = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : (input as Request).url;
    const method = (init?.method ?? 'GET').toUpperCase();
    const hit = Object.keys(routes)
      .sort((a, b) => b.length - a.length)
      .find((k) => {
        const [m, p] = k.includes(' ') ? k.split(' ') : ['GET', k];
        return m === method && url.includes(p);
      });
    if (!hit) return new Response('{}', { status: 404 });
    return routes[hit](init);
  });
  vi.stubGlobal('fetch', f);
  return f;
}

const json = (b: unknown, status = 200): Handler => () => new Response(JSON.stringify(b), { status });

function bodyOf(f: ReturnType<typeof stub>, method: string, path: string) {
  const call = f.mock.calls.find((c) => String(c[0]).includes(path) && (c[1]?.method ?? 'GET').toUpperCase() === method);
  return call ? JSON.parse(String(call[1]!.body)) : undefined;
}

function wrap(el: React.ReactElement) {
  return render(<SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>{el}</SWRConfig>);
}

beforeEach(() => {
  setAccessToken('t');
  setCsrfToken('c');
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('platform default', () => {
  it('a pack can have its own action, sent as an override', async () => {
    const f = stub({
      '/api/guard/settings': json({ guard: null }),
      '/api/guard/packs': json(PACKS),
      'PUT /api/guard/settings': (init) => new Response(JSON.stringify({ guard: JSON.parse(String(init!.body)).guard })),
    });
    wrap(<GuardPlatformTab />);
    fireEvent.click(await screen.findByRole('radio', { name: G.mode.sanitize }));
    fireEvent.change(screen.getByLabelText(G.packs.actionLabel('Private keys')), { target: { value: 'block' } });
    expect(screen.getByText(/PEM private key/)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: G.save }));
    await waitFor(() => expect(bodyOf(f, 'PUT', '/api/guard/settings')).toBeDefined());
    expect(bodyOf(f, 'PUT', '/api/guard/settings').guard.overrides).toEqual({ 'private-keys': 'block' });
  });

  it('starts off, and saving sanitize with the chosen packs sends one PUT', async () => {
    const f = stub({
      '/api/guard/settings': json({ guard: null }),
      '/api/guard/packs': json(PACKS),
      'PUT /api/guard/settings': (init) => new Response(JSON.stringify({ guard: JSON.parse(String(init!.body)).guard })),
    });
    wrap(<GuardPlatformTab />);
    const off = (await screen.findByRole('radio', { name: G.mode.off })) as HTMLInputElement;
    expect(off.checked).toBe(true);
    fireEvent.click(screen.getByRole('radio', { name: G.mode.sanitize }));
    fireEvent.click(screen.getByRole('checkbox', { name: /E-mail and phone/ }));
    fireEvent.click(screen.getByRole('button', { name: G.save }));
    await waitFor(() => expect(bodyOf(f, 'PUT', '/api/guard/settings')).toBeDefined());
    const sent = bodyOf(f, 'PUT', '/api/guard/settings');
    expect(sent.guard.mode).toBe('sanitize');
    expect(sent.guard.packs).toEqual(['cloud-api-keys', 'private-keys', 'pii-contact']);
    expect(sent.guard.restore).toBe('all');
  });

  it('turning it off saves null', async () => {
    const f = stub({
      '/api/guard/settings': json({ guard: { mode: 'block', packs: ['cloud-api-keys'], rules: [], overrides: {}, restore: 'all', trailer: 'new_only', fail: 'closed' } }),
      '/api/guard/packs': json(PACKS),
      'PUT /api/guard/settings': json({ guard: null }),
    });
    wrap(<GuardPlatformTab />);
    expect(((await screen.findByRole('radio', { name: G.mode.block })) as HTMLInputElement).checked).toBe(true);
    fireEvent.click(screen.getByRole('radio', { name: G.mode.off }));
    fireEvent.click(screen.getByRole('button', { name: G.save }));
    await waitFor(() => expect(bodyOf(f, 'PUT', '/api/guard/settings')).toEqual({ guard: null }));
  });

  it('a custom pattern rule is added and a rejected one shows the server reason', async () => {
    const f = stub({
      '/api/guard/settings': json({ guard: null }),
      '/api/guard/packs': json(PACKS),
      'PUT /api/guard/settings': json({ detail: 'nested repetition can take exponential time' }, 422),
    });
    wrap(<GuardPlatformTab />);
    fireEvent.click(await screen.findByRole('radio', { name: G.mode.sanitize }));
    fireEvent.click(screen.getByRole('button', { name: G.rules.addPattern }));
    fireEvent.change(screen.getByLabelText(G.rules.idLabel), { target: { value: 'corp-host' } });
    fireEvent.change(screen.getByLabelText(G.rules.patternLabel), { target: { value: '(a+)+$' } });
    fireEvent.click(screen.getByRole('button', { name: G.save }));
    expect(await screen.findByText(/exponential time/)).toBeTruthy();
    expect(bodyOf(f, 'PUT', '/api/guard/settings').guard.rules).toEqual([{ id: 'corp-host', type: 'regex', regex: '(a+)+$' }]);
  });
});

const TOKEN: GuardCardProps['token'] = {
  id: 'tok1',
  guard: null,
  guard_effective: { mode: 'sanitize', packs: ['cloud-api-keys'], rules: [], overrides: {}, restore: 'all', trailer: 'new_only', fail: 'closed', source: { mode: 'platform' } },
};

describe('a key', () => {
  it('shows what it inherits and can be switched off for this key', async () => {
    const f = stub({ '/api/guard/packs': json(PACKS), 'PATCH /api/tokens/tok1': json({}) , '/api/guard/events': json({ events: [], totals: {} }) });
    const onSaved = vi.fn();
    wrap(<GuardCard token={TOKEN} onSaved={onSaved} />);
    expect(((await screen.findByRole('radio', { name: G.key.inherit })) as HTMLInputElement).checked).toBe(true);
    expect(await screen.findByText('Sanitize · Cloud and API keys')).toBeTruthy();
    fireEvent.click(screen.getByRole('radio', { name: G.key.off }));
    fireEvent.click(screen.getByRole('button', { name: G.save }));
    await waitFor(() => expect(bodyOf(f, 'PATCH', '/api/tokens/tok1')).toEqual({ guard: { off: true } }));
    expect(onSaved).toHaveBeenCalled();
  });

  it('own settings send an override', async () => {
    const f = stub({ '/api/guard/packs': json(PACKS), 'PATCH /api/tokens/tok1': json({}), '/api/guard/events': json({ events: [], totals: {} }) });
    wrap(<GuardCard token={TOKEN} onSaved={() => {}} />);
    fireEvent.click(await screen.findByRole('radio', { name: G.key.own }));
    fireEvent.click(screen.getByRole('radio', { name: G.mode.block }));
    fireEvent.click(screen.getByRole('button', { name: G.save }));
    await waitFor(() => expect(bodyOf(f, 'PATCH', '/api/tokens/tok1')).toBeDefined());
    const sent = bodyOf(f, 'PATCH', '/api/tokens/tok1');
    expect(sent.guard.override.mode).toBe('block');
    expect(sent.guard.override.packs).toEqual(['cloud-api-keys']);
  });

  it('try it checks the draft settings and shows masked findings only', async () => {
    const f = stub({
      '/api/guard/packs': json(PACKS),
      '/api/guard/events': json({ events: [], totals: {} }),
      'POST /api/guard/test': json({ ms: 0.4, preview: 'export K=⟦LMW-SECRET-7c41e09b2a⟧', findings: [{ rule_id: 'aws-access-key', label: 'AWS access key', start: 9, end: 29, length: 20, masked: 'AKIA… (20 chars)', action: 'sanitize' }] }),
    });
    wrap(<GuardCard token={TOKEN} onSaved={() => {}} />);
    fireEvent.change(await screen.findByLabelText(G.tryIt.label), { target: { value: 'export K=AKIAIOSFODNN7EXAMPLE' } });
    fireEvent.click(screen.getByRole('button', { name: G.tryIt.run }));
    expect(await screen.findByText(/⟦LMW-SECRET-7c41e09b2a⟧/)).toBeTruthy();
    expect(screen.getByText(/AKIA… \(20 chars\)/)).toBeTruthy();
    const sent = bodyOf(f, 'POST', '/api/guard/test');
    expect(sent.text).toBe('export K=AKIAIOSFODNN7EXAMPLE');
    expect(sent.guard.mode).toBe('sanitize');
  });

  it('the log lists actions for this key with lengths and fingerprints, never values', async () => {
    const f = stub({
      '/api/guard/packs': json(PACKS),
      '/api/guard/events': json({
        events: [{ ts: 1_793_300_000, request_id: 'r', sess_ref: null, action: 'blocked', rule_id: 'aws-access-key', role: 'tool', msg_index: 14, kind: 'tool result', length: 40, ph: '0d93aa51fe', relayed: false }],
        totals: { sanitized: 312, blocked: 3, restored: 41, error: 0, distinct_values: 17 },
      }),
    });
    wrap(<GuardCard token={TOKEN} onSaved={() => {}} />);
    const log = await screen.findByRole('table', { name: G.log.title });
    expect(within(log).getByText('0d93aa51fe')).toBeTruthy();
    expect(within(log).getByText(/tool result, message 14/)).toBeTruthy();
    expect(screen.getByText(/312/)).toBeTruthy();
    const url = String(f.mock.calls.find((c) => String(c[0]).includes('/api/guard/events'))![0]);
    expect(url).toContain('token=tok1');
  });
});
