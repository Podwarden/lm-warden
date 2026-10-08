import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { SWRConfig } from 'swr';
import type { ReactNode } from 'react';
import { RouterSwitch } from '@/components/router/switch';
import { useRouterOverview, ROUTER_KEYS } from '@/components/router/use-router-data';
import { pill, routeTone, strip } from '@/components/router/styles';
import {
  DECISIONS_HEALTHY,
  STATS_HEALTHY,
  TOKENS_RELAY,
  TOKENS_TRUNCATED,
  TOKENS_URL,
  baseRoutes,
  callsTo,
  json,
  stubFetch,
} from './router-test-utils';

function Probe() {
  const d = useRouterOverview();
  return (
    <div>
      <span data-testid="rules">{d.rules?.length ?? 'loading'}</span>
      <span data-testid="enabled">{d.settings ? String(d.settings.enabled) : 'loading'}</span>
      <span data-testid="since">{d.stats ? String(d.stats.since) : 'loading'}</span>
      <span data-testid="decisions">{d.decisions?.length ?? 'loading'}</span>
      <span data-testid="relay">{`${d.relay.relayKeys}/${d.relay.relayKnown}/${d.relay.relayLoading}`}</span>
      <span data-testid="stats-error">{String(d.errors.stats)}</span>
      <button onClick={() => void d.refresh()}>refresh</button>
    </div>
  );
}

afterEach(cleanup);

function wrap(ui: ReactNode) {
  return render(<SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>{ui}</SWRConfig>);
}

describe('useRouterOverview', () => {
  it('uses the overview keys (decisions?limit=20, tokens?limit=500)', () => {
    expect(ROUTER_KEYS).toMatchObject({
      rules: '/api/router/rules',
      settings: '/api/router/settings',
      stats: '/api/router/stats',
      decisions: '/api/router/decisions?limit=20',
      tokens: TOKENS_URL,
    });
  });

  it('loads all five sources and derives relay info', async () => {
    const fetch = stubFetch(
      baseRoutes({
        'GET /api/router/stats': () => json(STATS_HEALTHY),
        'GET /api/router/decisions?limit=20': () => json(DECISIONS_HEALTHY),
        [`GET ${TOKENS_URL}`]: () => json(TOKENS_RELAY),
      }),
    );
    wrap(<Probe />);
    await waitFor(() => expect(screen.getByTestId('relay')).toHaveTextContent('1/true/false'));
    await waitFor(() => expect(screen.getByTestId('decisions')).toHaveTextContent('3'));
    expect(screen.getByTestId('rules')).toHaveTextContent('2');
    expect(screen.getByTestId('enabled')).toHaveTextContent('false');
    expect(screen.getByTestId('since')).toHaveTextContent('2026-10-04T09:12:00Z');
    expect(screen.getByTestId('stats-error')).toHaveTextContent('false');
    expect(callsTo(fetch, `GET ${TOKENS_URL}`).length).toBeGreaterThan(0);
  });

  it('a truncated token list is not a verified answer', async () => {
    stubFetch(baseRoutes({ [`GET ${TOKENS_URL}`]: () => json(TOKENS_TRUNCATED) }));
    wrap(<Probe />);
    await waitFor(() => expect(screen.getByTestId('relay')).toHaveTextContent('0/false/false'));
  });

  it('surfaces a stats error', async () => {
    stubFetch(baseRoutes({ 'GET /api/router/stats': () => json({ detail: 'boom' }, 500) }));
    wrap(<Probe />);
    await waitFor(() => expect(screen.getByTestId('stats-error')).toHaveTextContent('true'));
  });

  it('refresh() re-fetches every source', async () => {
    const fetch = stubFetch(baseRoutes());
    wrap(<Probe />);
    await waitFor(() => expect(screen.getByTestId('rules')).toHaveTextContent('2'));
    const before = callsTo(fetch, 'GET /api/router/rules').length;
    await act(async () => {
      fireEvent.click(screen.getByText('refresh'));
    });
    await waitFor(() => expect(callsTo(fetch, 'GET /api/router/rules').length).toBeGreaterThan(before));
  });
});

describe('RouterSwitch', () => {
  it('is a real checkbox with role=switch and its label', () => {
    const onChange = vi.fn();
    render(<RouterSwitch checked={false} onChange={onChange} label="Routing" testId="router-enabled" />);
    const sw = screen.getByRole('switch', { name: 'Routing' });
    expect(sw).toHaveAttribute('type', 'checkbox');
    expect(sw).toHaveAttribute('aria-checked', 'false');
    expect(sw).toHaveAttribute('data-testid', 'router-enabled');
    fireEvent.click(sw);
    expect(onChange).toHaveBeenCalledWith(true);
  });

  it('reflects checked, honours disabled and aria-describedby', () => {
    const onChange = vi.fn();
    render(
      <RouterSwitch checked onChange={onChange} label="Enable rule claude-haiku*" size="sm" disabled describedBy="d1" />,
    );
    const sw = screen.getByRole('switch', { name: 'Enable rule claude-haiku*' });
    expect(sw).toBeChecked();
    expect(sw).toHaveAttribute('aria-checked', 'true');
    expect(sw).toBeDisabled();
    expect(sw).toHaveAttribute('aria-describedby', 'd1');
    // (jsdom dispatches clicks on disabled inputs, unlike browsers; `disabled` is the contract.)
    expect(onChange).not.toHaveBeenCalled();
  });
});

describe('router styles', () => {
  it('maps routes to the semantic tones with a word', () => {
    expect(routeTone('local')).toMatchObject({ tone: 'ok', label: 'local' });
    expect(routeTone('passthrough')).toMatchObject({ tone: 'info', label: 'Anthropic' });
    expect(routeTone('fallback')).toMatchObject({ tone: 'amber', label: 'fell back' });
    expect(routeTone('refused')).toMatchObject({ tone: 'danger', label: 'refused' });
    expect(routeTone('error')).toMatchObject({ tone: 'danger', label: 'error' });
    expect(routeTone('weird')).toMatchObject({ tone: 'idle', label: 'weird' });
  });

  it('uses tokens only, never slate/emerald', () => {
    const all = [
      ...(['ok', 'amber', 'danger', 'info', 'idle'] as const).map(pill),
      ...(['neutral', 'ok', 'amber', 'danger'] as const).map(strip),
    ].join(' ');
    expect(all).not.toMatch(/slate-|emerald-|red-\d|amber-\d/);
    expect(pill('danger')).toMatch(/vw-danger-fg/);
    expect(strip('ok')).toMatch(/vw-ok-bg/);
  });
});
