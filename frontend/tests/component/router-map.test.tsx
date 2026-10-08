import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, within } from '@testing-library/react';
import { SWRConfig } from 'swr';
import { RoutingMap, type RoutingMapProps } from '@/components/router/routing-map';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import type { RouterSettingsOut, RouterStatsOut, RuleOut } from '@/lib/router';
import {
  RULE_A,
  RULE_B,
  SETTINGS,
  STATS_HEALTHY,
  STATS_BREAKER_OPEN,
  NOW,
  json,
  stubFetch,
  callsTo,
  baseRoutes,
} from './router-test-utils';

// Moved here from router-page.test.tsx (plan T3b): order, enable, delete and
// the empty-state copy now belong to the routing map.

const RULE_C: RuleOut = {
  ...RULE_A,
  id: 'rc',
  position: 2,
  pattern: 'claude-opus-4*',
  target_model_id: 'm3',
  target_served_name: 'llama',
  target_status: 'registered',
  enabled: false,
};

function renderMap(props: Partial<RoutingMapProps> = {}) {
  const onChange = vi.fn();
  const onSettingsChange = vi.fn();
  const utils = render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <RoutingMap
        rules={[RULE_B, RULE_A] as RuleOut[]}
        settings={SETTINGS as RouterSettingsOut}
        stats={undefined}
        onChange={onChange}
        onSettingsChange={onSettingsChange}
        now={NOW}
        {...props}
      />
    </SWRConfig>,
  );
  return { ...utils, onChange, onSettingsChange };
}

const body = (fn: ReturnType<typeof vi.fn>, key: string) =>
  JSON.parse((callsTo(fn, key)[0][1] as RequestInit).body as string);

function ruleItems() {
  const list = screen.getByTestId('router-rules');
  return within(list)
    .getAllByRole('listitem')
    .filter((li) => li.hasAttribute('data-rule-id'));
}

function openMenu(pattern: string) {
  fireEvent.click(screen.getByRole('button', { name: `Actions for ${pattern}` }));
  return screen.getByRole('menu');
}

describe('RoutingMap', () => {
  beforeEach(() => {
    setAccessToken('jwt');
    setCsrfToken('csrf');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('is an ordered list in position order, with the rest row last', () => {
    stubFetch(baseRoutes());
    renderMap();
    const list = screen.getByTestId('router-rules');
    expect(list.tagName).toBe('OL');
    expect(list).toHaveAttribute('aria-label', 'Rules, first match wins');
    const rows = ruleItems();
    expect(rows.map((r) => r.getAttribute('data-rule-id'))).toEqual(['ra', 'rb']);
    expect(rows[0]).toHaveTextContent('claude-haiku*');
    expect(rows[0]).toHaveTextContent('qwen');
    expect(rows[0]).toHaveTextContent('loaded');
    expect(rows[1]).toHaveTextContent('claude-sonnet-4-5');
    const items = within(list).getAllByRole('listitem');
    expect(items[items.length - 1]).toBe(screen.getByTestId('router-rest'));
  });

  it('Move up/down are disabled at the ends and PUT the full new order', async () => {
    const fn = stubFetch(
      baseRoutes({ 'PUT /api/router/rules/order': () => json({ rules: [RULE_B, RULE_A] }) }),
    );
    const { onChange } = renderMap();
    expect(screen.getByRole('button', { name: 'Move claude-haiku* up' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Move claude-sonnet-4-5 down' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Move claude-haiku* down' }));
    await waitFor(() => expect(callsTo(fn, 'PUT /api/router/rules/order')).toHaveLength(1));
    expect(body(fn, 'PUT /api/router/rules/order')).toEqual({ ids: ['rb', 'ra'] });
    const init = callsTo(fn, 'PUT /api/router/rules/order')[0][1] as RequestInit;
    expect(new Headers(init.headers).get('Content-Type')).toBe('application/json');
    await waitFor(() => expect(onChange).toHaveBeenCalled());
  });

  it('after a reorder, focus stays on the moved row (its other arrow at the end)', async () => {
    stubFetch(baseRoutes({ 'PUT /api/router/rules/order': () => json({ rules: [RULE_B, RULE_A] }) }));
    const props = {
      settings: SETTINGS as RouterSettingsOut,
      onChange: vi.fn(),
      onSettingsChange: vi.fn(),
      now: NOW,
    };
    const { rerender } = render(<RoutingMap {...props} rules={[RULE_A, RULE_B] as RuleOut[]} />);
    fireEvent.click(screen.getByRole('button', { name: 'Move claude-haiku* down' }));
    await waitFor(() => expect(props.onChange).toHaveBeenCalled());
    rerender(
      <RoutingMap
        {...props}
        rules={[{ ...RULE_B, position: 0 }, { ...RULE_A, position: 1 }] as RuleOut[]}
      />,
    );
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Move claude-haiku* up' })).toHaveFocus(),
    );
  });

  it('after a delete, focus moves to the next row\'s ⋯ button', async () => {
    stubFetch(baseRoutes({ 'DELETE /api/router/rules/ra': () => new Response(null, { status: 204 }) }));
    const props = {
      settings: SETTINGS as RouterSettingsOut,
      onChange: vi.fn(),
      onSettingsChange: vi.fn(),
      now: NOW,
    };
    const { rerender } = render(<RoutingMap {...props} rules={[RULE_A, RULE_B] as RuleOut[]} />);
    fireEvent.click(screen.getByRole('button', { name: 'Actions for claude-haiku*' }));
    fireEvent.click(screen.getByRole('menuitem', { name: 'Delete rule claude-haiku*' }));
    fireEvent.click(within(await screen.findByRole('dialog')).getByRole('button', { name: 'Delete' }));
    await waitFor(() => expect(props.onChange).toHaveBeenCalled());
    rerender(<RoutingMap {...props} rules={[{ ...RULE_B, position: 0 }] as RuleOut[]} />);
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Actions for claude-sonnet-4-5' })).toHaveFocus(),
    );
  });

  it('the row switch PATCHes {enabled:false}', async () => {
    const fn = stubFetch(
      baseRoutes({ 'PATCH /api/router/rules/ra': () => json({ ...RULE_A, enabled: false }) }),
    );
    renderMap();
    const sw = screen.getByRole('switch', { name: 'Enable rule claude-haiku*' });
    expect(sw).toBeChecked();
    fireEvent.click(sw);
    await waitFor(() => expect(callsTo(fn, 'PATCH /api/router/rules/ra')).toHaveLength(1));
    expect(body(fn, 'PATCH /api/router/rules/ra')).toEqual({ enabled: false });
  });

  it('a reorder/toggle failure shows the server detail in an alert', async () => {
    stubFetch(
      baseRoutes({ 'PATCH /api/router/rules/ra': () => json({ detail: 'rule is locked' }, 409) }),
    );
    renderMap();
    fireEvent.click(screen.getByRole('switch', { name: 'Enable rule claude-haiku*' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('rule is locked');
  });

  it('Delete via the ⋯ menu confirms in a Modal, then DELETEs', async () => {
    const fn = stubFetch(
      baseRoutes({ 'DELETE /api/router/rules/rb': () => new Response(null, { status: 204 }) }),
    );
    const { onChange } = renderMap();
    const menu = openMenu('claude-sonnet-4-5');
    fireEvent.click(within(menu).getByRole('menuitem', { name: 'Delete rule claude-sonnet-4-5' }));
    expect(screen.queryByRole('menu')).toBeNull();
    expect(callsTo(fn, 'DELETE /api/router/rules/rb')).toHaveLength(0);
    const dlg = await screen.findByRole('dialog');
    fireEvent.click(within(dlg).getByRole('button', { name: 'Delete' }));
    await waitFor(() => expect(callsTo(fn, 'DELETE /api/router/rules/rb')).toHaveLength(1));
    await waitFor(() => expect(onChange).toHaveBeenCalled());
  });

  it('Edit from the menu opens the rule dialog prefilled', async () => {
    stubFetch(baseRoutes());
    renderMap();
    const menu = openMenu('claude-haiku*');
    fireEvent.click(within(menu).getByRole('menuitem', { name: 'Edit rule claude-haiku*' }));
    const dlg = await screen.findByRole('dialog', { name: 'Edit rule' });
    expect(within(dlg).getByLabelText('Pattern')).toHaveValue('claude-haiku*');
  });

  it('clicking the pattern opens Edit', async () => {
    stubFetch(baseRoutes());
    renderMap();
    fireEvent.click(within(ruleItems()[1]).getByText('claude-sonnet-4-5'));
    const dlg = await screen.findByRole('dialog', { name: 'Edit rule' });
    expect(within(dlg).getByLabelText('Pattern')).toHaveValue('claude-sonnet-4-5');
  });

  it('the ⋯ menu is keyboard operable: arrows move, Escape closes and returns focus', () => {
    stubFetch(baseRoutes());
    renderMap();
    const trigger = screen.getByRole('button', { name: 'Actions for claude-haiku*' });
    expect(trigger).toHaveAttribute('aria-haspopup', 'menu');
    expect(trigger).toHaveAttribute('aria-expanded', 'false');
    fireEvent.keyDown(trigger, { key: 'ArrowDown' });
    const menu = screen.getByRole('menu');
    expect(trigger).toHaveAttribute('aria-expanded', 'true');
    const items = within(menu).getAllByRole('menuitem');
    expect(items[0]).toHaveFocus();
    fireEvent.keyDown(menu, { key: 'ArrowDown' });
    expect(items[1]).toHaveFocus();
    fireEvent.keyDown(menu, { key: 'ArrowUp' });
    fireEvent.keyDown(menu, { key: 'ArrowUp' });
    expect(items[items.length - 1]).toHaveFocus();
    fireEvent.keyDown(menu, { key: 'Escape' });
    expect(screen.queryByRole('menu')).toBeNull();
    expect(trigger).toHaveFocus();
  });

  it('the menu also offers move and enable (the mobile path)', async () => {
    const fn = stubFetch(
      baseRoutes({ 'PUT /api/router/rules/order': () => json({ rules: [RULE_B, RULE_A] }) }),
    );
    renderMap();
    const menu = openMenu('claude-sonnet-4-5');
    expect(within(menu).getByRole('menuitem', { name: 'Move down' })).toHaveAttribute('aria-disabled', 'true');
    expect(within(menu).getByRole('menuitem', { name: 'Pause rule' })).toBeInTheDocument();
    fireEvent.click(within(menu).getByRole('menuitem', { name: 'Move up' }));
    await waitFor(() => expect(callsTo(fn, 'PUT /api/router/rules/order')).toHaveLength(1));
    expect(body(fn, 'PUT /api/router/rules/order')).toEqual({ ids: ['rb', 'ra'] });
  });

  it('rest row reflects passthrough_unmatched and Change PATCHes it', async () => {
    const fn = stubFetch(
      baseRoutes({
        'PATCH /api/router/settings': () => json({ ...SETTINGS, passthrough_unmatched: false }),
      }),
    );
    const { onSettingsChange } = renderMap();
    const rest = screen.getByTestId('router-rest');
    expect(rest).toHaveTextContent('Every other Claude model and /v1 path');
    expect(rest).toHaveTextContent('Anthropic');
    expect(rest).toHaveTextContent("on the user's own login");
    fireEvent.click(within(rest).getByRole('button', { name: /Change/ }));
    const dlg = await screen.findByRole('dialog');
    expect(within(dlg).getByRole('radio', { name: /Pass through to Anthropic/ })).toBeChecked();
    fireEvent.click(within(dlg).getByRole('radio', { name: /Answer 404/ }));
    fireEvent.click(within(dlg).getByRole('button', { name: 'Apply' }));
    await waitFor(() => expect(callsTo(fn, 'PATCH /api/router/settings')).toHaveLength(1));
    expect(body(fn, 'PATCH /api/router/settings')).toEqual({ passthrough_unmatched: false });
    await waitFor(() => expect(onSettingsChange).toHaveBeenCalled());
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
  });

  it('rest row with pass-through off reads 404 (not relayed)', () => {
    stubFetch(baseRoutes());
    renderMap({ settings: { ...SETTINGS, passthrough_unmatched: false } as RouterSettingsOut });
    expect(screen.getByTestId('router-rest')).toHaveTextContent('404 Not found (not relayed)');
  });

  it('pass-through dialog shows the server detail on failure and keeps the choice', async () => {
    stubFetch(
      baseRoutes({ 'PATCH /api/router/settings': () => json({ detail: 'nope' }, 422) }),
    );
    renderMap();
    fireEvent.click(within(screen.getByTestId('router-rest')).getByRole('button', { name: /Change/ }));
    const dlg = await screen.findByRole('dialog');
    fireEvent.click(within(dlg).getByRole('radio', { name: /Answer 404/ }));
    fireEvent.click(within(dlg).getByRole('button', { name: 'Apply' }));
    expect(await within(dlg).findByRole('alert')).toHaveTextContent('nope');
    expect(within(dlg).getByRole('radio', { name: /Answer 404/ })).toBeChecked();
  });

  it('per-rule counts come from stats; "—" when a rule has none', () => {
    stubFetch(baseRoutes());
    renderMap({
      rules: [RULE_A, RULE_B, RULE_C] as RuleOut[],
      stats: STATS_HEALTHY as RouterStatsOut,
    });
    const [a, b, c] = ruleItems();
    expect(within(a).getByTestId('rule-traffic')).toHaveTextContent('1,284 local · 6 fell back');
    expect(within(b).getByTestId('rule-traffic')).toHaveTextContent('296 local · 2 refused');
    expect(within(c).getByTestId('rule-traffic')).toHaveTextContent('—');
    expect(within(screen.getByTestId('router-rest')).getByTestId('rule-traffic')).toHaveTextContent(
      '311 passed through',
    );
  });

  it('without stats every row shows "—"', () => {
    stubFetch(baseRoutes());
    renderMap();
    for (const r of ruleItems()) expect(within(r).getByTestId('rule-traffic')).toHaveTextContent('—');
  });

  it('the breaker-open row is tinted and says it is falling back now', () => {
    stubFetch(baseRoutes());
    renderMap({ stats: STATS_BREAKER_OPEN as RouterStatsOut });
    const [a, b] = ruleItems();
    expect(a).toHaveAttribute('data-incident', 'true');
    expect(a).toHaveTextContent('breaker open · 41 s');
    expect(a).toHaveTextContent('Falling back to Anthropic now');
    expect(a).not.toHaveTextContent('If it fails');
    expect(b).not.toHaveAttribute('data-incident');
  });

  it('a long breaker window reads in minutes, like the strip ("10 min", not "600 s")', () => {
    stubFetch(baseRoutes());
    const stats = structuredClone(STATS_BREAKER_OPEN) as RouterStatsOut;
    stats.targets = stats.targets.map((t) =>
      t.breaker === 'open' ? { ...t, open_until: new Date(NOW + 600_000).toISOString() } : t,
    );
    renderMap({ stats });
    expect(ruleItems()[0]).toHaveTextContent('breaker open · 10 min');
  });

  it('a breaker-open refuse rule says it is refusing with 529 now', () => {
    stubFetch(baseRoutes());
    renderMap({
      rules: [{ ...RULE_A, fallback: false }] as RuleOut[],
      stats: STATS_BREAKER_OPEN as RouterStatsOut,
    });
    expect(ruleItems()[0]).toHaveTextContent('Refusing with 529 now');
  });

  it('a deleted target gets the danger "model deleted" pill', () => {
    stubFetch(baseRoutes());
    renderMap();
    const pill = within(ruleItems()[1]).getByText('model deleted');
    expect(pill.className).toContain('text-vw-danger-fg');
  });

  it('a paused rule is struck through and says so', () => {
    stubFetch(baseRoutes());
    renderMap({ rules: [RULE_A, RULE_C] as RuleOut[] });
    const c = ruleItems()[1];
    expect(c).toHaveTextContent('rule paused');
    expect(within(c).getByRole('switch', { name: 'Enable rule claude-opus-4*' })).not.toBeChecked();
  });

  it('meta shows the failure path and only non-default options', () => {
    stubFetch(baseRoutes());
    renderMap();
    const [a, b] = ruleItems();
    expect(a).toHaveTextContent('If it fails: Anthropic');
    expect(a).toHaveTextContent('Thinking off');
    expect(a).not.toHaveTextContent('Thinking passed through');
    expect(a).not.toHaveTextContent('max_tokens ≥');
    expect(b).toHaveTextContent('If it fails: refuse (529, Claude Code retries)');
    expect(b).toHaveTextContent('Thinking passed through');
    expect(b).toHaveTextContent('max_tokens ≥ 64');
  });

  it('each row carries an sr-only sentence', () => {
    stubFetch(baseRoutes());
    renderMap({ stats: STATS_HEALTHY as RouterStatsOut });
    const [a, b] = ruleItems();
    expect(within(a).getByTestId('rule-sentence')).toHaveTextContent(
      'Rule 1: claude-haiku* goes to qwen, loaded. If it fails, falls back to Anthropic. 1,284 local, 6 fell back.',
    );
    expect(within(b).getByTestId('rule-sentence')).toHaveTextContent(
      'Rule 2: claude-sonnet-4-5 goes to a deleted model. If it fails, refuses with 529. 296 local, 2 refused.',
    );
    expect(within(a).getByTestId('rule-sentence').className).toContain('sr-only');
  });

  it('empty rules: copy follows passthrough_unmatched, and the rest row stays', () => {
    stubFetch(baseRoutes());
    renderMap({ rules: [] });
    expect(
      screen.getByText('With routing on, every Claude model would go to Anthropic.', { exact: false }),
    ).toBeInTheDocument();
    expect(screen.getByTestId('router-rest')).toBeInTheDocument();
    cleanup();
    stubFetch(baseRoutes());
    renderMap({ rules: [], settings: { ...SETTINGS, passthrough_unmatched: false } as RouterSettingsOut });
    expect(
      screen.getByText('With routing on, every Claude model would get 404.', { exact: false }),
    ).toBeInTheDocument();
  });

  it('Add rule delegates to onAddRule when given', () => {
    stubFetch(baseRoutes());
    const onAddRule = vi.fn();
    renderMap({ onAddRule });
    fireEvent.click(screen.getByRole('button', { name: 'Add rule' }));
    expect(onAddRule).toHaveBeenCalledWith(undefined);
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('Add rule opens its own dialog without onAddRule', async () => {
    stubFetch(baseRoutes());
    renderMap({ rules: [] });
    fireEvent.click(screen.getByRole('button', { name: 'Add a rule' }));
    expect(await screen.findByRole('dialog', { name: 'Add rule' })).toBeInTheDocument();
  });

  it('renders row skeletons while rules load', () => {
    stubFetch(baseRoutes());
    renderMap({ rules: undefined });
    expect(screen.getByTestId('router-rules-loading')).toBeInTheDocument();
  });
});

describe('RoutingMap — review fixes', () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('Move up from the ⋯ menu (phones, arrows hidden) returns focus to that row\'s ⋯ button', async () => {
    stubFetch(baseRoutes({ 'PUT /api/router/rules/order': () => json({ rules: [RULE_B, RULE_A] }) }));
    const props = {
      settings: SETTINGS as RouterSettingsOut,
      onChange: vi.fn(),
      onSettingsChange: vi.fn(),
      now: NOW,
    };
    const { rerender } = render(<RoutingMap {...props} rules={[RULE_A, RULE_B] as RuleOut[]} />);
    fireEvent.click(screen.getByRole('button', { name: 'Actions for claude-sonnet-4-5' }));
    fireEvent.click(within(screen.getByRole('menu')).getByRole('menuitem', { name: 'Move up' }));
    await waitFor(() => expect(props.onChange).toHaveBeenCalled());
    rerender(
      <RoutingMap {...props} rules={[{ ...RULE_B, position: 0 }, { ...RULE_A, position: 1 }] as RuleOut[]} />,
    );
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Actions for claude-sonnet-4-5' })).toHaveFocus(),
    );
  });

  it('a half-open breaker is amber "next request probes", not a failing-now incident', () => {
    stubFetch(baseRoutes());
    const stats = structuredClone(STATS_BREAKER_OPEN) as RouterStatsOut;
    stats.targets = stats.targets.map((t) => ({ ...t, breaker: 'half_open', open_until: null }));
    renderMap({ stats });
    const [a] = ruleItems();
    expect(a).not.toHaveAttribute('data-incident');
    expect(a).toHaveTextContent('next request probes');
    expect(a).not.toHaveTextContent('Falling back to Anthropic now');
  });
});
