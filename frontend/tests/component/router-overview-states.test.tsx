// Router overview redesign T3a (plan §2, §4.2–§4.7): the master switch, the
// state strip, the setup checklist, the traffic rail and the latest feed.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, within } from '@testing-library/react';
import { SWRConfig } from 'swr';
import RouterPage from '@/app/router/page';
import { RouterSetup } from '@/components/router/router-setup';
import { TrafficPanel } from '@/components/router/traffic-panel';
import { LatestRequests } from '@/components/router/latest-requests';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import { setupSteps, FALLBACKS_HREF, type DecisionOut } from '@/lib/router';
import {
  RULE_A,
  SETTINGS,
  STATS_EMPTY,
  STATS_HEALTHY,
  STATS_BREAKER_OPEN,
  DECISIONS_HEALTHY,
  TOKENS_NONE,
  TOKENS_RELAY,
  TOKENS_URL,
  decision,
  json,
  stubFetch,
  callsTo,
  baseRoutes,
  type Handler,
} from './router-test-utils';

vi.mock('next/navigation', () => ({
  usePathname: () => '/router',
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
}));

function renderPage() {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <RouterPage />
    </SWRConfig>,
  );
}

const body = (fn: ReturnType<typeof vi.fn>, key: string, i = 0) =>
  JSON.parse((callsTo(fn, key)[i][1] as RequestInit).body as string);

const ON = { ...SETTINGS, enabled: true };
const ago = (ms: number) => new Date(Date.now() - ms).toISOString();
/** DECISIONS_HEALTHY, re-stamped relative to the real clock. */
const freshDecisions = (msAgo = 4000) => ({
  decisions: DECISIONS_HEALTHY.decisions.map((d, i) => ({ ...d, ts: ago(msAgo + i * 1000) })),
});

/** Healthy: one loaded rule, routing on, traffic, a relay key. */
function healthy(over: Record<string, Handler> = {}): Record<string, Handler> {
  return baseRoutes({
    'GET /api/router/rules': () => json({ rules: [RULE_A] }),
    'GET /api/router/settings': () => json(ON),
    'GET /api/router/stats': () => json(STATS_HEALTHY),
    'GET /api/router/decisions?limit=20': () => json(freshDecisions()),
    [`GET ${TOKENS_URL}`]: () => json(TOKENS_RELAY),
    ...over,
  });
}

/** First run: off, no rules, no relay key, no traffic. */
function setup(over: Record<string, Handler> = {}): Record<string, Handler> {
  return baseRoutes({ 'GET /api/router/rules': () => json({ rules: [] }), ...over });
}

const strip = () => screen.getByTestId('router-strip');

beforeEach(() => {
  setAccessToken('jwt');
  setCsrfToken('csrf');
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('master switch (plan §4.2)', () => {
  it('off → on PATCHes {enabled:true} at once, with no Save click', async () => {
    let enabled = false;
    const fn = stubFetch(
      baseRoutes({
        'GET /api/router/rules': () => json({ rules: [RULE_A] }),
        'GET /api/router/settings': () => json({ ...SETTINGS, enabled }),
        'PATCH /api/router/settings': (init) => {
          enabled = JSON.parse(init!.body as string).enabled;
          return json({ ...SETTINGS, enabled });
        },
      }),
    );
    renderPage();
    const sw = await screen.findByRole('switch', { name: 'Routing' });
    expect(sw).not.toBeChecked();
    expect(sw).toHaveAttribute('data-testid', 'router-enabled');
    expect(screen.getByText('Routing is off')).toBeInTheDocument();
    fireEvent.click(sw);
    await waitFor(() => expect(callsTo(fn, 'PATCH /api/router/settings')).toHaveLength(1));
    expect(body(fn, 'PATCH /api/router/settings')).toEqual({ enabled: true });
    await waitFor(() => expect(screen.getByRole('switch', { name: 'Routing' })).toBeChecked());
    expect(screen.getByText('Routing is on')).toBeInTheDocument();
    // No Save involved: the settings section (and its Save) is still collapsed.
    expect(screen.queryByRole('button', { name: 'Save' })).not.toBeInTheDocument();
  });

  it('on → off with a decision under 10 min old asks first; Cancel sends nothing', async () => {
    const fn = stubFetch(
      healthy({ 'PATCH /api/router/settings': () => json({ ...ON, enabled: false }) }),
    );
    renderPage();
    const sw = await screen.findByRole('switch', { name: 'Routing' });
    await waitFor(() => expect(sw).toBeChecked());
    await screen.findByText(/Routing normally/);
    sw.focus();
    fireEvent.click(sw);
    const dlg = await screen.findByRole('dialog', { name: 'Turn routing off?' });
    expect(dlg).toHaveTextContent(/404 for Claude models/);
    fireEvent.click(within(dlg).getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    expect(callsTo(fn, 'PATCH /api/router/settings')).toHaveLength(0);
    expect(sw).toBeChecked();
    await waitFor(() => expect(document.activeElement).toBe(sw));

    fireEvent.click(sw);
    const again = await screen.findByRole('dialog', { name: 'Turn routing off?' });
    fireEvent.click(within(again).getByRole('button', { name: 'Turn off' }));
    await waitFor(() => expect(callsTo(fn, 'PATCH /api/router/settings')).toHaveLength(1));
    expect(body(fn, 'PATCH /api/router/settings')).toEqual({ enabled: false });
  });

  it('on → off with no recent decision turns off immediately', async () => {
    const fn = stubFetch(
      healthy({
        'GET /api/router/decisions?limit=20': () => json({ decisions: [decision({ ts: ago(11 * 60_000) })] }),
        'PATCH /api/router/settings': () => json({ ...ON, enabled: false }),
      }),
    );
    renderPage();
    const sw = await screen.findByRole('switch', { name: 'Routing' });
    await waitFor(() => expect(sw).toBeChecked());
    await screen.findByText(/Routing normally/);
    fireEvent.click(sw);
    await waitFor(() => expect(callsTo(fn, 'PATCH /api/router/settings')).toHaveLength(1));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('a failed PATCH reverts the switch and raises an alert with the server detail', async () => {
    stubFetch(
      baseRoutes({
        'GET /api/router/rules': () => json({ rules: [RULE_A] }),
        'PATCH /api/router/settings': () => json({ detail: 'database is locked' }, 500),
      }),
    );
    renderPage();
    const sw = await screen.findByRole('switch', { name: 'Routing' });
    fireEvent.click(sw);
    const alert = await within(strip()).findByRole('alert');
    expect(alert).toHaveTextContent('database is locked');
    expect(screen.getByRole('switch', { name: 'Routing' })).not.toBeChecked();
    expect(screen.getByRole('switch', { name: 'Routing' })).not.toBeDisabled();
  });
});

describe('state strip (plan §4.1)', () => {
  it('setup: a neutral status, the checklist, and no Activity link', async () => {
    stubFetch(setup());
    renderPage();
    await waitFor(() => expect(within(strip()).getByRole('status')).toHaveTextContent('Not set up yet.'));
    expect(within(strip()).queryByRole('alert')).not.toBeInTheDocument();
    expect(await screen.findByTestId('router-setup')).toHaveTextContent('0 of 5 done');
    expect(screen.getByRole('heading', { name: 'How routing works' })).toBeInTheDocument();
  });

  it('ok: "Routing normally" as a status; the checklist is gone; traffic and latest show', async () => {
    stubFetch(healthy());
    renderPage();
    await waitFor(() => expect(within(strip()).getByRole('status')).toHaveTextContent(/Routing normally\./));
    expect(strip()).toHaveTextContent(/83 % answered locally since/);
    expect(screen.queryByTestId('router-setup')).not.toBeInTheDocument();
    expect(screen.getByTestId('router-traffic')).toBeInTheDocument();
    expect(screen.getByTestId('router-latest')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Activity/ })).toHaveAttribute('href', '/router/stats');
  });

  it('waiting: routing on with no request yet', async () => {
    stubFetch(
      healthy({
        'GET /api/router/stats': () => json(STATS_EMPTY),
        'GET /api/router/decisions?limit=20': () => json({ decisions: [] }),
      }),
    );
    renderPage();
    await waitFor(() =>
      expect(within(strip()).getByRole('status')).toHaveTextContent('Routing is on — waiting for the first request.'),
    );
    expect(within(strip()).getByRole('link', { name: 'Connect Claude Code' })).toHaveAttribute('href', '#connect');
  });

  it('amber: no relay key — Create a relay key opens the dialog with relay on', async () => {
    stubFetch(healthy({ [`GET ${TOKENS_URL}`]: () => json(TOKENS_NONE) }));
    renderPage();
    const alert = await within(strip()).findByRole('alert');
    await waitFor(() => expect(alert).toHaveTextContent('No key may relay to Anthropic.'));
    expect(alert).toHaveTextContent('relay_not_allowed');
    fireEvent.click(within(alert).getByRole('button', { name: 'Create a relay key' }));
    const dlg = await screen.findByRole('dialog', { name: 'Create API token' });
    expect(within(dlg).getByLabelText('Name')).toHaveValue('claude-code');
    expect(within(dlg).getByTestId('token-anthropic-relay')).toBeChecked();
  });

  it('danger: an open breaker on a fall-back rule, with its three actions', async () => {
    const fn = stubFetch(
      healthy({
        'GET /api/router/stats': () => json(STATS_BREAKER_OPEN),
        'PATCH /api/router/rules/ra': () => json({ ...RULE_A, enabled: false }),
      }),
    );
    renderPage();
    const alert = await within(strip()).findByRole('alert');
    await waitFor(() => expect(alert).toHaveTextContent('qwen is failing — its breaker is open.'));
    expect(alert).toHaveTextContent('are going to Anthropic instead (the rule falls back)');
    expect(within(alert).getByRole('link', { name: 'Open qwen' })).toHaveAttribute('href', '/models/m1');
    expect(within(alert).getByRole('link', { name: 'See recent fallbacks' })).toHaveAttribute('href', FALLBACKS_HREF);
    fireEvent.click(within(alert).getByRole('button', { name: 'Pause this rule' }));
    await waitFor(() => expect(callsTo(fn, 'PATCH /api/router/rules/ra')).toHaveLength(1));
    expect(body(fn, 'PATCH /api/router/rules/ra')).toEqual({ enabled: false });
  });

  it('danger: a refuse rule says 529', async () => {
    stubFetch(
      healthy({
        'GET /api/router/rules': () => json({ rules: [{ ...RULE_A, fallback: false }] }),
        'GET /api/router/stats': () => json(STATS_BREAKER_OPEN),
      }),
    );
    renderPage();
    const alert = await within(strip()).findByRole('alert');
    await waitFor(() => expect(alert).toHaveTextContent('are refused with 529 (Claude Code retries).'));
  });

  it('several issues: the headline plus an "and N more" disclosure', async () => {
    stubFetch(
      healthy({
        'GET /api/router/stats': () => json(STATS_BREAKER_OPEN),
        [`GET ${TOKENS_URL}`]: () => json(TOKENS_NONE),
      }),
    );
    renderPage();
    const more = await within(strip()).findByRole('button', { name: /and \d+ more/ });
    expect(more).toHaveAttribute('aria-expanded', 'false');
    expect(strip()).not.toHaveTextContent('No key may relay to Anthropic.');
    fireEvent.click(more);
    expect(more).toHaveAttribute('aria-expanded', 'true');
    expect(strip()).toHaveTextContent('No key may relay to Anthropic.');
  });

  it('a stats fetch error is its own danger strip with Retry, never "no traffic"', async () => {
    stubFetch(healthy({ 'GET /api/router/stats': () => json({ detail: 'nope' }, 500) }));
    renderPage();
    const alert = await within(strip()).findByRole('alert');
    await waitFor(() => expect(alert).toHaveTextContent("Can't load router status."));
    expect(within(alert).getByRole('button', { name: 'Retry' })).toBeInTheDocument();
  });
});

describe('setup checklist (plan §4.3)', () => {
  it('Create a relay key → the snippet carries the new plaintext, not the placeholder', async () => {
    const fn = stubFetch(
      setup({
        'POST /api/tokens': () =>
          json({ id: 'tn', name: 'claude-code', plaintext: 'vw_secret_123', prefix: 'vw_secr', preview: 'vw_secr…', expires_at: null }, 201),
      }),
    );
    renderPage();
    const steps = await screen.findByTestId('router-setup');
    await waitFor(() => expect(within(steps).getByRole('button', { name: 'Create a relay key' })).toBeEnabled());
    fireEvent.click(within(steps).getByRole('button', { name: 'Create a relay key' }));
    const dlg = await screen.findByRole('dialog', { name: 'Create API token' });
    expect(within(dlg).getByTestId('token-anthropic-relay')).toBeChecked();
    fireEvent.click(within(dlg).getByRole('button', { name: 'Create' }));
    await waitFor(() => expect(callsTo(fn, 'POST /api/tokens')).toHaveLength(1));
    expect(body(fn, 'POST /api/tokens')).toMatchObject({ name: 'claude-code', anthropic_relay: true });
    fireEvent.click(await screen.findByRole('button', { name: 'Done' }));
    const pre = await screen.findByTestId('router-snippet');
    expect(pre.textContent).toContain('vw_secret_123');
    expect(pre.textContent).not.toContain('vw_YOUR_KEY');
    expect(screen.getByTestId('router-connect')).toHaveTextContent(/Shown once/);
  });

  it('focus lands on the checklist heading when the step that opened a dialog completes', async () => {
    let created = false;
    stubFetch(
      setup({
        [`GET ${TOKENS_URL}`]: () => json(created ? TOKENS_RELAY : TOKENS_NONE),
        'POST /api/tokens': () => {
          created = true;
          return json({ id: 't5', name: 'claude-code', plaintext: 'vw_secret_123', prefix: 'vw_secr', preview: 'vw_secr…', expires_at: null }, 201);
        },
      }),
    );
    renderPage();
    const steps = await screen.findByTestId('router-setup');
    await waitFor(() => expect(within(steps).getByRole('button', { name: 'Create a relay key' })).toBeEnabled());
    const opener = within(steps).getByRole('button', { name: 'Create a relay key' });
    opener.focus();
    fireEvent.click(opener);
    const dlg = await screen.findByRole('dialog', { name: 'Create API token' });
    fireEvent.click(within(dlg).getByRole('button', { name: 'Create' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Done' }));
    // The relay step is done, so its button is gone; focus must not fall to <body>.
    await waitFor(() => expect(within(steps).queryByRole('button', { name: 'Create a relay key' })).toBeNull());
    await waitFor(() => expect(document.activeElement).toBe(screen.getByRole('heading', { name: 'Set up in five steps' })));
  });

  it('after a preset rule is created, focus lands on the checklist heading, not <body>', async () => {
    let rules: unknown[] = [];
    stubFetch(
      setup({
        'GET /api/router/rules': () => json({ rules }),
        'POST /api/router/rules': () => {
          rules = [RULE_A];
          return json(RULE_A, 201);
        },
      }),
    );
    renderPage();
    const steps = await screen.findByTestId('router-setup');
    const preset = within(steps).getByRole('button', { name: 'claude-haiku*' });
    preset.focus();
    fireEvent.click(preset);
    const dialog = await screen.findByRole('dialog');
    fireEvent.click(within(dialog).getByRole('button', { name: 'Local model' }));
    fireEvent.mouseDown(await screen.findByRole('option', { name: 'qwen (loaded)' }));
    fireEvent.click(within(dialog).getByRole('button', { name: 'Create rule' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    await waitFor(() => expect(within(steps).queryByRole('button', { name: 'claude-haiku*' })).toBeNull());
    await waitFor(() => expect(document.activeElement).toBe(screen.getByRole('heading', { name: 'Set up in five steps' })));
  });

  it('a preset opens the rule dialog', async () => {
    stubFetch(setup());
    renderPage();
    const steps = await screen.findByTestId('router-setup');
    fireEvent.click(within(steps).getByRole('button', { name: 'claude-sonnet*' }));
    const dialog = await screen.findByRole('dialog');
    // The page owns the add dialog, so the preset prefills the pattern.
    expect(within(dialog).getByLabelText('Pattern')).toHaveValue('claude-sonnet*');
    expect(screen.getAllByRole('dialog')).toHaveLength(1);
  });

  it('step 4 "Turn on" uses the master switch handler', async () => {
    const fn = stubFetch(
      baseRoutes({
        'GET /api/router/rules': () => json({ rules: [RULE_A] }),
        'PATCH /api/router/settings': () => json(ON),
      }),
    );
    renderPage();
    const steps = await screen.findByTestId('router-setup');
    fireEvent.click(within(steps).getByRole('button', { name: 'Turn on' }));
    await waitFor(() => expect(callsTo(fn, 'PATCH /api/router/settings')).toHaveLength(1));
    expect(body(fn, 'PATCH /api/router/settings')).toEqual({ enabled: true });
  });
});

describe('RouterSetup', () => {
  const noop = () => {};
  function view(decisions: DecisionOut[]) {
    const steps = setupSteps({
      settings: { ...SETTINGS },
      rules: [],
      stats: decisions.length ? STATS_HEALTHY : STATS_EMPTY,
      decisions,
      relay: { relayKeys: 0, relayKnown: true, relayLoading: false },
    });
    return (
      <RouterSetup
        steps={steps}
        decisions={decisions}
        onAddRule={noop}
        onCreateRelayKey={noop}
        onTurnOn={noop}
        turning={false}
      />
    );
  }

  it('step 1 is current; nothing is ticked with nothing done', () => {
    render(view([]));
    const items = within(screen.getByTestId('router-setup')).getAllByRole('listitem');
    expect(items).toHaveLength(5);
    expect(items[0]).toHaveAttribute('aria-current', 'step');
    expect(screen.getByText('0 of 5 done')).toBeInTheDocument();
  });

  it('the first-request status flips when decisions arrive', () => {
    const { rerender } = render(view([]));
    expect(screen.getByRole('status')).toHaveTextContent('Waiting for the first request…');
    rerender(view([decision() as DecisionOut]));
    const s = screen.getByRole('status');
    expect(s).toHaveTextContent('First request:');
    expect(s).toHaveTextContent('claude-haiku-4-5 → qwen');
    expect(s).toHaveTextContent('local');
    expect(s).toHaveTextContent('820 ms');
  });

  it('an unverifiable token list makes no claim about relay keys', () => {
    const steps = setupSteps({
      settings: { ...SETTINGS }, rules: [], stats: STATS_EMPTY, decisions: [],
      relay: { relayKeys: 0, relayKnown: false, relayLoading: false },
    });
    render(<RouterSetup steps={steps} decisions={[]} onAddRule={noop} onCreateRelayKey={noop} onTurnOn={noop} turning={false} />);
    const step2 = within(screen.getByTestId('router-setup')).getAllByRole('listitem')[1];
    expect(step2).not.toHaveTextContent('No key has');
    expect(step2).toHaveTextContent(/couldn.t check/i);
  });
});

describe('TrafficPanel (plan §4.7)', () => {
  it('since=null: one empty state, no legend', () => {
    render(<TrafficPanel stats={STATS_EMPTY} alarm={false} />);
    const card = screen.getByTestId('router-traffic');
    expect(card).toHaveTextContent('No requests yet');
    expect(within(card).queryByRole('img')).not.toBeInTheDocument();
    expect(within(card).queryByRole('list')).not.toBeInTheDocument();
  });

  it('legend counts and percentages from totals; reasons when present', () => {
    render(<TrafficPanel stats={STATS_HEALTHY} alarm={false} />);
    const card = screen.getByTestId('router-traffic');
    expect(within(card).getByRole('img')).toHaveAttribute(
      'aria-label',
      'Local 1,580, Anthropic 311, fell back 6, refused or error 3 (of 1,900)',
    );
    const rows = within(card).getAllByRole('listitem');
    expect(rows[0]).toHaveTextContent(/Local\s*1,580\s*83 %/);
    expect(rows[1]).toHaveTextContent(/Anthropic \(pass-through\)\s*311\s*16 %/);
    expect(rows[2]).toHaveTextContent(/Fell back to Anthropic\s*6\s*0\.3 %/);
    expect(rows[3]).toHaveTextContent(/Refused or error\s*3\s*0\.2 %/);
    expect(card).toHaveTextContent('Why it fell back or was refused');
    expect(card).toHaveTextContent('first_byte_timeout');
  });

  it('no reasons block when by_reason is empty; "since reset" when the total is 0', () => {
    render(
      <TrafficPanel
        stats={{ ...STATS_HEALTHY, by_reason: {}, totals: { local: 0, passthrough: 0, fallback: 0, refused: 0, error: 0 } }}
        alarm={false}
      />,
    );
    const card = screen.getByTestId('router-traffic');
    expect(card).not.toHaveTextContent('Why it fell back or was refused');
    expect(card).toHaveTextContent('No requests since reset');
  });
});

describe('LatestRequests (plan §4.7)', () => {
  it('shows the newest 4 with route words, reasons and the Activity link', () => {
    const many = [
      decision({ route: 'fallback', reason: 'breaker_open', model_out: 'claude-haiku-4-5' }),
      decision({ route: 'passthrough', model_in: 'claude-opus-4-1', model_out: null, token_name: 'ci-runner' }),
      decision(),
      decision({ route: 'refused', reason: 'target_missing', model_out: null }),
      decision({ model_in: 'claude-fifth' }),
    ] as DecisionOut[];
    render(<LatestRequests decisions={many} />);
    const card = screen.getByTestId('router-latest');
    const items = within(card).getAllByRole('listitem');
    expect(items).toHaveLength(4);
    expect(items[0]).toHaveTextContent('fell back');
    expect(items[0]).toHaveTextContent('breaker_open');
    expect(items[0]).not.toHaveTextContent('→');
    expect(items[1]).toHaveTextContent('Anthropic');
    expect(items[1]).toHaveTextContent('ci-runner');
    expect(items[2]).toHaveTextContent('claude-haiku-4-5 → qwen');
    expect(items[2]).toHaveTextContent('820 ms');
    expect(card).not.toHaveTextContent('claude-fifth');
    expect(within(card).getByRole('link', { name: /All activity and latency/ })).toHaveAttribute('href', '/router/stats');
  });

  it('empty: waiting for the first request', () => {
    render(<LatestRequests decisions={[]} />);
    expect(screen.getByTestId('router-latest')).toHaveTextContent('Waiting for the first request');
  });
});

describe('review fixes — the checklist never hides traffic evidence (#1)', () => {
  it('configured router switched off still shows Traffic and Latest (no checklist)', async () => {
    stubFetch(healthy({ 'GET /api/router/settings': () => json(SETTINGS) }));
    renderPage();
    await waitFor(() => expect(within(strip()).getByRole('status')).toHaveTextContent('Routing is off.'));
    const traffic = await screen.findByTestId('router-traffic');
    expect(traffic).toHaveTextContent('Why it fell back or was refused');
    expect(traffic).toHaveTextContent('first_byte_timeout');
    expect(screen.getByTestId('router-latest')).toBeInTheDocument();
    expect(screen.queryByTestId('router-setup')).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'How routing works' })).not.toBeInTheDocument();
  });

  it('on with a rule and no request since start (restart/reset): no checklist; Traffic and Latest render', async () => {
    stubFetch(
      healthy({
        'GET /api/router/stats': () => json({ ...STATS_EMPTY, enabled: true }),
        'GET /api/router/decisions?limit=20': () => json({ decisions: [] }),
      }),
    );
    renderPage();
    await waitFor(() =>
      expect(within(strip()).getByRole('status')).toHaveTextContent('Routing is on — waiting for the first request.'),
    );
    expect(screen.queryByTestId('router-setup')).not.toBeInTheDocument();
    expect(screen.getByTestId('router-traffic')).toBeInTheDocument();
    expect(screen.getByTestId('router-latest')).toHaveTextContent('Waiting for the first request');
  });

  it('off with a rule and nothing seen yet: the checklist (onboarding) and the traffic rail both show', async () => {
    stubFetch(baseRoutes({ 'GET /api/router/rules': () => json({ rules: [RULE_A] }) }));
    renderPage();
    expect(await screen.findByTestId('router-setup')).toBeInTheDocument();
    expect(await screen.findByTestId('router-traffic')).toBeInTheDocument();
    expect(screen.getByTestId('router-latest')).toBeInTheDocument();
  });

  it('first run with no rules: the explainer, no traffic rail', async () => {
    stubFetch(setup());
    renderPage();
    expect(await screen.findByRole('heading', { name: 'How routing works' })).toBeInTheDocument();
    expect(screen.queryByTestId('router-traffic')).not.toBeInTheDocument();
  });
});

describe('review fixes — master switch', () => {
  it('off → on never asks, even with a decision under 10 min old', async () => {
    const fn = stubFetch(
      healthy({
        'GET /api/router/settings': () => json(SETTINGS),
        'PATCH /api/router/settings': () => json(ON),
      }),
    );
    renderPage();
    await screen.findByTestId('router-latest');
    await waitFor(() => expect(screen.getByTestId('router-latest')).toHaveTextContent('laptop'));
    fireEvent.click(screen.getByRole('switch', { name: 'Routing' }));
    await waitFor(() => expect(callsTo(fn, 'PATCH /api/router/settings')).toHaveLength(1));
    expect(body(fn, 'PATCH /api/router/settings')).toEqual({ enabled: true });
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('a failed action can be dismissed, and a later success clears it', async () => {
    let fail = true;
    stubFetch(
      baseRoutes({
        'GET /api/router/rules': () => json({ rules: [RULE_A] }),
        'PATCH /api/router/settings': () => (fail ? json({ detail: 'database is locked' }, 500) : json(ON)),
      }),
    );
    renderPage();
    const sw = await screen.findByRole('switch', { name: 'Routing' });
    fireEvent.click(sw);
    const alert = await within(strip()).findByRole('alert');
    expect(alert).toHaveTextContent("That didn't work.");
    fireEvent.click(within(alert).getByRole('button', { name: 'Dismiss' }));
    await waitFor(() => expect(strip()).not.toHaveTextContent("That didn't work."));

    fireEvent.click(screen.getByRole('switch', { name: 'Routing' }));
    await within(strip()).findByRole('alert');
    fail = false;
    fireEvent.click(screen.getByRole('switch', { name: 'Routing' }));
    await waitFor(() => expect(screen.getByRole('switch', { name: 'Routing' })).toBeChecked());
    expect(strip()).not.toHaveTextContent("That didn't work.");
  });

  it('a failed action clears itself after a while', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      stubFetch(
        baseRoutes({
          'GET /api/router/rules': () => json({ rules: [RULE_A] }),
          'PATCH /api/router/settings': () => json({ detail: 'database is locked' }, 500),
        }),
      );
      renderPage();
      fireEvent.click(await screen.findByRole('switch', { name: 'Routing' }));
      await waitFor(() => expect(strip()).toHaveTextContent("That didn't work."));
      await vi.advanceTimersByTimeAsync(31_000);
      await waitFor(() => expect(strip()).not.toHaveTextContent("That didn't work."));
    } finally {
      vi.useRealTimers();
    }
  });
});

describe('review fixes — settings draft survives switch / pass-through (#2)', () => {
  it('toggling the master switch keeps an unsaved Timeouts edit', async () => {
    let enabled = false;
    stubFetch(
      baseRoutes({
        'GET /api/router/rules': () => json({ rules: [RULE_A] }),
        'GET /api/router/settings': () => json({ ...SETTINGS, enabled }),
        'PATCH /api/router/settings': (init) => {
          enabled = JSON.parse(init!.body as string).enabled;
          return json({ ...SETTINGS, enabled });
        },
      }),
    );
    renderPage();
    fireEvent.click(await screen.findByRole('button', { name: /Timeouts, breaker and upstream/ }));
    const field = screen.getByLabelText(/First-byte timeout/);
    fireEvent.change(field, { target: { value: '45' } });
    expect(screen.getByText('1 unsaved change')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('switch', { name: 'Routing' }));
    await waitFor(() => expect(screen.getByRole('switch', { name: 'Routing' })).toBeChecked());
    expect(screen.getByLabelText(/First-byte timeout/)).toHaveValue(45);
    expect(screen.getByText('1 unsaved change')).toBeInTheDocument();
  });

  it('a draft equal to the fresh server value is dropped (no phantom unsaved change)', async () => {
    let s = { ...SETTINGS };
    const fn = stubFetch(
      baseRoutes({
        'GET /api/router/rules': () => json({ rules: [RULE_A] }),
        'GET /api/router/settings': () => json(s),
        'PATCH /api/router/settings': (init) => {
          s = { ...s, ...JSON.parse(init!.body as string) };
          return json(s);
        },
      }),
    );
    renderPage();
    fireEvent.click(await screen.findByRole('button', { name: /Timeouts, breaker and upstream/ }));
    fireEvent.change(screen.getByLabelText(/First-byte timeout/), { target: { value: '45' } });
    // Someone else saves 45 and flips the switch; our refetch brings both.
    s = { ...s, local_header_timeout_s: 45 };
    fireEvent.click(screen.getByRole('switch', { name: 'Routing' }));
    await waitFor(() => expect(callsTo(fn, 'PATCH /api/router/settings')).toHaveLength(1));
    await waitFor(() => expect(screen.getByRole('switch', { name: 'Routing' })).toBeChecked());
    expect(screen.getByText('No unsaved changes')).toBeInTheDocument();
  });

  it('typing a field back to its saved value leaves nothing to save', async () => {
    stubFetch(baseRoutes({ 'GET /api/router/rules': () => json({ rules: [RULE_A] }) }));
    renderPage();
    fireEvent.click(await screen.findByRole('button', { name: /Timeouts, breaker and upstream/ }));
    const field = screen.getByLabelText(/First-byte timeout/);
    fireEvent.change(field, { target: { value: '45' } });
    expect(screen.getByText('1 unsaved change')).toBeInTheDocument();
    fireEvent.change(field, { target: { value: '30' } });
    expect(screen.getByText('No unsaved changes')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled();
  });
});

describe('review fixes — inline key creation', () => {
  it('a key created with relay unticked shows a neutral "New key" pill, not "New relay key"', async () => {
    stubFetch(
      setup({
        'POST /api/tokens': () =>
          json({ id: 'tn', name: 'claude-code', plaintext: 'vw_plain_9', prefix: 'vw_plai', preview: 'vw_plai…', expires_at: null }, 201),
      }),
    );
    renderPage();
    const steps = await screen.findByTestId('router-setup');
    await waitFor(() => expect(within(steps).getByRole('button', { name: 'Create a relay key' })).toBeEnabled());
    fireEvent.click(within(steps).getByRole('button', { name: 'Create a relay key' }));
    const dlg = await screen.findByRole('dialog', { name: 'Create API token' });
    fireEvent.click(within(dlg).getByTestId('token-anthropic-relay'));
    fireEvent.click(within(dlg).getByRole('button', { name: 'Create' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Done' }));
    const connect = await screen.findByTestId('router-connect');
    await waitFor(() => expect(connect).toHaveTextContent('New key'));
    expect(connect).not.toHaveTextContent('New relay key');
    expect(connect).toHaveTextContent(/can.t relay/i);
  });

  it('the Local only tab uses the freshly created key too', async () => {
    stubFetch(
      setup({
        'POST /api/tokens': () =>
          json({ id: 'tn', name: 'claude-code', plaintext: 'vw_secret_123', prefix: 'vw_secr', preview: 'vw_secr…', expires_at: null }, 201),
      }),
    );
    renderPage();
    const steps = await screen.findByTestId('router-setup');
    await waitFor(() => expect(within(steps).getByRole('button', { name: 'Create a relay key' })).toBeEnabled());
    fireEvent.click(within(steps).getByRole('button', { name: 'Create a relay key' }));
    const dlg = await screen.findByRole('dialog', { name: 'Create API token' });
    fireEvent.click(within(dlg).getByRole('button', { name: 'Create' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Done' }));
    fireEvent.click(await screen.findByRole('tab', { name: /Local only/ }));
    const pre = screen.getByTestId('router-snippet');
    expect(pre.textContent).toContain('ANTHROPIC_AUTH_TOKEN=vw_secret_123');
    expect(pre.textContent).not.toContain('vw_YOUR_KEY');
  });
});
