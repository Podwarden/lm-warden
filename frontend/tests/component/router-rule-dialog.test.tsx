import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, within } from '@testing-library/react';
import { SWRConfig } from 'swr';
import { RuleDialog } from '@/components/router/rule-dialog';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import { RULE_A, json, stubFetch, callsTo, baseRoutes } from './router-test-utils';

function renderDialog(props: Partial<React.ComponentProps<typeof RuleDialog>> = {}) {
  const onClose = vi.fn();
  const onSaved = vi.fn();
  render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <RuleDialog open rule={null} onClose={onClose} onSaved={onSaved} {...props} />
    </SWRConfig>,
  );
  return { onClose, onSaved };
}

async function pickModel(name: RegExp) {
  fireEvent.click(await screen.findByRole('button', { name: 'Local model' }, { timeout: 5000 }));
  fireEvent.mouseDown(await screen.findByRole('option', { name }, { timeout: 5000 }));
}

describe('RuleDialog', () => {
  beforeEach(() => {
    setAccessToken('jwt');
    setCsrfToken('csrf');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('lists models from the envelope and POSTs the full body', async () => {
    const fn = stubFetch(
      baseRoutes({ 'POST /api/router/rules': () => json(RULE_A, 201) }),
    );
    const { onSaved } = renderDialog();
    fireEvent.change(screen.getByLabelText('Pattern'), { target: { value: 'claude-haiku*' } });
    fireEvent.click(await screen.findByRole('button', { name: 'Local model' }, { timeout: 5000 }));
    expect(await screen.findByRole('option', { name: 'qwen (loaded)' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'llama (registered)' })).toBeInTheDocument();
    fireEvent.mouseDown(screen.getByRole('option', { name: 'qwen (loaded)' }));
    fireEvent.click(screen.getByRole('button', { name: 'Create rule' }));
    await waitFor(() => expect(callsTo(fn, 'POST /api/router/rules')).toHaveLength(1));
    const init = callsTo(fn, 'POST /api/router/rules')[0][1] as RequestInit;
    expect(JSON.parse(init.body as string)).toEqual({
      pattern: 'claude-haiku*',
      target_model_id: 'm1',
      enabled: true,
      fallback: true,
      strip_thinking: true,
      min_max_tokens: 0,
    });
    await waitFor(() => expect(onSaved).toHaveBeenCalled());
  });

  it('shows the server 422 text for pattern "*"', async () => {
    stubFetch(
      baseRoutes({
        'POST /api/router/rules': () =>
          json({ detail: 'pattern: must not match every model' }, 422),
      }),
    );
    const { onSaved } = renderDialog();
    fireEvent.change(screen.getByLabelText('Pattern'), { target: { value: '*' } });
    await pickModel(/qwen/);
    fireEvent.click(screen.getByRole('button', { name: 'Create rule' }));
    expect(await screen.findByRole('alert', {}, { timeout: 5000 })).toHaveTextContent('pattern: must not match every model');
    expect(onSaved).not.toHaveBeenCalled();
  });

  it('edit PATCHes only changed keys', async () => {
    const fn = stubFetch(
      baseRoutes({ 'PATCH /api/router/rules/ra': () => json({ ...RULE_A, pattern: 'claude-haiku-4*' }) }),
    );
    renderDialog({ rule: RULE_A });
    expect(screen.getByLabelText('Pattern')).toHaveValue('claude-haiku*');
    fireEvent.change(screen.getByLabelText('Pattern'), { target: { value: 'claude-haiku-4*' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save rule' }));
    await waitFor(() => expect(callsTo(fn, 'PATCH /api/router/rules/ra')).toHaveLength(1));
    const init = callsTo(fn, 'PATCH /api/router/rules/ra')[0][1] as RequestInit;
    expect(JSON.parse(init.body as string)).toEqual({ pattern: 'claude-haiku-4*' });
  });

  it('failure is a radio group; Refuse sends fallback:false', async () => {
    const fn = stubFetch(baseRoutes({ 'POST /api/router/rules': () => json(RULE_A, 201) }));
    renderDialog();
    const group = screen.getByRole('radiogroup', { name: 'If the local model fails' });
    expect(within(group).getByRole('radio', { name: 'Fall back to Anthropic' })).toBeChecked();
    fireEvent.click(within(group).getByRole('radio', { name: 'Refuse with 529' }));
    fireEvent.change(screen.getByLabelText('Pattern'), { target: { value: 'claude-haiku*' } });
    await pickModel(/qwen/);
    fireEvent.click(screen.getByRole('button', { name: 'Create rule' }));
    await waitFor(() => expect(callsTo(fn, 'POST /api/router/rules')).toHaveLength(1));
    const init = callsTo(fn, 'POST /api/router/rules')[0][1] as RequestInit;
    expect(JSON.parse(init.body as string)).toMatchObject({ fallback: false });
  });

  it('Model quirks is a collapsed disclosure holding thinking and min max_tokens', () => {
    stubFetch(baseRoutes());
    renderDialog();
    const quirks = screen.getByText('Model quirks').closest('details')!;
    expect(quirks).not.toHaveAttribute('open');
    expect(within(quirks).getByLabelText('Disable thinking')).toBeChecked();
    expect(within(quirks).getByLabelText('Minimum max_tokens')).toHaveValue(0);
  });

  it('Model quirks starts open when the rule has non-default quirks', () => {
    stubFetch(baseRoutes());
    renderDialog({ rule: { ...RULE_A, strip_thinking: false, min_max_tokens: 64 } });
    expect(screen.getByText('Model quirks').closest('details')).toHaveAttribute('open');
    expect(screen.getByLabelText('Minimum max_tokens')).toHaveValue(64);
  });

  it('caps Minimum max_tokens at the server limit and disables Save with a message', async () => {
    stubFetch(baseRoutes());
    renderDialog({ rule: { ...RULE_A, min_max_tokens: 64 } });
    const input = screen.getByLabelText('Minimum max_tokens');
    expect(input).toHaveAttribute('max', '131072');
    fireEvent.change(input, { target: { value: '131073' } });
    expect(screen.getByText('Must be at most 131,072.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /save|create rule/i })).toBeDisabled();
    fireEvent.change(input, { target: { value: '131072' } });
    expect(screen.queryByText('Must be at most 131,072.')).toBeNull();
  });

  it('initialPattern prefills a new rule', () => {
    stubFetch(baseRoutes());
    renderDialog({ initialPattern: 'claude-sonnet*' });
    expect(screen.getByLabelText('Pattern')).toHaveValue('claude-sonnet*');
  });

  it('previews which known Claude ids a pattern matches', () => {
    stubFetch(baseRoutes());
    renderDialog();
    const input = screen.getByLabelText('Pattern');
    fireEvent.change(input, { target: { value: 'claude-haiku*' } });
    const preview = screen.getByTestId('pattern-preview');
    expect(preview).toHaveTextContent('Matches e.g. claude-haiku-4-5, claude-haiku-4-5-20251001');
    expect(input).toHaveAttribute('aria-describedby', expect.stringContaining(preview.id));
    fireEvent.change(input, { target: { value: '*haiku*' } });
    expect(preview).toHaveTextContent('claude-3-5-haiku-latest');
    fireEvent.change(input, { target: { value: 'claude-sonnet-4-5' } });
    expect(preview).toHaveTextContent('Exact match');
    fireEvent.change(input, { target: { value: 'Claude-haiku*' } });
    expect(preview).toHaveTextContent('Matches none of the known Claude ids');
  });

  it('orders loaded models first in the dropdown', async () => {
    stubFetch(
      baseRoutes({
        'GET /api/models': () =>
          json({
            models: [
              { id: 'm3', served_model_name: 'llama', status: 'registered' },
              { id: 'm1', served_model_name: 'qwen', status: 'loaded' },
            ],
          }),
      }),
    );
    renderDialog();
    fireEvent.click(await screen.findByRole('button', { name: 'Local model' }, { timeout: 5000 }));
    await screen.findByRole('option', { name: 'qwen (loaded)' });
    expect(screen.getAllByRole('option').map((o) => o.textContent)).toEqual([
      'qwen (loaded)',
      'llama (registered)',
    ]);
  });

  it('a 422 on pattern moves focus to the pattern field', async () => {
    stubFetch(
      baseRoutes({
        'POST /api/router/rules': () => json({ detail: 'pattern: must not match every model' }, 422),
      }),
    );
    renderDialog();
    fireEvent.change(screen.getByLabelText('Pattern'), { target: { value: '*' } });
    await pickModel(/qwen/);
    fireEvent.click(screen.getByRole('button', { name: 'Create rule' }));
    await screen.findByRole('alert', {}, { timeout: 5000 });
    await waitFor(() => expect(screen.getByLabelText('Pattern')).toHaveFocus());
    expect(screen.getByLabelText('Pattern')).toHaveValue('*');
  });

  it('has no dialog landmarks leaking when closed', () => {
    stubFetch(baseRoutes());
    render(
      <SWRConfig value={{ provider: () => new Map() }}>
        <RuleDialog open={false} rule={null} onClose={() => {}} onSaved={() => {}} />
      </SWRConfig>,
    );
    expect(within(document.body).queryByRole('dialog')).toBeNull();
  });
});
