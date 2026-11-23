/**
 * Where isolation shows outside the settings (plan 2, task 14): the session's turns,
 * the router's rules, and the models list.
 */
import { describe, it, expect, afterEach } from 'vitest';
import { cleanup, render, screen, within } from '@testing-library/react';
import { SWRConfig } from 'swr';

import { TurnTable } from '@/components/sessions/turn-table';
import { RoutingMap } from '@/components/router/routing-map';
import { ModelCard, type ModelRow } from '@/components/models/model-card';
import type { RouterSettingsOut, RuleOut } from '@/lib/router';
import { RULE_A, RULE_B, SETTINGS, NOW, stubFetch, baseRoutes } from './router-test-utils';
import { turn } from './session-fixture';

afterEach(() => cleanup());

describe('session turns', () => {
  it('marks isolated turns and notes the first one after a switch', () => {
    const turns = [
      turn('a', 'c1', 12, 100, { isolated: false }),
      turn('b', 'c1', 13, 110, { isolated: false }),
      turn('c', 'c1', 14, 120, { isolated: true }),
      turn('d', 'c1', 15, 130, { isolated: true }),
    ];
    render(<TurnTable turns={turns} title="Turns" selection={{ requestId: null } as never} onSelect={() => {}} />);
    const table = screen.getByTestId('turn-table');
    const row = (id: string) => table.querySelector(`tr[data-req="${id}"]`)!;
    expect(within(row('a') as HTMLElement).queryByText('isolated')).toBeNull();
    expect(within(row('c') as HTMLElement).getByText('isolated')).toBeInTheDocument();
    expect(within(row('d') as HTMLElement).getByText('isolated')).toBeInTheDocument();
    const notes = within(table).getAllByText('This key switched to isolated here, so the next turn computed its prompt again.');
    expect(notes).toHaveLength(1);
  });

  it('rows written before the mark existed carry none', () => {
    render(
      <TurnTable turns={[turn('a', 'c1', 1, 100), turn('b', 'c1', 2, 110)]} title="Turns" selection={{ requestId: null } as never} onSelect={() => {}} />,
    );
    expect(screen.queryByText('isolated')).toBeNull();
    expect(screen.queryByText(/switched to isolated/)).toBeNull();
  });
});

function renderMap(rules: RuleOut[]) {
  stubFetch(baseRoutes());
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <RoutingMap rules={rules} settings={SETTINGS as RouterSettingsOut} stats={undefined} onChange={() => {}} onSettingsChange={() => {}} now={NOW} />
    </SWRConfig>,
  );
}

function ruleItems() {
  return within(screen.getByTestId('router-rules'))
    .getAllByRole('listitem')
    .filter((li) => li.hasAttribute('data-rule-id'));
}

describe('router rules', () => {
  it('a relayed rule shows "not isolated at the provider", with the reason on hover', () => {
    renderMap([{ ...RULE_A, fallback_cache_isolation: 'not_isolated_at_provider' }] as RuleOut[]);
    const mark = within(ruleItems()[0]).getByTestId('rule-not-isolated');
    expect(mark).toHaveTextContent('not isolated at the provider');
    expect(mark.getAttribute('title')).toBe(
      'Isolated keys may use this rule, but the provider may still cache their prompts. Their requests are sent without cache marks.',
    );
  });

  it('a rule to a local model that cannot isolate says isolated keys are refused', () => {
    renderMap([
      { ...RULE_A, cache_isolation: 'unsupported' },
      { ...RULE_B, target_served_name: 'qwen', target_status: 'loaded', cache_isolation: 'isolated' },
    ] as RuleOut[]);
    const [a, b] = ruleItems();
    expect(within(a).getByTestId('rule-isolated-refused')).toHaveTextContent('isolated keys refused');
    expect(within(b).queryByTestId('rule-isolated-refused')).toBeNull();
  });
});

function row(over: Partial<ModelRow> = {}): ModelRow {
  return {
    id: 'm1', served_model_name: 'qwen', hf_repo: 'org/repo', hf_revision: 'main', gpu_indices: [0],
    tensor_parallel_size: 1, status: 'loaded', pulled_bytes: 0, pulled_total: null, last_error: null, ...over,
  };
}

describe('models list', () => {
  it('a model whose engine cannot keep keys apart carries "isolated keys refused"', () => {
    render(<ModelCard model={row({ backend: 'llamacpp', cache_isolation: 'unsupported', kv_store: 'off' })} />);
    expect(screen.getByTestId('model-isolation-refused')).toHaveTextContent('isolated keys refused');
    expect(screen.getByTestId('model-kv-store')).toHaveTextContent('KV cache store not available');
  });

  it('a vLLM model shows its KV cache store and no refusal', () => {
    render(<ModelCard model={row({ backend: 'vllm', cache_isolation: 'isolated', kv_store: 'ram+nvme' })} />);
    expect(screen.queryByTestId('model-isolation-refused')).toBeNull();
    expect(screen.getByTestId('model-kv-store')).toHaveTextContent('KV cache store: RAM + disk');
  });
});
