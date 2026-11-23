// Live counters across the app sit in <LiveNumber> slots (shared convention
// ui.live-numbers; owner request 2026-11-23: a growing counter must not push
// the text after it). jsdom has no layout, so these tests pin that each live
// figure is a LiveNumber (tabular figures, reserved width) with its unit
// outside it; the real-browser check measures the positions.

import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen, within } from '@testing-library/react';

import { LoadProgressCompact } from '@/components/models/load-progress/load-progress-compact';
import { VitalsTiles } from '@/components/models/load-progress/vitals-tiles';
import { CombinedFigure, LastUpdated } from '@/components/stats/live-panels';
import { NowCard, type NowCardProps } from '@/components/stats/redesign/now-card';
import type { LoadProgressOut } from '@/lib/load-progress';

afterEach(() => cleanup());

/** The LiveNumber inside `el` whose text is `text`. */
function live(el: HTMLElement, text: string): HTMLElement {
  const hit = [...el.querySelectorAll<HTMLElement>('[data-live-number]')].find((n) => n.textContent === text);
  if (!hit) throw new Error(`no live number "${text}" in: ${el.innerHTML.slice(0, 400)}`);
  expect(hit).toHaveClass('num-live');
  expect(hit.style.minWidth).toMatch(/ch$/);
  return hit;
}

const now = (o: Partial<NowCardProps> = {}): NowCardProps => ({
  modelId: 'm1',
  modelName: 'qwen',
  state: 'Running',
  running: 3,
  engineWaiting: 1,
  wardenQueue: 0,
  genNow: 1234,
  prefillNow: 5678,
  kvPct: 0.42,
  maxModelLen: 32768,
  windowP50Ttft: 0.4,
  windowP50Decode: 50,
  windowN: 10,
  typicalPrefill: null,
  lastFinishedAt: null,
  spark: [],
  ...o,
});

describe('live numbers outside Prompt Lens', () => {
  it('Stats Now card: rates, running/waiting/queue and KV % are live slots', () => {
    render(<NowCard {...now()} />);
    const card = screen.getByTestId('now-card');
    live(card, '1.2k');
    live(card, '3');
    live(card, '1');
    live(card, '42');
  });

  it('Stats "updated N s ago" counts in a slot', () => {
    render(<LastUpdated ts={new Date().toISOString()} />);
    const n = live(document.body, '0');
    expect(n.parentElement).toHaveTextContent(/updated 0\s*s ago/);
  });

  it('Stats combined figure: the value is a slot, the unit outside it', () => {
    render(<CombinedFigure label="Generation" metric={{ value: 1234, reporting: 1, total: 1 }} format={(n) => String(n)} unit="tok/s" testid="fig" />);
    const n = live(screen.getByTestId('fig'), '1234');
    expect(n).not.toHaveTextContent('tok/s');
  });

  it('model load progress: elapsed is a slot', () => {
    const p = {
      model_id: 'm1', backend: 'vllm', layout: null, elapsed_s: 30, state: 'progressing', stalled_for_s: 0,
      give_up_at: null, reason: null, hint: null, usual_total_s: null, ready_eta: null, current_step: 'WEIGHTS',
      steps: [], vitals: { series: [], last_log_age_s: null, last_log_line: null, log_quiet_normal: false, gpu_mem_total_mib: null, gpu_mem_supported: true },
    } as unknown as LoadProgressOut;
    render(<LoadProgressCompact progress={p} />);
    live(screen.getByTestId('load-progress-compact'), '30s');
  });

  it('model load vitals: CPU cores busy is a slot', () => {
    render(
      <VitalsTiles
        vitals={{ series: [{ t: 1, cpu_cores: 2.5, gpu_mem_mib: null }], last_log_age_s: 3, last_log_line: 'x', log_quiet_normal: false, gpu_mem_total_mib: null, gpu_mem_supported: true } as never}
      />,
    );
    live(within(screen.getByTestId('vital-cpu-headline')).getByText('2.5').parentElement!, '2.5');
  });
});
