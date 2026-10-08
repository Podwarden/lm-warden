// The header instrument cluster with more than one model loaded.
//
// Operator report: two models were serving — llama-3.1-8b on vLLM/GPU 1 and
// qwen3.8-27b on llama.cpp/GPU 0 — and the chip named qwen3.8-27b alone. The
// SQL behind it ended in LIMIT 1, so no frontend change could have shown both;
// app/header/routes_api.py now reports every model and this file pins the
// render half.
//
// Designed for N, not for two: the operator may load more, so the cluster caps
// what it draws inline and folds the rest into a counter, while the tooltip and
// the accessible name still enumerate every model.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import type {
  HeaderMetricsFrame,
  HeaderMetricsState,
} from '@/lib/header-metrics-stream';
import {
  activeModelsOf,
  worstModelStatus,
  type HeaderActiveModel,
} from '@/lib/header-models';

let mockState: HeaderMetricsState = {
  status: 'connecting',
  frame: null,
  errorCode: null,
};

vi.mock('@/lib/header-metrics-stream', () => ({
  useHeaderMetrics: () => mockState,
}));

// eslint-disable-next-line @typescript-eslint/no-require-imports
import { HeaderMetrics } from '@/components/header-metrics';

function model(
  name: string,
  status: HeaderActiveModel['status'] = 'loaded',
): HeaderActiveModel {
  return { id: `id-${name}`, served_model_name: name, status };
}

function frame(overrides: Partial<HeaderMetricsFrame> = {}): HeaderMetricsFrame {
  return {
    ts: '2026-09-02T19:01:02.345Z',
    gpus: [
      {
        index: 0,
        name: 'Quadro RTX 5000',
        memory_used_mib: 14000,
        memory_total_mib: 16376,
        utilization_pct: 90,
      },
      {
        index: 1,
        name: 'NVIDIA RTX A4000',
        memory_used_mib: 7000,
        memory_total_mib: 16376,
        utilization_pct: 10,
      },
    ],
    vram_used_mib: 21000,
    vram_total_mib: 32752,
    vram_pct: 64,
    gpu_util_pct: 90,
    active_models: [model('llama-3.1-8b'), model('qwen3.8-27b')],
    active_model: 'llama-3.1-8b',
    active_model_id: 'id-llama-3.1-8b',
    active_model_status: 'loaded',
    probe_error: null,
    ...overrides,
  };
}

function renderConnected(f: HeaderMetricsFrame) {
  mockState = { status: 'connected', frame: f, errorCode: null };
  return render(<HeaderMetrics />);
}

describe('activeModelsOf', () => {
  it('returns every model the frame carries', () => {
    expect(activeModelsOf(frame())).toHaveLength(2);
  });

  it('is empty when the box is idle', () => {
    expect(
      activeModelsOf(
        frame({ active_models: [], active_model: null, active_model_id: null }),
      ),
    ).toEqual([]);
  });

  it('reads a pre-multi-model frame as exactly one model', () => {
    // A ui pod newer than its api pod. The singular fields are the whole
    // contract there, and dropping to "idle" would be a lie about a serving
    // box during a rollout.
    const legacy = frame();
    delete (legacy as Partial<HeaderMetricsFrame>).active_models;
    expect(activeModelsOf(legacy)).toEqual([
      { id: 'id-llama-3.1-8b', served_model_name: 'llama-3.1-8b', status: 'loaded' },
    ]);
  });

  it('assumes loaded when even the status key predates the frame', () => {
    const ancient = frame();
    delete (ancient as Partial<HeaderMetricsFrame>).active_models;
    delete (ancient as Partial<HeaderMetricsFrame>).active_model_status;
    expect(activeModelsOf(ancient)[0].status).toBe('loaded');
  });
});

describe('worstModelStatus', () => {
  it('is null with no models', () => {
    expect(worstModelStatus([])).toBeNull();
  });

  it('lets one failed engine outrank three healthy ones', () => {
    // The cluster has ONE accent colour for N models, so it summarises. A
    // summary that reports the best of its inputs hides the only one worth
    // acting on.
    expect(
      worstModelStatus([
        model('a'),
        model('b'),
        model('c'),
        model('d', 'failed'),
      ]),
    ).toBe('failed');
  });

  it('ranks failed above loading above loaded', () => {
    expect(worstModelStatus([model('a'), model('b', 'loading')])).toBe('loading');
    expect(worstModelStatus([model('a', 'loading'), model('b', 'failed')])).toBe(
      'failed',
    );
  });
});

describe('HeaderMetrics — N loaded models', () => {
  beforeEach(() => {
    mockState = { status: 'connecting', frame: null, errorCode: null };
  });
  afterEach(() => cleanup());

  it('names both models when two are loaded', () => {
    renderConnected(frame());
    const chip = screen.getByTestId('header-metrics-models');
    expect(chip).toHaveTextContent('llama-3.1-8b');
    expect(chip).toHaveTextContent('qwen3.8-27b');
  });

  it('renders one chip per model, not one chip for the fleet', () => {
    renderConnected(frame());
    expect(screen.getAllByTestId('header-metrics-model-chip')).toHaveLength(2);
  });

  it('caps the inline list and counts the rest, so N stays legible', () => {
    // Six models must not push the brand block off a laptop viewport. The
    // widget's width has to be bounded at any N.
    renderConnected(
      frame({
        active_models: [
          model('a'),
          model('b'),
          model('c'),
          model('d'),
          model('e'),
          model('f'),
        ],
      }),
    );
    expect(screen.getAllByTestId('header-metrics-model-chip')).toHaveLength(2);
    expect(screen.getByTestId('header-metrics-model-overflow')).toHaveTextContent(
      '+4',
    );
  });

  it('never loses a folded model — the tooltip still names every one', () => {
    renderConnected(
      frame({ active_models: [model('a'), model('b'), model('c'), model('d')] }),
    );
    const tooltip = screen.getByTestId('header-metrics').getAttribute('title') ?? '';
    for (const name of ['a', 'b', 'c', 'd']) {
      expect(tooltip).toContain(name);
    }
  });

  it('names every model in the accessible label too', () => {
    renderConnected(frame());
    const label = screen.getByTestId('header-metrics').getAttribute('aria-label') ?? '';
    expect(label).toContain('llama-3.1-8b');
    expect(label).toContain('qwen3.8-27b');
  });

  it('goes red when any one of several models has failed', () => {
    renderConnected(
      frame({
        active_models: [model('serving'), model('crashed', 'failed')],
        active_model: 'serving',
        active_model_status: 'loaded',
      }),
    );
    expect(screen.getByTestId('header-metrics')).toHaveAttribute(
      'data-model-status',
      'failed',
    );
  });

  it('still says idle with no models at all', () => {
    renderConnected(
      frame({ active_models: [], active_model: null, active_model_id: null }),
    );
    expect(screen.getByTestId('header-metrics-models')).toHaveTextContent('idle');
    expect(screen.queryByTestId('header-metrics-model-overflow')).toBeNull();
  });

  it('renders a pre-multi-model frame as the single model it describes', () => {
    const legacy = frame();
    delete (legacy as Partial<HeaderMetricsFrame>).active_models;
    renderConnected(legacy);
    expect(screen.getAllByTestId('header-metrics-model-chip')).toHaveLength(1);
    expect(screen.getByTestId('header-metrics-models')).toHaveTextContent(
      'llama-3.1-8b',
    );
  });

  it('labels the GPU readout as the busiest card, not as "the GPU"', () => {
    // 90% and 10% across two cards. "GPU 90%" alone reads as though one
    // number described the box; with several cards it describes one of them,
    // and which one has to be sayable.
    renderConnected(frame());
    expect(screen.getByTestId('header-metrics-gpu-pct')).toHaveTextContent('90');
    const tooltip = screen.getByTestId('header-metrics').getAttribute('title') ?? '';
    expect(tooltip).toMatch(/busiest/i);
    expect(tooltip).toContain('Quadro RTX 5000');
    expect(tooltip).toContain('NVIDIA RTX A4000');
  });
});

// The cluster is also navigation: VRAM and GPU open the Stats page, and each
// model chip opens that model's page (owner request, 2026-10-05).
describe('HeaderMetrics links', () => {
  afterEach(() => cleanup());

  it('links VRAM and GPU to /stats and each chip to its model page', () => {
    mockState = {
      status: 'connected',
      frame: frame({
        active_models: [model('qwen3.8-27b', 'loading'), model('llama-3.1-8b')],
      } as Partial<HeaderMetricsFrame>),
      errorCode: null,
    };
    render(<HeaderMetrics />);
    expect(screen.getByTestId('header-metrics-vram-link').getAttribute('href')).toBe('/stats');
    expect(screen.getByTestId('header-metrics-gpu-link').getAttribute('href')).toBe('/stats');
    const chips = screen.getAllByTestId('header-metrics-model-chip');
    expect(chips.map((c) => c.getAttribute('href'))).toEqual([
      '/models/id-qwen3.8-27b',
      '/models/id-llama-3.1-8b',
    ]);
  });
});
