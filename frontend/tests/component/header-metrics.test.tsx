// HeaderMetrics widget — pins the contract for the nav-bar instrument
// cluster shipped in S2 of the overhaul epic.
//
//   - Renders three readouts: VRAM %, GPU %, model label.
//   - data-status mirrors the underlying stream status.
//   - Accent color transitions: idle (slate) → loaded (emerald) →
//     reconnecting (amber) → terminal (red).
//   - Hidden on narrow viewports via the `hidden md:inline-flex`
//     responsive classes (we assert on the className, not on computed
//     layout — jsdom doesn't run media queries).
//   - Probe error in the frame degrades to amber without killing the
//     stream.
//
// We mock the singleton hook directly (vi.mock of
// `@/lib/header-metrics-stream`) so the widget renders under each
// status state without spinning a real EventSource. The companion
// singleton test (`header-metrics-stream.test.ts`) exercises the
// EventSource subscription contract end to end.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import type {
  HeaderMetricsFrame,
  HeaderMetricsState,
} from '@/lib/header-metrics-stream';

// Mutable state the mocked hook returns — each test patches this then
// renders. Reset to a sane default in `beforeEach` so leakage between
// tests is impossible.
let mockState: HeaderMetricsState = {
  status: 'connecting',
  frame: null,
  errorCode: null,
};

vi.mock('@/lib/header-metrics-stream', () => ({
  useHeaderMetrics: () => mockState,
}));

// Imported AFTER the mock so the component picks up the mocked hook.
// eslint-disable-next-line @typescript-eslint/no-require-imports
import { HeaderMetrics } from '@/components/header-metrics';

function frame(overrides: Partial<HeaderMetricsFrame> = {}): HeaderMetricsFrame {
  return {
    ts: '2026-05-23T19:01:02.345Z',
    gpus: [
      {
        index: 0,
        name: 'NVIDIA RTX A4000',
        memory_used_mib: 12450,
        memory_total_mib: 16376,
        utilization_pct: 87,
      },
    ],
    vram_used_mib: 12450,
    vram_total_mib: 16376,
    vram_pct: 76,
    gpu_util_pct: 87,
    active_model: null,
    active_model_id: null,
    probe_error: null,
    ...overrides,
  };
}

describe('HeaderMetrics', () => {
  beforeEach(() => {
    mockState = { status: 'connecting', frame: null, errorCode: null };
  });
  afterEach(() => {
    cleanup();
  });

  it('renders VRAM %, GPU % and "idle" placeholder when no model is loaded', () => {
    mockState = { status: 'connected', frame: frame(), errorCode: null };
    render(<HeaderMetrics />);
    const root = screen.getByTestId('header-metrics');
    expect(root).toBeInTheDocument();
    expect(root).toHaveAttribute('data-status', 'connected');
    // Values are right-aligned via padStart(2, ' '); textContent collapses
    // whitespace in some queries, so we match the digits inside the
    // dedicated testid wrappers.
    expect(screen.getByTestId('header-metrics-vram-pct').textContent).toContain(
      '76',
    );
    expect(screen.getByTestId('header-metrics-gpu-pct').textContent).toContain(
      '87',
    );
    expect(screen.getByTestId('header-metrics-models')).toHaveTextContent('idle');
  });

  it('shows the active model name when one is loaded and switches to emerald', () => {
    mockState = {
      status: 'connected',
      frame: frame({ active_model: 'gpt-oss-20b', active_model_id: 'abc' }),
      errorCode: null,
    };
    render(<HeaderMetrics />);
    expect(screen.getByTestId('header-metrics-models')).toHaveTextContent(
      'gpt-oss-20b',
    );
    // Accent + dot both carry the emerald color when a model is loaded.
    const root = screen.getByTestId('header-metrics');
    expect(root.className).toContain('text-emerald-400');
  });

  it('shows "loading" while a model is still coming up', () => {
    // Regression: a 27B load takes minutes, during which the row is
    // 'loading' but the badge used to read "idle" — the opposite of true.
    mockState = {
      status: 'connected',
      frame: frame({
        active_model: 'qwen3.8-27b',
        active_model_id: 'abc',
        active_model_status: 'loading',
      }),
      errorCode: null,
    };
    render(<HeaderMetrics />);
    expect(screen.getByTestId('header-metrics-models')).toHaveTextContent(
      'loading',
    );
    const root = screen.getByTestId('header-metrics');
    expect(root).toHaveAttribute('data-model-status', 'loading');
    expect(root.className).toContain('text-sky-400');
  });

  it('goes red and reads "error" when the model has failed', () => {
    mockState = {
      status: 'connected',
      frame: frame({
        active_model: 'qwen3.8-27b',
        active_model_id: 'abc',
        active_model_status: 'failed',
      }),
      errorCode: null,
    };
    render(<HeaderMetrics />);
    expect(screen.getByTestId('header-metrics-models')).toHaveTextContent(
      'error',
    );
    const root = screen.getByTestId('header-metrics');
    expect(root).toHaveAttribute('data-model-status', 'failed');
    expect(root.className).toContain('text-red-400');
  });

  it('keeps the failed model red even when a probe error is present', () => {
    // A dead engine outranks a degraded nvidia-smi probe: the amber
    // "instruments are unreliable" hint must not mask a hard failure.
    mockState = {
      status: 'connected',
      frame: frame({
        active_model: 'qwen3.8-27b',
        active_model_status: 'failed',
        probe_error: 'nvidia-smi unavailable',
      }),
      errorCode: null,
    };
    render(<HeaderMetrics />);
    const root = screen.getByTestId('header-metrics');
    expect(root.className).toContain('text-red-400');
    expect(root.className).not.toContain('text-amber-400');
  });

  it('treats a name with no status field as loaded (older API build)', () => {
    // UI and API ship as separate images; a UI newer than its API sees no
    // active_model_status at all and must still render the old meaning.
    mockState = {
      status: 'connected',
      frame: frame({ active_model: 'gpt-oss-20b' }),
      errorCode: null,
    };
    render(<HeaderMetrics />);
    expect(screen.getByTestId('header-metrics-models')).toHaveTextContent(
      'gpt-oss-20b',
    );
    const root = screen.getByTestId('header-metrics');
    expect(root).toHaveAttribute('data-model-status', 'loaded');
    expect(root.className).toContain('text-emerald-400');
  });

  it('paints the cluster amber while reconnecting', () => {
    mockState = {
      status: 'reconnecting',
      frame: frame({ active_model: 'gpt-oss-20b' }),
      errorCode: 502,
    };
    render(<HeaderMetrics />);
    const root = screen.getByTestId('header-metrics');
    expect(root).toHaveAttribute('data-status', 'reconnecting');
    expect(root.className).toContain('text-amber-400');
  });

  it('paints the cluster amber when a probe error is reported', () => {
    mockState = {
      status: 'connected',
      frame: frame({ active_model: 'gpt-oss-20b', probe_error: 'nvidia-smi unavailable' }),
      errorCode: null,
    };
    render(<HeaderMetrics />);
    const root = screen.getByTestId('header-metrics');
    // Status is still "connected" — the stream itself is fine — but the
    // probe payload triggers the degraded visual.
    expect(root).toHaveAttribute('data-status', 'connected');
    expect(root.className).toContain('text-amber-400');
  });

  // #255: a revoked GPU grant leaves a model serving while nvidia-smi in the
  // container fails. The badge must say that, not show an idle 0 %.
  const NVML = 'nvidia-smi exit 255: Failed to initialize NVML: Unknown Error';

  it('shows "--" and leads the tooltip with the reason when GPU telemetry is lost', () => {
    mockState = {
      status: 'connected',
      frame: frame({
        gpus: [],
        vram_used_mib: null,
        vram_total_mib: null,
        vram_pct: null,
        gpu_util_pct: null,
        active_models: [{ id: 'm1', served_model_name: 'qwen', status: 'loaded' }],
        active_model: 'qwen',
        active_model_id: 'm1',
        active_model_status: 'loaded',
        probe_error: NVML,
      }),
      errorCode: null,
    };
    render(<HeaderMetrics />);
    const root = screen.getByTestId('header-metrics');
    expect(screen.getByTestId('header-metrics-vram-pct').textContent).toContain('--');
    expect(screen.getByTestId('header-metrics-gpu-pct').textContent).toContain('--');
    const title = root.getAttribute('title') ?? '';
    expect(title.split('\n')[0]).toBe(`GPU telemetry unavailable: ${NVML}`);
    // The model is still listed: it is serving, and that is half the story.
    expect(title).toContain('qwen');
    expect(title).not.toMatch(/VRAM 0|0 \/ 0 GiB/);
    expect(root.getAttribute('aria-label')).toContain('GPU telemetry unavailable');
    expect(root.getAttribute('aria-label')).not.toContain('VRAM');
  });

  it('never shows a number beside a probe error, even from an older API that sent 0', () => {
    mockState = {
      status: 'connected',
      frame: frame({ gpus: [], vram_used_mib: 0, vram_total_mib: 0, vram_pct: 0, gpu_util_pct: 0, probe_error: NVML }),
      errorCode: null,
    };
    render(<HeaderMetrics />);
    expect(screen.getByTestId('header-metrics-vram-pct').textContent).toContain('--');
    expect(screen.getByTestId('header-metrics-gpu-pct').textContent).toContain('--');
  });

  it('collapses to red "offline" on terminal-error', () => {
    mockState = {
      status: 'terminal-error',
      frame: null,
      errorCode: 401,
    };
    render(<HeaderMetrics />);
    const root = screen.getByTestId('header-metrics');
    expect(root).toHaveAttribute('data-status', 'terminal-error');
    expect(root.className).toContain('text-red-400');
    expect(screen.getByTestId('header-metrics-models')).toHaveTextContent(
      'offline',
    );
    // Percentages drop to the "--" placeholder so the operator doesn't
    // see stale numbers after a session ends.
    expect(screen.getByTestId('header-metrics-vram-pct').textContent).toContain(
      '--',
    );
    expect(screen.getByTestId('header-metrics-gpu-pct').textContent).toContain(
      '--',
    );
  });

  it('keeps the responsive `hidden md:inline-flex` guard so it disappears on phones', () => {
    mockState = { status: 'connected', frame: frame(), errorCode: null };
    render(<HeaderMetrics />);
    const root = screen.getByTestId('header-metrics');
    // jsdom won't apply Tailwind, but the class contract is the
    // surface area the integration test pins.
    expect(root.className).toContain('hidden');
    expect(root.className).toContain('md:inline-flex');
  });
});
