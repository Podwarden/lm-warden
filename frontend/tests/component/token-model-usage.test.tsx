import { describe, it, expect, vi, beforeAll, afterAll, afterEach } from 'vitest';
process.env.TZ = 'UTC';
import { render, screen, cleanup, fireEvent, within } from '@testing-library/react';
import { ModelUsageCard, sharePct } from '@/components/tokens/detail/model-usage-card';
import { UsageCharts, tokenLines } from '@/components/tokens/detail/usage-charts';
import { MODEL_COLORS, modelColor, modelStyle, SERIES } from '@/components/tokens/detail/styles';
import {
  binIsSplit, fillBins, startsBeforeByModel, variantLabel, variantLabels,
} from '@/lib/token-series';
import { modelUsage, NOW, seriesFixture, splitSeriesFixture, variantUsage } from './token-fixtures';

beforeAll(() => {
  Object.defineProperty(HTMLElement.prototype, 'clientWidth', { configurable: true, get: () => 600 });
  if (!('PointerEvent' in window)) (window as unknown as { PointerEvent: typeof MouseEvent }).PointerEvent = MouseEvent;
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} });
});
afterAll(() => {
  delete (HTMLElement.prototype as unknown as { clientWidth?: number }).clientWidth;
  vi.unstubAllGlobals();
});
afterEach(() => cleanup());

const RECT = { left: 52, top: 8, width: 540, height: 178, right: 592, bottom: 186, x: 52, y: 8, toJSON: () => ({}) } as DOMRect;

describe('per-model helpers', () => {
  it('variantLabel: engine, quantization, dtype, revision, context; defaults left out', () => {
    expect(variantLabel(variantUsage())).toBe('vLLM 0.11.0 · awq · rev a1b2c3d · 32k ctx');
    expect(variantLabel(variantUsage({ engine_vllm_version: null, dtype: 'bfloat16', hf_revision: 'main', max_model_len: 40960 })))
      .toBe('vLLM v0.11.0 · awq · bfloat16 · 40k ctx');
    expect(variantLabel(variantUsage({ backend: 'llamacpp', engine_vllm_version: null, engine_image_tag: null, quantization: null, hf_revision: null, max_model_len: null })))
      .toBe('llama.cpp');
    expect(variantLabel(variantUsage({ backend: null, engine_vllm_version: null, engine_image_tag: null, quantization: null, hf_revision: null, max_model_len: null })))
      .toBe('default settings');
  });

  it('variantLabel shows what the launch actually ran: baked engine version, image tag, resolved commit', () => {
    const unpinned = { engine_channel: null, engine_vllm_version: null, quantization: null, max_model_len: null };
    // In-container: the engine version baked into the warden image.
    expect(variantLabel(variantUsage({ ...unpinned, engine_image_tag: null, engine_version: '0.10.1', hf_revision: 'main', hf_commit: null })))
      .toBe('vLLM 0.10.1');
    // Docker driver, no pin: the tag of the default image that ran.
    expect(variantLabel(variantUsage({ ...unpinned, engine_image_tag: 'v0.10.2', hf_revision: 'main', hf_commit: 'main' })))
      .toBe('vLLM v0.10.2');
    // `main` resolved to a commit: the short commit is shown.
    expect(variantLabel(variantUsage({ ...unpinned, engine_image_tag: 'v0.10.2', hf_revision: 'main', hf_commit: 'f00dfacecafebeef0123456789abcdef01234567' })))
      .toBe('vLLM v0.10.2 · rev f00dfac');
    expect(variantLabel(variantUsage({ backend: 'llamacpp', engine_vllm_version: null, engine_version: 'b10731', engine_image_tag: null, quantization: null, hf_revision: 'main', hf_commit: null, max_model_len: null })))
      .toBe('llama.cpp b10731');
  });

  it('variantLabels disambiguates variants that summarise alike', () => {
    const a = variantUsage({ variant_id: '1111111aaaa' });
    const b = variantUsage({ variant_id: '2222222bbbb' });
    const c = variantUsage({ variant_id: '3333333cccc', quantization: 'fp8' });
    expect(variantLabels([a, b, c])).toEqual([
      'vLLM 0.11.0 · awq · rev a1b2c3d · 32k ctx · #1111111',
      'vLLM 0.11.0 · awq · rev a1b2c3d · 32k ctx · #2222222',
      'vLLM 0.11.0 · fp8 · rev a1b2c3d · 32k ctx',
    ]);
  });

  it('a bin is split only from the first per-model row on', () => {
    expect(binIsSplit(100, null)).toBe(false);
    expect(binIsSplit(100, 100 * 60)).toBe(true);
    expect(binIsSplit(99, 100 * 60 - 1)).toBe(false); // straddles: would under-count
    expect(startsBeforeByModel(splitSeriesFixture())).toBe(true);
    const s = splitSeriesFixture();
    expect(startsBeforeByModel({ ...s, by_model_since: s.from_minute * 60 })).toBe(false);
    expect(startsBeforeByModel(seriesFixture())).toBe(false);
  });

  it('sharePct', () => {
    expect([0, 0.004, 0.37, 0.996, 1].map(sharePct)).toEqual(['0%', '<1%', '37%', '100%', '100%']);
  });

  it('six model colours from theme tokens; past six the colours repeat with a different line style', () => {
    expect(MODEL_COLORS).toHaveLength(6);
    expect(new Set(MODEL_COLORS).size).toBe(6);
    expect(modelColor(0)).toBe('rgb(var(--vw-model-1))');
    expect(modelColor(6)).toBe(modelColor(0));
    const looks = Array.from({ length: 18 }, (_, i) => { const st = modelStyle(i); return `${st.color}|${st.dash}`; });
    expect(new Set(looks).size).toBe(18);
    expect([modelStyle(0).mark, modelStyle(6).mark, modelStyle(12).mark]).toEqual(['solid', 'dashed', 'dotted']);
  });
});

describe('ModelUsageCard', () => {
  it('lists models busiest first with compact numbers, full values in titles, and share', () => {
    render(<ModelUsageCard series={splitSeriesFixture()} nowSec={NOW} />);
    const card = screen.getByRole('region', { name: 'Usage by model' });
    const headers = within(card).getAllByRole('columnheader').map((h) => h.textContent);
    expect(headers).toEqual(['Model', 'Requests', 'Prefill', 'Generation', 'Total', 'Share']);
    const rows = within(card).getAllByRole('row').slice(1);
    expect(rows).toHaveLength(2);
    expect(rows[0]).toHaveTextContent('qwen3-8b');
    expect(rows[1]).toHaveTextContent('llama-3.1-8b');
    const cells = within(rows[0]).getAllByRole('cell');
    expect(cells.map((c) => c.textContent).slice(1)).toEqual(['20', '1.2M', '20k', '1.2M', '80%']);
    expect(cells[2]).toHaveAttribute('title', '1,200,000');
    expect(cells[4]).toHaveAttribute('title', '1,220,000');
    // A single variant shows its settings under the name.
    expect(rows[1]).toHaveTextContent('llama.cpp · 4k ctx');
  });

  it('a model with two variants expands into one row per variant', () => {
    render(<ModelUsageCard series={splitSeriesFixture()} nowSec={NOW} />);
    expect(screen.queryAllByTestId('variant-row')).toHaveLength(0);
    const toggle = screen.getByRole('button', { name: /2 variants of qwen3-8b/ });
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    expect(toggle).toHaveTextContent('2 variants');
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    const vrows = screen.getAllByTestId('variant-row');
    expect(vrows).toHaveLength(2);
    expect(vrows[0]).toHaveTextContent('vLLM 0.11.0 · awq · rev a1b2c3d · 32k ctx');
    expect(within(vrows[0]).getAllByRole('cell').map((c) => c.textContent).slice(1)).toEqual(['15', '1.0M', '15k', '1.0M', '66%']);
    expect(vrows[1]).toHaveTextContent('vLLM 0.9.0 · fp8 · 8k ctx');
    expect(within(vrows[1]).getAllByRole('cell')[0].querySelector('[title]')?.getAttribute('title'))
      .toBe('first served Sep 17, 09:00 · variant bbbbbbbbbbbbbbbb');
    fireEvent.click(toggle);
    expect(screen.queryAllByTestId('variant-row')).toHaveLength(0);
  });

  it('says where the breakdown starts when the window starts before it', () => {
    render(<ModelUsageCard series={splitSeriesFixture()} nowSec={NOW} />);
    expect(screen.getByText('Per-model breakdown recorded since Sep 15, 00:00.')).toBeInTheDocument();
  });

  it('no footnote once the whole window is covered', () => {
    const s = splitSeriesFixture();
    render(<ModelUsageCard series={{ ...s, by_model_since: s.from_minute * 60 - 60 }} nowSec={NOW} />);
    expect(screen.queryByText(/^Per-model breakdown recorded since/)).toBeNull();
  });

  it('empty period', () => {
    render(<ModelUsageCard series={seriesFixture()} nowSec={NOW} />);
    expect(screen.getByText('No per-model data in this period yet.')).toBeInTheDocument();
    expect(screen.queryByRole('table')).toBeNull();
  });

  it('empty period that also starts before recording began says both', () => {
    render(<ModelUsageCard series={seriesFixture({ by_model_since: NOW - 3600 })} nowSec={NOW} />);
    expect(screen.getByText('No per-model data in this period yet.')).toBeInTheDocument();
    expect(screen.getByText('Per-model breakdown recorded since Sep 18, 12:21.')).toBeInTheDocument();
  });
});

// Explicit 20s per-test timeout (default is 5s): recharts in jsdom on a
// loaded CI runner has taken ~4.4s per test here.
describe('tokens chart per model', { timeout: 20_000 }, () => {
  it('one model: the single total line, exactly as before', () => {
    const s = seriesFixture({ by_model: [modelUsage()], by_model_since: 0 });
    const t = tokenLines(s, 'prompt_per_min', SERIES.prompt, NOW);
    expect(t.split).toBe(false);
    expect(t.lines).toEqual([{ key: 'prompt_per_min', color: SERIES.prompt }]);
    render(<UsageCharts series={s} nowSec={NOW} />);
    expect(screen.queryByTestId('model-legend')).toBeNull();
    expect(screen.getByTestId('chart-tokens').querySelectorAll('.recharts-line-curve')).toHaveLength(1);
  });

  it('several models: a line each, plus the pre-split total, with a legend and footnote', () => {
    const s = splitSeriesFixture();
    const t = tokenLines(s, 'prompt_per_min', SERIES.prompt, NOW);
    expect(t.lines.map((l) => [l.key, l.color, l.dashed ?? false])).toEqual([
      ['all-models', SERIES.muted, true],
      ['m0', modelColor(0), false],
      ['m1', modelColor(1), false],
    ]);
    const [early, late] = fillBins(s).filter((b) => b.requests > 0);
    const get = (i: number, b: typeof early) => t.lines[i].get!(b);
    expect([get(0, early), get(1, early), get(2, early)]).toEqual([95_000, null, null]);
    expect([get(0, late), get(1, late), get(2, late)]).toEqual([null, 45_000, 15_000]);
    // an idle split bin draws the models at zero, not as a gap
    const idle = fillBins(s).find((b) => b.minute > late.minute - 600 && b.requests === 0)!;
    expect(get(1, idle)).toBe(0);
    expect(t.note).toBe('Not split by model before Sep 15, 00:00.');

    render(<UsageCharts series={s} nowSec={NOW} />);
    const legend = screen.getByTestId('model-legend');
    expect(legend).toHaveTextContent('All models');
    expect(legend).toHaveTextContent('qwen3-8b');
    expect(legend).toHaveTextContent('llama-3.1-8b');
    expect(screen.getByTestId('chart-tokens').querySelectorAll('.recharts-line-curve')).toHaveLength(3);
    expect(screen.getByText('Not split by model before Sep 15, 00:00.')).toBeInTheDocument();
  });

  it('no pre-split line or footnote once the window is fully split', () => {
    const s = splitSeriesFixture();
    const t = tokenLines({ ...s, by_model_since: s.from_minute * 60 }, 'completion_per_min', SERIES.completion, NOW);
    expect(t.lines.map((l) => l.key)).toEqual(['m0', 'm1']);
    expect(t.note).toBeNull();
  });

  it('tooltip: the total, a row per model with traffic, the busiest minute', () => {
    const s = splitSeriesFixture();
    const late = s.bins[1];
    const x = RECT.left + ((late.minute + 15 - s.from_minute) / (s.to_minute - s.from_minute)) * RECT.width;
    render(<UsageCharts series={s} nowSec={NOW} />);
    fireEvent.click(screen.getByRole('button', { name: 'Generation' }));
    const panel = screen.getByTestId('chart-tokens');
    const hit = within(panel).getByTestId('chart-hit');
    vi.spyOn(hit, 'getBoundingClientRect').mockReturnValue(RECT);
    fireEvent.pointerMove(hit, { clientX: x }); // the last bin with data
    const tip = within(panel).getByRole('tooltip');
    const rows = [...tip.querySelectorAll('.flex.justify-between')].map((r) => r.textContent);
    expect(rows).toEqual(['Generation / min900', 'qwen3-8b700', 'llama-3.1-8b200', 'Busiest minute2.0k']);
  });

  it('a seventh model gets a dashed line, legend marker and hollow card swatch', () => {
    const base = splitSeriesFixture();
    const models = Array.from({ length: 7 }, (_, i) => modelUsage({ model_id: `id-${i}`, model: `model-${i}` }));
    const s = { ...base, by_model: models, by_model_since: base.from_minute * 60 };
    const t = tokenLines(s, 'prompt_per_min', SERIES.prompt, NOW);
    expect(t.lines[6]).toMatchObject({ key: 'm6', color: modelColor(0), dash: '5 4' });
    expect(t.lines[0].dash).toBeUndefined();
    expect(t.legend[6]).toMatchObject({ label: 'model-6', mark: 'dashed' });
    render(<ModelUsageCard series={s} nowSec={NOW} />);
    const marks = [...document.querySelectorAll('[data-mark]')].map((e) => e.getAttribute('data-mark'));
    expect(marks).toEqual(['solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'dashed']);
  });

  it('tooltip before the split keeps the old rows', () => {
    render(<UsageCharts series={splitSeriesFixture()} nowSec={NOW} />);
    const panel = screen.getByTestId('chart-tokens');
    const hit = within(panel).getByTestId('chart-hit');
    vi.spyOn(hit, 'getBoundingClientRect').mockReturnValue(RECT);
    fireEvent.pointerMove(hit, { clientX: 53 }); // the first bins
    const tip = within(panel).getByRole('tooltip');
    expect(tip).toHaveTextContent('Prefill / min');
    expect(tip).not.toHaveTextContent('qwen3-8b');
  });
});
