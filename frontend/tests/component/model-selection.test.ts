import { describe, it, expect, beforeEach } from 'vitest';
import {
  canDeselect,
  toggleModel,
  reconcileSelection,
  modelsQueryParam,
  MODEL_SELECTION_KEY,
} from '@/lib/model-selection';
import {
  sumReported,
  isPartial,
  combineThroughput,
  liveFramesOf,
  type LiveEngineFrame,
} from '@/lib/live-stats';

// ---------------------------------------------------------------------------
// The selection rule.
//
// At least one model stays selected. An empty selection has two plausible
// readings — "everything" and "nothing" — and every surface downstream (stats
// numbers, live panels, god mode) would have to pick one. So the state can
// never reach it.
//
// DISABLE, do not silently re-select. A checkbox that unticks and re-ticks
// itself leaves the operator unable to tell whether their click registered; a
// disabled one says the rule out loud. This is the same rule the rest of the
// audit's findings turn on: an affordance and its effect gated on the same
// condition.
// ---------------------------------------------------------------------------

describe('canDeselect — at least one model always stays selected', () => {
  it('refuses to unselect the last one', () => {
    expect(canDeselect(['a'], 'a')).toBe(false);
  });

  it('allows unselecting either of two', () => {
    expect(canDeselect(['a', 'b'], 'a')).toBe(true);
    expect(canDeselect(['a', 'b'], 'b')).toBe(true);
  });

  it('is about the LAST one, not about any particular model', () => {
    // A rule keyed on a specific id (say, "never unselect the first") would
    // pass the case above and be wrong the moment the operator unselects in a
    // different order.
    expect(canDeselect(['b'], 'b')).toBe(false);
    expect(canDeselect(['a', 'b', 'c'], 'b')).toBe(true);
  });
});

describe('toggleModel', () => {
  it('unticking the last model is a no-op, not an empty selection', () => {
    expect(toggleModel(['a'], 'a')).toEqual(['a']);
  });

  it('returns the same array identity when the change is refused', () => {
    // So a React setState is a genuine no-op rather than a re-render that
    // could read as a flicker on a page repainting every two seconds.
    const s = ['a'];
    expect(toggleModel(s, 'a')).toBe(s);
  });

  it('unticks a model when others remain', () => {
    expect(toggleModel(['a', 'b'], 'a')).toEqual(['b']);
  });

  it('ticks a model back on, sorted', () => {
    expect(toggleModel(['b'], 'a')).toEqual(['a', 'b']);
  });

  it('supports any combination, not just one or all', () => {
    let s = ['a', 'b', 'c', 'd'];
    s = toggleModel(s, 'b');
    s = toggleModel(s, 'd');
    expect(s).toEqual(['a', 'c']);
  });

  it('can be driven down to exactly one and no further', () => {
    let s = ['a', 'b', 'c'];
    s = toggleModel(s, 'a');
    s = toggleModel(s, 'b');
    expect(s).toEqual(['c']);
    s = toggleModel(s, 'c');
    expect(s).toEqual(['c']);
  });
});

describe('reconcileSelection — the fleet changes under a stored selection', () => {
  beforeEach(() => window.localStorage.clear());

  it('selects everything on a first visit', () => {
    expect(reconcileSelection(null, ['b', 'a'])).toEqual(['a', 'b']);
  });

  it('keeps a remembered subset', () => {
    expect(reconcileSelection(['a'], ['a', 'b'])).toEqual(['a']);
  });

  it('drops a model that is no longer loaded', () => {
    // Otherwise an unloaded model keeps narrowing every number invisibly.
    expect(reconcileSelection(['a', 'gone'], ['a', 'b'])).toEqual(['a']);
  });

  it('falls back to everything when the whole stored fleet is gone', () => {
    // Picking one arbitrarily would show a narrowed number the operator never
    // asked for; "all" is the honest default for a dashboard.
    expect(reconcileSelection(['gone'], ['a', 'b'])).toEqual(['a', 'b']);
  });

  it('is empty when nothing is loaded', () => {
    expect(reconcileSelection(['a'], [])).toEqual([]);
  });

  it('never yields an empty selection while models exist', () => {
    for (const stored of [null, [], ['ghost'], ['a']]) {
      expect(reconcileSelection(stored, ['a', 'b']).length).toBeGreaterThan(0);
    }
  });
});

describe('modelsQueryParam — absent and empty are different questions', () => {
  it('joins a selection', () => {
    expect(modelsQueryParam(['a', 'b'])).toBe('a,b');
  });

  it('is null for an empty selection, so no parameter is sent', () => {
    // The API reads an ABSENT ?models as "the whole deployment" and an EMPTY
    // one as a client bug (400). Sending "" would turn a still-loading page
    // into an error.
    expect(modelsQueryParam([])).toBeNull();
  });
});

it('uses one storage key so every page agrees on the selection', () => {
  expect(MODEL_SELECTION_KEY).toBe('vw.stats.models');
});

// ---------------------------------------------------------------------------
// Combining across a mixed selection.
//
// `null` means "this engine does not report it", never zero. llama.cpp
// publishes no KV gauge, no preemption counter and no latency histograms;
// vLLM publishes all of them. A multi-model selection is a fresh chance to
// reintroduce the `?? 0` bug sub-project C fixed — once per tile.
// ---------------------------------------------------------------------------

describe('sumReported — an absent metric is not zero', () => {
  it('sums the models that report', () => {
    expect(sumReported([2, 3])).toEqual({ value: 5, reporting: 2, total: 2 });
  });

  it('skips an absent value instead of adding zero', () => {
    // The headline case: vLLM reports 12 tok/s, llama.cpp reports nothing.
    // The total is 12 FROM ONE MODEL — not 12 from two, and not 6 on average.
    expect(sumReported([12, null])).toEqual({
      value: 12,
      reporting: 1,
      total: 2,
    });
  });

  it('is null — never 0 — when no selected model reports it', () => {
    // 0 would read as "measured, and idle". The truth is "nobody here
    // measures this", and the tile must render an em dash for it.
    expect(sumReported([null, null])).toEqual({
      value: null,
      reporting: 0,
      total: 2,
    });
  });

  it('treats undefined and NaN as absent too', () => {
    // undefined arrives from a frame missing a whole sub-object; NaN from a
    // JSON round trip that lost a number. Neither is a measurement.
    expect(sumReported([undefined, NaN, 4]).value).toBe(4);
    expect(sumReported([undefined, NaN, 4]).reporting).toBe(1);
  });

  it('distinguishes a reported zero from an absent value', () => {
    // A genuinely idle engine reporting 0 IS a measurement and must count.
    expect(sumReported([0, null])).toEqual({ value: 0, reporting: 1, total: 2 });
  });

  it('is empty-safe', () => {
    expect(sumReported([])).toEqual({ value: null, reporting: 0, total: 0 });
  });
});

describe('isPartial — the tile has to be able to say so', () => {
  it('is true when some but not all models reported', () => {
    expect(isPartial(sumReported([1, null]))).toBe(true);
  });

  it('is false when everyone reported', () => {
    expect(isPartial(sumReported([1, 2]))).toBe(false);
  });

  it('is false when nobody reported — that is absent, not partial', () => {
    expect(isPartial(sumReported([null, null]))).toBe(false);
  });
});

function engineFrame(
  name: string,
  overrides: Partial<LiveEngineFrame> = {},
): LiveEngineFrame {
  return {
    ts: '2026-09-02T19:00:00Z',
    model: name,
    model_id: `id-${name}`,
    backend: 'vllm',
    max_model_len: 8192,
    engine: {
      num_requests_running: 1,
      num_requests_waiting: 0,
      waiting_by_reason: {},
      kv_cache_usage_perc: 0.4,
      kv_tokens_used: 100,
      kv_tokens_total: 250,
      engine_sleep_state: 0,
      preemptions_total: 0,
      preemptions_per_s: 0,
    },
    throughput: {
      prompt_tokens_per_s: 10,
      generation_tokens_per_s: 20,
      prompt_tokens_total: 1000,
      generation_tokens_total: 2000,
    },
    cache: {
      prefix_hit_rate: null,
      prefix_hit_rate_cumulative: null,
      mm_hit_rate_cumulative: null,
      external_prefix_hit_rate_cumulative: null,
    },
    latency: {
      ttft_p50: null, ttft_p90: null, ttft_p99: null, ttft_mean: null,
      itl_p50: null, itl_p99: null, tpot_p50: null,
      e2e_p50: null, e2e_p90: null, e2e_p99: null,
    },
    finished: {},
    scrape_error: null,
    ...overrides,
  };
}

describe('combineThroughput — a mixed vLLM + llama.cpp selection', () => {
  it('adds two reporting engines', () => {
    const out = combineThroughput([engineFrame('a'), engineFrame('b')]);
    expect(out.generation_tokens_per_s.value).toBe(40);
    expect(out.generation_tokens_per_s.reporting).toBe(2);
  });

  it('does not report zero for the engine that stays silent', () => {
    // The exact shape of the operator's deployment: one vLLM model reporting
    // throughput, one llama.cpp model whose /metrics carries no such counter.
    const quiet = engineFrame('llamacpp-one', {
      backend: 'llamacpp',
      throughput: {
        prompt_tokens_per_s: null,
        generation_tokens_per_s: null,
        prompt_tokens_total: null,
        generation_tokens_total: null,
      },
    });
    const out = combineThroughput([engineFrame('vllm-one'), quiet]);
    expect(out.generation_tokens_per_s.value).toBe(20);
    expect(out.generation_tokens_per_s.reporting).toBe(1);
    expect(out.generation_tokens_per_s.total).toBe(2);
    expect(isPartial(out.generation_tokens_per_s)).toBe(true);
  });

  it('reports null, not 0, when neither engine measures it', () => {
    const quiet = {
      prompt_tokens_per_s: null,
      generation_tokens_per_s: null,
      prompt_tokens_total: null,
      generation_tokens_total: null,
    };
    const out = combineThroughput([
      engineFrame('a', { throughput: quiet }),
      engineFrame('b', { throughput: quiet }),
    ]);
    expect(out.generation_tokens_per_s.value).toBeNull();
  });

  it('combines the request counters too', () => {
    const out = combineThroughput([
      engineFrame('a'),
      engineFrame('b', {
        // engine is nullable at the type level now (a null frame); the
        // fixture always builds one, hence the assertion.
        engine: { ...engineFrame('b').engine!, num_requests_running: 4 },
      }),
    ]);
    expect(out.running.value).toBe(5);
  });

  it('survives a block with no engine sub-object at all', () => {
    // What a null frame from a failed scrape looks like. It must contribute
    // nothing rather than throwing inside a render.
    const broken = engineFrame('down', {
      engine: undefined as unknown as LiveEngineFrame['engine'],
      throughput: undefined as unknown as LiveEngineFrame['throughput'],
    });
    const out = combineThroughput([engineFrame('up'), broken]);
    expect(out.generation_tokens_per_s.value).toBe(20);
    expect(out.generation_tokens_per_s.reporting).toBe(1);
  });
});

describe('liveFramesOf', () => {
  it('returns the per-model list when the API sends one', () => {
    const f = engineFrame('a', { models: [engineFrame('a'), engineFrame('b')] });
    expect(liveFramesOf(f).map((x) => x.model)).toEqual(['a', 'b']);
  });

  it('reads a pre-multi-model frame as the one model it describes', () => {
    // A ui newer than its api. Blanking the dashboard during a rollout, while
    // the engine serves perfectly well, is the worse failure.
    expect(liveFramesOf(engineFrame('a')).map((x) => x.model)).toEqual(['a']);
  });

  it('is empty when nothing is loaded', () => {
    const idle = engineFrame('x', { model: null, model_id: null });
    expect(liveFramesOf(idle)).toEqual([]);
    expect(liveFramesOf(null)).toEqual([]);
  });
});
