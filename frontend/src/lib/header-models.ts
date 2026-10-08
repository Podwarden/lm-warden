// The header chip's model list: types plus the two pure functions that turn a
// frame into something renderable.
//
// Deliberately NOT in header-metrics-stream.ts. That module owns the
// EventSource singleton and is mocked wholesale by the widget's component
// tests (`vi.mock('@/lib/header-metrics-stream')`), so a pure helper living
// beside the hook would be replaced by `undefined` in exactly the tests that
// need it. Pure logic that a mocked module would swallow belongs outside it.

export type HeaderModelStatus = "loaded" | "loading" | "failed";

export interface HeaderActiveModel {
  id: string;
  served_model_name: string;
  status: HeaderModelStatus;
}

/** The subset of a header frame these helpers read. */
export interface HeaderModelSource {
  active_models?: HeaderActiveModel[];
  active_model: string | null;
  active_model_id: string | null;
  active_model_status?: HeaderModelStatus | null;
}

/**
 * Every model in a frame, however old the API that produced it.
 *
 * Exported so the widget renders from ONE shape instead of branching on which
 * contract it received. A frame from an api pod predating `active_models`
 * carries only the singular fields, and the correct reading of those is "one
 * model, whose status is `active_model_status`, or `loaded` if that key is
 * absent too" — the pre-status contract's only meaning. The ui and api ship as
 * separate images and can skew in either direction.
 */
export function activeModelsOf(
  frame: HeaderModelSource | null | undefined,
): HeaderActiveModel[] {
  if (!frame) return [];
  if (Array.isArray(frame.active_models)) return frame.active_models;
  if (!frame.active_model) return [];
  return [
    {
      id: frame.active_model_id ?? frame.active_model,
      served_model_name: frame.active_model,
      status: frame.active_model_status ?? "loaded",
    },
  ];
}

// Worst-first, because the cluster's single accent colour is a SUMMARY of N
// models and a summary must not be reassuring. One crashed engine beside three
// healthy ones is a red cluster; each chip still carries its own dot, so the
// operator can see which one.
const STATUS_SEVERITY: Record<HeaderModelStatus, number> = {
  failed: 0,
  loading: 1,
  loaded: 2,
};

/** The most alarming status among `models`, or null when there are none. */
export function worstModelStatus(
  models: HeaderActiveModel[],
): HeaderModelStatus | null {
  let worst: HeaderModelStatus | null = null;
  for (const m of models) {
    if (worst === null || STATUS_SEVERITY[m.status] < STATUS_SEVERITY[worst]) {
      worst = m.status;
    }
  }
  return worst;
}

/**
 * How many model chips the cluster renders inline before collapsing the rest
 * into a "+N" counter.
 *
 * Two, because the pill sits in the nav bar beside a brand block and a menu
 * button, and a fleet of six would push those off a laptop viewport. The
 * overflow counter keeps the widget's width bounded at any N while the title
 * and the aria-label still enumerate every model — the information is never
 * lost, only folded.
 */
export const HEADER_MODELS_INLINE = 2;
