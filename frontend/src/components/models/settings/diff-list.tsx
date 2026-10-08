// Shared diff helpers for the preset confirm and the suggest result.
//
// computeDiff produces a stable ordered list of {key, before, after} rows
// (PATCHABLE_KEYS order); keys whose `after` deep-equals `before` are skipped,
// so an empty diff means "nothing to apply" and the Apply button disables.
// Rows show the field LABEL (MODEL_HINTS), not the key; the key stays in
// `data-diff-key` for tests and tooling.

import {
  PATCHABLE_KEYS,
  eqValue,
  formatSettingValue,
  labelFor,
  type Draft,
} from "@/lib/model-settings";

export interface DiffRow {
  key: string;
  before: unknown;
  after: unknown;
}

export function computeDiff(
  draft: Draft,
  sparse: Record<string, unknown>,
): DiffRow[] {
  const rows: DiffRow[] = [];
  for (const k of PATCHABLE_KEYS) {
    if (!Object.prototype.hasOwnProperty.call(sparse, k)) continue;
    const before = draft[k];
    const after = sparse[k];
    if (eqValue(before, after)) continue;
    rows.push({ key: k, before, after });
  }
  return rows;
}

// Same wording as the save bar; an empty string still needs a visible mark.
const show = (key: string, v: unknown) => formatSettingValue(key, v) || "(empty)";

export function DiffList({
  rows,
  testIdPrefix,
}: {
  rows: DiffRow[];
  testIdPrefix: string;
}) {
  if (rows.length === 0) {
    return (
      <p
        className="mt-2 text-[12.5px] text-chat-muted"
        data-testid={`${testIdPrefix}-empty`}
      >
        No changes — your draft already matches.
      </p>
    );
  }
  return (
    <ul
      className="mt-2 grid gap-1.5 text-[13px]"
      data-testid={`${testIdPrefix}-list`}
    >
      {rows.map((r) => (
        <li
          key={r.key}
          className="grid grid-cols-1 items-baseline gap-x-2.5 gap-y-0.5 sm:grid-cols-[minmax(140px,220px)_1fr]"
          data-testid={`${testIdPrefix}-row`}
          data-diff-key={r.key}
        >
          <span className="font-medium text-chat-fg">{labelFor(r.key)}</span>
          <span className="!font-mono text-[12.5px] tabular-nums text-chat-muted [overflow-wrap:anywhere]">
            <s>{show(r.key, r.before)}</s>
            <span aria-hidden="true"> → </span>
            <span className="sr-only"> to </span>
            <b className="font-medium text-vw-amber-fg">{show(r.key, r.after)}</b>
          </span>
        </li>
      ))}
    </ul>
  );
}

/**
 * Small neutral button used by the settings panels (Copy, Suggest values,
 * Cancel / Apply). Neutral on purpose: the page's one accent belongs to the
 * Save button and the focus ring (plan §5).
 */
const PANEL_BUTTON_BASE =
  "inline-flex min-h-7 items-center justify-center gap-1.5 whitespace-nowrap rounded-lg border bg-transparent px-2.5 py-0.5 text-[12.5px] text-chat-fg hover:bg-chat-surface-2/60 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-chat-accent disabled:cursor-not-allowed disabled:opacity-[.45]";
export const PANEL_BUTTON_CLASS = `${PANEL_BUTTON_BASE} border-vw-rule-soft font-medium`;
/** The confirming action of a popover (Apply): same chrome, firmer border. */
export const PANEL_BUTTON_STRONG_CLASS = `${PANEL_BUTTON_BASE} border-chat-fg/40 font-semibold`;
