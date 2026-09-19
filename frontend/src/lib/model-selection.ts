"use client";
// Which loaded models a page's numbers cover.
//
// One selection, used by /stats — the operator
// picks once and every panel answers about the same models. A per-page
// selection would let two pages disagree about what "the numbers" mean while
// both were open, which is the same defect as a control whose effect is gated
// on something other than the control.
//
// THE RULE THAT SHAPES THIS FILE: at least one model stays selected. An empty
// selection has two plausible readings — "everything" and "nothing" — and
// every surface downstream would have to pick one. So the state can never
// reach it: the last remaining checkbox is DISABLED rather than silently
// re-ticked. Silently re-selecting would mean the operator clicks, sees the box
// stay ticked, and cannot tell whether the click registered.
//
// The pure functions are separate from the hook so the rule can be tested
// without a DOM, a clock, or localStorage.
import { useCallback, useEffect, useMemo, useState } from "react";

/** localStorage key. Shared across the three pages on purpose. */
export const MODEL_SELECTION_KEY = "vw.stats.models";

/**
 * Can `id` be unticked, given what is currently selected?
 *
 * False only for the last one standing. Callers render the checkbox
 * `disabled` on false — the affordance and the effect are then gated on the
 * same condition, which is the whole point.
 */
export function canDeselect(selected: readonly string[], id: string): boolean {
  return !(selected.length === 1 && selected[0] === id);
}

/**
 * Tick or untick `id`. Returns the SAME array identity when the change is
 * refused, so a React state setter is a no-op rather than a pointless re-render
 * that could look like a flicker.
 */
export function toggleModel(selected: readonly string[], id: string): string[] {
  if (selected.includes(id)) {
    if (!canDeselect(selected, id)) return selected as string[];
    return selected.filter((x) => x !== id);
  }
  return [...selected, id].sort();
}

/**
 * Fold a remembered selection onto the models that actually exist now.
 *
 * The fleet changes under a stored selection: models are loaded, unloaded,
 * deleted, renamed. Three cases, and the third is the one worth stating:
 *
 *   - nothing available -> empty. There is no model to select; the pages
 *     render their "nothing loaded" state rather than a selector.
 *   - some of the stored ids still exist -> keep exactly those. A model that
 *     went away must not keep narrowing the numbers invisibly.
 *   - NONE of them exist (first visit, or the whole fleet was replaced) ->
 *     select everything. "All" is the honest default for a dashboard: the
 *     alternative, picking one arbitrarily, would show a number the operator
 *     never asked to narrow.
 */
export function reconcileSelection(
  stored: readonly string[] | null,
  available: readonly string[],
): string[] {
  if (available.length === 0) return [];
  const kept = (stored ?? []).filter((id) => available.includes(id));
  return kept.length > 0 ? [...kept].sort() : [...available].sort();
}

/**
 * Serialise for `?models=`. Empty selection -> `null`, meaning "send no
 * parameter at all": the API treats an absent `models` as the whole
 * deployment and an EMPTY one as a client bug (400), and this is the one place
 * that distinction is encoded.
 */
export function modelsQueryParam(selected: readonly string[]): string | null {
  return selected.length > 0 ? selected.join(",") : null;
}

function readStored(): string[] | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(MODEL_SELECTION_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed)
      ? parsed.filter((x): x is string => typeof x === "string")
      : null;
  } catch {
    // Private mode, blocked site data, or a value some other build wrote.
    // A remembered selection is a convenience; failing to read one must never
    // stop the page rendering.
    return null;
  }
}

function writeStored(ids: readonly string[]): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(MODEL_SELECTION_KEY, JSON.stringify(ids));
  } catch {
    /* quota, private mode — see readStored */
  }
}

export interface ModelSelection {
  /** Currently selected ids, sorted. Never empty while `available` is not. */
  selected: string[];
  /** Tick/untick. Refused for the last remaining model. */
  toggle: (id: string) => void;
  /** Select every available model. */
  selectAll: () => void;
  /** Is this id the last one standing? Render its checkbox disabled. */
  isLocked: (id: string) => boolean;
  /** `?models=` value, or null when no parameter should be sent. */
  queryParam: string | null;
}

/**
 * The shared selection, reconciled against `available` and persisted.
 *
 * `available` is the loaded-model id list from whichever endpoint the page
 * already fetches. Pass `[]` while it is still loading: the selection then
 * stays empty, `queryParam` is null, and the page fetches unfiltered — which
 * is the correct thing to show before we know what is loaded, and avoids a
 * first render that narrows to a guess and then widens.
 */
export function useModelSelection(available: readonly string[]): ModelSelection {
  const [selected, setSelected] = useState<string[]>([]);

  // Keyed on CONTENT, not identity: `available` is derived from a SWR response
  // and gets a new array identity on every poll, so an identity-keyed effect
  // would re-run -- and re-write localStorage -- every few seconds.
  //
  // JSON rather than a joined string, because there is no separator character
  // a model id is guaranteed not to contain, and a key collision here would
  // silently stop reconciling when the fleet changed.
  const availableKey = useMemo(
    () => JSON.stringify([...available].sort()),
    [available],
  );
  const availableList = useMemo(
    () => JSON.parse(availableKey) as string[],
    [availableKey],
  );

  useEffect(() => {
    setSelected((current) =>
      reconcileSelection(
        current.length > 0 ? current : readStored(),
        availableList,
      ),
    );
  }, [availableList]);

  useEffect(() => {
    if (selected.length > 0) writeStored(selected);
  }, [selected]);

  const toggle = useCallback((id: string) => {
    setSelected((s) => toggleModel(s, id));
  }, []);

  const selectAll = useCallback(() => {
    setSelected(availableList);
  }, [availableList]);

  const isLocked = useCallback(
    (id: string) => !canDeselect(selected, id),
    [selected],
  );

  return {
    selected,
    toggle,
    selectAll,
    isLocked,
    queryParam: modelsQueryParam(selected),
  };
}
