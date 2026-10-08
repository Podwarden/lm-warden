"use client";

import { useId, useState, type ReactNode } from "react";

/**
 * A collapsible section with a thin top border + uppercase title (S4 design
 * principle: instrument-panel aesthetic — discrete sections of related knobs
 * rather than one flat list).
 *
 * Each section ships with:
 *   - A bold uppercase title (left).
 *   - An optional subtitle (right) — used for the per-section restart-warning
 *     summary on the model settings page.
 *   - A children block that can be hidden via the disclosure triangle.
 *
 * Native `<details>` would handle keyboard a11y for free but doesn't give us
 * a slot for the right-aligned subtitle, so we drive expansion ourselves with
 * `aria-expanded` + a button summary.
 */
export interface SettingsSectionProps {
  /** Accessible name of the toggle (`aria-label`) and the testId slug. */
  title: string;
  subtitle?: ReactNode;
  /** Initial expansion state. Default true — collapsing is opt-in. */
  defaultOpen?: boolean;
  /** Stable test id; defaults to the title slugified. */
  testId?: string;
  children: ReactNode;

  // ---- Opt-in (model settings redesign). All default off; a section that
  // ---- passes none of them renders exactly as it always has.

  /** Value summary shown only while collapsed, next to the title and OUTSIDE
   *  the toggle. Linked to the toggle with `aria-describedby`, so it is a
   *  description, never part of the toggle's name. */
  summary?: ReactNode;
  /** Renders "N unsaved" beside the toggle (outside its accessible name).
   *  Nothing is rendered for 0 / undefined. */
  dirtyCount?: number;
  /** Controlled expansion. When set, `defaultOpen` is ignored and the
   *  section only changes when the parent changes this prop — which is how
   *  the page force-expands a section holding an error or a fresh preset. */
  open?: boolean;
  /** Called with the requested state on every toggle click, controlled or
   *  not. */
  onOpenChange?: (open: boolean) => void;
  /** "card": token-styled card chrome (chat-* / vw-* scales, no slate). */
  variant?: "card";
  /** Wrap the toggle in a heading of this level (WAI-ARIA accordion
   *  pattern: heading > button). The section is then `aria-labelledby` it. */
  as?: "h3";
  /** Visible title text when it should differ from `title` — e.g. visible
   *  "Memory & context" while the toggle keeps `aria-label="Memory"`. */
  displayTitle?: string;
}

/** Plain-props rendering the global settings tabs use. Kept as its own
 *  branch so its markup and classes cannot drift with the card variant. */
function LegacySection({
  title,
  subtitle,
  testId,
  children,
  open,
  toggle,
  bodyId,
}: {
  title: string;
  subtitle?: ReactNode;
  testId: string;
  children: ReactNode;
  open: boolean;
  toggle: () => void;
  bodyId: string;
}) {
  const id = testId;
  return (
    <section
      data-testid={id}
      className="border-t border-slate-700/60 pt-4 first:border-t-0 first:pt-0"
    >
      <div className="flex w-full items-baseline justify-between gap-3">
        <button
          type="button"
          onClick={toggle}
          aria-expanded={open}
          aria-controls={bodyId}
          aria-label={title}
          data-testid={`${id}-toggle`}
          className="flex items-center gap-2 text-left focus:outline-none focus:ring-1 focus:ring-amber-600 rounded-sm"
        >
          <span
            aria-hidden
            className={
              "inline-block w-3 text-[10px] text-slate-500 transition-transform " +
              (open ? "rotate-90" : "")
            }
          >
            ▶
          </span>
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            {title}
          </span>
        </button>
        {subtitle && (
          // Subtitle lives OUTSIDE the toggle button so its content does NOT
          // become part of the button's accessible name. Crucial because the
          // model-settings page renders a "1 unsaved" subtitle that would
          // otherwise match `getByRole('button', { name: /save/i })` and
          // collide with the page-level Save action button.
          <span
            className="text-[11px] text-slate-500"
            data-testid={`${id}-subtitle`}
          >
            {subtitle}
          </span>
        )}
      </div>
      <div
        id={bodyId}
        data-testid={`${id}-body`}
        hidden={!open}
        className="mt-3 space-y-4"
      >
        {children}
      </div>
    </section>
  );
}

export function SettingsSection({
  title,
  subtitle,
  defaultOpen = true,
  testId,
  children,
  summary,
  dirtyCount,
  open: openProp,
  onOpenChange,
  variant,
  as: HeadingTag,
  displayTitle,
}: SettingsSectionProps) {
  const [openState, setOpenState] = useState(defaultOpen);
  const controlled = openProp !== undefined;
  const open = controlled ? openProp : openState;
  const bodyId = useId();
  const headingId = useId();
  const summaryId = useId();
  const id = testId ?? `settings-section-${slugify(title)}`;

  const toggle = () => {
    const next = !open;
    if (!controlled) setOpenState(next);
    onOpenChange?.(next);
  };

  const enhanced =
    variant !== undefined ||
    HeadingTag !== undefined ||
    displayTitle !== undefined ||
    summary !== undefined ||
    dirtyCount !== undefined;

  if (!enhanced) {
    return (
      <LegacySection
        title={title}
        subtitle={subtitle}
        testId={id}
        open={open}
        toggle={toggle}
        bodyId={bodyId}
      >
        {children}
      </LegacySection>
    );
  }

  const card = variant === "card";
  const showSummary = !open && summary !== undefined && summary !== null && summary !== "";

  const button = (
    <button
      type="button"
      onClick={toggle}
      aria-expanded={open}
      aria-controls={bodyId}
      aria-label={title}
      aria-describedby={showSummary ? summaryId : undefined}
      data-testid={`${id}-toggle`}
      className={
        card
          ? "flex min-h-6 min-w-0 items-center gap-2.5 rounded-md text-left focus:outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-chat-accent"
          : "flex items-center gap-2 text-left focus:outline-none focus:ring-1 focus:ring-amber-600 rounded-sm"
      }
    >
      {card ? (
        <svg
          aria-hidden="true"
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth={1.8}
          className={
            "h-4 w-4 flex-none text-chat-muted transition-transform motion-reduce:transition-none " +
            (open ? "rotate-90" : "")
          }
        >
          <path d="m6 3.5 4.5 4.5L6 12.5" />
        </svg>
      ) : (
        <span
          aria-hidden
          className={
            "inline-block w-3 text-[10px] text-slate-500 transition-transform " +
            (open ? "rotate-90" : "")
          }
        >
          ▶
        </span>
      )}
      <span
        className={
          card
            ? "whitespace-nowrap text-[15px] font-semibold text-chat-fg"
            : "text-xs font-semibold uppercase tracking-wider text-slate-300"
        }
      >
        {displayTitle ?? title}
      </span>
    </button>
  );

  return (
    <section
      data-testid={id}
      data-open={open ? "true" : "false"}
      aria-labelledby={HeadingTag ? headingId : undefined}
      className={
        card
          ? "rounded-xl border border-vw-rule-soft/70 bg-chat-surface/55"
          : "border-t border-slate-700/60 pt-4 first:border-t-0 first:pt-0"
      }
    >
      <div
        className={
          card
            ? "flex flex-wrap items-center gap-x-2.5 gap-y-1 px-4 py-3.5"
            : "flex w-full items-baseline justify-between gap-3"
        }
      >
        {HeadingTag ? (
          <HeadingTag id={headingId} className="m-0 flex min-w-0 text-[15px] font-semibold">
            {button}
          </HeadingTag>
        ) : (
          button
        )}
        {showSummary && (
          // Outside the toggle: a description (aria-describedby), not a name.
          // Wraps under the title on phones, indented past the chevron.
          <span
            id={summaryId}
            data-testid={`${id}-summary`}
            className={
              card
                ? "min-w-0 flex-1 truncate text-[12.5px] leading-[1.45] tabular-nums text-chat-muted max-sm:basis-full max-sm:whitespace-normal max-sm:pl-[26px] max-sm:[overflow-wrap:anywhere]"
                : "text-[11px] text-slate-500"
            }
          >
            {summary}
          </span>
        )}
        {subtitle && (
          <span
            className={card ? "text-[12.5px] text-chat-muted" : "text-[11px] text-slate-500"}
            data-testid={`${id}-subtitle`}
          >
            {subtitle}
          </span>
        )}
        {dirtyCount !== undefined && dirtyCount > 0 && (
          // Outside the toggle so "unsaved" never joins a button name (the
          // page's single /save/i button must stay unique).
          <span
            data-testid={`${id}-dirty`}
            className={
              card
                ? "ml-auto whitespace-nowrap text-xs font-semibold tabular-nums text-vw-amber-fg"
                : "ml-auto text-[11px] text-slate-500"
            }
          >
            {dirtyCount} unsaved
          </span>
        )}
      </div>
      <div
        id={bodyId}
        data-testid={`${id}-body`}
        hidden={!open}
        // `[&[hidden]]:hidden`: the `flex` class would otherwise beat the
        // browser's [hidden] rule and show a collapsed body.
        className={card ? "flex flex-col gap-[18px] px-4 pb-[18px] pt-1 [&[hidden]]:hidden" : "mt-3 space-y-4"}
      >
        {children}
      </div>
    </section>
  );
}

function slugify(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");
}
