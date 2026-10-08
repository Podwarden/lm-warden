"use client";

// Effective command (plan §4.5): the FULL argv, argv[0] included, that the
// engine would be invoked with for the *saved* settings (preview port 10000),
// from GET /api/models/{id}/effective-argv.
//
// - One flag per line: a token starting with "-<letter>" begins a new line,
//   the non-flag tokens after it stay on that line.
// - Token colours: binary/subcommand in accent, flags in fg, values muted, and
//   the trailing tokens that equal the saved `extra_args` in ok-fg ("yours").
// - `stale` adds the amber note: the argv is server-computed from SAVED
//   settings, so a dirty draft is not in it yet.
// - Wraps (`pre-wrap`, `overflow-wrap:anywhere`), never scrolls sideways.
// - Changed tokens flash with `row-flash-emerald` (reduced-motion gated in
//   globals.css), keyed by index against the previous fetch.

import { Fragment, useEffect, useId, useMemo, useRef, useState } from "react";
import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";
import { copyToClipboard } from "@/lib/utils";
import { LLAMACPP_MANAGED_FLAGS } from "@/lib/llamacpp-args";
import { Skeleton } from "@/components/ui/skeleton";
import { PANEL_BUTTON_CLASS } from "./diff-list";

export interface EffectiveArgvResponse {
  argv: string[];
}

export type ArgvTokenKind = "bin" | "flag" | "val" | "yours";

export interface ArgvToken {
  /** Index into the original argv (the flash is keyed by it). */
  index: number;
  text: string;
  kind: ArgvTokenKind;
}

const FLAG_RE = /^--?[A-Za-z]/;
// llama.cpp flags that live in extra_args but are set by the engine controls
// (flag + its value). In the argv tail they are "from the fields", not yours.
const MANAGED_FLAGS: ReadonlySet<string> = new Set(
  LLAMACPP_MANAGED_FLAGS.flatMap((f) => [f.flag, ...f.aliases]),
);
// A subcommand like `serve`: a bare lowercase word right after the binary.
const SUBCOMMAND_RE = /^[a-z][a-z0-9-]*$/;

function isFlag(tok: string): boolean {
  return FLAG_RE.test(tok);
}

/**
 * How many trailing argv tokens are the operator's own extra args. Tries the
 * saved list verbatim first, then with each entry whitespace-split (an entry
 * stored as "--max-num-seqs 32" reaches the argv as two tokens). 0 when the
 * argv does not end with them.
 */
export function yoursCount(argv: readonly string[], extraArgs?: readonly string[]): number {
  if (!extraArgs || extraArgs.length === 0) return 0;
  const candidates = [
    [...extraArgs],
    extraArgs.flatMap((a) => a.trim().split(/\s+/)).filter(Boolean),
  ];
  for (const tail of candidates) {
    if (tail.length === 0 || tail.length > argv.length) continue;
    const off = argv.length - tail.length;
    if (tail.every((t, i) => argv[off + i] === t)) return tail.length;
  }
  return 0;
}

/** Group argv into display lines of classified tokens (pure; exported for reuse). */
export function argvLines(argv: readonly string[], extraArgs?: readonly string[]): ArgvToken[][] {
  const yoursFrom = argv.length - yoursCount(argv, extraArgs);
  // Inside the tail, a managed flag and the value after it are not yours.
  const managed = new Set<number>();
  for (let i = yoursFrom; i < argv.length; i++) {
    if (!MANAGED_FLAGS.has(argv[i])) continue;
    managed.add(i);
    if (i + 1 < argv.length && !isFlag(argv[i + 1])) managed.add(++i);
  }
  const lines: ArgvToken[][] = [];
  let current: ArgvToken[] = [];
  argv.forEach((text, index) => {
    const flag = isFlag(text);
    if (flag && current.length > 0) {
      lines.push(current);
      current = [];
    }
    let kind: ArgvTokenKind;
    if (index >= yoursFrom && !managed.has(index)) kind = "yours";
    else if (index === 0 || (index === 1 && !flag && SUBCOMMAND_RE.test(text))) kind = "bin";
    else kind = flag ? "flag" : "val";
    current.push({ index, text, kind });
  });
  if (current.length > 0) lines.push(current);
  return lines;
}

const KIND_CLASS: Record<ArgvTokenKind, string> = {
  bin: "font-semibold text-chat-accent",
  flag: "text-chat-fg",
  val: "text-chat-muted",
  yours: "text-vw-ok-fg",
};

export interface EffectiveArgvPanelProps {
  modelId: string;
  /** The SAVED `extra_args` (the operator's own); their argv tail is tinted. */
  extraArgs?: readonly string[];
  /** The draft has unsaved changes, which this command does not reflect yet. */
  stale?: boolean;
  className?: string;
}

export function EffectiveArgvPanel({
  modelId,
  extraArgs,
  stale = false,
  className,
}: EffectiveArgvPanelProps) {
  const { data, error, isLoading } = useSWR<EffectiveArgvResponse>(
    `/api/models/${modelId}/effective-argv`,
    authFetchJSON,
    { revalidateOnFocus: false },
  );
  const headingId = useId();

  // Highlight tokens that changed vs. the previous fetch. The prior argv lives
  // in a ref so the comparison survives SWR mutations without an extra render.
  const prevArgv = useRef<string[] | null>(null);
  const [changedSet, setChangedSet] = useState<Set<number>>(new Set());
  useEffect(() => {
    if (!data) return;
    const prev = prevArgv.current;
    if (!prev) {
      prevArgv.current = data.argv;
      return;
    }
    const changed = new Set<number>();
    for (let i = 0; i < data.argv.length; i++) {
      if (data.argv[i] !== prev[i]) changed.add(i);
    }
    setChangedSet(changed);
    prevArgv.current = data.argv;
    // row-flash-emerald is a 2s keyframe; clear after it completes.
    const t = setTimeout(() => setChangedSet(new Set()), 2100);
    return () => clearTimeout(t);
  }, [data]);

  const lines = useMemo(
    () => (data ? argvLines(data.argv, extraArgs) : []),
    [data, extraArgs],
  );
  const hasYours = lines.some((l) => l.some((t) => t.kind === "yours"));

  const [copied, setCopied] = useState(false);
  function onCopy() {
    if (!data) return;
    // copyToClipboard falls back to execCommand on LAN-HTTP (#149). The flash
    // is best-effort: there is no toast surface here.
    void copyToClipboard(data.argv.join(" ")).catch(() => {});
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <section
      aria-labelledby={headingId}
      data-testid="effective-argv-panel"
      className={`rounded-xl border border-vw-rule-soft/70 bg-chat-surface/55${className ? ` ${className}` : ""}`}
    >
      <div className="flex items-center justify-between gap-2 px-3.5 pt-3">
        <h2 id={headingId} className="m-0 text-[15px] font-semibold text-chat-fg">
          Effective command
        </h2>
        <button
          type="button"
          className={PANEL_BUTTON_CLASS}
          onClick={onCopy}
          disabled={!data}
          data-testid="effective-argv-copy"
        >
          <svg
            width="13"
            height="13"
            viewBox="0 0 14 14"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            aria-hidden="true"
          >
            <rect x="4.5" y="4.5" width="7.5" height="7.5" rx="1.5" />
            <path d="M9.5 4.5V3A1.5 1.5 0 0 0 8 1.5H3A1.5 1.5 0 0 0 1.5 3v5A1.5 1.5 0 0 0 3 9.5h1.5" />
          </svg>
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <p
        className="mx-3.5 mb-2.5 mt-1 text-[12.5px] leading-[1.45] text-chat-muted"
        data-testid="effective-argv-subtitle"
      >
        {/* "Why is Extra args empty but the command so long?" -- answer it
            next to the thing that prompts the question. */}
        The <em>whole</em> command from the saved settings plus your extra args.{" "}
        <code className="font-mono text-[12px]">--port 10000</code> is a placeholder.
      </p>
      {stale && (
        <p
          className="mx-3.5 mb-3 text-[12.5px] leading-[1.45] text-vw-amber-fg"
          data-testid="effective-argv-stale"
        >
          Your unsaved changes are not in this command yet.
        </p>
      )}
      {error && (
        <p
          className="mx-3.5 mb-3 text-[12.5px] text-vw-danger-fg"
          data-testid="effective-argv-error"
        >
          Could not compute the effective command
          {error instanceof Error ? `: ${error.message}` : "."}
        </p>
      )}
      {isLoading && !data && (
        <div className="mx-2.5 mb-2.5">
          <Skeleton className="h-24 w-full" data-testid="effective-argv-skeleton" />
        </div>
      )}
      {data && (
        <pre
          data-testid="effective-argv-pre"
          aria-label="Effective command"
          className="mx-2.5 mb-2.5 max-h-[420px] overflow-auto whitespace-pre-wrap rounded-lg border border-vw-rule-soft/60 bg-vw-dock px-3 py-2.5 font-mono text-[12.5px] [&_span]:!font-mono leading-[1.6] [overflow-wrap:anywhere]"
        >
          <code>
            {lines.map((line, li) => (
              <Fragment key={line[0].index}>
                {li > 0 && "\n"}
                <span data-testid="effective-argv-line">
                  {line.map((tok, ti) => (
                    <Fragment key={tok.index}>
                      {ti > 0 && " "}
                      <span
                        data-testid={
                          changedSet.has(tok.index)
                            ? "effective-argv-tok-changed"
                            : "effective-argv-tok"
                        }
                        data-kind={tok.kind}
                        className={`${KIND_CLASS[tok.kind]}${changedSet.has(tok.index) ? " row-flash-emerald" : ""}`}
                      >
                        {tok.text}
                      </span>
                    </Fragment>
                  ))}
                </span>
              </Fragment>
            ))}
          </code>
        </pre>
      )}
      {data && hasYours && (
        <p
          className="mx-3.5 mb-3 flex flex-wrap gap-3 text-[12px] text-chat-muted"
          data-testid="effective-argv-legend"
        >
          <span>
            <i aria-hidden="true" className="mr-1.5 inline-block h-2 w-2 rounded-sm bg-chat-fg" />
            from the fields
          </span>
          <span>
            <i aria-hidden="true" className="mr-1.5 inline-block h-2 w-2 rounded-sm bg-vw-ok-fg" />
            your extra args
          </span>
        </p>
      )}
    </section>
  );
}
