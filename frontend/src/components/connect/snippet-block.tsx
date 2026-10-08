"use client";

// Snippet + Copy (plan §3 #4, §5.6, §7). One file at a time behind chip tabs;
// a `fields` file (Cline, Cursor: settings screens, no file) renders as a
// labelled value list with one Copy per value. Long lines do not wrap: the
// block scrolls inside itself, is keyboard-focusable, and says so when it
// overflows.

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { cn, copyToClipboard } from "@/lib/utils";
import type { ConnectFile } from "@/lib/connect";
import { Tabs, panelId, tabId } from "./tabs";
import { ICON_BTN, META, SNIPPET } from "./styles";

const useIsoLayoutEffect = typeof window === "undefined" ? useEffect : useLayoutEffect;

type CopyState = "idle" | "copied" | "failed";

function useCopy(resetKey: string) {
  const [state, setState] = useState<CopyState>("idle");
  const [which, setWhich] = useState<string | null>(null);
  useEffect(() => {
    setState("idle");
    setWhich(null);
  }, [resetKey]);
  useEffect(() => {
    if (state !== "copied") return;
    const t = setTimeout(() => setState("idle"), 1500);
    return () => clearTimeout(t);
  }, [state, which]);
  async function copy(text: string, id: string) {
    setWhich(id);
    try {
      await copyToClipboard(text);
      setState("copied");
    } catch {
      setState("failed");
    }
  }
  return { state, which, copy };
}

function CopyIcon({ done }: { done: boolean }) {
  return done ? (
    <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
      <path d="M3.5 8.5 6.5 11.5 12.5 4.5" />
    </svg>
  ) : (
    <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
      <rect x="5" y="5" width="9" height="9" rx="1.5" />
      <path d="M11 5V3.5A1.5 1.5 0 0 0 9.5 2h-6A1.5 1.5 0 0 0 2 3.5v6A1.5 1.5 0 0 0 3.5 11H5" />
    </svg>
  );
}

/** True when the element's content is wider than its box. */
function useOverflowX(ref: React.RefObject<HTMLElement | null>, dep: string): boolean {
  const [over, setOver] = useState(false);
  useIsoLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setOver(el.scrollWidth > el.clientWidth + 1);
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref, dep]);
  return over;
}

function parseFields(text: string): { label: string; value: string }[] {
  return text
    .split("\n")
    .filter(Boolean)
    .map((line) => {
      const i = line.indexOf(": ");
      return i < 0 ? { label: line, value: "" } : { label: line.slice(0, i), value: line.slice(i + 2) };
    });
}

interface Props {
  files: ConnectFile[];
  /** Renders a file to the exact text shown and copied. */
  render: (file: ConnectFile) => string;
  idBase: string;
}

export function SnippetBlock({ files, render, idBase }: Props) {
  const [picked, setPicked] = useState<string | null>(null);
  const file = files.find((f) => f.id === picked) ?? files[0];
  const text = file ? render(file) : "";
  const { state, which, copy } = useCopy(`${file?.id}`);
  const preRef = useRef<HTMLPreElement | null>(null);
  const overflow = useOverflowX(preRef, text);

  if (!file) return null;
  const fields = file.language === "fields";

  return (
    <div className="min-w-0">
      {/* Always a tab strip, even for one file, so "File" is the same control everywhere. */}
      <Tabs
        items={files.map((f) => ({ id: f.id, label: f.label }))}
        selected={file.id}
        onSelect={setPicked}
        label="File"
        idBase={idBase}
        variant="chip"
        className="mb-2"
      />
      <div role="tabpanel" id={panelId(idBase)} aria-labelledby={tabId(idBase, file.id)} className="min-w-0">
        {file.path && (
          <p className={cn(META, "m-0 mb-1.5")}>
            Save as <code className="!font-mono text-[12.5px] text-chat-fg [overflow-wrap:anywhere]">{file.path}</code>
          </p>
        )}
        {fields ? (
          <dl data-testid="connect-snippet" className="m-0 grid gap-2 rounded-lg border border-vw-rule-soft/60 bg-vw-dock p-3">
            {parseFields(text).map((f) => (
              <div key={f.label} className="flex min-w-0 items-center gap-2">
                <div className="min-w-0 flex-1">
                  <dt className="text-[12.5px] text-chat-muted">{f.label}</dt>
                  <dd className="m-0 !font-mono text-[12.5px] [overflow-wrap:anywhere]">{f.value}</dd>
                </div>
                <button
                  type="button"
                  aria-label={`Copy ${f.label}`}
                  onClick={() => void copy(f.value, f.label)}
                  className={ICON_BTN}
                >
                  <CopyIcon done={state === "copied" && which === f.label} />
                </button>
              </div>
            ))}
          </dl>
        ) : (
          <div className="relative min-w-0">
            <pre
              ref={preRef}
              data-testid="connect-snippet"
              tabIndex={overflow ? 0 : -1}
              role={overflow ? "region" : undefined}
              aria-label={overflow ? `${file.label}, scrolls sideways` : undefined}
              className={SNIPPET}
            >
              {text}
            </pre>
            <button
              type="button"
              aria-label={`Copy ${file.label}`}
              onClick={() => void copy(text, file.id)}
              className={cn(ICON_BTN, "absolute right-2 top-2 bg-vw-dock")}
            >
              <CopyIcon done={state === "copied" && which === file.id} />
            </button>
            {overflow && (
              <p className={cn(META, "m-0 mt-1 flex items-center gap-1.5")}>
                <svg viewBox="0 0 16 16" className="h-3.5 w-3.5 flex-none" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
                  <path d="M2 8h12M10.5 4.5 14 8l-3.5 3.5" />
                </svg>
                Long lines: scroll the block sideways. Copy takes everything.
              </p>
            )}
          </div>
        )}
        <span role="status" className="sr-only">
          {state === "copied" ? "Copied" : ""}
        </span>
        {state === "failed" && <p className="mb-0 mt-1.5 text-[12.5px] text-vw-danger-fg">Copy failed — select the text.</p>}
      </div>
    </div>
  );
}
