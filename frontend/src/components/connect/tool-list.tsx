"use client";

// The tool list (plan §3, §6, §7): one tablist, vertical in a rail card from
// 1100 px and a horizontal scrolling strip below. Section labels are visual
// only (a tablist owns tabs and nothing else); every tab carries its support
// level in its accessible name, so nothing depends on the label or colour.

import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { groupTools, SUPPORT, type ConnectClient } from "@/lib/connect";
import { nextTabIndex, panelId, tabId } from "./tabs";
import { BAND_LABEL, pill, SEC, TOOL_TAB, TOOL_TAB_ON } from "./styles";

export const TOOLS_ID = "connect-tool";

/** "Anthropic SDK (Python, TypeScript, curl)" → name + a muted second line. */
function splitName(name: string): [string, string | null] {
  const m = /^(.*?)\s*\((.+)\)$/.exec(name);
  return m ? [m[1], m[2]] : [name, null];
}

function useWide(): boolean {
  const [wide, setWide] = useState(true);
  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const mq = window.matchMedia("(min-width: 1100px)");
    const on = () => setWide(mq.matches);
    on();
    mq.addEventListener?.("change", on);
    return () => mq.removeEventListener?.("change", on);
  }, []);
  return wide;
}

/** One pill per section when its tools share a support level (they do in the
 *  catalogue); the "Not supported" section's label already says it. */
function groupPill(clients: ConnectClient[]) {
  const levels = new Set(clients.map((c) => c.support));
  if (levels.size !== 1) return null;
  const level = clients[0].support;
  if (level === "unsupported") return null;
  const s = SUPPORT[level];
  return <span className={pill(s.tone)}>{s.label}</span>;
}

interface Props {
  clients: ConnectClient[];
  selected: string;
  onSelect: (id: string) => void;
  className?: string;
}

export function ToolList({ clients, selected, onSelect, className }: Props) {
  const refs = useRef<Map<string, HTMLButtonElement>>(new Map());
  const wide = useWide();
  const groups = groupTools(clients);
  const order = groups.flatMap((g) => g.clients.map((c) => c.id));

  // Keep the selected tab in view inside the horizontal strip.
  useEffect(() => {
    if (wide) return;
    refs.current.get(selected)?.scrollIntoView?.({ block: "nearest", inline: "nearest" });
  }, [selected, wide]);

  function onKey(e: React.KeyboardEvent, id: string) {
    const next = nextTabIndex(e.key, order.indexOf(id), order.length);
    if (next === null) return;
    e.preventDefault();
    onSelect(order[next]);
    refs.current.get(order[next])?.focus();
  }

  return (
    <div data-testid="connect-tools" className={cn(SEC, "py-1.5 max-[1099px]:py-0", className)}>
      <div
        role="tablist"
        aria-label="Tools"
        aria-orientation={wide ? "vertical" : "horizontal"}
        className="relative flex flex-col max-[1099px]:flex-row max-[1099px]:items-stretch max-[1099px]:overflow-x-auto max-[1099px]:[scrollbar-width:thin]"
      >
        {groups.map((g, gi) => (
          <div key={g.section} className="contents">
            <div
              aria-hidden="true"
              className={cn("flex flex-wrap items-center gap-x-2 gap-y-1 px-4 pb-1 pt-3 max-[1099px]:hidden", gi === 0 && "pt-2")}
            >
              <span data-testid="connect-group" className={cn(BAND_LABEL, "whitespace-nowrap")}>
                {g.label}
              </span>
              {groupPill(g.clients)}
            </div>
            {gi > 0 && <span aria-hidden="true" className="my-2.5 hidden w-px flex-none bg-vw-rule-soft/60 max-[1099px]:block" />}
            {g.clients.map((c) => {
              const on = c.id === selected;
              const [name, sub] = splitName(c.name);
              const s = SUPPORT[c.support];
              return (
                <button
                  key={c.id}
                  ref={(el) => {
                    if (el) refs.current.set(c.id, el);
                    else refs.current.delete(c.id);
                  }}
                  type="button"
                  role="tab"
                  data-tool={c.id}
                  id={tabId(TOOLS_ID, c.id)}
                  aria-selected={on}
                  aria-controls={panelId(TOOLS_ID)}
                  tabIndex={on ? 0 : -1}
                  onClick={() => onSelect(c.id)}
                  onKeyDown={(e) => onKey(e, c.id)}
                  className={cn(TOOL_TAB, on && TOOL_TAB_ON)}
                >
                  <span className="min-w-0 flex-1">
                    <span className="block">{name}</span>
                    {sub && (
                      <span className="block text-[12px] font-normal text-chat-muted max-[1099px]:hidden">{sub}</span>
                    )}
                  </span>
                  <span className="sr-only">, {s.label}</span>
                </button>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
