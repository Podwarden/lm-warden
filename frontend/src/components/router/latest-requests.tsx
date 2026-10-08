"use client";

// Rail "Latest requests" (plan §4.7): the newest 4 of the 20 decisions the
// overview polls. Each says when, which model went where, the route as a word
// next to its colour, the reason code, latency and key name.

import Link from "next/link";
import { cn } from "@/lib/utils";
import { ACTIVITY_HREF, fmtInt, type DecisionOut } from "@/lib/router";
import { CARD, LINK, routeTone, ROUTE_LABEL, ROUTE_SWATCH } from "./styles";

export const LATEST_SHOWN = 4;

function fmtTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

interface Props {
  decisions: DecisionOut[] | undefined;
  className?: string;
}

export function LatestRequests({ decisions, className }: Props) {
  const shown = decisions?.slice(0, LATEST_SHOWN);
  return (
    <section aria-labelledby="router-latest-h" data-testid="router-latest" className={cn(CARD, className)}>
      <h2 id="router-latest-h" className="m-0 text-[15px] font-semibold">
        Latest requests
      </h2>
      {shown === undefined ? (
        <div aria-hidden="true" className="mt-2 grid gap-2">
          {Array.from({ length: LATEST_SHOWN }, (_, i) => (
            <div key={i} className="h-9 animate-pulse rounded bg-chat-surface-2/60 motion-reduce:animate-none" />
          ))}
        </div>
      ) : shown.length === 0 ? (
        <p className="mb-0 mt-2 text-[13px] text-chat-muted">Waiting for the first request.</p>
      ) : (
        <ul className="m-0 mt-2 list-none p-0">
          {shown.map((d, i) => {
            const rt = routeTone(d.route);
            const to = d.route === "local" && d.model_out ? ` → ${d.model_out}` : "";
            return (
              <li
                key={`${d.ts}-${i}`}
                className="grid grid-cols-[58px_minmax(0,1fr)] gap-x-2.5 gap-y-0.5 border-t border-vw-rule-soft/35 py-[7px] text-[12.5px]"
              >
                <time dateTime={d.ts} className="tabular-nums text-chat-muted">
                  {fmtTime(d.ts)}
                </time>
                <span className="!font-mono text-[12px] [overflow-wrap:anywhere]">
                  {(d.model_in ?? d.path) + to}
                </span>
                <span className="col-start-2 flex flex-wrap items-center gap-2 text-chat-muted">
                  <span className={cn(ROUTE_LABEL, rt.fg)}>
                    <span aria-hidden="true" className={ROUTE_SWATCH} />
                    {rt.label}
                  </span>
                  {d.reason && <code className="font-mono text-[12px] [overflow-wrap:anywhere]">{d.reason}</code>}
                  <span className="tabular-nums">{fmtInt(d.latency_ms)} ms</span>
                  {d.token_name && <span className="[overflow-wrap:anywhere]">{d.token_name}</span>}
                </span>
              </li>
            );
          })}
        </ul>
      )}
      <Link href={ACTIVITY_HREF} className={cn(LINK, "mt-2.5 inline-block text-[13px]")}>
        All activity and latency →
      </Link>
    </section>
  );
}
