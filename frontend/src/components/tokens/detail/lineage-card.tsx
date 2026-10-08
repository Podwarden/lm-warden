// Rotation history (spec §4.2; mockup `.lineage`). Oldest first; earlier and
// later keys link to their own pages; this token is "This token".
import Link from "next/link";
import type { TokenLineageEntry } from "@/lib/token-series";
import { day, fmtDateTime, parseSqliteUtc } from "@/lib/token-format";
import { cn } from "@/lib/utils";
import { CARD, CARD_TITLE } from "./styles";

export function lineageMeta(e: TokenLineageEntry, nowSec: number): string {
  const created = parseSqliteUtc(e.created_at) ?? 0;
  // Mockup: other keys show the day only, this token shows day and time.
  const createdTxt = e.is_self ? fmtDateTime(created, nowSec) : day(created);
  const rotated = parseSqliteUtc(e.rotated_at);
  if (rotated == null) return `Created ${createdTxt} · in use`;
  return `Created ${createdTxt} · rotated ${fmtDateTime(rotated, nowSec)} · ${e.is_revoked ? "cut off" : "in grace"}`;
}

export function LineageCard({ lineage, nowSec }: { lineage: TokenLineageEntry[]; nowSec: number }) {
  if (lineage.length <= 1) return null;
  return (
    <section className={CARD} aria-label="Rotation history">
      <h2 className={CARD_TITLE}>Rotation history</h2>
      <ol className="m-0 list-none p-0 text-[13px]">
        {lineage.map((e, i) => {
          const last = i === lineage.length - 1;
          return (
            <li key={e.id} className={cn("relative pl-5", last ? "pb-0" : "pb-3")}>
              {/* 12px, not 8: the mockup's `* { box-sizing: border-box }` does not
                  reach ::before, so its 8px dot plus 2px border renders 12px wide. */}
              <span
                aria-hidden
                className={cn(
                  "absolute left-1 top-[7px] h-3 w-3 rounded-full border-2",
                  e.is_self ? "border-chat-accent bg-chat-accent" : "border-chat-dim bg-chat-page",
                )}
              />
              {!last && <span aria-hidden className="absolute bottom-0 left-2 top-[18px] w-0.5 bg-chat-rule" />}
              {e.is_self ? (
                <span className="font-semibold">This token</span>
              ) : (
                <Link href={`/tokens/${encodeURIComponent(e.id)}`} className="font-medium text-chat-fg no-underline hover:underline">
                  {e.name}
                </Link>
              )}
              <div className="text-[12px] text-chat-dim">{lineageMeta(e, nowSec)}</div>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
