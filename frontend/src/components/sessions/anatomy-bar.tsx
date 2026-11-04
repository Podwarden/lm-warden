// The prompt-anatomy bar: what a prompt was made of (from cache, dead weight, recomputed, new, cache unknown).

import { ANATOMY_CLASS, anatomyWidths } from "@/lib/sessions/anatomy";
import { S } from "@/lib/sessions/strings";
import type { Anatomy } from "@/lib/sessions/types";

import { fmtTokens } from "./format";

export function AnatomyBar({ a, size = "sm", label }: { a: Anatomy; size?: "sm" | "lg"; label?: string }) {
  const segs = anatomyWidths(a);
  const parts = segs.map((s) => S.anatomy.part(S.anatomy[s.key], Math.round(s.pct))).join(", ");
  const title = segs.map((s) => S.anatomy.title(S.anatomy[s.key], fmtTokens(a[s.key]))).join("\n");
  return (
    <div
      role="img"
      aria-label={label ?? S.anatomy.aria(parts || "—")}
      title={title || undefined}
      className={`flex w-full overflow-hidden rounded-sm bg-chat-rule ${size === "lg" ? "h-3" : "h-2"}`}
    >
      {segs.map((s) => (
        <i key={s.key} data-seg={s.key} style={{ width: `${s.pct}%` }} className={`block h-full ${ANATOMY_CLASS[s.key]}`} />
      ))}
    </div>
  );
}
