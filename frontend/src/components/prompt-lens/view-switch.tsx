"use client";

// The Prompt Lens page's view switch (analysis spec 16.1): Requests · Sessions
// · Trends · Leaderboard. Trends and the Leaderboard are listed only when their
// routes exist (`trends`, `leaderboard` props; the page turns them on when it
// knows); the Leaderboard is the admin's alone and is never offered to a key
// holder, whatever the props say (D-4).

import type { LensView } from "@/lib/prompt-lens/scorecard";
import type { LensScope } from "@/lib/prompt-lens/stream";
import { cn } from "@/lib/utils";

const LABEL: Record<LensView, string> = {
  requests: "Requests",
  sessions: "Sessions",
  trends: "Trends",
  leaderboard: "Leaderboard",
};

export interface ViewSwitchProps {
  scope: LensScope;
  view: LensView;
  onChange: (v: LensView) => void;
  /** The Trends route answers (stage 3). */
  trends?: boolean;
  /** The Leaderboard route answers (stage 6); admin only. */
  leaderboard?: boolean;
  phone?: boolean;
}

export function viewsFor({ scope, trends = false, leaderboard = false }: Pick<ViewSwitchProps, "scope" | "trends" | "leaderboard">): LensView[] {
  const out: LensView[] = ["requests", "sessions"];
  if (trends) out.push("trends");
  if (leaderboard && scope === "admin") out.push("leaderboard");
  return out;
}

export function ViewSwitch({ scope, view, onChange, trends, leaderboard, phone }: ViewSwitchProps) {
  const views = viewsFor({ scope, trends, leaderboard });
  return (
    <div role="tablist" aria-label="Prompt Lens view" data-testid="prompt-lens-view-switch" className="flex flex-none items-center gap-0.5">
      {views.map((v) => (
        <button
          key={v}
          type="button"
          role="tab"
          aria-selected={view === v}
          data-view={v}
          onClick={() => onChange(v)}
          className={cn(
            "cursor-pointer rounded-[5px] border border-transparent bg-transparent px-2 py-[3px] text-[12px] text-chat-muted hover:text-chat-fg",
            "aria-selected:border-chat-accent/60 aria-selected:bg-chat-surface aria-selected:text-chat-fg",
            phone && "min-h-9",
          )}
        >
          {LABEL[v]}
        </button>
      ))}
    </div>
  );
}
