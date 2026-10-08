"use client";

// Error boundary for /stats.
//
// There was none, and the page it guards renders a live SSE frame whose
// blocks can legitimately arrive half-null (`_null_frame` on a scrape error).
// The render path now guards those, but a dashboard whose whole job is to
// stay up during incidents must not take the app down when a frame — or a
// future edit — surprises it. Next.js renders this in place of the segment
// and `reset()` re-renders it, which for this page also reopens its SWR
// fetches and the shared SSE subscription.

import { useEffect } from "react";

export default function StatsError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Surface the real error for the operator's console; the UI copy stays calm.
    console.error("stats page crashed:", error);
  }, [error]);

  return (
    <div
      role="alert"
      data-testid="stats-error-boundary"
      className="rounded-lg border border-chat-negative/40 bg-chat-negative/10 p-6"
    >
      <p className="text-base font-medium text-chat-fg">
        The stats page hit an error while rendering.
      </p>
      <p className="mt-1.5 text-sm text-chat-muted">
        The engines and the proxy are unaffected — this is a display failure,
        not an outage. {error.message ? `(${error.message})` : ""}
      </p>
      <button
        type="button"
        onClick={reset}
        className="mt-4 rounded-md border border-chat-rule bg-chat-surface px-3 py-1.5 text-sm text-chat-fg transition-colors hover:bg-chat-surface-2"
      >
        Try again
      </button>
    </div>
  );
}
