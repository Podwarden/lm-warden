// In-flight table: the SESSION column and the ESTIMATED prefix-cache hit
// (blue segment in the context bar, "cached" pill). vLLM reports the real hit
// only after a request ends, so the UI must say "estimated".

import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup, within } from "@testing-library/react";
import { LiveRequestsPanel, PrefillModelLine } from "@/components/stats/live-panels";
import type { LiveRequestRow } from "@/lib/live-stats";

function row(over: Partial<LiveRequestRow> = {}): LiveRequestRow {
  return {
    id: "r1",
    token_id: "t1",
    token_name: "dev",
    client_ip: "10.0.0.1",
    model: "model-a",
    path: "/v1/chat/completions",
    prompt_tokens: 100_000,
    completion_tokens: 0,
    context_tokens: 100_000,
    max_model_len: 200_000,
    context_pct: 0.5,
    elapsed_s: 1.2,
    phase: "prefill",
    orphan: false,
    session_id: null,
    cache_est_tokens: null,
    cache_est_pct: null,
    prefill_slow: false,
    queued_s: null,
    ...over,
  };
}

const show = (rows: LiveRequestRow[]) =>
  render(<LiveRequestsPanel rows={rows} error={null} isLoading={false} />);

describe("in-flight table: session + cache estimate", () => {
  afterEach(cleanup);

  it("shows the first 8 chars of the session id, full id in the title", () => {
    const id = "0b1c2d3e-aaaa-bbbb-cccc-ddddeeeeffff";
    show([row({ session_id: id })]);
    const cell = screen.getByTestId("live-session");
    expect(cell.textContent).toBe("0b1c2d3e");
    expect(cell.getAttribute("title")).toBe(id);
  });

  it("names the signal in the tooltip, and the parent of a subagent", () => {
    show([
      row({
        session_id: "ses_a1B2c3D4e5F6g7H8i9J0k1L2m3",
        session_source: "x_session_id",
        parent_session_id: "ses_parentparent01",
      }),
    ]);
    const title = screen.getByTestId("live-session").getAttribute("title") ?? "";
    expect(title).toContain("ses_a1B2c3D4e5F6g7H8i9J0k1L2m3");
    expect(title).toContain("from X-Session-Id");
    expect(title).toContain("subagent of ses_parentparent01");
  });

  it("an inferred conversation shows a dash, dimmed, with its own tooltip", () => {
    show([row({ session_id: null, session_source: "prompt_hash" })]);
    const cell = screen.getByTestId("live-session");
    expect(cell.textContent).toBe("—");
    expect(cell.className).toContain("opacity-60");
    expect(cell.getAttribute("title")).toBe("conversation inferred from first message");
    cleanup();
    show([row({ session_id: null, session_source: null })]);
    expect(screen.getByTestId("live-session").getAttribute("title")).toBeNull();
    expect(screen.getByTestId("live-session").className).not.toContain("opacity-60");
  });

  it("shows a dash when the session is unknown", () => {
    show([row()]);
    expect(screen.getByTestId("live-session").textContent).toBe("—");
  });

  it("renders no cache segment and the amber prefill pill without an estimate", () => {
    show([row()]);
    expect(screen.queryByTestId("cache-est-segment")).toBeNull();
    expect(screen.getByText("prefill")).toBeInTheDocument();
    expect(screen.queryByText("≈ cached")).toBeNull();
  });

  it("draws the blue segment sized to cache_est_tokens / max_model_len with an estimate tooltip", () => {
    show([row({ cache_est_tokens: 80_000, cache_est_pct: 0.8 })]);
    const seg = screen.getByTestId("cache-est-segment");
    expect(seg.style.width).toBe("40%");
    expect(seg.parentElement?.getAttribute("title")).toMatch(/likely in this replica's prefix cache/);
    expect(seg.parentElement?.getAttribute("title")).toMatch(/estimated/);
  });

  it("describes cached, fresh and total on the context meter", () => {
    show([row({ cache_est_tokens: 80_000, cache_est_pct: 0.8 })]);
    const label = screen.getByRole("meter").getAttribute("aria-label") ?? "";
    expect(label).toMatch(/cached \(estimated\)/);
    expect(label).toMatch(/fresh/);
    expect(label).toMatch(/of 200[.0-9]*k/i);
  });
})

const label = (text: RegExp) => screen.getByText(text) as HTMLElement; // the coloured pill
const wrapper = (text: RegExp) => label(text).closest("span[title]") as HTMLElement; // title + aria

describe("in-flight table: phase pill is the lifecycle, badges carry cache and cause", () => {
  afterEach(cleanup);

  it.each([
    ["queued", "queued", "text-chat-muted"],
    ["prefill", "prefill", "text-chat-warn"],
    ["waiting", "waiting", "text-chat-muted"],
    ["thinking", "thinking", "text-vw-prompt"],
    ["tool_call", "tool call", "text-vw-model-6"],
    ["answering", "answering", "text-chat-positive"],
    ["decode", "answering", "text-chat-positive"], // legacy rows
  ])("%s renders as '%s' with an icon", (phase, text, tone) => {
    show([row({ phase })]);
    const pill = label(new RegExp(`^${text}`));
    expect(pill.className).toContain(tone);
    expect(pill.querySelector("svg")).not.toBeNull();
    expect(wrapper(new RegExp(`^${text}`)).getAttribute("aria-label")).toContain(text);
  });

  it("a queued row shows its seconds", () => {
    show([row({ phase: "queued", elapsed_s: 4.2 })]);
    expect(screen.getByText("queued 4.2s")).toBeInTheDocument();
  });

  it("shows the cache badge from 50% in prefill AND later phases, not below, not queued/waiting", () => {
    show([row({ cache_est_tokens: 50_000, cache_est_pct: 0.96 })]);
    expect(screen.getByTestId("badge-cache").textContent).toBe("96%");
    expect(screen.queryByText("≈ cached")).toBeNull();
    cleanup();
    show([row({ phase: "answering", cache_est_tokens: 90_000, cache_est_pct: 0.9 })]);
    expect(screen.getByTestId("badge-cache").textContent).toBe("90%");
    cleanup();
    show([row({ cache_est_tokens: 40_000, cache_est_pct: 0.49 })]);
    expect(screen.queryByTestId("badge-cache")).toBeNull();
    cleanup();
    show([row({ phase: "waiting", cache_est_pct: 0.9 })]);
    expect(screen.queryByTestId("badge-cache")).toBeNull();
  });

  it("dims the cache badge when the estimate was decayed", () => {
    show([row({ cache_est_pct: 0.8, est_decayed: true })]);
    expect(screen.getByTestId("badge-cache").className).toContain("opacity-50");
    cleanup();
    show([row({ cache_est_pct: 0.8, est_decayed: false })]);
    expect(screen.getByTestId("badge-cache").className).not.toContain("opacity-50");
  });

  it("queued cause: hourglass, ahead count and replica, with the full sentence", () => {
    show([
      row({
        cache_est_tokens: 36_000,
        cache_est_pct: 0.96,
        slow_cause: "queued",
        ahead_count: 3,
        ahead_fresh_tokens: 52_000,
        dp_rank: 3,
        expected_start_s: 9.2,
        prefill_tok_s: 5100,
        prefill_rate_source: "learned",
        est_accuracy: 0.94,
      }),
    ]);
    const cause = screen.getByTestId("badge-cause");
    expect(cause.textContent).toBe("3·r3");
    expect(cause.getAttribute("data-cause")).toBe("queued");
    const title = wrapper(/^prefill/).getAttribute("title") ?? "";
    expect(title).toMatch(/Queued behind 3 prompts \(≈52.0k tokens\) on replica 3\./);
    expect(title).toMatch(/≈96% of this prompt is likely cached \(estimate right 94% of the time\)/);
    expect(title).toMatch(/Expected start ~9 s at the learned 5.1k tok\/s/);
  });

  it("queued cause without a replica prints only the count", () => {
    show([row({ slow_cause: "queued", ahead_count: 1, ahead_fresh_tokens: 1000, dp_rank: null })]);
    expect(screen.getByTestId("badge-cause").textContent).toBe("1");
    expect(wrapper(/^prefill/).getAttribute("title")).toMatch(/Queued behind 1 prompt \(/);
  });

  it("evicted_likely and slow causes carry their own badge text and explanation", () => {
    show([row({ slow_cause: "evicted_likely", cache_est_pct: 0.9, prefill_slow: true, elapsed_s: 27.4 })]);
    expect(screen.getByTestId("badge-cause").textContent).toBe("evict?");
    expect(wrapper(/^prefill/).getAttribute("title")).toMatch(/probably evicted/);
    cleanup();
    show([row({ slow_cause: "slow", prefill_slow: true, elapsed_s: 27.4 })]);
    expect(screen.getByTestId("badge-cause").textContent).toBe("slow");
    const title = wrapper(/^prefill/).getAttribute("title") ?? "";
    expect(title).toMatch(/No token yet after 27s/);
    expect(title).toMatch(/engine's queue/);
  });

  it("combines cache and cause badges in one row, text always present", () => {
    show([row({ cache_est_pct: 0.96, slow_cause: "slow", prefill_slow: true, elapsed_s: 9 })]);
    expect(screen.getByTestId("badge-cache").textContent).toBe("96%");
    expect(screen.getByTestId("badge-cause").textContent).toBe("slow");
  });

  it("shows a cause badge only during prefill", () => {
    show([row({ phase: "answering", slow_cause: "slow", prefill_slow: true })]);
    expect(screen.queryByTestId("badge-cause")).toBeNull();
  });

  it("a legacy row without the new fields still renders", () => {
    const legacy = row({ phase: "prefill" }) as Partial<LiveRequestRow>;
    delete legacy.slow_cause;
    delete legacy.ahead_count;
    show([legacy as LiveRequestRow]);
    expect(screen.queryByTestId("badge-cause")).toBeNull();
  });
});

describe("Prefill model line", () => {
  afterEach(cleanup);
  const snap = (over = {}) => ({
    updated_at: 1,
    hint_tok_s: 2000,
    models: {
      m1: {
        rate_tok_s: 5100,
        rate_source: "learned" as const,
        samples: 412,
        estimate_accuracy: { samples: 90, hit_ratio: 0.94 },
        gap_buckets: [
          { label: "0-30s", from_s: 0, to_s: 30, samples: 50, hit_ratio: 0.98 },
          { label: "30-120s", from_s: 30, to_s: 120, samples: 0, hit_ratio: null },
          { label: "5-15m", from_s: 300, to_s: 900, samples: 12, hit_ratio: 0.3 },
        ],
        ...over,
      },
    },
  });

  it("shows the learned rate, accuracy and the gap table", () => {
    render(<PrefillModelLine snapshot={snap()} />);
    const t = screen.getByTestId("prefill-model").textContent ?? "";
    expect(t).toMatch(/5\.1k tok\/s learned from 412 requests/);
    expect(t).toMatch(/right 94% of the time/);
    expect(t).toMatch(/0-30s 98% · 5-15m 30%/);
  });

  it("says assumed while there are too few samples, and hides when empty", () => {
    render(
      <PrefillModelLine
        snapshot={snap({ rate_tok_s: null, rate_source: "hint", samples: 7 })}
      />,
    );
    expect(screen.getByTestId("prefill-model").textContent).toMatch(/2\.0k tok\/s assumed \(7 of 20/);
    cleanup();
    render(<PrefillModelLine snapshot={{ updated_at: null, hint_tok_s: 2000, models: {} }} />);
    expect(screen.queryByTestId("prefill-model")).toBeNull();
  });
});
