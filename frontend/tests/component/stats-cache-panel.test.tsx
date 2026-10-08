import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { CachePanel } from "@/components/stats/cache-panel";
import type { CacheResponse, CacheSummary } from "@/lib/cache-obs";

const s: CacheSummary = {
  model_id: "m", model: "qwen", backend: "llamacpp", requests: 100, prompt_tokens: 100_000,
  cached_tokens: 40_000, known_prompt_tokens: 100_000, measured_requests: 100, estimated_requests: 0,
  reused_tokens: 40_000, reusable_tokens: 50_000,
  outcomes: { hit: 60, partial: 0, lost: 30, misrouted: 0, diverged: 0, cold: 10 },
  by_rank: [], prefill_saved_s: 125, rate_source: "learned", top_diverging: [],
};
const data = (x: Partial<CacheSummary> = {}): CacheResponse => ({ range: "1h", since_epoch: 0, models: [{ ...s, ...x }] });

afterEach(cleanup);

describe("CachePanel", () => {
  it("shows the three numbers and the engine-specific hint", () => {
    render(<CachePanel data={data()} />);
    expect(screen.getByTestId("cache-from")).toHaveTextContent("40%");
    expect(screen.getByTestId("cache-efficiency")).toHaveTextContent("80%");
    expect(screen.getByTestId("cache-saved")).toHaveTextContent(/2m/);
    expect(screen.getByTestId("cache-hint")).toHaveTextContent(/slot/);
  });
  it("marks estimates", () => {
    render(<CachePanel data={data({ estimated_requests: 100, measured_requests: 0, backend: "mlx" })} />);
    expect(screen.getByTestId("cache-from")).toHaveTextContent("≈40%");
    expect(screen.getByTestId("cache-from")).toHaveAttribute("title", expect.stringMatching(/estimated from TTFT/));
  });
  it("renders dashes, not zeros, when nothing is known", () => {
    render(<CachePanel data={data({ measured_requests: 0, estimated_requests: 0, cached_tokens: 0, reusable_tokens: 0, prefill_saved_s: null })} />);
    expect(screen.getByTestId("cache-from")).toHaveTextContent("—");
    expect(screen.getByTestId("cache-efficiency")).toHaveTextContent("—");
  });
  it("prefill saved is a dash, not 0s, when nothing is known", () => {
    render(<CachePanel data={data({ measured_requests: 0, estimated_requests: 0, cached_tokens: 0, prefill_saved_s: 0 })} />);
    expect(screen.getByTestId("cache-saved")).toHaveTextContent("—");
  });
  it("marks estimated prefill saved like the other numbers", () => {
    render(<CachePanel data={data({ estimated_requests: 100, measured_requests: 0, backend: "mlx" })} />);
    const el = screen.getByTestId("cache-saved");
    expect(el).toHaveTextContent("≈2m");
    expect(el).toHaveAttribute("title", expect.stringMatching(/estimated from TTFT/));
    expect(el.className).toMatch(/dotted/);
  });
  it("shows a legend for outcomes with a count", () => {
    render(<CachePanel data={data()} />);
    const lg = screen.getByTestId("cache-legend");
    expect(lg).toHaveTextContent("hit 60");
    expect(lg).toHaveTextContent("lost 30");
    expect(lg).not.toHaveTextContent("partial");
  });
  it("marks prefill saved as assumed while the rate is only the configured hint", () => {
    render(<CachePanel data={data({ rate_source: "hint" })} />);
    const el = screen.getByTestId("cache-saved");
    expect(el).toHaveTextContent("≈2m");
    expect(el).toHaveAttribute(
      "title",
      "assumed prefill rate (VW_PREFILL_TOK_S_HINT); the warden has not learned this box's rate yet",
    );
    expect(screen.getByTestId("cache-from")).toHaveTextContent(/^40%$/);
  });
  it("a learned rate shows prefill saved plain", () => {
    render(<CachePanel data={data()} />);
    expect(screen.getByTestId("cache-saved")).toHaveTextContent(/^2m$/);
  });
  it("states coverage when some requests have no cache figures", () => {
    render(<CachePanel data={data({ requests: 120, measured_requests: 90, estimated_requests: 10 })} />);
    expect(screen.getByTestId("cache-coverage")).toHaveTextContent("cache figures from 100 of 120 requests");
  });
  it("no coverage line when every request has figures", () => {
    render(<CachePanel data={data()} />);
    expect(screen.queryByTestId("cache-coverage")).toBeNull();
  });
  it("notes requests whose prefix broke deep inside the prompt", () => {
    render(<CachePanel data={data({ deep_diverged: 7 })} />);
    expect(screen.getByTestId("cache-deep-diverged")).toHaveTextContent("7 hit or partial requests broke their prefix mid-prompt");
  });
  it("no deep-divergence line at 0 or when absent", () => {
    render(<CachePanel data={data({ deep_diverged: 0 })} />);
    expect(screen.queryByTestId("cache-deep-diverged")).toBeNull();
    cleanup();
    render(<CachePanel data={data()} />);
    expect(screen.queryByTestId("cache-deep-diverged")).toBeNull();
  });
});
