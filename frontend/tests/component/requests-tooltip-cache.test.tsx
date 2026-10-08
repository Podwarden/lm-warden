import { describe, it, expect } from "vitest";
import { cacheLine, divergeLine } from "@/components/stats/requests-chart";

const row = (x: object) => ({ prompt_tokens: 10_200, ...x }) as never;

describe("cacheLine", () => {
  it("measured", () => expect(cacheLine(row({ cache_outcome: "hit", cached_tokens: 9_800, cached_source: "engine" }))).toBe("hit · 9.8k / 10.2k"));
  it("estimated", () => expect(cacheLine(row({ cache_outcome: "partial", cached_ttft_est_tokens: 5_000, cached_source: "estimated" }))).toBe("partial · ≈5.0k / 10.2k"));
  it("unknown is null", () => expect(cacheLine(row({}))).toBeNull());
  it("uses the UI labels, not the raw keys", () => {
    expect(cacheLine(row({ cache_outcome: "misrouted", cached_tokens: 0, cached_source: "engine" }))).toBe("other replica · 0 / 10.2k");
    expect(cacheLine(row({ cache_outcome: "diverged", cached_tokens: 0, cached_source: "engine" }))).toBe("prefix changed · 0 / 10.2k");
    expect(cacheLine(row({ cache_outcome: "cold", cached_tokens: 0, cached_source: "engine" }))).toBe("nothing to reuse · 0 / 10.2k");
  });
});

describe("divergeLine", () => {
  it("names the message, counted from 1", () => expect(divergeLine(row({ diverged_at: 3 }))).toBe("prefix broke at message #4"));
  it("the tools element", () => expect(divergeLine(row({ diverged_at: -1 }))).toBe("prefix broke at tools"));
  it("no evidence is null", () => {
    expect(divergeLine(row({ diverged_at: null }))).toBeNull();
    expect(divergeLine(row({}))).toBeNull();
  });
});
