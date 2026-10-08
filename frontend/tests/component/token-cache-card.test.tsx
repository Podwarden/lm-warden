import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, waitFor } from "@testing-library/react";
import { SWRConfig } from "swr";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";
import { CacheCard } from "@/components/tokens/detail/cache-card";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

const payload = {
  token_id: "t1", range: "24h", since_epoch: 0, models: [{
    model_id: "m", model: "qwen", backend: "vllm", requests: 10, prompt_tokens: 10_000,
    cached_tokens: 2_000, measured_requests: 10, estimated_requests: 0, reusable_tokens: 8_000,
    outcomes: { hit: 2, partial: 0, lost: 0, misrouted: 3, diverged: 5, cold: 0 },
    top_diverging: [{ token_id: "other-id", token_name: "other-key", requests: 5 }],
    by_rank: [], prefill_saved_s: 1, rate_source: "hint" }],
};

function mount(ui: React.ReactElement, res: () => Response | Promise<Response>) {
  setAccessToken("test-jwt");
  setCsrfToken("test-csrf");
  const fetchMock = vi.fn(async () => res());
  vi.stubGlobal("fetch", fetchMock);
  render(<SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>{ui}</SWRConfig>);
  return fetchMock;
}
const ok = () => new Response(JSON.stringify(payload), { status: 200, headers: { "content-type": "application/json" } });

describe("CacheCard", () => {
  it("fetches the own-lens endpoint, names its window, and leaks nothing", async () => {
    const f = mount(<CacheCard tokenId="t1" range="24h" />, ok);
    await waitFor(() => expect(screen.getByTestId("cache-hint")).toHaveTextContent(/your prompt/i));
    expect(String((f.mock.calls as unknown[][])[0][0])).toContain("/api/tokens/t1/cache?range=24h");
    const card = screen.getByTestId("token-cache-card");
    expect(card).toHaveTextContent("Prompt cache · last 24h");
    expect(screen.queryByTestId("cache-range-note")).toBeNull();
    expect(card).not.toHaveTextContent(/other-key|other-id|other replica/);
  });
  it("says so when the page selection is not supported", async () => {
    mount(<CacheCard tokenId="t1" range="24h" unsupportedSelection />, ok);
    expect(screen.getByTestId("cache-range-note")).toHaveTextContent(
      "showing the last 24h; this card supports 1h, 24h and 7d",
    );
  });
  it("shows an unavailable message on failure", async () => {
    mount(<CacheCard tokenId="t1" range="7d" />, () => new Response("{}", { status: 500 }));
    await waitFor(() => expect(screen.getByTestId("cache-error")).toHaveTextContent("Cache data unavailable."));
  });
});
