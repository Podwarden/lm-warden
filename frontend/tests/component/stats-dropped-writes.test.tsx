import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { DroppedWritesNote } from "@/components/stats/dropped-writes-note";
import type { StatsV2Bookkeeping } from "@/lib/stats-v2";

const data = (over: Partial<StatsV2Bookkeeping> = {}): StatsV2Bookkeeping => ({
  since_epoch: 1_700_000_000,
  sites: {
    ledger_key_dropped: { count: 2, requests_lost: 5 },
    ledger_batch_dropped: { count: 1, requests_lost: 7 },
  },
  total: { count: 3, requests_lost: 12 },
  ...over,
});

afterEach(cleanup);

describe("DroppedWritesNote", () => {
  it("is hidden while nothing has been dropped", () => {
    render(<DroppedWritesNote data={data({ sites: {}, total: { count: 0, requests_lost: 0 } })} />);
    expect(screen.queryByTestId("stats-dropped-writes")).toBeNull();
  });
  it("is hidden before the response arrives", () => {
    render(<DroppedWritesNote data={undefined} />);
    expect(screen.queryByTestId("stats-dropped-writes")).toBeNull();
  });
  it("states the count and the requests lost", () => {
    render(<DroppedWritesNote data={data()} />);
    const el = screen.getByTestId("stats-dropped-writes");
    expect(el).toHaveTextContent(
      "3 stats writes dropped since start (12 requests' accounting lost)",
    );
    expect(el.getAttribute("title")).toMatch(/ledger_key_dropped: 2, 5 lost/);
  });
  it("leaves an unknown requests-lost out rather than showing 0", () => {
    render(
      <DroppedWritesNote
        data={data({
          sites: { touch_last_used: { count: 1, requests_lost: null } },
          total: { count: 1, requests_lost: null },
        })}
      />,
    );
    const el = screen.getByTestId("stats-dropped-writes");
    expect(el).toHaveTextContent(/^1 stats write dropped since start$/);
  });
});
