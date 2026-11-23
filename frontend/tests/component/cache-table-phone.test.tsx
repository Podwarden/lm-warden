// Phone layout of the Model files table (Models > Model files). Below the
// `sm` breakpoint (640 px) the table must not scroll sideways or spill past
// its card: the header row hides, each repo becomes a stacked card (grid),
// the repo id wraps anywhere, and Size / Last used / Owned by carry their own
// labels (the header that named them is gone). jsdom has no layout, so this
// pins the class contract; the 375 px headless check measures the result.
import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup, within } from "@testing-library/react";
import { CacheTable, type CachedRepoView } from "@/components/stats/cache-table";

const row = (repo: string, extra: Partial<CachedRepoView> = {}): CachedRepoView => ({
  repo,
  path: `/hf/models--${repo.replace("/", "--")}`,
  size_bytes: 16_400_000_000,
  last_modified: Date.now() / 1000 - 3600,
  matched_models: [],
  ...extra,
});

const classes = (el: Element | null) => (el?.getAttribute("class") ?? "").split(/\s+/);

describe("CacheTable — phone layout", () => {
  afterEach(cleanup);

  it("stacks each repo into a card below sm, with no sideways scroll", () => {
    render(
      <CacheTable
        repos={[
          row("TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ-very-long-repository-name"),
          row("facebook/opt-125m", {
            matched_models: [{ id: "m1", served_model_name: "opt", status: "loaded" } as never],
          }),
        ]}
        onMutate={() => {}}
      />,
    );
    const table = screen.getByRole("table");
    // The wrapper only scrolls from sm up; on a phone nothing overflows.
    expect(classes(table.parentElement)).not.toContain("overflow-x-auto");
    expect(classes(table.parentElement)).toContain("sm:overflow-x-auto");
    expect(classes(table)).toContain("max-sm:block");
    expect(classes(table.querySelector("thead"))).toContain("max-sm:hidden");
    expect(classes(table.querySelector("tbody"))).toContain("max-sm:block");

    const rows = Array.from(table.querySelectorAll("tbody > tr")) as HTMLElement[];
    expect(rows).toHaveLength(2);
    for (const tr of rows) {
      expect(classes(tr)).toContain("max-sm:grid");
      const cells = within(tr).getAllByRole("cell");
      expect(cells).toHaveLength(5);
      // Repo id: full width, wraps anywhere.
      expect(classes(cells[0])).toContain("max-sm:col-span-2");
      expect(classes(cells[0])).toContain("break-all");
      // The hidden header's names move onto the cells.
      expect(cells.slice(1, 4).map((c) => c.getAttribute("data-label"))).toEqual([
        "Size",
        "Last used",
        "Owned by",
      ]);
      for (const c of cells.slice(1, 4)) {
        expect(classes(c)).toContain("max-sm:before:content-[attr(data-label)]");
      }
      // Owned by and the actions take the full card width.
      expect(classes(cells[3])).toContain("max-sm:col-span-2");
      expect(classes(cells[4])).toContain("max-sm:col-span-2");
    }
  });
});
