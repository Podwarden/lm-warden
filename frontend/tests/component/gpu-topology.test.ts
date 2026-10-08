// The flat row: layoutTopology puts the end node and every card on one
// line for every card count (no ring, no two-row split), and the panel
// stays short instead of growing a second row. The SVG drawing itself is
// covered in gpu-interconnect.test.tsx.

import { describe, it, expect } from "vitest";
import {
  arcLabelPoint,
  arcPath,
  bondAdjacentOrder,
  layoutTopology,
  type NvlinkPair,
} from "@/lib/gpu-topology";

describe("layoutTopology", () => {
  it("sits the hub and every card on one line, for 1, 2, 4 and 8 cards", () => {
    for (const n of [1, 2, 4, 8]) {
      const layout = layoutTopology(n);
      expect(layout.nodes, `n=${n}`).toHaveLength(n);
      const ys = [layout.hub.y, ...layout.nodes.map((p) => p.y)];
      const spread = Math.max(...ys) - Math.min(...ys);
      expect(spread, `n=${n}: ${JSON.stringify(ys)}`).toBeLessThanOrEqual(2);
    }
  });

  it("keeps the panel short: a small, constant height for every n up to 8", () => {
    const heights = [1, 2, 4, 8].map((n) => layoutTopology(n).height);
    for (const h of heights) expect(h).toBeLessThanOrEqual(170);
    expect(new Set(heights).size).toBe(1);
  });

  it("grows the row wider instead of stacking a second row", () => {
    const widths = [1, 2, 4, 8].map((n) => layoutTopology(n).width);
    for (let i = 1; i < widths.length; i++) {
      expect(widths[i]).toBeGreaterThan(widths[i - 1]);
    }
    expect(layoutTopology(1).nodes).toHaveLength(1);
    expect(layoutTopology(8).nodes).toHaveLength(8);
  });
});

describe("arcPath / arcLabelPoint", () => {
  it("joins the two box tops, and the label sits at the apex above the curve", () => {
    const from = { x: 90, y: 83 };
    const to = { x: 246, y: 85 };
    const d = arcPath(from, to);
    expect(d.startsWith("M 90.0 83.0")).toBe(true);
    expect(d.endsWith(" 246.0 85.0")).toBe(true);
    const label = arcLabelPoint(from, to);
    expect(label.x).toBe((from.x + to.x) / 2);
    expect(label.y).toBeLessThan(Math.min(from.y, to.y));
    expect(label.y).toBeGreaterThan(0);
  });
});

describe("bondAdjacentOrder", () => {
  const pair = (a: number, b: number): NvlinkPair => ({ a, b, code: "NV2" });

  it("is the identity when nothing is bonded", () => {
    expect(bondAdjacentOrder(4, [])).toEqual([0, 1, 2, 3]);
  });

  it("keeps already-adjacent bonded pairs where they are", () => {
    expect(bondAdjacentOrder(4, [pair(0, 1), pair(2, 3)])).toEqual([0, 1, 2, 3]);
  });

  it("reorders crossing bonds so each pair sits next to its partner", () => {
    expect(bondAdjacentOrder(4, [pair(0, 2), pair(1, 3)])).toEqual([0, 2, 1, 3]);
  });

  it("keeps a bond chain contiguous, in order", () => {
    expect(bondAdjacentOrder(4, [pair(0, 1), pair(1, 2)])).toEqual([0, 1, 2, 3]);
  });

  it("sits a card bonded to two partners between them", () => {
    expect(bondAdjacentOrder(3, [pair(0, 1), pair(0, 2)])).toEqual([1, 0, 2]);
  });

  it("always returns a permutation of the matrix rows", () => {
    for (const order of [
      bondAdjacentOrder(8, [pair(1, 5), pair(2, 7)]),
      bondAdjacentOrder(2, [pair(0, 1)]),
      bondAdjacentOrder(1, []),
    ]) {
      expect([...order].sort((x, y) => x - y)).toEqual(order.map((_, i) => i));
    }
  });
});
