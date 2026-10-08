// Where a 24h reference label sits, and why it is computed rather than fixed.
//
// Two real failures on the deployed page:
//   * GPU utilisation pinned at 100% put the peak label ABOVE a line already
//     at the top of the scale, so the panel clipped it;
//   * power draw with a 511 W median under a 555 W peak drew both labels
//     against the right edge, where they overlapped.

import { describe, it, expect } from "vitest";
import { refLabelPosition } from "@/components/stats/v2-charts";

describe("24h reference label placement", () => {
  it("drops the label below a line at the top of the scale, so it is not clipped", () => {
    // GPU utilisation: peak 100 in a domain of 100.
    expect(refLabelPosition(100, 100, "Right")).toBe("insideBottomRight");
  });

  it("keeps the label above a line low in the scale, so it is not clipped at the floor", () => {
    expect(refLabelPosition(5, 100, "Left")).toBe("insideTopLeft");
  });

  it("puts peak and median on OPPOSITE edges, so close values cannot overlap", () => {
    // The power-draw case: 511 W median, 555 W peak, domain ceiling 599.
    const peak = refLabelPosition(555, 599, "Right");
    const median = refLabelPosition(511, 599, "Left");
    expect(peak.endsWith("Right")).toBe(true);
    expect(median.endsWith("Left")).toBe(true);
    expect(peak).not.toBe(median);
  });

  it("survives a zero or missing ceiling without dividing by it", () => {
    expect(refLabelPosition(42, 0, "Right")).toBe("insideTopRight");
  });

  it("flips at the threshold, not at the exact top, so a near-top line is also safe", () => {
    // 0.9 of the domain is still 'high' — the label would otherwise sit half
    // outside the plot.
    expect(refLabelPosition(90, 100, "Right")).toBe("insideBottomRight");
    expect(refLabelPosition(80, 100, "Right")).toBe("insideTopRight");
  });
});
