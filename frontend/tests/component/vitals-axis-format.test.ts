import { describe, expect, it } from "vitest";
import { formatAxisMiB } from "@/components/models/load-progress/vitals-tiles";

describe("formatAxisMiB", () => {
  it("keeps GiB labels short below a thousand GiB", () => {
    expect(formatAxisMiB(0)).toBe("0G");
    expect(formatAxisMiB(86 * 1024)).toBe("86G");
    expect(formatAxisMiB(860 * 1024)).toBe("860G");
  });
  it("switches to TiB so 8-GPU totals fit the axis", () => {
    expect(formatAxisMiB(2149 * 1024)).toBe("2.1T");
    expect(formatAxisMiB(1720 * 1024)).toBe("1.7T");
  });
});
