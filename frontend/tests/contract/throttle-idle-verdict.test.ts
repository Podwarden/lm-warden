/**
 * An idle GPU must not be reported as throttled.
 *
 * NVML sets `sw_power_cap` whenever the driver is holding clocks below the
 * maximum -- including when it does so simply because there is no work. It
 * also sets a dedicated `GpuIdle` bit saying exactly that, which the backend
 * already parses into `throttle.idle`.
 *
 * `throttleVerdict` ignored that bit, so a healthy idle card rendered as
 * "Power cap - clock held at 6%": the 6% is nothing but sm_clock/sm_clock_max
 * at idle (180 of 3090 MHz), and "held" implies something is capping it. Both
 * halves read as a hardware problem on a card that is simply doing nothing.
 *
 * Genuine faults must still surface even when the idle bit is set.
 */

import { describe, expect, it } from "vitest";

import { throttleVerdict } from "@/lib/system-info";

const NONE = {
  mask: null,
  sw_thermal: false,
  hw_thermal: false,
  sw_power_cap: false,
  hw_slowdown: false,
  hw_power_brake: false,
  idle: false,
};

describe("throttleVerdict", () => {
  it("says nothing about an idle card the driver has clocked down", () => {
    const v = throttleVerdict(
      { ...NONE, sw_power_cap: true, idle: true },
      180,
      3090,
    );
    expect(v).toBeNull();
  });

  it("still reports a real power cap when the card is not idle", () => {
    const v = throttleVerdict({ ...NONE, sw_power_cap: true }, 1800, 3090);
    expect(v).not.toBeNull();
    expect(v!.reasons).toContain("Power cap");
  });

  it("still reports a hardware fault even while the idle bit is set", () => {
    const v = throttleVerdict(
      { ...NONE, hw_thermal: true, idle: true },
      180,
      3090,
    );
    expect(v).not.toBeNull();
    expect(v!.severity).toBe("fault");
  });

  it("does not describe a normal clock as 'held'", () => {
    const v = throttleVerdict({ ...NONE, sw_power_cap: true }, 1800, 3090);
    expect(v!.text).not.toContain("held");
  });
});
