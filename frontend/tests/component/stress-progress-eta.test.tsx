// The running phase: what it says while a multi-hour run is in flight.
//
// Reported with a screenshot on 2026-09-04: a live run showed "0 probe(s)",
// "No probes recorded yet", and a bar rendered FULL. A full bar reads as
// "finished", which is the worst of the three possible wrong answers, and
// observations are not written until the run ends.
import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { StressRunProgress } from "@/components/stress/stress-run-progress";
import type { StressRun } from "@/components/stress/types";

afterEach(cleanup);

const running = (progress: unknown): StressRun =>
  ({ id: "r1", status: "running", progress } as unknown as StressRun);

describe("StressRunProgress", () => {
  it("says which phase is running and what it is probing", () => {
    render(
      <StressRunProgress
        run={running({ phase: "length", note: "probing 8,192 tokens",
                       completed: 3, total: 12, eta_s: 480 })}
      />,
    );
    expect(screen.getByTestId("stress-phase")).toHaveTextContent("length");
    expect(screen.getByTestId("stress-phase")).toHaveTextContent(
      "probing 8,192 tokens",
    );
  });

  it("shows the ETA coarsely, and scoped to the phase", () => {
    render(
      <StressRunProgress
        run={running({ phase: "length", completed: 3, total: 12, eta_s: 480 })}
      />,
    );
    // 480s -> "8 min", not "8 m 0 s": the estimate is one phase's pace over
    // its own remainder and does not deserve second-level precision.
    expect(screen.getByTestId("stress-eta")).toHaveTextContent("8 min");
    expect(screen.getByTestId("stress-eta")).toHaveTextContent("this phase");
  });

  it("omits the ETA when the phase cannot bound its own remainder", () => {
    render(
      <StressRunProgress
        run={running({ phase: "sweep", completed: 1, total: null, eta_s: null })}
      />,
    );
    expect(screen.queryByTestId("stress-eta")).toBeNull();
  });

  it("reports real progress rather than a full bar", () => {
    render(
      <StressRunProgress
        run={running({ phase: "baseline", completed: 3, total: 6 })}
      />,
    );
    expect(screen.getByTestId("stress-progress-count")).toHaveTextContent(
      "3 / 6",
    );
    expect(screen.getByRole("progressbar")).toHaveAttribute(
      "aria-valuenow",
      "50",
    );
  });

  it("announces busy, not 0%, when nothing is known yet", () => {
    // aria-valuenow is OMITTED rather than set to 0: a screen reader should
    // say "busy", not report a percentage nobody measured.
    render(<StressRunProgress run={running(null)} />);
    expect(screen.getByRole("progressbar")).not.toHaveAttribute("aria-valuenow");
  });
});
