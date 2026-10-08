import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { StatCard } from "@/components/stat-card";

afterEach(() => {
  cleanup();
});

describe("StatCard", () => {
  it("renders label and value", () => {
    render(<StatCard label="VRAM" value="12,000 / 32,000" />);
    expect(screen.getByText("VRAM")).toBeInTheDocument();
    expect(screen.getByText("12,000 / 32,000")).toBeInTheDocument();
  });

  it("renders the unit when provided", () => {
    render(<StatCard label="Power" value="250" unit="W" />);
    expect(screen.getByTestId("stat-card-unit")).toHaveTextContent("W");
  });

  it("omits the unit element when no unit prop is given", () => {
    render(<StatCard label="Power" value="—" />);
    expect(screen.queryByTestId("stat-card-unit")).toBeNull();
  });

  it("renders the hint when provided", () => {
    render(
      <StatCard label="GPU util" value="80" unit="%" hint="max across GPUs" />,
    );
    expect(screen.getByTestId("stat-card-hint")).toHaveTextContent(
      "max across GPUs",
    );
  });

  it("omits the hint element when no hint prop is given", () => {
    render(<StatCard label="Tokens/s" value="14.0" />);
    expect(screen.queryByTestId("stat-card-hint")).toBeNull();
  });

  it("forwards the title attribute for tooltips", () => {
    render(
      <StatCard
        label="Power"
        value="—"
        title="No GPU reports power on this host"
      />,
    );
    const card = screen.getByTestId("stat-card");
    expect(card).toHaveAttribute(
      "title",
      "No GPU reports power on this host",
    );
  });

  it("accepts a numeric ReactNode value", () => {
    render(<StatCard label="Active" value={3} />);
    expect(screen.getByText("3")).toBeInTheDocument();
  });
});
