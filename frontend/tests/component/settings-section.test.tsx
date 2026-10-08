import { describe, it, expect, vi, afterEach } from "vitest";
import React from "react";
import { render, screen, fireEvent, cleanup, within } from "@testing-library/react";
import { SettingsSection } from "@/components/settings-section";

afterEach(cleanup);

// The global settings tabs and today's model settings page render
// SettingsSection with no new props. That rendering must not move.
describe("SettingsSection — default rendering is unchanged", () => {
  it("keeps the legacy classes when no variant is given", () => {
    render(
      <SettingsSection title="Memory" testId="section-memory">
        <p>body</p>
      </SettingsSection>,
    );
    const section = screen.getByTestId("section-memory");
    expect(section.tagName).toBe("SECTION");
    expect(section.className).toBe(
      "border-t border-slate-700/60 pt-4 first:border-t-0 first:pt-0",
    );
    const toggle = screen.getByTestId("section-memory-toggle");
    expect(toggle.className).toBe(
      "flex items-center gap-2 text-left focus:outline-none focus:ring-1 focus:ring-amber-600 rounded-sm",
    );
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(toggle).not.toHaveAttribute("aria-describedby");
    expect(screen.getByTestId("section-memory-body").className).toBe("mt-3 space-y-4");
    // No heading, no aria-labelledby, no summary/dirty nodes.
    expect(screen.queryByRole("heading")).toBeNull();
    expect(section).not.toHaveAttribute("aria-labelledby");
    expect(screen.queryByTestId("section-memory-summary")).toBeNull();
    expect(screen.queryByTestId("section-memory-dirty")).toBeNull();
  });

  it("still toggles uncontrolled and honours defaultOpen", () => {
    render(
      <SettingsSection title="Advanced" defaultOpen={false}>
        <p>body</p>
      </SettingsSection>,
    );
    const toggle = screen.getByRole("button", { name: "Advanced" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByTestId("settings-section-advanced-body")).not.toBeVisible();
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByTestId("settings-section-advanced-body")).toBeVisible();
  });
});

describe("SettingsSection — summary", () => {
  it("is shown only while collapsed and describes the toggle", () => {
    render(
      <SettingsSection
        title="Memory"
        testId="section-memory"
        defaultOpen={false}
        summary="0.90 of VRAM · 32,768 tokens"
      >
        <p>body</p>
      </SettingsSection>,
    );
    const toggle = screen.getByRole("button", { name: "Memory" });
    const summary = screen.getByTestId("section-memory-summary");
    expect(summary).toHaveTextContent("0.90 of VRAM · 32,768 tokens");
    expect(toggle).toHaveAccessibleDescription("0.90 of VRAM · 32,768 tokens");
    // The summary is a description, never part of the name.
    expect(toggle).toHaveAccessibleName("Memory");

    fireEvent.click(toggle);
    expect(screen.queryByTestId("section-memory-summary")).toBeNull();
    expect(toggle).not.toHaveAttribute("aria-describedby");
  });
});

describe("SettingsSection — dirty count", () => {
  it("renders 'N unsaved' outside the toggle's accessible name", () => {
    render(
      <SettingsSection title="Memory" testId="section-memory" dirtyCount={2}>
        <p>body</p>
      </SettingsSection>,
    );
    expect(screen.getByTestId("section-memory-dirty")).toHaveTextContent("2 unsaved");
    const toggle = screen.getByRole("button", { name: "Memory" });
    expect(toggle).toHaveAccessibleName("Memory");
    expect(toggle).not.toContainElement(screen.getByTestId("section-memory-dirty"));
    // The page's single Save button must stay the only /save/i button.
    expect(screen.queryByRole("button", { name: /save/i })).toBeNull();
  });

  it("renders nothing when the count is zero", () => {
    render(
      <SettingsSection title="Memory" testId="section-memory" dirtyCount={0}>
        <p>body</p>
      </SettingsSection>,
    );
    expect(screen.queryByTestId("section-memory-dirty")).toBeNull();
  });
});

describe("SettingsSection — controlled open", () => {
  it("follows the open prop and reports toggles through onOpenChange", () => {
    const onOpenChange = vi.fn();
    const { rerender } = render(
      <SettingsSection title="Layout" testId="section-compute" open={false} onOpenChange={onOpenChange}>
        <p>body</p>
      </SettingsSection>,
    );
    const toggle = screen.getByRole("button", { name: "Layout" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");

    fireEvent.click(toggle);
    expect(onOpenChange).toHaveBeenCalledWith(true);
    // Controlled: the parent has not changed `open`, so nothing moves.
    expect(toggle).toHaveAttribute("aria-expanded", "false");

    // Parent force-expands (e.g. a preset touched a key in this section).
    rerender(
      <SettingsSection title="Layout" testId="section-compute" open onOpenChange={onOpenChange}>
        <p>body</p>
      </SettingsSection>,
    );
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByTestId("section-compute-body")).toBeVisible();
  });

  it("calls onOpenChange in uncontrolled mode too", () => {
    const onOpenChange = vi.fn();
    render(
      <SettingsSection title="Layout" onOpenChange={onOpenChange}>
        <p>body</p>
      </SettingsSection>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Layout" }));
    expect(onOpenChange).toHaveBeenCalledWith(false);
    expect(screen.getByRole("button", { name: "Layout" })).toHaveAttribute("aria-expanded", "false");
  });
});

describe("SettingsSection — card variant with an h3 heading", () => {
  function renderCard(extra: Partial<React.ComponentProps<typeof SettingsSection>> = {}) {
    return render(
      <SettingsSection
        title="Memory"
        displayTitle="Memory & context"
        as="h3"
        variant="card"
        testId="section-memory"
        {...extra}
      >
        <p>body</p>
      </SettingsSection>,
    );
  }

  it("renders the visible heading as an h3 that labels the section", () => {
    renderCard();
    const h3 = screen.getByRole("heading", { level: 3 });
    expect(h3).toHaveTextContent("Memory & context");
    const section = screen.getByTestId("section-memory");
    expect(section).toHaveAttribute("aria-labelledby", h3.id);
  });

  it("keeps aria-label as the toggle's name so /memory/i lookups resolve", () => {
    renderCard();
    const section = screen.getByTestId("section-memory");
    const toggle = within(section).getByRole("button", { name: /memory/i });
    expect(toggle).toHaveAccessibleName("Memory");
    expect(toggle).toHaveAttribute("aria-controls", screen.getByTestId("section-memory-body").id);
  });

  it("uses token classes, not slate, for the card chrome", () => {
    renderCard({ summary: "x", dirtyCount: 1, defaultOpen: false });
    const section = screen.getByTestId("section-memory");
    expect(section.className).toContain("rounded-xl");
    expect(section.className).toContain("border-vw-rule-soft/70");
    expect(section.className).toContain("bg-chat-surface/55");
    expect(section.outerHTML).not.toMatch(/slate-/);
  });

  it("collapsing never moves focus", () => {
    renderCard();
    const toggle = screen.getByRole("button", { name: "Memory" });
    toggle.focus();
    fireEvent.click(toggle);
    expect(document.activeElement).toBe(toggle);
  });
});
