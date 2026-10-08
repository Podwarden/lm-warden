// The interconnect graph: three renderings from a `topo -m` matrix, and
// the distinction between them. Rendered directly with known props — the
// section test covers the wiring from /api/system/gpus.
//
// The 4-card and 2-card matrices are real; the bridged-pair and switch
// fabric matrices are the shapes `topo -m` documents for hardware we do not
// have. Card names never appear here — only indices and lane widths.

import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup, within } from "@testing-library/react";
import { GpuInterconnect, type InterconnectCard } from "@/components/stats/gpu-interconnect";
import { classifyTopology, layoutTopology } from "@/lib/gpu-topology";

function cards(n: number, overrides: Record<number, Partial<InterconnectCard>> = {}) {
  const m = new Map<number, InterconnectCard>();
  for (let i = 0; i < n; i++) {
    m.set(i, { memory_total_mib: 16376, width_current: 16, width_max: 16, ...overrides[i] });
  }
  return m;
}

function matrix(n: number, cell: (i: number, j: number) => string): string[][] {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? "X" : cell(i, j))),
  );
}

const HUB_4 = { indices: [0, 1, 2, 3], matrix: matrix(4, () => "PHB") };
const PAIRS_4 = {
  indices: [0, 1, 2, 3],
  matrix: matrix(4, (i, j) => ((i < 2) === (j < 2) ? "NV2" : "PHB")),
};
const FABRIC_8 = { indices: [0, 1, 2, 3, 4, 5, 6, 7], matrix: matrix(8, () => "NV12") };

function edges() {
  return screen.getAllByTestId("system-gpu-interconnect-edge");
}
function kinds() {
  return edges().map((e) => e.getAttribute("data-kind"));
}

describe("GpuInterconnect", () => {
  afterEach(cleanup);

  it("is titled 'Server Architecture', and says so to the screen reader", () => {
    render(<GpuInterconnect topology={HUB_4} cards={cards(4)} />);
    const section = screen.getByTestId("system-gpu-interconnect");
    expect(section.querySelector("h3")?.textContent).toBe("Server Architecture");
    expect(section.getAttribute("aria-label")).toBe("Server Architecture");
    expect(section.getAttribute("aria-label") ?? "").not.toContain("How the cards reach each other");
  });

  it("no NVLink: a flat row with a lane-labelled arc per card and NO card-to-card edge", () => {
    // GPU 2 is on a x8 riser.
    render(<GpuInterconnect topology={HUB_4} cards={cards(4, { 2: { width_current: 8 } })} />);
    const svg = screen.getByTestId("system-gpu-interconnect-svg");
    expect(svg.getAttribute("data-kind")).toBe("hub");

    // Exactly four edges, every one to the CPU. PHB means the pair has no
    // direct path, so a GPU-to-GPU line would assert hardware that does
    // not exist.
    expect(edges()).toHaveLength(4);
    for (const e of edges()) expect(e.getAttribute("data-to")).toBe("cpu");
    expect(kinds().filter((k) => k === "nvlink")).toHaveLength(0);

    const hub = screen.getByTestId("system-gpu-interconnect-hub");
    expect(hub.getAttribute("data-kind")).toBe("cpu");
    expect(hub.textContent).toContain("CPU");
    expect(hub.textContent).toContain("PCIe host bridge");

    // Each spoke is labelled with that card's negotiated lane width, and
    // the narrow card's spoke is the dashed fault-coloured one.
    const lanes = screen.getAllByTestId("system-gpu-interconnect-lanes");
    expect(lanes.map((l) => l.textContent)).toEqual(["x16", "x16", "x8", "x16"]);
    const narrow = edges().filter((e) => e.getAttribute("data-kind") === "spoke-narrow");
    expect(narrow).toHaveLength(1);
    expect(narrow[0].getAttribute("data-from")).toBe("2");
    expect(narrow[0].getAttribute("class")).toContain("negative");
    expect(narrow[0].getAttribute("class")).toContain("dasharray");
    const full = edges().filter((e) => e.getAttribute("data-kind") === "spoke");
    expect(full).toHaveLength(3);
    expect(full[0].getAttribute("class")).not.toContain("negative");

    // Four nodes, one per card, with the card's VRAM.
    const nodes = screen.getAllByTestId("system-gpu-interconnect-node");
    expect(nodes.map((n) => n.getAttribute("data-gpu-index"))).toEqual(["0", "1", "2", "3"]);
    expect(nodes[0].textContent).toContain("GPU 0");
    expect(nodes[0].textContent).toContain("16 GiB");

    expect(screen.getByTestId("system-gpu-interconnect-legend").textContent).toContain(
      "No card here has a link to another",
    );
  });

  it("bridged pairs: thick NV# edges card-to-card plus a thin PCIe spoke for every card", () => {
    render(<GpuInterconnect topology={PAIRS_4} cards={cards(4)} />);
    expect(screen.getByTestId("system-gpu-interconnect-svg").getAttribute("data-kind")).toBe("bridged");

    // 4 spokes + 2 bonds.
    expect(kinds().filter((k) => k === "spoke")).toHaveLength(4);
    const bonds = edges().filter((e) => e.getAttribute("data-kind") === "nvlink");
    expect(bonds).toHaveLength(2);
    expect(bonds.map((b) => [b.getAttribute("data-from"), b.getAttribute("data-to")])).toEqual([
      ["0", "1"],
      ["2", "3"],
    ]);
    // Bonds are thick and accent-coloured; spokes stay thin.
    expect(bonds[0].getAttribute("class")).toContain("stroke-[4]");
    expect(bonds[0].getAttribute("class")).toContain("accent");
    // The bond is labelled with its code.
    const svg = screen.getByTestId("system-gpu-interconnect-svg");
    expect(within(svg).getAllByText("NV2")).toHaveLength(2);
    // Still a CPU hub — the unpaired pairs reach each other through it.
    expect(screen.getByTestId("system-gpu-interconnect-hub").getAttribute("data-kind")).toBe("cpu");
    expect(screen.getByTestId("system-gpu-interconnect-legend").textContent).toContain(
      "Thick lines are direct NVLink bonds",
    );
  });

  it("full fabric: one NVSwitch node they all join, never a mesh", () => {
    render(<GpuInterconnect topology={FABRIC_8} cards={cards(8)} />);
    expect(screen.getByTestId("system-gpu-interconnect-svg").getAttribute("data-kind")).toBe("fabric");

    // Eight thick spokes to the switch, not 28 chords.
    expect(edges()).toHaveLength(8);
    expect(new Set(kinds())).toEqual(new Set(["fabric-spoke"]));
    for (const e of edges()) expect(e.getAttribute("data-to")).toBe("nvswitch");
    expect(kinds().filter((k) => k === "nvlink")).toHaveLength(0);

    const hub = screen.getByTestId("system-gpu-interconnect-hub");
    expect(hub.getAttribute("data-kind")).toBe("nvswitch");
    expect(hub.textContent).toContain("NVSwitch");
    expect(hub.textContent).toContain("NV12 to every card");
    // No lane labels on a fabric: the spokes are NVLink, not PCIe.
    expect(screen.queryAllByTestId("system-gpu-interconnect-lanes")).toHaveLength(0);
    expect(screen.getByTestId("system-gpu-interconnect-legend").textContent).toContain(
      "rather than 28 separate lines",
    );
  });

  it("two bonded cards are a bridged pair, not a switch", () => {
    render(
      <GpuInterconnect
        topology={{ indices: [0, 1], matrix: [["X", "NV4"], ["NV4", "X"]] }}
        cards={cards(2)}
      />,
    );
    expect(screen.getByTestId("system-gpu-interconnect-svg").getAttribute("data-kind")).toBe("bridged");
    expect(kinds().filter((k) => k === "nvlink")).toHaveLength(1);
    expect(screen.getByTestId("system-gpu-interconnect-hub").getAttribute("data-kind")).toBe("cpu");
  });

  it("a lane width the driver did not report is said, not drawn as x0", () => {
    render(
      <GpuInterconnect
        topology={{ indices: [0, 1], matrix: [["X", "PHB"], ["PHB", "X"]] }}
        cards={cards(2, { 1: { width_current: null, width_max: null } })}
      />,
    );
    const lanes = screen.getAllByTestId("system-gpu-interconnect-lanes");
    expect(lanes[1].textContent).toBe("lanes not reported");
    expect(lanes[1].textContent).not.toMatch(/x0/);
    // Unknown is not "narrow".
    expect(kinds()).toEqual(["spoke", "spoke"]);
  });

  it("scrolls rather than clips past four cards: one wider row, never two", () => {
    render(<GpuInterconnect topology={FABRIC_8} cards={cards(8)} />);
    expect(screen.getByTestId("system-gpu-interconnect-scroll").className).toContain("overflow-x-auto");
    const layout = layoutTopology(8);
    // 8 cards make one wider row, not two short stacked ones.
    expect(new Set(layout.nodes.map((n) => n.y)).size).toBe(1);
    expect(layout.nodes[0].y).toBe(layout.hub.y);
    expect(layout.width).toBeGreaterThan(layoutTopology(4).width);
  });

  it("says the interconnect is not reported when there is no matrix", () => {
    render(<GpuInterconnect topology={null} cards={cards(2)} />);
    expect(screen.getByTestId("system-gpu-interconnect-not-reported").textContent).toBe(
      "interconnect not reported",
    );
    render(<GpuInterconnect topology={null} cards={cards(2)} unavailableReason="nvidia-smi unavailable" />);
    expect(screen.getAllByTestId("system-gpu-interconnect-not-reported")[1].textContent).toBe(
      "interconnect not reported (nvidia-smi unavailable)",
    );
  });
});

describe("classifyTopology", () => {
  it("names the three shapes and does not cap the card count", () => {
    expect(classifyTopology(HUB_4.matrix).kind).toBe("hub");
    expect(classifyTopology(PAIRS_4.matrix)).toMatchObject({
      kind: "bridged",
      nvPairs: [
        { a: 0, b: 1, code: "NV2" },
        { a: 2, b: 3, code: "NV2" },
      ],
    });
    expect(classifyTopology(FABRIC_8.matrix)).toMatchObject({ kind: "fabric", fabricCode: "NV12" });
    // Every PCIe distance code means the same thing: no direct link.
    for (const code of ["PIX", "PXB", "PHB", "NODE", "SYS"]) {
      expect(classifyTopology(matrix(3, () => code)).kind).toBe("hub");
    }
    // 16 cards on a switch: still one node, 16 spokes.
    const sixteen = classifyTopology(matrix(16, () => "NV18"));
    expect(sixteen.kind).toBe("fabric");
    expect(sixteen.n).toBe(16);
    expect(layoutTopology(16).nodes).toHaveLength(16);
  });

  it("a fabric with one unlinked card is bridged, not a fabric", () => {
    const m = matrix(4, (i, j) => (i === 3 || j === 3 ? "SYS" : "NV12"));
    expect(classifyTopology(m).kind).toBe("bridged");
    expect(classifyTopology(m).nvPairs).toHaveLength(3);
  });
});
