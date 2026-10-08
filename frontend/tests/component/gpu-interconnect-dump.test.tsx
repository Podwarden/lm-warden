// Screenshot-equivalent check for the Server Architecture panel: renders
// the four shapes the panel can take and dumps each as a standalone SVG
// under .svg-dumps/, so a layout change is visible in the file, not just
// in the test count. The style block inlines the retro-dark palette the
// app themes with; open the dumps and eyeball the row.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it, expect } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { GpuInterconnect, type InterconnectCard } from "@/components/stats/gpu-interconnect";
import type { GpuTopologyMatrix } from "@/lib/system-info";

const DUMP_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "..", ".svg-dumps");

// Retro-dark, the theme the panel ships under (src/app/globals.css).
const STYLE = `
  svg { background: #1C1409; }
  [class*="fill-chat-fg"] { fill: #FFFBEB; }
  [class*="fill-chat-dim"] { fill: #8A7C66; }
  [class*="fill-chat-muted"] { fill: #C4B49A; }
  [class*="fill-chat-accent"] { fill: #fbbf24; }
  [class*="fill-chat-negative"] { fill: #f87171; }
  [class*="fill-chat-page"] { fill: #1C1409; }
  [class*="fill-chat-surface-2"] { fill: #3E2E16; }
  [class*="stroke-chat-accent"] { stroke: #fbbf24; }
  [class*="stroke-chat-dim"] { stroke: #8A7C66; }
  [class*="stroke-chat-rule"] { stroke: #4A3820; }
  [class*="stroke-chat-negative"] { stroke: #f87171; }
  [class*="stroke-[4]"] { stroke-width: 4; }
  [class*="stroke-[2.5]"] { stroke-width: 2.5; }
  [class*="stroke-[1.5]"] { stroke-width: 1.5; }
  [class*="stroke-2"] { stroke-width: 2; }
  [class*="dasharray"] { stroke-dasharray: 5 3; }
  [class*="font-mono"] { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  [class*="font-sans"] { font-family: 'DM Sans', system-ui, sans-serif; }
  [class*="font-semibold"] { font-weight: 600; }
  [class*="text-[11.5px]"] { font-size: 11.5px; }
  [class*="text-[10px]"] { font-size: 10px; }
  [class*="text-[9.5px]"] { font-size: 9.5px; }
`;

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

interface DumpCase {
  name: string;
  kind: string;
  n: number;
  topology: GpuTopologyMatrix;
  cards: Map<number, InterconnectCard>;
}

const CASES: DumpCase[] = [
  {
    name: "n1-hub",
    kind: "hub",
    n: 1,
    topology: { indices: [0], matrix: matrix(1, () => "PHB") },
    cards: cards(1),
  },
  {
    // GPU 2 on a x8 riser, so the dashed fault arc gets drawn too.
    name: "n4-hub",
    kind: "hub",
    n: 4,
    topology: { indices: [0, 1, 2, 3], matrix: matrix(4, () => "PHB") },
    cards: cards(4, { 2: { width_current: 8 } }),
  },
  {
    name: "n4-bridged",
    kind: "bridged",
    n: 4,
    topology: {
      indices: [0, 1, 2, 3],
      matrix: matrix(4, (i, j) => ((i < 2) === (j < 2) ? "NV2" : "PHB")),
    },
    cards: cards(4),
  },
  {
    name: "n8-fabric",
    kind: "fabric",
    n: 8,
    topology: { indices: [0, 1, 2, 3, 4, 5, 6, 7], matrix: matrix(8, () => "NV12") },
    cards: cards(8),
  },
];

function dump(c: DumpCase): string {
  const html = renderToStaticMarkup(<GpuInterconnect topology={c.topology} cards={c.cards} />);
  const svg = html.slice(html.indexOf("<svg"), html.indexOf("</svg>") + "</svg>".length);
  const styled = svg.replace(/^<svg([^>]*)>/, `<svg$1><style>${STYLE}</style>`);
  return `<?xml version="1.0" encoding="UTF-8"?>\n${styled}\n`;
}

describe("Server Architecture SVG dumps", () => {
  for (const c of CASES) {
    it(`${c.name}: writes a standalone SVG with ${c.n} card(s) on one row`, () => {
      const file = join(DUMP_DIR, `${c.name}.svg`);
      mkdirSync(DUMP_DIR, { recursive: true });
      writeFileSync(file, dump(c), "utf8");

      const text = readFileSync(file, "utf8");
      expect(existsSync(file)).toBe(true);
      expect(text).toContain(`data-kind="${c.kind}"`);
      const nodeCount = text.split('data-testid="system-gpu-interconnect-node"').length - 1;
      expect(nodeCount, "one node box per card").toBe(c.n);
      expect(text).toContain('data-testid="system-gpu-interconnect-hub"');
      // One row: every node rect sits at the same y.
      const ys = [...text.matchAll(/data-testid="system-gpu-interconnect-node"[\s\S]*?<rect[^>]*y="([\d.]+)"/g)].map(
        (m) => m[1],
      );
      expect(new Set(ys).size).toBe(1);
      if (c.kind === "bridged") expect(text).toContain('data-kind="nvlink"');
    });
  }
});
