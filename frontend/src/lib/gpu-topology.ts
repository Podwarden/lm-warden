// Which card reaches which, and over what — the pure half of the
// interconnect graph on /stats. Input is the `nvidia-smi topo -m` matrix
// as the API relays it (legend codes verbatim); output is a classification
// and a layout the SVG component draws without further judgement.
//
// The legend, and what it means for the picture:
//   X              self
//   NV#            a bonded set of # NVLinks — a DIRECT card-to-card path
//   PIX PXB PHB    no direct link; routed through PCIe at increasing
//   NODE SYS       distance (bridge, host bridge, NUMA node, socket)
//
// Three renderings, and the distinction matters:
//   hub      no NVLink anywhere. Every card runs to the CPU's PCIe host
//            bridge and reaches its neighbours through it. NO card-to-card
//            edge is drawn — PHB means the pair has no direct path, and a
//            line between them would assert hardware that does not exist.
//   bridged  some pairs NVLink-bonded (thick direct edges labelled NV#);
//            every card still has its thin PCIe arc to the host bridge.
//   fabric   every pair NVLink: the cards are on an NVSwitch. Drawn as the
//            one switch they all join, not a mesh — 8 cards would be 28
//            lines and 16 would be 120, and the switch IS the hardware.
//
// The picture is one horizontal row: the host (CPU, or the NVSwitch on a
// fabric) as the left-hand end node, the cards to its right, each joined
// to the end by the arc above the row. One row for every n — eight cards
// make a wider row, never two short stacked ones; the container scrolls
// when the row outgrows it.

export type TopologyKind = "hub" | "bridged" | "fabric";

export interface NvlinkPair {
  /** Row/column positions into the matrix (NOT GPU indices). */
  a: number;
  b: number;
  /** The legend code, e.g. "NV2". */
  code: string;
}

export interface TopologyClass {
  kind: TopologyKind;
  n: number;
  /** Direct NVLink bonds, each pair once (a < b). Empty for `hub`. */
  nvPairs: NvlinkPair[];
  /** For `fabric`: the code every pair reports (e.g. "NV12"). */
  fabricCode: string | null;
}

const NV_CODE = /^NV\d*$/;

export function isNvlinkCode(code: string): boolean {
  return NV_CODE.test(code.trim());
}

export function classifyTopology(matrix: string[][]): TopologyClass {
  const n = matrix.length;
  const nvPairs: NvlinkPair[] = [];
  let offDiagonal = 0;
  let nvCells = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      offDiagonal++;
      const code = (matrix[i]?.[j] ?? "").trim();
      if (!isNvlinkCode(code)) continue;
      nvCells++;
      if (j > i) nvPairs.push({ a: i, b: j, code });
    }
  }
  // Two bonded cards are a pair, not a switch: a fabric needs three or
  // more cards with every pair linked.
  if (n > 2 && offDiagonal > 0 && nvCells === offDiagonal) {
    return { kind: "fabric", n, nvPairs, fabricCode: nvPairs[0]?.code ?? null };
  }
  if (nvPairs.length > 0) {
    return { kind: "bridged", n, nvPairs, fabricCode: null };
  }
  return { kind: "hub", n, nvPairs: [], fabricCode: null };
}

export interface Point {
  x: number;
  y: number;
}

export interface TopologyLayout {
  width: number;
  height: number;
  /** The end node (CPU, or the NVSwitch on a fabric): left end of the row. */
  hub: Point;
  /** One entry per matrix row, in matrix order. All share the row's y. */
  nodes: Point[];
}

export const NODE_W = 92;
export const NODE_H = 38;
export const HUB_W = 132;
export const HUB_H = 42;

const MARGIN = 24;
const HUB_GAP = 44;
const CARD_GAP = 30;
const ROW_Y = 104;
export const ROW_HEIGHT = 150;
/** How high the arcs rise: control points on this line above the canvas. */
const ARC_TOP = 10;

/** One horizontal row: the end node at the left end, the cards to its
 *  right, all on the same line. No upper bound on n: 16-GPU hosts exist
 *  and MIG can add instances; the container scrolls horizontally when the
 *  row outgrows it. */
export function layoutTopology(n: number): TopologyLayout {
  const width = MARGIN * 2 + HUB_W + HUB_GAP + n * NODE_W + (n - 1) * CARD_GAP;
  const firstNodeX = MARGIN + HUB_W + HUB_GAP;
  const nodes: Point[] = [];
  for (let i = 0; i < n; i++) {
    nodes.push({ x: firstNodeX + i * (NODE_W + CARD_GAP) + NODE_W / 2, y: ROW_Y });
  }
  return { width, height: ROW_HEIGHT, hub: { x: MARGIN + HUB_W / 2, y: ROW_Y }, nodes };
}

/** The arc above the row that joins a box to the end node: it leaves each
 *  top edge vertically and meets the other the same way, so the lane label
 *  at its apex never sits on a sloping line. */
export function arcPath(from: Point, to: Point): string {
  return (
    `M ${from.x.toFixed(1)} ${from.y.toFixed(1)} ` +
    `C ${from.x.toFixed(1)} ${ARC_TOP}, ${to.x.toFixed(1)} ${ARC_TOP}, ` +
    `${to.x.toFixed(1)} ${to.y.toFixed(1)}`
  );
}

/** The apex of that arc, nudged clear of the curve — where the lane label
 *  sits. (The cubic's midpoint: (P0 + 3·P1 + 3·P2 + P3) / 8.) */
export function arcLabelPoint(from: Point, to: Point): Point {
  return {
    x: (from.x + to.x) / 2,
    y: (from.y + to.y + 6 * ARC_TOP) / 8 - 5,
  };
}

/** The order the cards sit in along the row: bonded pairs (and chains of
 *  bonds) next to their partners, so a thick NVLink edge only has to cross
 *  the gap between two boxes. Each bond component is a path or cycle — a
 *  point-to-point bridge links two cards, so no card has more than two
 *  partners — and it is walked from an end to stay contiguous; a card
 *  bonded to two partners lands between them. Unbonded cards follow in
 *  index order. The result is always a permutation of `0..n-1`. */
export function bondAdjacentOrder(n: number, nvPairs: NvlinkPair[]): number[] {
  const adj: number[][] = Array.from({ length: n }, () => []);
  for (const { a, b } of nvPairs) {
    adj[a].push(b);
    adj[b].push(a);
  }
  const used = new Set<number>();
  const order: number[] = [];
  for (let i = 0; i < n; i++) {
    if (used.has(i)) continue;
    // This card's whole bond component (itself, when nothing is bonded).
    const component: number[] = [];
    const stack = [i];
    while (stack.length > 0) {
      const node = stack.pop()!;
      if (used.has(node) || component.includes(node)) continue;
      component.push(node);
      for (const nb of adj[node]) {
        if (!used.has(nb) && !component.includes(nb)) stack.push(nb);
      }
    }
    // Walk from an end of the component so a chain keeps its true order;
    // a cycle has no ends, so walk from its lowest card.
    const ends = component.filter((c) => adj[c].length <= 1);
    let cur = (ends.length > 0 ? ends : component).sort((a, b) => a - b)[0];
    while (!used.has(cur)) {
      used.add(cur);
      order.push(cur);
      const next = adj[cur].find((nb) => !used.has(nb));
      if (next === undefined) break;
      cur = next;
    }
  }
  return order;
}
