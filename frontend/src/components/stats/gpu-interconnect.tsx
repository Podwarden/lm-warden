// "Server Architecture" — the interconnect graph under the GPU cards on
// /stats. A relationship BETWEEN cards, so unlike the per-card specs it
// genuinely belongs outside the tiles.
//
// Source is `nvidia-smi topo -m`, relayed by /api/system/gpus as a matrix
// of legend codes; the classification and layout live in lib/gpu-topology
// so they can be tested without a DOM. The picture is one horizontal row:
// the host as the left end node, the cards to its right, each joined by an
// arc above the row. Three renderings (see that file):
//
//   hub      every card runs to a "CPU · PCIe host bridge" end node. No
//            card-to-card edge — PHB means no direct path exists. Each
//            arc is labelled with the card's negotiated lane width, and a
//            card getting fewer lanes than it can use gets a dashed
//            fault-coloured arc.
//   bridged  thick direct edges card-to-card labelled NV# (bonded pairs
//            sit next to each other in the row), plus the thin PCIe arcs
//            for the rest.
//   fabric   one NVSwitch end node they all join — never a mesh.
//
// Theme tokens only (chat-*). Edge KINDS differ by weight and dash, not
// by colour alone, because retro-dark shares one amber between positive
// and warn; the fault colour is reserved for the reduced-width spoke.

import {
  HUB_H,
  HUB_W,
  NODE_H,
  NODE_W,
  arcLabelPoint,
  arcPath,
  bondAdjacentOrder,
  classifyTopology,
  layoutTopology,
} from "@/lib/gpu-topology";
import { NOT_REPORTED, type GpuTopologyMatrix } from "@/lib/system-info";
import { mibToGib } from "@/lib/stats-v2";

/** What the graph needs to know about each card, keyed by GPU index. */
export interface InterconnectCard {
  memory_total_mib: number | null;
  width_current: number | null;
  width_max: number | null;
}

export function GpuInterconnect({
  topology,
  cards,
  unavailableReason,
}: {
  topology: GpuTopologyMatrix | null;
  cards: Map<number, InterconnectCard>;
  /** Why there is no live feed at all, when there is none. */
  unavailableReason?: string;
}) {
  return (
    <section
      data-testid="system-gpu-interconnect"
      aria-label="Server Architecture"
      className="flex min-w-0 flex-col gap-2.5 rounded-lg border border-chat-rule bg-chat-surface/50 p-4"
    >
      <h3 className="text-[13px] font-semibold text-chat-fg">Server Architecture</h3>
      {topology === null || topology.indices.length === 0 ? (
        <p
          data-testid="system-gpu-interconnect-not-reported"
          className="text-xs italic text-chat-dim"
          title={
            unavailableReason ??
            "nvidia-smi topo -m printed no GPU matrix — an old driver, or a card the driver cannot place."
          }
        >
          interconnect {NOT_REPORTED}
          {unavailableReason ? ` (${unavailableReason})` : ""}
        </p>
      ) : (
        <InterconnectGraph topology={topology} cards={cards} />
      )}
    </section>
  );
}

function InterconnectGraph({
  topology,
  cards,
}: {
  topology: GpuTopologyMatrix;
  cards: Map<number, InterconnectCard>;
}) {
  const cls = classifyTopology(topology.matrix);
  const n = cls.n;
  const layout = layoutTopology(n);
  const { hub } = layout;
  // Cards sit along the row in bondAdjacentOrder so an NVLink-bonded pair
  // is next to its partner; posOf maps back to matrix row.
  const posOf = bondAdjacentOrder(n, cls.nvPairs).map((row, slot) => layout.nodes[slot]);
  const hubTop = { x: hub.x, y: hub.y - HUB_H / 2 };
  const fabric = cls.kind === "fabric";

  const hubLabel = fabric ? "NVSwitch" : "CPU";
  const hubSub = fabric ? `${cls.fabricCode ?? "NVLink"} to every card` : "PCIe host bridge";

  return (
    <>
      <div className="overflow-x-auto" data-testid="system-gpu-interconnect-scroll">
        <svg
          data-testid="system-gpu-interconnect-svg"
          data-kind={cls.kind}
          width={layout.width}
          height={layout.height}
          viewBox={`0 0 ${layout.width} ${layout.height}`}
          role="img"
          aria-label={`Interconnect for ${n} cards: ${describeKind(cls.kind)}`}
          className="mx-auto block"
        >
          {/* Arcs above the row: to the switch when there is one (thick,
              they are NVLink), otherwise to the host bridge (thin, labelled
              with the lanes that path actually gets). */}
          {topology.indices.map((gpuIndex, i) => {
            const pos = posOf[i];
            const card = cards.get(gpuIndex);
            const width = card?.width_current ?? null;
            const widthMax = card?.width_max ?? null;
            const narrow = !fabric && width !== null && widthMax !== null && width < widthMax;
            const kind = fabric ? "fabric-spoke" : narrow ? "spoke-narrow" : "spoke";
            const nodeTop = { x: pos.x, y: pos.y - NODE_H / 2 };
            const label = arcLabelPoint(hubTop, nodeTop);
            return (
              <g key={`spoke-${gpuIndex}`}>
                <path
                  data-testid="system-gpu-interconnect-edge"
                  data-kind={kind}
                  data-from={gpuIndex}
                  data-to={fabric ? "nvswitch" : "cpu"}
                  d={arcPath(hubTop, nodeTop)}
                  fill="none"
                  className={
                    fabric
                      ? "stroke-chat-accent stroke-[4]"
                      : narrow
                        ? "stroke-chat-negative stroke-2 [stroke-dasharray:5_3]"
                        : "stroke-chat-rule stroke-2"
                  }
                />
                {!fabric && (
                  <text
                    data-testid="system-gpu-interconnect-lanes"
                    data-gpu-index={gpuIndex}
                    x={label.x.toFixed(1)}
                    y={label.y.toFixed(1)}
                    textAnchor="middle"
                    className={`font-mono text-[10px] ${narrow ? "fill-chat-negative" : "fill-chat-dim"}`}
                  >
                    {width === null ? `lanes ${NOT_REPORTED}` : `x${width}`}
                  </text>
                )}
              </g>
            );
          })}

          {/* Bridged pairs get the direct edge they physically have. Bonded
              pairs sit next to each other, so only the gap between the two
              boxes shows. */}
          {!fabric &&
            cls.nvPairs.map(({ a, b, code }) => {
              const pa = posOf[a];
              const pb = posOf[b];
              const mx = (pa.x + pb.x) / 2;
              const my = (pa.y + pb.y) / 2;
              return (
                <g key={`nv-${a}-${b}`}>
                  <line
                    data-testid="system-gpu-interconnect-edge"
                    data-kind="nvlink"
                    data-from={topology.indices[a]}
                    data-to={topology.indices[b]}
                    x1={pa.x.toFixed(1)}
                    y1={pa.y.toFixed(1)}
                    x2={pb.x.toFixed(1)}
                    y2={pb.y.toFixed(1)}
                    fill="none"
                    className="stroke-chat-accent stroke-[4]"
                  />
                  <text
                    x={mx.toFixed(1)}
                    y={(my - 7).toFixed(1)}
                    textAnchor="middle"
                    className="fill-chat-accent font-mono text-[10px]"
                  >
                    {code}
                  </text>
                </g>
              );
            })}

          <g data-testid="system-gpu-interconnect-hub" data-kind={fabric ? "nvswitch" : "cpu"}>
            <rect
              x={hub.x - HUB_W / 2}
              y={hub.y - HUB_H / 2}
              width={HUB_W}
              height={HUB_H}
              rx={7}
              className={
                fabric
                  ? "fill-chat-page stroke-chat-accent stroke-[2.5]"
                  : "fill-chat-page stroke-chat-dim stroke-[1.5]"
              }
            />
            <text
              x={hub.x}
              y={hub.y - 2}
              textAnchor="middle"
              className="fill-chat-fg font-sans text-[11.5px] font-semibold"
            >
              {hubLabel}
            </text>
            <text
              x={hub.x}
              y={hub.y + 12}
              textAnchor="middle"
              className="fill-chat-dim font-mono text-[9.5px]"
            >
              {hubSub}
            </text>
          </g>

          {topology.indices.map((gpuIndex, i) => {
            const pos = posOf[i];
            const vram = cards.get(gpuIndex)?.memory_total_mib ?? null;
            return (
              <g key={`node-${gpuIndex}`} data-testid="system-gpu-interconnect-node" data-gpu-index={gpuIndex}>
                <rect
                  x={(pos.x - NODE_W / 2).toFixed(1)}
                  y={(pos.y - NODE_H / 2).toFixed(1)}
                  width={NODE_W}
                  height={NODE_H}
                  rx={7}
                  className="fill-chat-surface-2 stroke-chat-rule stroke-[1.5]"
                />
                <text
                  x={pos.x.toFixed(1)}
                  y={(pos.y - 2).toFixed(1)}
                  textAnchor="middle"
                  className="fill-chat-fg font-sans text-[11.5px] font-semibold"
                >
                  GPU {gpuIndex}
                </text>
                <text
                  x={pos.x.toFixed(1)}
                  y={(pos.y + 11).toFixed(1)}
                  textAnchor="middle"
                  className="fill-chat-dim font-mono text-[9.5px]"
                >
                  {vram === null ? "" : `${Math.round(Number(mibToGib(vram)))} GiB`}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      <p
        data-testid="system-gpu-interconnect-legend"
        className="max-w-[66ch] text-[11.5px] leading-relaxed text-chat-dim"
      >
        <Legend kind={cls.kind} n={n} />
      </p>
    </>
  );
}

function describeKind(kind: "hub" | "bridged" | "fabric"): string {
  switch (kind) {
    case "fabric":
      return "every card on one NVSwitch";
    case "bridged":
      return "some pairs bonded by NVLink, the rest through the PCIe host bridge";
    case "hub":
      return "no NVLink, every card through the PCIe host bridge";
  }
}

function Code({ children }: { children: string }) {
  return <b className="font-mono font-medium text-chat-muted">{children}</b>;
}

function Legend({ kind, n }: { kind: "hub" | "bridged" | "fabric"; n: number }) {
  if (kind === "fabric") {
    return (
      <>
        Every card is on an <Code>NVSwitch</Code>, so any pair talks directly at full NVLink
        speed. Drawn as the one switch they all join rather than {(n * (n - 1)) / 2} separate
        lines, because that is the hardware and the picture stays readable.
      </>
    );
  }
  if (kind === "bridged") {
    return (
      <>
        Thick lines are direct <Code>NVLink</Code> bonds — those pairs talk card-to-card,
        bypassing the CPU. Pairs without one still reach each other through the host bridge,
        at the lane width on their thin line.
      </>
    );
  }
  return (
    <>
      No card here has a link to another. Every one runs to the CPU&apos;s PCIe host bridge (
      <Code>PHB</Code>) and reaches its neighbours through it — the slow path, worth knowing
      before splitting one model across {n} cards, because every activation exchanged between
      them takes it. Each label is that card&apos;s negotiated lane width; a dashed line is a
      card getting fewer lanes than it can use.
    </>
  );
}
