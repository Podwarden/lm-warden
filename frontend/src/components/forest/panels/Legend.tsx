"use client";

import { GlassDetails, openWhenWide } from "./glass";
import { GHOST_CROWN, GHOST_SOIL, GHOST_STONE } from "../engine/ghostColors";

// Legend text for the voxel look (Plan 4: forest-styles.html, the chosen style) in the live-only view (spec §10.6).
// Every row names something the scene draws, in its colours (voxels.ts `cubeColors`, ground.ts sprites, quiet.ts).

/** A few small square blocks in the colours of a voxel part. */
const Blocks = ({ colors }: { colors: string[] }) => (
  <span aria-hidden="true" className="flex w-[22px] flex-none gap-px">
    {colors.map((c, i) => (
      <span key={i} className="h-1.5 w-1.5" style={{ background: c }} />
    ))}
  </span>
);

const Row = ({ sw, children }: { sw: React.ReactNode; children: React.ReactNode }) => (
  <li className="flex items-center gap-2">
    {sw}
    <span>{children}</span>
  </li>
);

const Head = ({ children }: { children: React.ReactNode }) => (
  <h3 className="mb-1 mt-2.5 text-[10px] uppercase tracking-[0.12em] text-[#cfc7b6] first:mt-0">{children}</h3>
);

const SPECIES: [string, string][] = [
  ["oak", "big system prompt, short replies"],
  ["baobab", "code-stuffed prompt (repo map)"],
  ["poplar", "small prompt, long answers"],
  ["shrub", "small prompt, short answers"],
  ["pine", "one long session, no subagents"],
  ["banyan", "orchestrator with many subagents"],
  ["willow", "cache thrash, prompts recomputed"],
  ["birch", "reasoning model, thinking-heavy"],
  ["palm", "one big-prompt call, no tools"],
  ["bamboo", "cron: same prompt, ~100% cached"],
  ["bonsai", "small local model, tight context"],
  ["cactus", "failing sessions, failed tools"],
  ["round crown", "none of the above"],
];

/**
 * Left, below the description (spec §6.4): the voxel look's parts (wood, crown, the cyan growth tint, blossom and
 * failed blocks, ghosts), the ground (daisy beds, mushrooms, quiet stones) and the species.
 */
export function Legend() {
  return (
    <GlassDetails
      title="Legend"
      storageKey="forest.legend.open"
      defaultOpen={openWhenWide}
      testid="forest-legend"
      bodyClassName="max-h-[calc(100vh-16rem)] overflow-y-auto pr-1 text-[#d3dad3]"
    >
      <p data-testid="forest-gesture-help" className="mb-2 text-[11px] leading-snug text-[#cfd3dc]">
        Trackpad: two fingers pan · pinch zoom · Shift+two fingers (or twist in Safari) rotate
      </p>
      <Head>A tree · a key&apos;s working spell</Head>
      <ul className="space-y-1">
        <Row sw={<Blocks colors={["#8f5a2c", "#7d4f27", "#9c6331"]} />}>
          Brown blocks: trunk and limbs, one limb per session — busier trees have thicker trunks
        </Row>
        <Row sw={<Blocks colors={["#3cbf3a", "#56c232", "#2a8a2a"]} />}>
          Green leaf blocks: the crown — it fills out with every turn and its context
        </Row>
        <Row sw={<Blocks colors={["#7fd6ff", "#4cc2ff", "#a6e4ff"]} />}>
          Cyan tint: blocks growing now (fades 12 s after)
        </Row>
        <Row sw={<Blocks colors={["#ff7fbf"]} />}>Pink block: a turn that answered the user</Row>
        <Row sw={<Blocks colors={["#b8902f"]} />}>Olive block: a failed tool call</Row>
        <Row sw={<Blocks colors={[GHOST_CROWN, GHOST_SOIL[0], GHOST_STONE[0]]} />}>
          Ghost blocks on a patch of soil and stones: a tree that moved to the front when its session came back
        </Row>
      </ul>
      <Head>On the ground</Head>
      <ul className="space-y-1">
        <Row sw={<Blocks colors={["#ffffff", "#ffd21f", "#ffffff"]} />}>Daisy bed: one-shot question and answer</Row>
        <Row sw={<Blocks colors={["#e8302c", "#ffffff", "#e8302c"]} />}>Red mushrooms: embeddings — prompt in, no text out</Row>
        <Row sw={<Blocks colors={["#c9c5b8", "#b4b0a4", "#9d998e"]} />}>
          Pale stone blocks on the shore: quiet hours, shortened (hover: how long)
        </Row>
      </ul>
      <Head>Species · what grew it (shapes the trunk and crown)</Head>
      <dl className="grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5 leading-snug">
        {SPECIES.map(([name, what]) => (
          <div key={name} className="contents">
            <dt className="font-medium text-[#cfc7b6]">{name}</dt>
            <dd>{what}</dd>
          </div>
        ))}
      </dl>
    </GlassDetails>
  );
}
