/**
 * The ghost's colours (no three.js: the legend imports them too). A tree that moved leaves a faint ghost of its crown
 * standing on a small patch of voxel soil with a few stones on it (flights.ts `ghostGround`).
 */
export const GHOST_CROWN = "#e8f0ff";
/** Soil blocks: flat tiles on the 0.5 grid. */
export const GHOST_SOIL = ["#6b4a2b", "#7a5532", "#5c3f24"] as const;
/** Stone blocks: small cubes on the soil, the quiet-hour stones' greys (quiet.ts). */
export const GHOST_STONE = ["#b4b0a4", "#9d998e"] as const;
