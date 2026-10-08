/**
 * README §4.4 — the spit, its terraces, the sea, the surf and the clouds, in the voxel look (Plan 4 Task 3). Ported from
 * the style mockup (forest-styles.html: `updateLand`'s toon colours, the stepped `HIL` hills, the cartoon `surfMat`, the
 * glossy toon `sea`, the boxy `CLB` clouds and `stepClouds`) on top of forest-real.html's strip (`land`/`updateSurf`).
 * - The land is one strip of fixed topology: every row is a fraction across (grass → sand → wet sand → under water), so
 *   the shore follows the spit's width without new geometry. It is rewritten only when the shore moved (hash of the
 *   widths) or the spit grew past the allocated length.
 * - Terraces (the mockup's stepped hills): boxes of one to three 0.5-high tiers, laid out by a hash of the world grid
 *   (`hillAt`), never by the data: they do not move as the spit grows, so what stands on them never changes height.
 *   They lie inside the grass of the narrowest spit (|z| ≤ TERRACE_ZMAX) and only where the strip does not taper.
 *   `groundHeight` is the land's own profile, or a terrace's top exactly; `cameraGround` is its continuous envelope.
 * - The sea is a flat toon sea (two blues, white wave bands parallel to the shore) that fades into the sky's horizon
 *   colour just before the far plane, so no edge of it is ever in frame. The surf is a crisp white swash line and white
 *   bands rolling in.
 * - Clouds are white boxes in three parallax layers tied to eye level (always above the horizon), one instanced mesh.
 */
import * as THREE from "three";
import { boxes } from "./voxels";

const STEP = 0.5;
/** Surf foam is full up to this camera distance and gone FOAM_FADE further (review I2: aliasing far off). */
const FOAM_NEAR = 250, FOAM_FADE = 250;
const ROWS = 56;
const SURF_R = 14;
/** The strip starts here, or further back when the range reaches behind it (follow-up C: `landSpan`'s `xStart`). */
export const LX0 = -30;
/** The strip's start is snapped down to a multiple of this (units), so it is reallocated rarely. */
const LX_SNAP = 200;
/** The strip starts this far before the shown range's oldest tree or flower (its taper lies in between). */
const LAND_BEFORE = 160;
/** Each end of the strip narrows and sinks under the sea over this many units. */
export const TAPER = 120;
/** How far the strip's ends sink (units), well under the sea. */
const TAPER_SINK = 2.5;

/**
 * The strip's end taper at x (1 inside, 0 at its ends): a smoothstep over TAPER units from the start, and over TAPER units
 * ending 20 units before the end. The shore's width is scaled by it and the land sinks by (1 − k)·TAPER_SINK.
 */
export function taper(x: number, lx0: number, lx1: number): number {
  const ss = (u: number) => {
    const v = Math.min(1, Math.max(0, u));
    return v * v * (3 - 2 * v);
  };
  return ss((x - lx0) / TAPER) * ss((lx1 - 20 - x) / TAPER);
}
/** Extra length allocated past the front, so the strip is reallocated rarely. */
const HEADROOM = 240;

const lcg = (seed: number) => {
  let q = seed;
  return () => (q = (q * 16807) % 2147483647) / 2147483647;
};
const hash = (a: number, b: number) => {
  const v = Math.sin(a * 127.1 + b * 311.7) * 43758.5453;
  return v - Math.floor(v);
};

// ---------------- terraces ----------------

/** The terrace grid's cell along x (units): at most one hill per cell and side, inside the cell. */
export const TERRACE_CELL = 14;
/** One tier's height (units): the voxel grid's 0.5. */
export const TERRACE_STEP = 0.5;
/** Terraces stay within this distance of the spit's axis: inside the grass of the narrowest spit (BASE_W − 1.8). */
export const TERRACE_ZMAX = 11.5;
/** A cell and side holds a hill this often. */
const TERRACE_P = 0.55;
/** Each tier's half-size against the bottom one (the mockup's `HIL` boxes: 2, 1.5, 0.9 → 1, 0.72, 0.45). */
const TIER_K = [1, 0.72, 0.45];
/** The camera's floor rises at most this steeply (units per unit) toward a terrace's edge. */
const CAM_SLOPE = 1;

/** One tier of a terrace: the box [x0, x1) × [z0, z1) whose top is at `top`. */
export interface Tier {
  x0: number;
  x1: number;
  z0: number;
  z1: number;
  top: number;
}
/** A stepped hill: its tiers from the bottom up (each inside the one below). */
export interface Hill {
  tiers: Tier[];
}

/** An integer hash of (i, j, k) in [0, 1): the terrace layout's only source of chance (world grid, never the data). */
function ihash(i: number, j: number, k: number): number {
  let h = Math.imul(i | 0, 0x27d4eb2d) ^ Math.imul((j | 0) + 0x632be5ab, 0x165667b1) ^ Math.imul((k | 0) + 0x5bd1e995, 0x9e3779b1);
  h ^= h >>> 15;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}
const snap = (v: number) => Math.round(v / STEP) * STEP;
const snapDown = (v: number) => Math.floor(v / STEP + 1e-9) * STEP;

const hillMemo = new Map<string, Hill | null>();
/** The hill in terrace cell `i` on `side` of the spit (north −1, south +1), or null: a pure function of (i, side). */
export function hillAt(i: number, side: -1 | 1): Hill | null {
  const key = `${i}|${side}`;
  const m = hillMemo.get(key);
  if (m !== undefined) return m;
  const H = (k: number) => ihash(i, side, k);
  let hill: Hill | null = null;
  if (H(0) < TERRACE_P) {
    const R = snap(3.2 * (0.75 + 0.6 * H(1))), RZ = Math.max(1.5, snap(R * 0.85));
    const n = H(2) < 0.2 ? 1 : H(2) < 0.6 ? 2 : 3;
    // x: inside the cell with 0.5 to spare at each end; z: inside TERRACE_ZMAX
    const cx = i * TERRACE_CELL + R + STEP + snapDown(H(3) * (TERRACE_CELL - 2 * R - 2 * STEP));
    const cz = side * snapDown(H(4) * (TERRACE_ZMAX - RZ));
    const tiers: Tier[] = [];
    for (let k = 0; k < n; k++) {
      const hx = Math.max(STEP, snap(R * TIER_K[k])), hz = Math.max(STEP, snap(RZ * TIER_K[k]));
      tiers.push({ x0: cx - hx, x1: cx + hx, z0: cz - hz, z1: cz + hz, top: TERRACE_STEP * (k + 1) });
    }
    hill = { tiers };
  }
  if (hillMemo.size > 20_000) hillMemo.clear();
  hillMemo.set(key, hill);
  return hill;
}

const inSpan = (h: Hill, xa: number, xb: number) => h.tiers[0].x0 >= xa && h.tiers[0].x1 <= xb;

/** Every hill lying wholly in [xa, xb], west to east. */
export function hillsIn(xa: number, xb: number): Hill[] {
  const out: Hill[] = [];
  if (!(xb > xa) || !Number.isFinite(xa) || !Number.isFinite(xb)) return out;
  for (let i = Math.floor(xa / TERRACE_CELL); i <= Math.floor(xb / TERRACE_CELL); i++)
    for (const side of [-1, 1] as const) {
      const h = hillAt(i, side);
      if (h && inSpan(h, xa, xb)) out.push(h);
    }
  return out;
}

/** Where a strip from lx0 to lx1 has terraces: between its tapers (a hill must lie wholly inside). */
export function terraceSpan(lx0: number, lx1: number): [number, number] {
  return [lx0 + TAPER, lx1 - 20 - TAPER];
}

/**
 * The terraces' height at (x, z): the top of the highest tier containing it, or 0 (no terrace). Only hills lying wholly
 * in [xa, xb] count. Pure in the world position: the spit's growth never changes it.
 */
export function terraceHeight(x: number, z: number, xa = -Infinity, xb = Infinity): number {
  if (!Number.isFinite(x) || !Number.isFinite(z) || Math.abs(z) >= TERRACE_ZMAX) return 0;
  const i = Math.floor(x / TERRACE_CELL); // a hill lies inside its own cell
  let y = 0;
  for (const side of [-1, 1] as const) {
    const h = hillAt(i, side);
    if (!h || !inSpan(h, xa, xb)) continue;
    for (const t of h.tiers) if (x >= t.x0 && x < t.x1 && z >= t.z0 && z < t.z1 && t.top > y) y = t.top;
  }
  return y;
}

// ---------------- the sea ----------------

/** One tile of the sea, in world units (the mockup's 1200-deep sea held 18 wave tiles). */
export const SEA_TILE = 1200 / 18;

/**
 * The sea plane under a view (review I2): centred under the camera, snapped to whole tiles, reaching `reach` units each
 * way (at least 600, rounded up to whole tiles). The scene passes the far plane's distance, so no edge of the sea is
 * ever in frame.
 */
export function seaFit(camX: number, camZ: number, reach: number): { x: number; z: number; size: number; tiles: number } {
  const r = Math.ceil(Math.max(600, Number.isFinite(reach) ? reach : 600) / SEA_TILE) * SEA_TILE;
  const snapT = (v: number) => (Number.isFinite(v) ? Math.round(v / SEA_TILE) * SEA_TILE : 0);
  return { x: snapT(camX), z: snapT(camZ), size: 2 * r, tiles: (2 * r) / SEA_TILE };
}

/** The sky's colour at the horizon (sky.ts SKY_FS), linear: the sea fades into it before the far plane. */
const HORIZON_GLSL = "pow(vec3(.66,.93,1.0),vec3(2.2))";

/**
 * Mockup's glossy toon sea (the plain, not the pixel, variant): two flat blues, wave bands parallel to the shore. Far
 * from the camera the bands fade out (sub-pixel, they alias into flickering dashes, as the surf's foam does), and just
 * before the far plane the sea fades into the sky's horizon colour: no edge is ever in frame.
 */
const SEA_VS = "varying vec3 wp;void main(){vec4 w=modelMatrix*vec4(position,1.0);wp=w.xyz;gl_Position=projectionMatrix*viewMatrix*w;}";
const SEA_FS = `uniform float uTime,uFar;varying vec3 wp;
void main(){vec3 P=wp;float T=uTime;
  float z=abs(P.z);vec3 c=mix(vec3(.16,.60,1.0),vec3(.07,.40,.92),step(34.0,z+3.0*sin(P.x*0.05)));
  float w=0.9*sin(P.x*0.16+T*0.6)+1.3*sin(P.x*0.05-T*0.25);
  float ph=fract((z+w)/9.0-T*0.05);float band=step(0.84,ph)*step(ph,0.94)*(1.0-step(40.0,z))*step(fract(P.x*0.025+floor((z+w)/9.0)*0.37),0.62);
  c=mix(c,vec3(.33,.72,1.0),(1.0-smoothstep(0.0,0.35,abs(normalize(cameraPosition-wp).y-0.08)))*0.25);
  float d=length(cameraPosition.xz-wp.xz);
  c=mix(c,vec3(1.),band*0.95*(1.0-smoothstep(300.0,700.0,d)));
  c=pow(c,vec3(2.2));
  c=mix(c,${HORIZON_GLSL},smoothstep(0.75*uFar,0.98*uFar,d));
  gl_FragColor=vec4(c,1.);}`;

const SURF_VS = "attribute float dist;varying float d;varying vec2 wp;void main(){d=dist;wp=position.xz;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}";
/**
 * Mockup's cartoon surf: a crisp white swash line on the sand and simple white wave bands rolling in over light
 * shallows. A tapered end's dist is pushed far negative, so it is discarded there. `uFoam` fades the bands far off.
 */
const SURF_FS = `uniform float uTime,uFoam;varying float d;varying vec2 wp;
void main(){float D=d;vec2 W=wp;float T=uTime;
  float sw=0.3*sin(T*0.8+W.x*0.07);if(d<sw-0.05)discard;
  float edge=step(abs(D-sw-0.12),0.16);
  float ph=fract(D*0.30-T*0.22+0.12*sin(W.x*0.21));float band=step(0.80,ph)*step(ph,0.90)*step(0.7,D)*(1.0-step(9.5,D));
  float foam=max(edge,band*uFoam);float ta=0.65*exp(-max(D,0.)/3.0)*step(sw,d);
  vec3 c=mix(vec3(.50,.90,1.0),vec3(1.),foam);
  gl_FragColor=vec4(pow(c,vec3(2.2)),max(ta,foam));}`;

/** Mockup `updateLand` colours: bright grass, pale sand (a band on the grass's edge), wet sand, shallows. */
const G1 = new THREE.Color("#4cc23a"), G2 = new THREE.Color("#57cc40"), SD = new THREE.Color("#fbeaa6");
const SW = new THREE.Color("#f0d684"), SU = new THREE.Color("#57b8e8");
/**
 * The stepped hills' colour, face-shaded (voxels.ts `shadeBox`: tops ×1.16, sides ×0.78–0.9): the tops come out the
 * grass's own green (G1 / 1.16), the risers darker, so each step reads (the target's terraces).
 */
const HILL_C = "#41a732";

/**
 * The strip never has more columns than this (about 15 700 units of spit: 29 days at normal speed; 48 h is ~2 800).
 * Position + uv + colour then stay under 60 MB even for a bad time (an absolute unix time read as relative once asked
 * for a 2 GB Float32Array).
 */
export const MAX_LAND_COLS = 32_000;

/**
 * The strip's ends and column count for a spit front at `xFront` whose shown range starts at `xStart`: it starts
 * LAND_BEFORE units before `xStart` (snapped down to LX_SNAP), or at LX0 without a range. Both ends taper into the sea
 * over TAPER units (`taper`): no square end is ever drawn (review I2). Null (do nothing) for a non-finite or negative front. Clamped to MAX_LAND_COLS,
 * keeping the front.
 */
export function landSpan(xFront: number, xStart?: number): { lx0: number; lx1: number; cols: number } | null {
  if (!Number.isFinite(xFront) || xFront < 0) return null;
  const lx1 = Math.min(xFront + 80 + HEADROOM, LX0 + MAX_LAND_COLS * STEP);
  // the strip starts LAND_BEFORE units before the range's start (its taper lies before that), snapped
  let lx0 = xStart !== undefined && Number.isFinite(xStart) ? Math.min(Math.floor((xStart - LAND_BEFORE) / LX_SNAP) * LX_SNAP, xFront - 200) : LX0;
  lx0 = Math.max(lx0, lx1 - MAX_LAND_COLS * STEP);
  return { lx0, lx1, cols: Math.ceil((lx1 - lx0) / STEP) };
}

export type WidthFn = (x: number, side: -1 | 1) => number;

/** Rows across the strip (fraction `a` of the half-width row range): grass to here, then dry and wet sand, then under
 * water. */
const GRASS_A = 0.6, SAND_A = 0.82;
/** From this fraction on, the grass rows are drawn as sand (the mockup's sand band on the grass's edge). */
const BAND_A = 0.55;
/** Grass noise, sand noise, beach drop and water depth (units). */
const GRASS_NZ = 0.04, SAND_NZ = 0.02, BEACH = 2.2, SHELF = 10;
/** The sea plane's height. */
export const SEA_Y = -0.3;

/**
 * One vertex of the strip (mockup `updateLand`): at fraction `a` (0 the spit's axis, 1 the far edge under water) of a
 * side whose shore is `w` from the axis, with noise `nz` in [0, 1]: its distance from the axis and its height.
 */
export function landVertex(a: number, w: number, nz: number): { z: number; y: number } {
  if (a <= GRASS_A) return { z: (a / GRASS_A) * w, y: GRASS_NZ * nz };
  if (a <= SAND_A) {
    const u = (a - GRASS_A) / (SAND_A - GRASS_A);
    return { z: w + u * BEACH, y: -0.2 * Math.pow(u, 1.6) + SAND_NZ * nz };
  }
  const u = (a - SAND_A) / (1 - SAND_A);
  return { z: w + BEACH + u * SHELF, y: -0.2 - 1.6 * u };
}

/**
 * The land strip's own height at (x, z), without terraces (spec follow-up A): its profile across the spit
 * (`landVertex`, taking the noise at its highest, so the mesh is never above it), or the sea where the strip is under
 * it or absent (before `x0`, the strip's start: `Land.startX()`). Pure: the scene passes the timeline's width at the
 * shown time, the same widths the strip is written from.
 */
export function landHeight(x: number, z: number, width: WidthFn, x0 = LX0): number {
  if (x < x0) return SEA_Y;
  const side: -1 | 1 = z < 0 ? -1 : 1, raw = width(x, side), w = Number.isFinite(raw) ? Math.max(0, raw) : 0, d = Math.abs(z);
  let y: number;
  if (d <= w) y = GRASS_NZ;
  else if (d <= w + BEACH) y = -0.2 * Math.pow((d - w) / BEACH, 1.6) + SAND_NZ;
  else if (d <= w + BEACH + SHELF) y = -0.2 - (1.6 * (d - w - BEACH)) / SHELF;
  else return SEA_Y;
  return Math.max(SEA_Y, y);
}

/**
 * The visible ground's height at (x, z): a terrace's top exactly where one stands (`terraceHeight`, over the strip from
 * `x0` to `x1` between its tapers: `Land.startX()`, `Land.endX()`), else the land's own profile (`landHeight`). The
 * camera's floor (`cameraGround`), tree bases, flora and the quiet stones all stand on it.
 */
export function groundHeight(x: number, z: number, width: WidthFn, x0 = LX0, x1 = Infinity): number {
  const L = landHeight(x, z, width, x0);
  if (x < x0) return L;
  const [a, b] = terraceSpan(x0, x1), t = terraceHeight(x, z, a, b);
  return t > 0 && t > L ? t : L;
}

/**
 * The camera's floor: `groundHeight` raised near each terrace by a slope of CAM_SLOPE down from its edges, so it never
 * steps (the clamp stays continuous: a small move never lifts the camera by a whole tier). ≥ groundHeight everywhere;
 * equal to it on a tier's top away from a higher tier, and on flat land away from the terraces.
 */
export function cameraGround(x: number, z: number, width: WidthFn, x0 = LX0, x1 = Infinity): number {
  let y = groundHeight(x, z, width, x0, x1);
  if (x < x0 || !Number.isFinite(x) || !Number.isFinite(z)) return y;
  const [a, b] = terraceSpan(x0, x1), i = Math.floor(x / TERRACE_CELL);
  for (let ii = i - 1; ii <= i + 1; ii++)
    for (const side of [-1, 1] as const) {
      const h = hillAt(ii, side);
      if (!h || !inSpan(h, a, b)) continue;
      for (const t of h.tiers) {
        if (t.top <= y) continue;
        const d = Math.hypot(Math.max(t.x0 - x, 0, x - t.x1), Math.max(t.z0 - z, 0, z - t.z1));
        y = Math.max(y, t.top - CAM_SLOPE * d);
      }
    }
  return y;
}

// ---------------- clouds ----------------

const CL_SPAN = 700;
/** Mockup `CL_LAYERS`: far / middle / near; nearer is bigger and faster (parallax). */
export const CL_LAYERS = [
  { z: -330, y: [30, 60], w: [60, 100], n: 8, v: 0.6 },
  { z: -200, y: [24, 46], w: [30, 55], n: 9, v: 1.3 },
  { z: -110, y: [22, 36], w: [14, 26], n: 8, v: 2.6 },
] as const;
/** Mockup `CLB`: a boxy cloud, 5 units wide at scale 1, its boxes centred on y = 0. */
const CLOUD_BOXES = [
  [2.4, 0.8, 1.1, 0, 0, 0, "#ffffff"], [1.4, 0.7, 1.1, -0.25, 0.7, 0, "#ffffff"], [0.9, 0.5, 1.1, 0.75, 0.55, 0, "#ffffff"],
  [0.8, 0.5, 1.1, 1.5, -0.1, 0, "#ffffff"], [0.7, 0.4, 1.1, -1.45, -0.15, 0, "#ffffff"],
] as const;
/** The cloud's lowest point below its centre, per unit of scale. */
export const CLOUD_BOTTOM = Math.max(...CLOUD_BOXES.map(([, h, , , y]) => h / 2 - y));
/** Eye level of a camera lower than this counts as this (clouds then stay a little up, as before). */
const CLOUD_EYE_MIN = 2;

export interface CloudSeed {
  layer: number;
  off: number;
  v: number;
  z: number;
  dz: number;
  /** Height of its centre above eye level. */
  y: number;
  /** Scale (the mockup's w / 5). */
  s: number;
}
/** Every cloud: a fixed draw (the mockup's seed 17). */
export const CLOUDS: readonly CloudSeed[] = (() => {
  const r = lcg(17), out: CloudSeed[] = [];
  CL_LAYERS.forEach((L, layer) => {
    for (let i = 0; i < L.n; i++) {
      const w = L.w[0] + r() * (L.w[1] - L.w[0]), off = r() * CL_SPAN;
      // the mockup's height over its ground, taken over eye level instead: always above the horizon
      const y = L.y[0] - 12 + w * 0.12 + r() * (L.y[1] - L.y[0]);
      out.push({ layer, off, v: L.v, z: L.z, dz: (r() - 0.5) * 40, y: Math.max(y, CLOUD_BOTTOM * (w / 5) + 4), s: w / 5 });
    }
  });
  return out;
})();

/** Cloud `i` at cloud time `t` for a view: drifting right to left around the view's target, tied to eye level. */
export function cloudPose(i: number, t: number, view: { camY: number; target: { x: number; z: number } }): { x: number; y: number; z: number; s: number } {
  const c = CLOUDS[i];
  const x = ((((c.off - t * c.v) % CL_SPAN) + CL_SPAN) % CL_SPAN) - CL_SPAN / 2;
  const eye = Math.max(Number.isFinite(view.camY) ? view.camY : 0, CLOUD_EYE_MIN);
  return { x: view.target.x + x, y: eye + c.y, z: view.target.z + c.z + c.dz, s: c.s };
}

// ---------------- the land ----------------

export interface Land {
  /**
   * Rewrites the shore for the spit up to `xFront` + 80 (and back to `xStart` − 80 when that is behind LX0) if the widths
   * moved; true when anything changed.
   */
  update(width: WidthFn, xFront: number, xStart?: number): boolean;
  /** Where the strip starts (x): before it is sea. */
  startX(): number;
  /** Where the strip ends (x). */
  endX(): number;
  /** Re-centres and sizes the sea under the camera (`seaFit`); true when it changed. */
  fitSea(camX: number, camZ: number, reach: number): boolean;
  /** Water and clouds tick at ~24 fps; true when the frame must be redrawn. */
  step(dt: number, view: { camY: number; target: THREE.Vector3; camDist?: number }): boolean;
  /** Terrace boxes drawn (tiers). */
  terraces(): number;
  dispose(): void;
}

type Toon = (o: THREE.MeshToonMaterialParameters) => THREE.Material;

export function createLand(scene: THREE.Scene, opts: { compact: boolean; shadows: boolean; toon?: Toon }): Land {
  const owned: THREE.Material[] = [];
  const toon: Toon = opts.toon ?? ((o) => {
    const m = new THREE.MeshToonMaterial(o);
    owned.push(m);
    return m;
  });
  const landMat = toon({ vertexColors: true });
  const hillMat = toon({ vertexColors: true });
  const cloudMat = toon({ vertexColors: true, emissive: "#cfeaff", emissiveIntensity: 0.35, fog: false });
  const SU_ = { uTime: { value: 0 }, uFoam: { value: 1 } };
  const surfMat = new THREE.ShaderMaterial({ uniforms: SU_, transparent: true, depthWrite: false, side: THREE.DoubleSide, vertexShader: SURF_VS, fragmentShader: SURF_FS });
  const SEA_U = { uTime: SU_.uTime, uFar: { value: 1800 } };
  // drawn after the land (transparent pass, alpha 1) so it hides the land under the water; the surf ribbon lies 0.012
  // above it and is drawn after it. The sea need not write depth (nothing transparent lies under it), so the two never
  // fight along the shore (review I2)
  const seaMat = new THREE.ShaderMaterial({ uniforms: SEA_U, fog: false, transparent: true, depthWrite: false, vertexShader: SEA_VS, fragmentShader: SEA_FS });
  let cols = 0, lx0 = LX0, lx1 = LX0, hashW = -1;
  let WN = new Float32Array(0), WS = new Float32Array(0), K = new Float32Array(0);
  let land: THREE.Mesh | null = null, surf: THREE.Mesh | null = null, hills: THREE.InstancedMesh | null = null;
  const hillGeo = boxes([[1, 1, 1, 0, 0.5, 0, HILL_C]]);
  // the sea: one unit plane, scaled and re-centred under the view (fitSea)
  const sea = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), seaMat);
  sea.rotation.x = -Math.PI / 2;
  scene.add(sea);
  let seaKey = "";
  const placeSea = (f: ReturnType<typeof seaFit>, reach: number): boolean => {
    const k = `${f.x}|${f.z}|${f.size}`;
    SEA_U.uFar.value = Math.max(600, Number.isFinite(reach) ? reach : 600);
    if (k === seaKey) return false;
    seaKey = k;
    sea.position.set(f.x, SEA_Y, f.z);
    sea.scale.set(f.size, f.size, 1);
    return true;
  };
  placeSea(seaFit(0, 0, 900), 900);

  /** The terraces between the strip's tapers: one box per tier, from under the sea up to its top. */
  const placeHills = () => {
    if (hills) (scene.remove(hills), hills.dispose());
    const [a, b] = terraceSpan(lx0, lx1), tiers = hillsIn(a, b).flatMap((h) => h.tiers);
    hills = new THREE.InstancedMesh(hillGeo, hillMat, Math.max(1, tiers.length));
    const m4 = new THREE.Matrix4();
    tiers.forEach((t, i) => {
      m4.makeScale(t.x1 - t.x0, t.top - SEA_Y, t.z1 - t.z0).setPosition((t.x0 + t.x1) / 2, SEA_Y, (t.z0 + t.z1) / 2);
      hills!.setMatrixAt(i, m4);
    });
    hills.count = tiers.length;
    hills.instanceMatrix.needsUpdate = true;
    hills.castShadow = hills.receiveShadow = opts.shadows;
    hills.computeBoundingSphere();
    scene.add(hills);
  };

  /** (Re)allocates the strip, the surf ribbon and the terraces for COLS columns from lx0. */
  const allocate = (span: { lx0: number; lx1: number; cols: number }) => {
    for (const m of [land, surf]) if (m) (scene.remove(m), m.geometry.dispose());
    ({ lx0, lx1, cols } = span);
    WN = new Float32Array(cols + 1);
    K = new Float32Array(cols + 1);
    WS = new Float32Array(cols + 1);
    const n = (cols + 1) * (ROWS + 1), lg = new THREE.BufferGeometry(), idx: number[] = [];
    lg.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(n * 3), 3));
    lg.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(n * 3), 3));
    for (let c = 0; c < cols; c++)
      for (let r = 0; r < ROWS; r++) {
        const a = c * (ROWS + 1) + r, b = (c + 1) * (ROWS + 1) + r;
        idx.push(a, a + 1, b, b, a + 1, b + 1);
      }
    lg.setIndex(idx);
    land = new THREE.Mesh(lg, landMat);
    land.receiveShadow = opts.shadows;
    const sn = (cols + 1) * (SURF_R + 1) * 2, sg = new THREE.BufferGeometry(), sidx: number[] = [];
    sg.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(sn * 3), 3));
    sg.setAttribute("dist", new THREE.Float32BufferAttribute(new Float32Array(sn), 1));
    for (let side = 0; side < 2; side++) {
      const o = side * (cols + 1) * (SURF_R + 1);
      for (let c = 0; c < cols; c++)
        for (let r = 0; r < SURF_R; r++) {
          const a = o + c * (SURF_R + 1) + r, b = o + (c + 1) * (SURF_R + 1) + r;
          if (side) sidx.push(a, b, a + 1, b, b + 1, a + 1);
          else sidx.push(a, a + 1, b, b, a + 1, b + 1);
        }
    }
    sg.setIndex(sidx);
    surf = new THREE.Mesh(sg, surfMat);
    surf.renderOrder = 2;
    surf.frustumCulled = false;
    scene.add(land, surf);
    placeHills();
    hashW = -1;
  };

  /** Mockup `updateLand` + `updateSurf`. */
  const rewrite = () => {
    if (!land || !surf) return;
    const g = land.geometry, P = g.attributes.position.array as Float32Array, Cc = g.attributes.color.array as Float32Array;
    const col = new THREE.Color();
    // the noise follows the column's world position, so a strip that starts elsewhere keeps its pattern
    const c0 = Math.round((lx0 - LX0) / STEP);
    for (let c = 0; c <= cols; c++) {
      const x = lx0 + c * STEP;
      for (let r = 0; r <= ROWS; r++) {
        const sgn = r < ROWS / 2 ? -1 : 1, a = Math.abs(r - ROWS / 2) / (ROWS / 2), w = sgn < 0 ? WN[c] : WS[c], nz = hash(c + c0, r);
        const { z, y } = landVertex(a, w, nz);
        if (a <= GRASS_A) {
          col.copy(G1).lerp(G2, nz);
          if (a > BAND_A) col.copy(SD);
        } else if (a <= SAND_A) col.copy(SD).lerp(SW, (a - GRASS_A) / (SAND_A - GRASS_A) > 0.5 ? 1 : 0);
        else col.copy(SW).lerp(SU, Math.min(1, ((a - SAND_A) / (1 - SAND_A)) * 1.6));
        const k = (c * (ROWS + 1) + r) * 3;
        P[k] = x;
        P[k + 1] = y - (1 - K[c]) * TAPER_SINK;
        P[k + 2] = sgn * z;
        Cc[k] = col.r;
        Cc[k + 1] = col.g;
        Cc[k + 2] = col.b;
      }
    }
    g.attributes.position.needsUpdate = true;
    g.attributes.color.needsUpdate = true;
    g.computeVertexNormals();
    g.computeBoundingSphere();
    const sg = surf.geometry, SP = sg.attributes.position.array as Float32Array, D = sg.attributes.dist.array as Float32Array;
    let k = 0;
    for (let side = 0; side < 2; side++) {
      const sgn = side ? 1 : -1, A = side ? WS : WN;
      for (let c = 0; c <= cols; c++) {
        const x = lx0 + c * STEP, wl = A[c] + 2.83;
        for (let r = 0; r <= SURF_R; r++) {
          const dd = -1.0 + Math.pow(r / SURF_R, 1.6) * 13;
          SP[k * 3] = x;
          SP[k * 3 + 1] = (dd < 0 ? -0.3 - dd * 0.16 + 0.012 : -0.288) - (1 - K[c]) * TAPER_SINK; // sinks with a tapered end
          SP[k * 3 + 2] = sgn * (wl + dd);
          D[k] = dd - (1 - K[c]) * 40; // a tapered end: the surf is discarded there (it would show through the sea)
          k++;
        }
      }
    }
    sg.attributes.position.needsUpdate = true;
    sg.attributes.dist.needsUpdate = true;
  };

  // clouds: boxes in three parallax layers, tied to eye level so they always sit above the horizon; one instanced mesh
  const cloudGeo = boxes(CLOUD_BOXES);
  const clouds = new THREE.InstancedMesh(cloudGeo, cloudMat, CLOUDS.length);
  clouds.frustumCulled = false;
  scene.add(clouds);
  let cloudT = 0, tick = 0, cloudsPlaced = false;
  const cm = new THREE.Matrix4();
  const placeClouds = (view: { camY: number; target: THREE.Vector3 }) => {
    for (let i = 0; i < CLOUDS.length; i++) {
      const p = cloudPose(i, cloudT, view);
      clouds.setMatrixAt(i, cm.makeScale(p.s, p.s, p.s).setPosition(p.x, p.y, p.z));
    }
    clouds.instanceMatrix.needsUpdate = true;
    cloudsPlaced = true;
  };

  return {
    update(width, xFront, xStart) {
      const span = landSpan(xFront, xStart);
      if (!span) return false; // a bad time: keep the strip as it is
      let changed = false;
      // grown past the front, or the range's start moved the strip's start (a range switch)
      if (!land || (xFront + 80 > lx1 && span.cols > cols) || span.lx0 !== lx0) {
        allocate(span);
        changed = true;
      }
      const fin = (w: number) => (Number.isFinite(w) ? w : 0);
      for (let c = 0; c <= cols; c++) {
        const x = lx0 + c * STEP;
        K[c] = taper(x, lx0, lx1);
        WN[c] = fin(width(x, -1)) * K[c];
        WS[c] = fin(width(x, 1)) * K[c];
      }
      let h = 0;
      for (let c = 0; c <= cols; c += 3) h = (h * 31 + Math.round(WN[c] * 20) + Math.round(WS[c] * 20) * 7) >>> 0;
      if (h !== hashW || changed) {
        hashW = h;
        rewrite();
        changed = true;
      }
      return changed;
    },
    startX: () => lx0,
    endX: () => (land ? lx1 : Infinity),
    fitSea: (camX, camZ, reach) => placeSea(seaFit(camX, camZ, reach), reach),
    step(dt, view) {
      tick += dt;
      if (tick < 0.042 && cloudsPlaced) return false;
      const t = tick;
      tick = 0;
      // compact (card) mode: no surf, sea or cloud animation; the clouds still follow the eye level, and nothing here
      // asks for a redraw (a camera move does)
      if (!opts.compact) {
        SU_.uTime.value += t;
        cloudT += t;
      }
      // far off (a wide range's fit) the foam lines are sub-pixel and alias into flickering dashes: fade them out
      const dist = view.camDist ?? 0, foam = Math.min(1, Math.max(0, 1 - (dist - FOAM_NEAR) / FOAM_FADE));
      const foamChanged = foam !== SU_.uFoam.value;
      SU_.uFoam.value = foam;
      placeClouds(view);
      return !opts.compact || foamChanged;
    },
    terraces: () => hills?.count ?? 0,
    dispose() {
      for (const m of [land, surf, sea]) if (m) (scene.remove(m), m.geometry.dispose());
      if (hills) (scene.remove(hills), hills.dispose());
      scene.remove(clouds);
      clouds.dispose();
      for (const g of [hillGeo, cloudGeo]) g.dispose();
      for (const x of [surfMat, seaMat, ...owned]) x.dispose();
    },
  };
}
