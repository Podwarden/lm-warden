/**
 * The voxel look's materials (Plan 4, the chosen style: cel 4 steps, saturation 1.4, no outlines, no pixel filter, cyan
 * growth tint). Growth is animated only in the shaders (`uNow` = the history time shown): a cube scales in from 0 over
 * SCALE_IN_S from its `iborn` and is tinted cyan for TINT_S, fading out (voxels.ts holds the CPU copies of both rules).
 * Ported from the style mockup (`GRAD`/`setCel`, `toon`, `growInject`, the `PULSE` tint, the saturation pass).
 * Created once per scene by `createMaterials` (no canvas: works in tests too).
 */
import * as THREE from "three";
import { CEL_STEPS, SATURATION, SCALE_IN_S, TINT_S, cubeGeometry } from "./voxels";
import { GHOST_CROWN } from "./ghostColors";

const f1 = (x: number) => x.toFixed(1);
/** The tip markers: white mixed 0.6 toward the growth tint's cyan (0.3, 0.75, 1.0). */
const TIP_COLOR = new THREE.Color(1, 1, 1).lerp(new THREE.Color(0.3, 0.75, 1.0), 0.6);

/**
 * The style mockup's saturation pass (its `QM` quad): one pass over the whole frame, on the linear render (no tone
 * mapping), before the sRGB encode. scene.ts adds it to the composer just before the OutputPass.
 */
export const SaturationShader = {
  uniforms: { tDiffuse: { value: null as THREE.Texture | null }, uSat: { value: SATURATION } },
  vertexShader: "varying vec2 vUv;void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}",
  fragmentShader:
    "uniform sampler2D tDiffuse;uniform float uSat;varying vec2 vUv;void main(){vec4 t=texture2D(tDiffuse,vUv);" +
    "float l=dot(t.rgb,vec3(0.2126,0.7152,0.0722));gl_FragColor=vec4(max(mix(vec3(l),t.rgb,uSat),0.0),t.a);}",
};

/**
 * The cube patch (`#include` chunk → text appended after it):
 * - scale-in: smoothstep of (uNow − iborn) / SCALE_IN_S about the cube's centre (before the instance matrix);
 * - growth tint: diffuse mixed 0.6 toward the mockup's cyan (0.3, 0.75, 1.0), weight 1 → 0 over TINT_S;
 * - cross-fade: a screen-space dither (interleaved gradient noise); the level fading in keeps h < uFade, the one fading
 *   out keeps h ≥ uFade (uFade = uFadeIn = 1: everything, the resting state).
 * Saturation is one pass over the whole frame (`SaturationShader`).
 */
export const VOXEL_PATCH = {
  vertexCommon: "attribute float iborn;uniform float uNow;varying float vGlow;",
  beginVertex:
    `float age=uNow-iborn;float gs=clamp(age/${f1(SCALE_IN_S)},0.0,1.0);gs=gs*gs*(3.0-2.0*gs);transformed*=max(gs,0.0001);` +
    `vGlow=step(0.0,age)*clamp(1.0-age/${f1(TINT_S)},0.0,1.0);`,
  fragmentCommon: "varying float vGlow;uniform float uFade;uniform float uFadeIn;",
  dither:
    "if(uFade<1.0){float h_=fract(52.9829189*fract(dot(gl_FragCoord.xy,vec2(0.06711056,0.00583715))));" +
    "if(uFadeIn>0.5?h_>=uFade:h_<uFade)discard;}",
  color: "diffuseColor.rgb=mix(diffuseColor.rgb,vec3(0.3,0.75,1.0),0.6*vGlow);",
} as const;

/** The whole cube patch as one string (for inspection and tests). */
export const VOXEL_SHADER_PATCH = Object.values(VOXEL_PATCH).join("\n");

export interface GrowUniforms {
  uNow: { value: number };
}

/** A cube material's own cross-fade state. */
export interface FadeUniforms {
  uFade: { value: number };
  uFadeIn: { value: number };
}

export type VoxelMaterial = THREE.MeshToonMaterial & { fade: FadeUniforms };

/** Mockup `GRAD` / `setCel(n)`: a nearest-sampled ramp of n steps from 0.4 to 1. */
export function celRamp(n = CEL_STEPS): THREE.DataTexture {
  const d = new Uint8Array(64);
  for (let i = 0; i < 64; i++) {
    const st = Math.min(n - 1, Math.floor((i / 64) * n));
    d[i] = Math.round(255 * (0.4 + (0.6 * st) / Math.max(1, n - 1)));
  }
  const t = new THREE.DataTexture(d, 64, 1, THREE.RedFormat);
  t.minFilter = t.magFilter = THREE.NearestFilter;
  t.generateMipmaps = false;
  t.needsUpdate = true;
  return t;
}

function voxelMaterial(U: GrowUniforms, grad: THREE.Texture, fadingIn: boolean | null): VoxelMaterial {
  const m = new THREE.MeshToonMaterial({ vertexColors: true, gradientMap: grad }) as VoxelMaterial;
  m.fade = { uFade: { value: fadingIn === null ? 1 : 0 }, uFadeIn: { value: fadingIn === false ? 0 : 1 } };
  const P = VOXEL_PATCH;
  m.onBeforeCompile = (sh) => {
    sh.uniforms.uNow = U.uNow;
    sh.uniforms.uFade = m.fade.uFade;
    sh.uniforms.uFadeIn = m.fade.uFadeIn;
    sh.vertexShader = sh.vertexShader
      .replace("#include <common>", "#include <common>\n" + P.vertexCommon)
      .replace("#include <begin_vertex>", "#include <begin_vertex>\n" + P.beginVertex);
    sh.fragmentShader = sh.fragmentShader
      .replace("#include <common>", "#include <common>\n" + P.fragmentCommon)
      .replace("#include <clipping_planes_fragment>", "#include <clipping_planes_fragment>\n" + P.dither)
      .replace("#include <color_fragment>", "#include <color_fragment>\n" + P.color);
  };
  m.customProgramCacheKey = () => "forest-voxel";
  return m;
}

/** A plain toon material of the look (the cel ramp, no growth): flora and the tip markers. */
const toonMaterial = (grad: THREE.Texture, o: THREE.MeshToonMaterialParameters = {}) => new THREE.MeshToonMaterial({ gradientMap: grad, ...o });

export interface Materials {
  uniforms: GrowUniforms;
  /** The cubes at rest (no cross-fade). */
  voxel: VoxelMaterial;
  /** A cube material for one level of a cross-fade (its own `fade` uniforms); dispose it when the fade ends. */
  fading(fadingIn: boolean): VoxelMaterial;
  /** Shadow depth of the cubes, scaled in like them. */
  voxelDepth: THREE.MeshDepthMaterial;
  /** The unit cube of every cell. */
  cube: THREE.BufferGeometry;
  /** The cel ramp (4 steps). */
  gradient: THREE.DataTexture;
  /** A toon material of the look (flora). */
  toon(o?: THREE.MeshToonMaterialParameters): THREE.MeshToonMaterial;
  /** Marks the highlighted session's joints: small voxel cubes in the growth tint's cyan (with `cube`). */
  tip: THREE.MeshToonMaterial;
  /** Ghost of a transplanted tree's old crown; one material per ghost (its own opacity). */
  ghost(): THREE.MeshBasicMaterial;
  /** A ghost's ground: soil and stone blocks (toon, white: the instance colours show; flights.ts `ghostGround`). */
  ghostGroundMat: THREE.MeshToonMaterial;
  dispose(): void;
}

export function createMaterials(): Materials {
  const uniforms: GrowUniforms = { uNow: { value: 1e12 } };
  const gradient = celRamp();
  const voxel = voxelMaterial(uniforms, gradient, null);
  const voxelDepth = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking });
  voxelDepth.onBeforeCompile = (sh) => {
    sh.uniforms.uNow = uniforms.uNow;
    sh.vertexShader = sh.vertexShader
      .replace("#include <common>", "#include <common>\nattribute float iborn;uniform float uNow;")
      .replace("#include <begin_vertex>", `#include <begin_vertex>\nfloat gs=clamp((uNow-iborn)/${f1(SCALE_IN_S)},0.0,1.0);gs=gs*gs*(3.0-2.0*gs);transformed*=max(gs,0.0001);`);
  };
  voxelDepth.customProgramCacheKey = () => "forest-voxel-depth";
  const cube = cubeGeometry();
  const tip = toonMaterial(gradient, { vertexColors: true, color: TIP_COLOR, emissive: TIP_COLOR, emissiveIntensity: 0.35 });
  const ghostGroundMat = toonMaterial(gradient, { vertexColors: true });
  const owned: THREE.Material[] = [];
  return {
    uniforms, voxel, voxelDepth, cube, gradient, tip, ghostGroundMat,
    fading: (fadingIn) => voxelMaterial(uniforms, gradient, fadingIn),
    toon: (o) => {
      const m = toonMaterial(gradient, o);
      owned.push(m);
      return m;
    },
    ghost: () => new THREE.MeshBasicMaterial({ color: GHOST_CROWN, transparent: true, opacity: 0.3, depthWrite: false }),
    dispose() {
      for (const m of [voxel, voxelDepth, tip, ghostGroundMat, ...owned]) m.dispose();
      gradient.dispose();
      cube.dispose();
    },
  };
}
