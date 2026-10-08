/**
 * README §4.4 — the sky and the light, in the voxel look (Plan 4: always day). Ported from the style mockup
 * (forest-styles.html: the cartoon `SKY_FS` dome with `sky: day`, its fog, ambient and key light, `placeSun`). There is
 * no sun, moon, stars or clock: the sky is a flat cyan dome, lighter toward the horizon, at every hour (this supersedes
 * README §4.4's day/night cycle). The key light keeps the mockup's direction and travels with the view (its shadow box).
 */
import * as THREE from "three";

export const SKY_VS = "varying vec3 p;void main(){p=normalize(position);gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}";
/**
 * Mockup `SKY_FS` by day: cyan, lighter toward the horizon; below it the horizon colour (the sea fades into the same
 * colour before the far plane, land.ts). Linear output (the frame is encoded to sRGB at the end).
 */
export const SKY_FS = `varying vec3 p;
void main(){vec3 d=normalize(p);float h=max(d.y,0.0);
  vec3 c=mix(vec3(.66,.93,1.0),vec3(.22,.66,1.0),smoothstep(0.0,0.45,h));
  gl_FragColor=vec4(pow(c,vec3(2.2)),1.);}`;

/** The style mockup's day fog colour (the sky's horizon). */
const DAY_FOG = "#a8e4ff";
/** The mockup's key light direction (12, 22, 10). */
const KEY_DIR = new THREE.Vector3(12, 22, 10).normalize();

export interface Sky {
  dome: THREE.Mesh;
  key: THREE.DirectionalLight;
  /** Moves the key light and its shadow box with the view; true when it moved (the shadow map must update). */
  follow(target: THREE.Vector3): boolean;
  dispose(): void;
}

export function createSky(scene: THREE.Scene, opts: { shadows: boolean }): Sky {
  const dome = new THREE.Mesh(
    new THREE.SphereGeometry(400, 48, 24),
    new THREE.ShaderMaterial({ side: THREE.BackSide, depthWrite: false, fog: false, vertexShader: SKY_VS, fragmentShader: SKY_FS }),
  );
  dome.frustumCulled = false;
  dome.renderOrder = -2;
  scene.add(dome);
  scene.background = new THREE.Color("#5cc8ff");
  // light day fog (the style mockup's day fog colour); its density follows the view (frame.ts viewDepth)
  scene.fog = new THREE.FogExp2(DAY_FOG, 0.0011);

  // the style mockup's day lights: a white ambient and a warm white key (no tinted hemisphere, no rim)
  const amb = new THREE.AmbientLight("#ffffff", 1.25);
  const key = new THREE.DirectionalLight("#fff4dc", 2.3);
  key.position.set(12, 22, 10);
  key.castShadow = opts.shadows;
  key.shadow.mapSize.set(2048, 2048);
  Object.assign(key.shadow.camera, { left: -45, right: 45, top: 45, bottom: -45, far: 120 });
  key.shadow.radius = 5;
  scene.add(amb, key, key.target);
  const at = new THREE.Vector3(1e9, 0, 0);

  return {
    dome,
    key,
    follow(target) {
      if (at.distanceTo(target) < 0.5) return false;
      at.copy(target);
      key.position.copy(at).addScaledVector(KEY_DIR, 60);
      key.target.position.copy(at);
      key.target.updateMatrixWorld();
      return true;
    },
    dispose() {
      scene.remove(dome, amb, key, key.target);
      dome.geometry.dispose();
      (dome.material as THREE.Material).dispose();
      key.shadow.map?.dispose();
    },
  };
}
