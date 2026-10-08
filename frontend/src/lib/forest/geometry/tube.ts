/**
 * The mockup's three.js maths without three.js: a tiny `V3` (THREE.Vector3's methods the mockup uses, same names).
 * The old look's colours, Catmull-Rom curve and `taperedTube` (forest-real.html l.240) are gone with the bark (Plan 4).
 */

export class V3 {
  constructor(public x = 0, public y = 0, public z = 0) {}
  clone(): V3 {
    return new V3(this.x, this.y, this.z);
  }
  set(x: number, y: number, z: number): V3 {
    this.x = x;
    this.y = y;
    this.z = z;
    return this;
  }
  setY(y: number): V3 {
    this.y = y;
    return this;
  }
  add(v: ReadonlyV3): V3 {
    return this.set(this.x + v.x, this.y + v.y, this.z + v.z);
  }
  sub(v: ReadonlyV3): V3 {
    return this.set(this.x - v.x, this.y - v.y, this.z - v.z);
  }
  addScaledVector(v: ReadonlyV3, s: number): V3 {
    return this.set(this.x + v.x * s, this.y + v.y * s, this.z + v.z * s);
  }
  multiplyScalar(s: number): V3 {
    return this.set(this.x * s, this.y * s, this.z * s);
  }
  dot(v: ReadonlyV3): number {
    return this.x * v.x + this.y * v.y + this.z * v.z;
  }
  length(): number {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
  }
  distanceToSquared(v: ReadonlyV3): number {
    const dx = this.x - v.x, dy = this.y - v.y, dz = this.z - v.z;
    return dx * dx + dy * dy + dz * dz;
  }
  distanceTo(v: ReadonlyV3): number {
    return Math.sqrt(this.distanceToSquared(v));
  }
  /** THREE: divides by `length() || 1`, so a zero vector stays zero. */
  normalize(): V3 {
    return this.multiplyScalar(1 / (this.length() || 1));
  }
  lerp(v: ReadonlyV3, a: number): V3 {
    return this.set(this.x + (v.x - this.x) * a, this.y + (v.y - this.y) * a, this.z + (v.z - this.z) * a);
  }
  cross(v: ReadonlyV3): V3 {
    return this.crossVectors(this, v);
  }
  crossVectors(a: ReadonlyV3, b: ReadonlyV3): V3 {
    const ax = a.x, ay = a.y, az = a.z, bx = b.x, by = b.y, bz = b.z;
    return this.set(ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx);
  }
  /** THREE `applyAxisAngle` (quaternion from a normalized axis, then `applyQuaternion`). */
  applyAxisAngle(axis: ReadonlyV3, angle: number): V3 {
    const s = Math.sin(angle / 2);
    return this.applyQuat(axis.x * s, axis.y * s, axis.z * s, Math.cos(angle / 2));
  }
  applyQuat(qx: number, qy: number, qz: number, qw: number): V3 {
    const vx = this.x, vy = this.y, vz = this.z;
    const tx = 2 * (qy * vz - qz * vy), ty = 2 * (qz * vx - qx * vz), tz = 2 * (qx * vy - qy * vx);
    return this.set(vx + qw * tx + qy * tz - qz * ty, vy + qw * ty + qz * tx - qx * tz, vz + qw * tz + qx * ty - qy * tx);
  }
  /** THREE `applyMatrix4(makeRotationAxis(axis, angle))` for a normalized axis. */
  rotateAxis(axis: ReadonlyV3, angle: number): V3 {
    const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
    const { x, y, z } = axis, vx = this.x, vy = this.y, vz = this.z;
    return this.set(
      (t * x * x + c) * vx + (t * x * y - s * z) * vy + (t * x * z + s * y) * vz,
      (t * x * y + s * z) * vx + (t * y * y + c) * vy + (t * y * z - s * x) * vz,
      (t * x * z - s * y) * vx + (t * y * z + s * x) * vy + (t * z * z + c) * vz,
    );
  }
}

/** A vector that may be read and cloned but never changed (shared constants, inputs). */
export type ReadonlyV3 = Readonly<Pick<V3, "x" | "y" | "z" | "clone" | "dot" | "length" | "distanceTo" | "distanceToSquared">>;

/** A frozen shared constant: a mutation throws (modules are strict), so nothing leaks between tree builds. */
export const frozen = (v: V3): ReadonlyV3 => Object.freeze(v);
