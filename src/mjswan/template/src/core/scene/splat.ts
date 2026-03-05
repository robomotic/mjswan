import * as THREE from 'three';
import { SplatMesh } from '@sparkjsdev/spark';
export type { SplatMesh };

export interface SplatConfig {
  name: string;
  url: string;
  scale?: number;
  xOffset?: number;
  yOffset?: number;
  zOffset?: number;
  /** Roll in degrees applied on top of the COLMAP→Three.js base rotation. */
  roll?: number;
  /** Pitch in degrees applied on top of the COLMAP→Three.js base rotation. */
  pitch?: number;
  /** Yaw in degrees applied on top of the COLMAP→Three.js base rotation. */
  yaw?: number;
  colliderUrl?: string;
  /** If true, shows scale and offset controls in the viewer control panel. */
  control?: boolean;
}

const DEG2RAD = Math.PI / 180;

export function loadSplat(config: SplatConfig, scene: THREE.Scene): SplatMesh {
  const splat = new SplatMesh({ url: config.url });

  const scale = config.scale ?? 1.0;
  const xOffset = config.xOffset ?? 0.0;
  const yOffset = config.yOffset ?? 0.0;
  const zOffset = config.zOffset ?? 0.0;
  const roll  = (config.roll  ?? 0.0) * DEG2RAD;
  const pitch = (config.pitch ?? 0.0) * DEG2RAD;
  const yaw   = (config.yaw   ?? 0.0) * DEG2RAD;

  splat.scale.setScalar(scale);

  // WorldLabs splats use COLMAP/OpenCV convention (Y-down, Z-into-scene).
  // Rotating 180° around X flips to Three.js convention (Y-up, Z-towards-viewer).
  // User roll/pitch/yaw are applied on top via quaternion composition.
  const baseQuat = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI, 0, 0));
  const userQuat = new THREE.Quaternion().setFromEuler(new THREE.Euler(pitch, yaw, roll));
  splat.quaternion.copy(baseQuat.multiply(userQuat));

  splat.position.set(xOffset * scale, zOffset * scale, yOffset * scale);

  scene.add(splat);
  return splat;
}

export function disposeSplat(splat: SplatMesh, scene: THREE.Scene): void {
  scene.remove(splat);
  splat.dispose?.();
}
