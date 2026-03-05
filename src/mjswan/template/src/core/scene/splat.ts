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
  colliderUrl?: string;
  /** If true, shows scale and offset controls in the viewer control panel. */
  control?: boolean;
}

export function loadSplat(config: SplatConfig, scene: THREE.Scene): SplatMesh {
  const splat = new SplatMesh({ url: config.url });

  const scale = config.scale ?? 1.0;
  const xOffset = config.xOffset ?? 0.0;
  const yOffset = config.yOffset ?? 0.0;
  const zOffset = config.zOffset ?? 0.0;

  splat.scale.setScalar(scale);

  // WorldLabs splats use COLMAP/OpenCV convention (Y-down, Z-into-scene).
  // Rotating 180° around X flips to Three.js convention (Y-up, Z-towards-viewer).
  splat.rotation.x = Math.PI;

  splat.position.set(xOffset * scale, zOffset * scale, yOffset * scale);

  scene.add(splat);
  return splat;
}

export function disposeSplat(splat: SplatMesh, scene: THREE.Scene): void {
  scene.remove(splat);
  splat.dispose?.();
}
