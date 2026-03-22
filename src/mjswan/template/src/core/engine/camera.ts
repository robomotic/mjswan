import * as THREE from 'three';
import type { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { MjData, MjModel } from 'mujoco';
import { mjcToThreeCoordinate } from '../scene/coordinate';

export type CameraConfig = {
  /** Initial camera position in MuJoCo coordinates [x, y, z]. */
  position?: [number, number, number];
  /** Initial look-at target in MuJoCo coordinates [x, y, z]. */
  target?: [number, number, number];
  /** Vertical field of view in degrees. */
  fov?: number;
  /** Body name for the orbit target to follow each frame. */
  trackBodyName?: string;
  /** Name of a MuJoCo camera defined in the scene XML. Locks to its pose. */
  mujocoCamera?: string;
};

export type CameraState = {
  /** Body index to track, or null. */
  trackBodyId: number | null;
  /** MuJoCo camera index to use as a fixed camera, or null. */
  fixedCamIndex: number | null;
  /** Previous body world position used to compute per-frame delta for parallel tracking. */
  prevBodyPos: THREE.Vector3 | null;
};

const DEFAULT_POSITION = new THREE.Vector3(2.0, 1.7, 1.7);
const DEFAULT_TARGET = new THREE.Vector3(0, 0.2, 0);
const DEFAULT_FOV = 45;

/**
 * Apply a CameraConfig after a scene loads.
 *
 * Converts MuJoCo coordinates (x forward, y left, z up) to Three.js
 * (x right, y up, z out). Returns the camera state that runtime.ts
 * must keep to drive per-frame updates.
 *
 * Mirrors the pattern of createLights() in lights.ts.
 */
export function applyCameraConfig(
  config: CameraConfig | null,
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  mjModel: MjModel | null
): CameraState {
  const state: CameraState = { trackBodyId: null, fixedCamIndex: null, prevBodyPos: null };
  controls.enabled = true;

  if (!config) {
    camera.fov = DEFAULT_FOV;
    camera.updateProjectionMatrix();
    camera.position.copy(DEFAULT_POSITION);
    controls.target.copy(DEFAULT_TARGET);
    controls.update();
    return state;
  }

  if (config.fov != null) {
    camera.fov = config.fov;
    camera.updateProjectionMatrix();
  }

  if (config.position != null) {
    camera.position.copy(mjcToThreeCoordinate(config.position));
  }

  if (config.target != null) {
    controls.target.copy(mjcToThreeCoordinate(config.target));
  }

  if (config.position != null || config.target != null) {
    controls.update();
  }

  if (config.trackBodyName && mjModel) {
    for (let b = 0; b < mjModel.nbody; b++) {
      if (mjModel.body(b).name === config.trackBodyName) {
        state.trackBodyId = b;
        break;
      }
    }
    if (state.trackBodyId === null) {
      console.warn(`[Camera] trackBodyName: body "${config.trackBodyName}" not found.`);
    }
  }

  if (config.mujocoCamera && mjModel) {
    for (let c = 0; c < mjModel.ncam; c++) {
      if (mjModel.cam(c).name === config.mujocoCamera) {
        state.fixedCamIndex = c;
        controls.enabled = false;
        break;
      }
    }
    if (state.fixedCamIndex === null) {
      console.warn(`[Camera] mujocoCamera: camera "${config.mujocoCamera}" not found.`);
    }
  }

  return state;
}

/**
 * Update the Three.js camera each frame for body tracking or fixed MuJoCo cameras.
 *
 * Must be called before controls.update().
 * Mirrors the pattern of updateLightsFromData() in lights.ts.
 */
export function updateCameraFromData(
  mjData: MjData,
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  state: CameraState
): void {
  if (state.fixedCamIndex !== null) {
    // Fixed MuJoCo camera: read world position and orientation from mjData.
    // cam_xpos: ncam×3 world positions.
    // cam_xmat: ncam×9 row-major rotation matrices (camera-to-world).
    //   Column 2 = camera Z axis in world frame; camera looks along -Z.
    const ci = state.fixedCamIndex;
    const pos = mjcToThreeCoordinate(mjData.cam_xpos.slice(ci * 3, ci * 3 + 3));
    camera.position.copy(pos);

    const m = mjData.cam_xmat.slice(ci * 9, ci * 9 + 9);
    const lookDirMJ = [-m[2], -m[5], -m[8]] as [number, number, number];
    camera.lookAt(pos.clone().add(mjcToThreeCoordinate(lookDirMJ)));
  } else if (state.trackBodyId !== null) {
    // Parallel tracking: translate both the camera and the orbit target by the
    // body's delta each frame, preserving the camera angle and zoom level.
    const b = state.trackBodyId;
    const bodyPos = mjcToThreeCoordinate(mjData.xpos.slice(b * 3, b * 3 + 3));
    if (state.prevBodyPos !== null) {
      const delta = bodyPos.clone().sub(state.prevBodyPos);
      camera.position.add(delta);
      controls.target.add(delta);
    }
    state.prevBodyPos = bodyPos;
  }
}
