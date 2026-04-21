import * as THREE from 'three';

import { getPosition, getQuaternion } from '../scene/scene';
import { type NpzEntry, loadNpz } from '../scene/npz';
import type { CommandConfigEntry, CommandTerm, CommandTermContext, CommandUiConfig } from './types';

export type TrackingMotionConfig = {
  name: string;
  path: string;
  fps: number;
  anchor_body_name: string;
  body_names: string[];
  dataset_joint_names?: string[];
  default?: boolean;
};

type LoadedTrackingMotion = TrackingMotionConfig & {
  jointPos: Float32Array[];
  jointVel: Float32Array[];
  bodyPosW: Float32Array[];
  bodyQuatW: Float32Array[];
  bodyLinVelW: Float32Array[];
  bodyAngVelW: Float32Array[];
  frameCount: number;
};

function normalizeQuat(quat: ArrayLike<number>): Float32Array {
  const length = Math.hypot(quat[0] ?? 1, quat[1] ?? 0, quat[2] ?? 0, quat[3] ?? 0) || 1.0;
  return new Float32Array([
    (quat[0] ?? 1) / length,
    (quat[1] ?? 0) / length,
    (quat[2] ?? 0) / length,
    (quat[3] ?? 0) / length,
  ]);
}

function splitFrames(entry: NpzEntry): Float32Array[] {
  const totalFrames = entry.shape[0] ?? 0;
  const width = entry.shape.length <= 1 ? 1 : entry.shape.slice(1).reduce((acc, v) => acc * v, 1);
  const frames: Float32Array[] = [];
  for (let i = 0; i < totalFrames; i++) {
    const out = new Float32Array(width);
    const start = i * width;
    for (let j = 0; j < width; j++) {
      out[j] = entry.data[start + j] ?? 0.0;
    }
    frames.push(out);
  }
  return frames;
}

function quatMultiply(a: ArrayLike<number>, b: ArrayLike<number>): Float32Array {
  const aw = a[0] ?? 1;
  const ax = a[1] ?? 0;
  const ay = a[2] ?? 0;
  const az = a[3] ?? 0;
  const bw = b[0] ?? 1;
  const bx = b[1] ?? 0;
  const by = b[2] ?? 0;
  const bz = b[3] ?? 0;
  return new Float32Array([
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ]);
}

function quatInverse(quat: ArrayLike<number>): Float32Array {
  const q = normalizeQuat(quat);
  return new Float32Array([q[0], -q[1], -q[2], -q[3]]);
}

function quatApply(quat: ArrayLike<number>, vec: ArrayLike<number>): Float32Array {
  const q = normalizeQuat(quat);
  const vx = vec[0] ?? 0;
  const vy = vec[1] ?? 0;
  const vz = vec[2] ?? 0;
  const tx = 2.0 * (q[2] * vz - q[3] * vy);
  const ty = 2.0 * (q[3] * vx - q[1] * vz);
  const tz = 2.0 * (q[1] * vy - q[2] * vx);
  const cx = q[2] * tz - q[3] * ty;
  const cy = q[3] * tx - q[1] * tz;
  const cz = q[1] * ty - q[2] * tx;
  return new Float32Array([
    vx + q[0] * tx + cx,
    vy + q[0] * ty + cy,
    vz + q[0] * tz + cz,
  ]);
}

function yawQuat(quat: ArrayLike<number>): Float32Array {
  const q = normalizeQuat(quat);
  const yaw = Math.atan2(
    2 * (q[0] * q[3] + q[1] * q[2]),
    1 - 2 * (q[2] * q[2] + q[3] * q[3]),
  );
  return new Float32Array([Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)]);
}

function setGhostMaterial(material: THREE.Material): THREE.Material {
  const next = material.clone();
  if ('transparent' in next) {
    next.transparent = true;
  }
  if ('opacity' in next) {
    next.opacity = 0.35;
  }
  if ('depthWrite' in next) {
    next.depthWrite = false;
  }
  if ('color' in next && next.color instanceof THREE.Color) {
    next.color = new THREE.Color('#7bd88f');
  }
  return next;
}

function hasRenderableMesh(object: THREE.Object3D): boolean {
  let found = false;
  object.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      found = true;
    }
  });
  return found;
}

export class TrackingCommand implements CommandTerm {
  private readonly context: CommandTermContext;
  private readonly motions: TrackingMotionConfig[];
  private readonly loadedMotions: Map<string, LoadedTrackingMotion>;
  private sampleHz: number;
  private readonly ghostRoot: THREE.Group | null;
  private readonly ghostBodies: Map<number, THREE.Group>;
  private readonly ghostData: import('mujoco').MjData | null;
  private refBodyPosW: Float32Array[];
  private refBodyQuatW: Float32Array[];
  private refBodyLinVelW: Float32Array[];
  private refBodyAngVelW: Float32Array[];
  private selectedMotionName: string | null;
  private selectedMotion: LoadedTrackingMotion | null;
  private selectedAnchorBodyIndex: number;
  private selectedRootBodyIndex: number;
  private datasetQposAdr: number[];
  private frameAccumulator: number;
  private referenceVisible: boolean;
  refJointPos: Float32Array[];
  refRootPos: Float32Array[];
  refRootQuat: Float32Array[];
  refIdx: number;
  refLen: number;
  nJoints: number;

  constructor(
    _termName: string,
    config: CommandConfigEntry,
    context: CommandTermContext,
  ) {
    this.context = context;
    this.motions = Array.isArray(config.motions) ? config.motions as TrackingMotionConfig[] : [];
    this.loadedMotions = new Map();
    this.sampleHz = 50.0;
    this.selectedMotionName =
      this.motions.find((motion) => motion.default)?.name ??
      this.motions[0]?.name ??
      null;
    this.selectedMotion = null;
    this.selectedAnchorBodyIndex = 0;
    this.selectedRootBodyIndex = 0;
    this.datasetQposAdr = [];
    this.frameAccumulator = 0.0;
    this.referenceVisible = true;
    this.refJointPos = [];
    this.refRootPos = [];
    this.refRootQuat = [];
    this.refIdx = 0;
    this.refLen = 0;
    this.nJoints = this.motions.find((motion) => motion.name === this.selectedMotionName)?.dataset_joint_names?.length ?? 0;

    this.ghostBodies = new Map();
    this.ghostData = context.mjModel ? new context.mujoco.MjData(context.mjModel) : null;
    this.ghostRoot = this.createGhostRoot();
    this.refBodyPosW = [];
    this.refBodyQuatW = [];
    this.refBodyLinVelW = [];
    this.refBodyAngVelW = [];
  }

  getCommand(): Float32Array {
    if (!this.selectedMotion || this.refLen === 0) {
      return new Float32Array(this.nJoints * 2);
    }
    const jointPos = this.refJointPos[this.refIdx] ?? new Float32Array(this.nJoints);
    const jointVel = this.selectedMotion.jointVel[this.refIdx] ?? new Float32Array(this.nJoints);
    const out = new Float32Array(jointPos.length + jointVel.length);
    out.set(jointPos, 0);
    out.set(jointVel, jointPos.length);
    return out;
  }

  getUiConfig(): CommandUiConfig | null {
    return null;
  }

  async setSelectedMotion(name: string | null): Promise<boolean> {
    const nextName = name ?? this.selectedMotionName;
    if (!nextName) {
      this.selectedMotionName = null;
      this.selectedMotion = null;
      this.refJointPos = [];
      this.refRootPos = [];
      this.refRootQuat = [];
      this.refBodyPosW = [];
      this.refBodyQuatW = [];
      this.refBodyLinVelW = [];
      this.refBodyAngVelW = [];
      this.refLen = 0;
      this.nJoints = 0;
      this.updateGhostPose();
      return false;
    }

    const config = this.motions.find((motion) => motion.name === nextName);
    if (!config) {
      return false;
    }

    const loaded = this.loadedMotions.get(nextName) ?? await this.loadMotion(config);
    this.loadedMotions.set(nextName, loaded);
    this.selectedMotionName = nextName;
    this.selectedMotion = loaded;
    this.selectedAnchorBodyIndex = Math.max(
      0,
      loaded.body_names.indexOf(loaded.anchor_body_name),
    );
    this.selectedRootBodyIndex = this.resolveMotionRootBodyIndex(loaded.body_names);
    this.datasetQposAdr = this.resolveQposAdr(loaded.dataset_joint_names ?? []);
    this.refJointPos = loaded.jointPos;
    this.refIdx = 0;
    this.refLen = loaded.frameCount;
    this.nJoints = loaded.jointPos[0]?.length ?? 0;
    this.frameAccumulator = 0.0;
    this.updateReferenceAlignment();
    this.applyReferenceStateToSim();
    this.updateGhostPose();
    return true;
  }

  setReferenceVisible(visible: boolean): void {
    this.referenceVisible = visible;
    if (this.ghostRoot) {
      this.ghostRoot.visible = visible && this.selectedMotion !== null;
    }
  }

  reset(): void {
    this.refIdx = 0;
    this.frameAccumulator = 0.0;
    this.updateReferenceAlignment();
    this.applyReferenceStateToSim();
    this.updateGhostPose();
  }

  update(dt: number): void {
    if (!this.selectedMotion || this.refLen <= 1) {
      return;
    }
    this.frameAccumulator += dt * this.sampleHz;
    while (this.frameAccumulator >= 1.0 && this.refIdx < this.refLen - 1) {
      this.refIdx += 1;
      this.frameAccumulator -= 1.0;
    }
    this.updateGhostPose();
  }

  updateDebugVisuals(): void {
    if (this.ghostRoot) {
      this.ghostRoot.visible = this.referenceVisible && this.selectedMotion !== null;
    }
  }

  dispose(): void {
    if (this.ghostRoot) {
      this.context.scene.remove(this.ghostRoot);
      this.ghostRoot.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          if (Array.isArray(obj.material)) {
            for (const material of obj.material) {
              material.dispose?.();
            }
          } else {
            obj.material?.dispose?.();
          }
        }
      });
    }
    this.ghostData?.delete?.();
  }

  isReady(): boolean {
    return this.selectedMotion !== null && this.refLen > 0;
  }

  getSelectedMotion(): LoadedTrackingMotion | null {
    return this.selectedMotion;
  }

  getSelectedMotionName(): string | null {
    return this.selectedMotionName;
  }

  getAnchorBodyName(): string | null {
    return this.selectedMotion?.anchor_body_name
      ?? this.motions.find((motion) => motion.name === this.selectedMotionName)?.anchor_body_name
      ?? null;
  }

  getBodyNames(): string[] {
    return this.selectedMotion?.body_names
      ?? this.motions.find((motion) => motion.name === this.selectedMotionName)?.body_names
      ?? [];
  }

  getAnchorBodyIndex(): number {
    return this.selectedAnchorBodyIndex;
  }

  getAnchorPos(frameIndex = this.refIdx): Float32Array | null {
    const motion = this.selectedMotion;
    if (!motion) {
      return null;
    }
    const frame = motion.bodyPosW[frameIndex];
    const alignedFrame = this.refBodyPosW[frameIndex];
    if (!frame || !alignedFrame) {
      return null;
    }
    const offset = this.selectedAnchorBodyIndex * 3;
    return alignedFrame.slice(offset, offset + 3);
  }

  getAnchorQuat(frameIndex = this.refIdx): Float32Array | null {
    const motion = this.selectedMotion;
    if (!motion) {
      return null;
    }
    const frame = motion.bodyQuatW[frameIndex];
    const alignedFrame = this.refBodyQuatW[frameIndex];
    if (!frame || !alignedFrame) {
      return null;
    }
    const offset = this.selectedAnchorBodyIndex * 4;
    return normalizeQuat(alignedFrame.slice(offset, offset + 4));
  }

  getBodyPosW(frameIndex = this.refIdx): Float32Array | null {
    const motion = this.selectedMotion;
    if (!motion) {
      return null;
    }
    const frame = this.refBodyPosW[frameIndex];
    return frame ? frame.slice() : null;
  }

  private createGhostRoot(): THREE.Group | null {
    const bodies = this.context.bodies ?? null;
    const mjModel = this.context.mjModel;
    if (!bodies || !mjModel) {
      return null;
    }
    const root = new THREE.Group();
    root.name = 'Tracking Ghost';
    root.visible = false;
    for (const [bodyId, body] of Object.entries(bodies)) {
      const numericBodyId = Number(bodyId);
      if (!this.isDynamicBody(numericBodyId)) {
        continue;
      }
      const clone = body.clone(true);
      clone.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          if (Array.isArray(obj.material)) {
            obj.material = obj.material.map(setGhostMaterial);
          } else {
            obj.material = setGhostMaterial(obj.material);
          }
        }
      });
      if (!hasRenderableMesh(clone)) {
        continue;
      }
      this.ghostBodies.set(numericBodyId, clone);
      root.add(clone);
    }
    this.context.scene.add(root);
    return root;
  }

  private isDynamicBody(bodyId: number): boolean {
    const mjModel = this.context.mjModel;
    if (!mjModel || bodyId <= 0 || bodyId >= mjModel.nbody) {
      return false;
    }

    let current = bodyId;
    while (current > 0) {
      if (mjModel.body_jntnum[current] > 0) {
        return true;
      }
      current = mjModel.body_parentid[current];
    }
    return false;
  }

  private async loadMotion(config: TrackingMotionConfig): Promise<LoadedTrackingMotion> {
    this.sampleHz = config.fps;
    const npz = await loadNpz(config.path);
    const required = ['joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w'] as const;
    for (const key of required) {
      if (!npz[key]) {
        throw new Error(`Motion asset is missing '${key}'`);
      }
    }
    const jointPos = splitFrames(npz['joint_pos']!);
    const jointVel = splitFrames(npz['joint_vel']!);
    const bodyPosW = splitFrames(npz['body_pos_w']!);
    const bodyQuatW = splitFrames(npz['body_quat_w']!);
    const bodyLinVelW = splitFrames(npz['body_lin_vel_w']!);
    const bodyAngVelW = splitFrames(npz['body_ang_vel_w']!);
    return { ...config, jointPos, jointVel, bodyPosW, bodyQuatW, bodyLinVelW, bodyAngVelW, frameCount: jointPos.length };
  }

  private updateReferenceAlignment(): void {
    const motion = this.selectedMotion;
    const mjModel = this.context.mjModel;
    const mjData = this.context.mjData;
    if (!motion || !mjModel || !mjData || motion.frameCount === 0) {
      this.refRootPos = [];
      this.refRootQuat = [];
      this.refBodyPosW = [];
      this.refBodyQuatW = [];
      this.refBodyLinVelW = [];
      this.refBodyAngVelW = [];
      return;
    }

    const refAnchorOffset = this.selectedAnchorBodyIndex * 3;
    const refAnchorQuatOffset = this.selectedAnchorBodyIndex * 4;
    const refRootOffset = this.selectedRootBodyIndex * 3;
    const refAnchorPos0 = motion.bodyPosW[0]?.slice(refAnchorOffset, refAnchorOffset + 3);
    const refAnchorQuat0 = motion.bodyQuatW[0]?.slice(refAnchorQuatOffset, refAnchorQuatOffset + 4);
    const refRootPos0 = motion.bodyPosW[0]?.slice(refRootOffset, refRootOffset + 3);
    const currentAnchor = this.getCurrentAnchorPose();
    const spawnZOffset = this.getSpawnZOffset();
    if (!refAnchorPos0 || !refAnchorQuat0 || !refRootPos0 || !currentAnchor) {
      this.refBodyPosW = motion.bodyPosW.map((frame) => frame.slice());
      this.refBodyQuatW = motion.bodyQuatW.map((frame) => frame.slice());
      this.refBodyLinVelW = motion.bodyLinVelW.map((frame) => frame.slice());
      this.refBodyAngVelW = motion.bodyAngVelW.map((frame) => frame.slice());
      this.refRootPos = this.refBodyPosW.map((frame) =>
        frame.slice(this.selectedRootBodyIndex * 3, this.selectedRootBodyIndex * 3 + 3),
      );
      this.refRootQuat = this.refBodyQuatW.map((frame) =>
        normalizeQuat(frame.slice(this.selectedRootBodyIndex * 4, this.selectedRootBodyIndex * 4 + 4)),
      );
      return;
    }

    const yawDelta = yawQuat(quatMultiply(currentAnchor.quat, quatInverse(refAnchorQuat0)));
    const rotatedAnchor = quatApply(yawDelta, refAnchorPos0);
    const rotatedRoot = quatApply(yawDelta, refRootPos0);
    const offset = new Float32Array([
      currentAnchor.pos[0] - rotatedAnchor[0],
      currentAnchor.pos[1] - rotatedAnchor[1],
      spawnZOffset + refRootPos0[2] - rotatedRoot[2],
    ]);

    this.refBodyPosW = motion.bodyPosW.map((frame) => {
      const aligned = new Float32Array(frame.length);
      for (let i = 0; i < frame.length; i += 3) {
        const rotated = quatApply(yawDelta, frame.slice(i, i + 3));
        aligned[i + 0] = offset[0] + rotated[0];
        aligned[i + 1] = offset[1] + rotated[1];
        aligned[i + 2] = offset[2] + rotated[2];
      }
      return aligned;
    });
    this.refBodyQuatW = motion.bodyQuatW.map((frame) => {
      const aligned = new Float32Array(frame.length);
      for (let i = 0; i < frame.length; i += 4) {
        aligned.set(normalizeQuat(quatMultiply(yawDelta, frame.slice(i, i + 4))), i);
      }
      return aligned;
    });
    this.refBodyLinVelW = motion.bodyLinVelW.map((frame) => {
      const aligned = new Float32Array(frame.length);
      for (let i = 0; i < frame.length; i += 3) {
        aligned.set(quatApply(yawDelta, frame.slice(i, i + 3)), i);
      }
      return aligned;
    });
    this.refBodyAngVelW = motion.bodyAngVelW.map((frame) => {
      const aligned = new Float32Array(frame.length);
      for (let i = 0; i < frame.length; i += 3) {
        aligned.set(quatApply(yawDelta, frame.slice(i, i + 3)), i);
      }
      return aligned;
    });
    this.refRootPos = this.refBodyPosW.map((frame) =>
      frame.slice(this.selectedRootBodyIndex * 3, this.selectedRootBodyIndex * 3 + 3),
    );
    this.refRootQuat = this.refBodyQuatW.map((frame) =>
      normalizeQuat(frame.slice(this.selectedRootBodyIndex * 4, this.selectedRootBodyIndex * 4 + 4)),
    );
  }

  private getCurrentAnchorPose(): { pos: Float32Array; quat: Float32Array } | null {
    const mjModel = this.context.mjModel;
    const mjData = this.context.mjData;
    const anchorName = this.getAnchorBodyName();
    if (!mjModel || !mjData || !anchorName) {
      return null;
    }
    const bodyId = this.findBodyIdByName(anchorName);
    if (bodyId < 0) {
      return null;
    }
    return {
      pos: mjData.xpos.slice(bodyId * 3, bodyId * 3 + 3),
      quat: normalizeQuat(mjData.xquat.slice(bodyId * 4, bodyId * 4 + 4)),
    };
  }

  private getSpawnZOffset(): number {
    const mjModel = this.context.mjModel;
    const mjData = this.context.mjData;
    if (!mjModel || !mjData) {
      return 0.0;
    }
    const freeJointIndex = this.findFreeJointIndex();
    if (freeJointIndex < 0) {
      return 0.0;
    }
    const qposAdr = mjModel.jnt_qposadr[freeJointIndex];
    return (mjData.qpos[qposAdr + 2] ?? 0.0) - (mjModel.qpos0[qposAdr + 2] ?? 0.0);
  }

  private applyReferenceStateToSim(): void {
    const mjModel = this.context.mjModel;
    const mjData = this.context.mjData;
    const motion = this.selectedMotion;
    if (!mjModel || !mjData || !motion || this.refLen === 0) {
      return;
    }

    const rootPos = this.refRootPos[this.refIdx];
    const rootQuat = this.refRootQuat[this.refIdx];
    const freeJointIndex = this.findFreeJointIndex();
    if (rootPos && rootQuat && freeJointIndex >= 0) {
      const qposAdr = mjModel.jnt_qposadr[freeJointIndex];
      const qvelAdr = mjModel.jnt_dofadr[freeJointIndex];
      mjData.qpos[qposAdr + 0] = rootPos[0] ?? 0.0;
      mjData.qpos[qposAdr + 1] = rootPos[1] ?? 0.0;
      mjData.qpos[qposAdr + 2] = rootPos[2] ?? 0.0;
      mjData.qpos[qposAdr + 3] = rootQuat[0] ?? 1.0;
      mjData.qpos[qposAdr + 4] = rootQuat[1] ?? 0.0;
      mjData.qpos[qposAdr + 5] = rootQuat[2] ?? 0.0;
      mjData.qpos[qposAdr + 6] = rootQuat[3] ?? 0.0;

      const linVel = this.refBodyLinVelW[this.refIdx]?.slice(
        this.selectedRootBodyIndex * 3,
        this.selectedRootBodyIndex * 3 + 3,
      );
      const angVel = this.refBodyAngVelW[this.refIdx]?.slice(
        this.selectedRootBodyIndex * 3,
        this.selectedRootBodyIndex * 3 + 3,
      );
      if (linVel && angVel) {
        mjData.qvel[qvelAdr + 0] = linVel[0] ?? 0.0;
        mjData.qvel[qvelAdr + 1] = linVel[1] ?? 0.0;
        mjData.qvel[qvelAdr + 2] = linVel[2] ?? 0.0;
        mjData.qvel[qvelAdr + 3] = angVel[0] ?? 0.0;
        mjData.qvel[qvelAdr + 4] = angVel[1] ?? 0.0;
        mjData.qvel[qvelAdr + 5] = angVel[2] ?? 0.0;
      }
    }

    const jointPos = this.refJointPos[this.refIdx] ?? new Float32Array(0);
    const jointVel = motion.jointVel[this.refIdx] ?? new Float32Array(0);
    for (let i = 0; i < this.datasetQposAdr.length && i < jointPos.length; i++) {
      mjData.qpos[this.datasetQposAdr[i]] = jointPos[i] ?? 0.0;
    }
    for (let i = 0; i < this.datasetQposAdr.length && i < jointVel.length; i++) {
      const dofAdr = this.resolveQvelAdrForQposAdr(this.datasetQposAdr[i]);
      if (dofAdr >= 0) {
        mjData.qvel[dofAdr] = jointVel[i] ?? 0.0;
      }
    }

    this.context.mujoco.mj_forward(mjModel, mjData);
  }

  private resolveQposAdr(jointNames: string[]): number[] {
    const mjModel = this.context.mjModel;
    if (!mjModel || jointNames.length === 0) {
      return [];
    }
    const resolved: number[] = [];
    for (const jointName of jointNames) {
      let adr = -1;
      for (let j = 0; j < mjModel.njnt; j++) {
        const modelJointName = mjModel.jnt(j).name;
        if (modelJointName === jointName || modelJointName.endsWith(`/${jointName}`)) {
          adr = mjModel.jnt_qposadr[j];
          break;
        }
      }
      if (adr >= 0) {
        resolved.push(adr);
      }
    }
    return resolved;
  }

  private resolveQvelAdrForQposAdr(qposAdr: number): number {
    const mjModel = this.context.mjModel;
    if (!mjModel) {
      return -1;
    }
    for (let j = 0; j < mjModel.njnt; j++) {
      if (mjModel.jnt_qposadr[j] === qposAdr) {
        return mjModel.jnt_dofadr[j];
      }
    }
    return -1;
  }

  private findBodyIdByName(bodyName: string): number {
    const mjModel = this.context.mjModel;
    if (!mjModel) {
      return -1;
    }
    for (let b = 0; b < mjModel.nbody; b++) {
      const name = mjModel.body(b).name;
      if (name === bodyName || name.endsWith(`/${bodyName}`)) {
        return b;
      }
    }
    return -1;
  }

  private resolveMotionRootBodyIndex(bodyNames: string[]): number {
    const mjModel = this.context.mjModel;
    if (!mjModel) {
      return 0;
    }
    const freeJointIndex = this.findFreeJointIndex();
    if (freeJointIndex < 0) {
      return 0;
    }
    const bodyId = mjModel.jnt_bodyid?.[freeJointIndex] ?? -1;
    if (bodyId < 0 || bodyId >= mjModel.nbody) {
      return 0;
    }
    const rootBodyName = mjModel.body(bodyId).name;
    const exact = bodyNames.indexOf(rootBodyName);
    if (exact >= 0) {
      return exact;
    }
    const suffix = bodyNames.findIndex((name) =>
      rootBodyName.endsWith(`/${name}`) || name.endsWith(`/${rootBodyName}`)
    );
    return suffix >= 0 ? suffix : 0;
  }

  private updateGhostPose(): void {
    if (!this.ghostRoot || !this.ghostData || !this.context.mjModel || !this.selectedMotion || !this.refLen) {
      if (this.ghostRoot) {
        this.ghostRoot.visible = false;
      }
      return;
    }

    const qpos = this.ghostData.qpos;
    qpos.set(this.context.mjModel.qpos0);

    const rootPos = this.refRootPos[this.refIdx];
    const rootQuat = this.refRootQuat[this.refIdx];
    const freeJointIndex = this.findFreeJointIndex();
    if (freeJointIndex >= 0) {
      const qposAdr = this.context.mjModel.jnt_qposadr[freeJointIndex];
      qpos[qposAdr + 0] = rootPos[0] ?? 0.0;
      qpos[qposAdr + 1] = rootPos[1] ?? 0.0;
      qpos[qposAdr + 2] = rootPos[2] ?? 0.0;
      qpos[qposAdr + 3] = rootQuat[0] ?? 1.0;
      qpos[qposAdr + 4] = rootQuat[1] ?? 0.0;
      qpos[qposAdr + 5] = rootQuat[2] ?? 0.0;
      qpos[qposAdr + 6] = rootQuat[3] ?? 0.0;
    }

    const jointPos = this.refJointPos[this.refIdx] ?? new Float32Array(0);
    for (let i = 0; i < this.datasetQposAdr.length && i < jointPos.length; i++) {
      qpos[this.datasetQposAdr[i]] = jointPos[i] ?? 0.0;
    }

    this.context.mujoco.mj_forward(this.context.mjModel, this.ghostData);

    for (const [bodyId, body] of this.ghostBodies) {
      getPosition(this.ghostData.xpos, bodyId, body.position);
      getQuaternion(this.ghostData.xquat, bodyId, body.quaternion);
    }
    this.ghostRoot.visible = this.referenceVisible;
  }

  private findFreeJointIndex(): number {
    const mjModel = this.context.mjModel;
    if (!mjModel) {
      return -1;
    }
    for (let i = 0; i < mjModel.njnt; i++) {
      if (mjModel.jnt_type[i] === 0) {
        return i;
      }
    }
    return -1;
  }
}
