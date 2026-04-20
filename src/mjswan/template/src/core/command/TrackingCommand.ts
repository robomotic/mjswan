import * as THREE from 'three';
import JSZip from 'jszip';

import { getPosition, getQuaternion } from '../scene/scene';
import type { CommandConfigEntry, CommandTerm, CommandTermContext, CommandUiConfig } from './types';

type ParsedNpy = {
  data: Float32Array | Float64Array;
  shape: number[];
};

export type TrackingMotionConfig = {
  name: string;
  path: string;
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

function parseShape(raw: string): number[] {
  const items = raw
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
  return items.map((item) => Number.parseInt(item, 10)).filter((value) => Number.isFinite(value));
}

function parseNpy(buffer: ArrayBuffer): ParsedNpy {
  const view = new DataView(buffer);
  const magic = new Uint8Array(buffer, 0, 6);
  const magicText = String.fromCharCode(...magic);
  if (magicText !== '\u0093NUMPY') {
    throw new Error('Unsupported .npy header');
  }
  const major = view.getUint8(6);
  const headerLength = major >= 2 ? view.getUint32(8, true) : view.getUint16(8, true);
  const headerOffset = major >= 2 ? 12 : 10;
  const headerBytes = new Uint8Array(buffer, headerOffset, headerLength);
  const header = new TextDecoder('latin1').decode(headerBytes);

  const descrMatch = header.match(/'descr':\s*'([^']+)'/);
  const fortranMatch = header.match(/'fortran_order':\s*(True|False)/);
  const shapeMatch = header.match(/'shape':\s*\(([^)]*)\)/);
  if (!descrMatch || !fortranMatch || !shapeMatch) {
    throw new Error('Unsupported .npy metadata');
  }
  if (fortranMatch[1] !== 'False') {
    throw new Error('Fortran-ordered arrays are not supported');
  }

  const descr = descrMatch[1];
  const shape = parseShape(shapeMatch[1]);
  const dataOffset = headerOffset + headerLength;

  if (descr === '<f4' || descr === '|f4') {
    return {
      data: new Float32Array(buffer, dataOffset),
      shape,
    };
  }
  if (descr === '<f8' || descr === '|f8') {
    return {
      data: new Float64Array(buffer, dataOffset),
      shape,
    };
  }
  throw new Error(`Unsupported .npy dtype: ${descr}`);
}

function frameCount(shape: number[]): number {
  return shape[0] ?? 0;
}

function frameWidth(shape: number[]): number {
  if (shape.length <= 1) {
    return 1;
  }
  return shape.slice(1).reduce((acc, value) => acc * value, 1);
}

function splitFrames(
  array: ParsedNpy,
): Float32Array[] {
  const totalFrames = frameCount(array.shape);
  const width = frameWidth(array.shape);
  const frames: Float32Array[] = [];
  for (let i = 0; i < totalFrames; i++) {
    const out = new Float32Array(width);
    const start = i * width;
    for (let j = 0; j < width; j++) {
      out[j] = array.data[start + j] ?? 0.0;
    }
    frames.push(out);
  }
  return frames;
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

export class TrackingCommand implements CommandTerm {
  private readonly context: CommandTermContext;
  private readonly motions: TrackingMotionConfig[];
  private readonly loadedMotions: Map<string, LoadedTrackingMotion>;
  private readonly sampleHz: number;
  private readonly ghostRoot: THREE.Group | null;
  private readonly ghostBodies: Map<number, THREE.Group>;
  private readonly ghostData: import('mujoco').MjData | null;
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
    this.selectedRootBodyIndex = 0;
    this.datasetQposAdr = this.resolveQposAdr(loaded.dataset_joint_names ?? []);
    this.refJointPos = loaded.jointPos;
    this.refRootPos = loaded.bodyPosW.map((frame) =>
      frame.slice(this.selectedRootBodyIndex * 3, this.selectedRootBodyIndex * 3 + 3),
    );
    this.refRootQuat = loaded.bodyQuatW.map((frame) =>
      normalizeQuat(
        frame.slice(this.selectedRootBodyIndex * 4, this.selectedRootBodyIndex * 4 + 4),
      ),
    );
    this.refIdx = 0;
    this.refLen = loaded.frameCount;
    this.nJoints = loaded.jointPos[0]?.length ?? 0;
    this.frameAccumulator = 0.0;
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
    if (!frame) {
      return null;
    }
    const offset = this.selectedAnchorBodyIndex * 3;
    return frame.slice(offset, offset + 3);
  }

  getAnchorQuat(frameIndex = this.refIdx): Float32Array | null {
    const motion = this.selectedMotion;
    if (!motion) {
      return null;
    }
    const frame = motion.bodyQuatW[frameIndex];
    if (!frame) {
      return null;
    }
    const offset = this.selectedAnchorBodyIndex * 4;
    return normalizeQuat(frame.slice(offset, offset + 4));
  }

  getBodyPosW(frameIndex = this.refIdx): Float32Array | null {
    const motion = this.selectedMotion;
    if (!motion) {
      return null;
    }
    const frame = motion.bodyPosW[frameIndex];
    return frame ? frame.slice() : null;
  }

  private createGhostRoot(): THREE.Group | null {
    const bodies = this.context.bodies ?? null;
    if (!bodies) {
      return null;
    }
    const root = new THREE.Group();
    root.name = 'Tracking Ghost';
    root.visible = false;
    for (const [bodyId, body] of Object.entries(bodies)) {
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
      this.ghostBodies.set(Number(bodyId), clone);
      root.add(clone);
    }
    this.context.scene.add(root);
    return root;
  }

  private async loadMotion(config: TrackingMotionConfig): Promise<LoadedTrackingMotion> {
    const response = await fetch(config.path, { cache: 'force-cache' });
    if (!response.ok) {
      throw new Error(`Failed to fetch motion asset: ${response.status}`);
    }
    const archive = await JSZip.loadAsync(await response.arrayBuffer());
    const arrays = await Promise.all([
      archive.file('joint_pos.npy')?.async('arraybuffer'),
      archive.file('joint_vel.npy')?.async('arraybuffer'),
      archive.file('body_pos_w.npy')?.async('arraybuffer'),
      archive.file('body_quat_w.npy')?.async('arraybuffer'),
      archive.file('body_lin_vel_w.npy')?.async('arraybuffer'),
      archive.file('body_ang_vel_w.npy')?.async('arraybuffer'),
    ]);
    if (arrays.some((entry) => entry === undefined)) {
      throw new Error('Motion asset is missing required arrays');
    }

    const jointPos = splitFrames(parseNpy(arrays[0]!));
    const jointVel = splitFrames(parseNpy(arrays[1]!));
    const bodyPosW = splitFrames(parseNpy(arrays[2]!));
    const bodyQuatW = splitFrames(parseNpy(arrays[3]!));
    const bodyLinVelW = splitFrames(parseNpy(arrays[4]!));
    const bodyAngVelW = splitFrames(parseNpy(arrays[5]!));

    return {
      ...config,
      jointPos,
      jointVel,
      bodyPosW,
      bodyQuatW,
      bodyLinVelW,
      bodyAngVelW,
      frameCount: jointPos.length,
    };
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
