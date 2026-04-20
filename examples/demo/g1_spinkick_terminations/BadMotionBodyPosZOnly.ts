import { getCommandManager } from '../command';
import { TrackingCommand } from '../command/TrackingCommand';
import { TerminationBase, type TerminationConfig } from './TerminationBase';
import type { PolicyState } from '../policy/types';

function normalizeQuatBodyPos(quat: ArrayLike<number>): [number, number, number, number] {
  const length = Math.hypot(quat[0] ?? 1, quat[1] ?? 0, quat[2] ?? 0, quat[3] ?? 0) || 1.0;
  return [
    (quat[0] ?? 1) / length,
    (quat[1] ?? 0) / length,
    (quat[2] ?? 0) / length,
    (quat[3] ?? 0) / length,
  ];
}

function quatConjugateBodyPos([w, x, y, z]: [number, number, number, number]): [number, number, number, number] {
  return [w, -x, -y, -z];
}

function quatMultiplyBodyPos(
  [aw, ax, ay, az]: [number, number, number, number],
  [bw, bx, by, bz]: [number, number, number, number],
): [number, number, number, number] {
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}

function quatApplyInvBodyPos(quat: ArrayLike<number>, vec: readonly [number, number, number]): [number, number, number] {
  const q = normalizeQuatBodyPos(quat);
  const qInv = quatConjugateBodyPos(q);
  const vQuat: [number, number, number, number] = [0, vec[0], vec[1], vec[2]];
  const rotated = quatMultiplyBodyPos(quatMultiplyBodyPos(qInv, vQuat), q);
  return [rotated[1], rotated[2], rotated[3]];
}

function getBodyIdByNameBodyPos(mjModel: import('mujoco').MjModel, bodyName: string): number {
  for (let i = 0; i < mjModel.nbody; i++) {
    if (mjModel.body(i).name === bodyName) {
      return i;
    }
  }
  return -1;
}

export class BadMotionBodyPosZOnly extends TerminationBase {
  private readonly threshold: number;
  private readonly bodyNames: string[] | null;

  constructor(config: TerminationConfig) {
    super(config);
    this.threshold = (config.params?.threshold as number | undefined) ?? Infinity;
    this.bodyNames = Array.isArray(config.params?.body_names)
      ? config.params!.body_names.filter((value): value is string => typeof value === 'string')
      : null;
  }

  evaluate(_state: PolicyState): boolean {
    const tracking = getCommandManager().getTerm('motion');
    const context = getCommandManager().getContext();
    const mjModel = context?.mjModel ?? null;
    const mjData = context?.mjData ?? null;
    if (!(tracking instanceof TrackingCommand) || !tracking.isReady() || !mjModel || !mjData) {
      return false;
    }

    const anchorName = tracking.getAnchorBodyName();
    const bodyNames = this.bodyNames && this.bodyNames.length > 0
      ? this.bodyNames
      : tracking.getBodyNames();
    if (!anchorName || bodyNames.length === 0) {
      return false;
    }

    const anchorId = getBodyIdByNameBodyPos(mjModel, anchorName);
    if (anchorId < 0) {
      return false;
    }
    const currentAnchorPos = mjData.xpos.slice(anchorId * 3, anchorId * 3 + 3);
    const currentAnchorQuat = normalizeQuatBodyPos(mjData.xquat.slice(anchorId * 4, anchorId * 4 + 4));

    const refAnchorPos = tracking.getAnchorPos();
    const refAnchorQuat = tracking.getAnchorQuat();
    const refBodyPosW = tracking.getBodyPosW();
    if (!refAnchorPos || !refAnchorQuat || !refBodyPosW) {
      return false;
    }

    const allTrackingBodies = tracking.getBodyNames();
    for (const bodyName of bodyNames) {
      const bodySlot = allTrackingBodies.indexOf(bodyName);
      const bodyId = getBodyIdByNameBodyPos(mjModel, bodyName);
      if (bodySlot < 0 || bodyId < 0) {
        continue;
      }

      const currentBodyPos = mjData.xpos.slice(bodyId * 3, bodyId * 3 + 3);
      const currentLocal = quatApplyInvBodyPos(currentAnchorQuat, [
        currentBodyPos[0] - currentAnchorPos[0],
        currentBodyPos[1] - currentAnchorPos[1],
        currentBodyPos[2] - currentAnchorPos[2],
      ]);

      const refOffset = bodySlot * 3;
      const refLocal = quatApplyInvBodyPos(refAnchorQuat, [
        refBodyPosW[refOffset + 0] - refAnchorPos[0],
        refBodyPosW[refOffset + 1] - refAnchorPos[1],
        refBodyPosW[refOffset + 2] - refAnchorPos[2],
      ]);

      if (Math.abs(refLocal[2] - currentLocal[2]) > this.threshold) {
        return true;
      }
    }

    return false;
  }
}
