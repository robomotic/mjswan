import { getCommandManager } from '../command';
import { TrackingCommand } from '../command/TrackingCommand';
import { TerminationBase, type TerminationConfig } from './TerminationBase';
import type { PolicyState } from '../policy/types';

function getBodyIdByNameAnchorOri(mjModel: import('mujoco').MjModel, bodyName: string): number {
  for (let i = 0; i < mjModel.nbody; i++) {
    const name = mjModel.body(i).name;
    if (name === bodyName || name.endsWith(`/${bodyName}`)) {
      return i;
    }
  }
  return -1;
}

function normalizeQuatAnchorOri(quat: ArrayLike<number>): [number, number, number, number] {
  const length = Math.hypot(quat[0] ?? 1, quat[1] ?? 0, quat[2] ?? 0, quat[3] ?? 0) || 1.0;
  return [
    (quat[0] ?? 1) / length,
    (quat[1] ?? 0) / length,
    (quat[2] ?? 0) / length,
    (quat[3] ?? 0) / length,
  ];
}

function quatConjugateAnchorOri([w, x, y, z]: [number, number, number, number]): [number, number, number, number] {
  return [w, -x, -y, -z];
}

function quatMultiplyAnchorOri(
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

function quatApplyInvAnchorOri(quat: ArrayLike<number>, vec: readonly [number, number, number]): [number, number, number] {
  const q = normalizeQuatAnchorOri(quat);
  const qInv = quatConjugateAnchorOri(q);
  const vQuat: [number, number, number, number] = [0, vec[0], vec[1], vec[2]];
  const rotated = quatMultiplyAnchorOri(quatMultiplyAnchorOri(qInv, vQuat), q);
  return [rotated[1], rotated[2], rotated[3]];
}

export class BadAnchorOri extends TerminationBase {
  private readonly threshold: number;

  constructor(config: TerminationConfig) {
    super(config);
    this.threshold = (config.params?.threshold as number | undefined) ?? Infinity;
  }

  evaluate(_state: PolicyState): boolean {
    const tracking = getCommandManager().getTerm('motion');
    const context = getCommandManager().getContext();
    const mjModel = context?.mjModel ?? null;
    const mjData = context?.mjData ?? null;
    if (!(tracking instanceof TrackingCommand) || !tracking.isReady() || !mjModel || !mjData) {
      return false;
    }
    const anchorQuat = tracking.getAnchorQuat();
    const anchorName = tracking.getAnchorBodyName();
    if (!anchorQuat || anchorQuat.length < 4 || !anchorName) {
      return false;
    }
    const anchorId = getBodyIdByNameAnchorOri(mjModel, anchorName);
    if (anchorId < 0) {
      return false;
    }
    const currentAnchorQuat = mjData.xquat.slice(anchorId * 4, anchorId * 4 + 4);
    const gravity: [number, number, number] = [0.0, 0.0, -1.0];
    const motionGravity = quatApplyInvAnchorOri(anchorQuat, gravity);
    const robotGravity = quatApplyInvAnchorOri(currentAnchorQuat, gravity);
    return Math.abs(motionGravity[2] - robotGravity[2]) > this.threshold;
  }
}
