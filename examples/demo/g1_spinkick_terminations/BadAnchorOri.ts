import { getCommandManager } from '../command';
import { TrackingCommand } from '../command/TrackingCommand';
import { TerminationBase, type TerminationConfig } from './TerminationBase';
import type { PolicyState } from '../policy/types';

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

  evaluate(state: PolicyState): boolean {
    const rootQuat = state.rootQuat;
    const tracking = getCommandManager().getTerm('motion');
    if (!(tracking instanceof TrackingCommand) || !tracking.isReady() || !rootQuat || rootQuat.length < 4) {
      return false;
    }
    const anchorQuat = tracking.getAnchorQuat();
    if (!anchorQuat || anchorQuat.length < 4) {
      return false;
    }
    const gravity: [number, number, number] = [0.0, 0.0, -1.0];
    const motionGravity = quatApplyInvAnchorOri(anchorQuat, gravity);
    const robotGravity = quatApplyInvAnchorOri(rootQuat, gravity);
    return Math.abs(motionGravity[2] - robotGravity[2]) > this.threshold;
  }
}
