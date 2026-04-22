export function getBodyIdByName(mjModel: import('mujoco').MjModel, bodyName: string): number {
  for (let i = 0; i < mjModel.nbody; i++) {
    const name = mjModel.body(i).name;
    if (name === bodyName || name.endsWith(`/${bodyName}`)) {
      return i;
    }
  }
  return -1;
}

export function normalizeQuat(quat: ArrayLike<number>): [number, number, number, number] {
  const length = Math.hypot(quat[0] ?? 1, quat[1] ?? 0, quat[2] ?? 0, quat[3] ?? 0) || 1.0;
  return [
    (quat[0] ?? 1) / length,
    (quat[1] ?? 0) / length,
    (quat[2] ?? 0) / length,
    (quat[3] ?? 0) / length,
  ];
}

export function quatConjugate([w, x, y, z]: [number, number, number, number]): [number, number, number, number] {
  return [w, -x, -y, -z];
}

export function quatMultiply(
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

export function quatApplyInv(quat: ArrayLike<number>, vec: readonly [number, number, number]): [number, number, number] {
  const q = normalizeQuat(quat);
  const qInv = quatConjugate(q);
  const vQuat: [number, number, number, number] = [0, vec[0], vec[1], vec[2]];
  const rotated = quatMultiply(quatMultiply(qInv, vQuat), q);
  return [rotated[1], rotated[2], rotated[3]];
}
