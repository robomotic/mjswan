import type {
  CommandTerm,
  CommandTermContext,
} from "./types";

export class HoldOnCommand implements CommandTerm {
  private output: Float32Array;
  private isHeld: boolean = true;
  private heldQpos: Float32Array;
  private qposAdrs: number[] = [];

  constructor(
    private termName: string,
    private config: any,
    private context: CommandTermContext
  ) {
    const jointNames: string[] = config.joint_names || [];
    this.output = new Float32Array(jointNames.length + 1);
    this.heldQpos = new Float32Array(jointNames.length);

    if (context.mjModel) {
      this.qposAdrs = jointNames.map(name => this.resolveJointQposAdr(context.mjModel, name));
    }

    // Capture the initial joint positions
    this.latchCurrentPositions();

    window.addEventListener("pointerdown", this.handlePointerDown);
    window.addEventListener("pointerup", this.handlePointerUp);
    window.addEventListener("pointercancel", this.handlePointerUp);
  }

  private latchCurrentPositions() {
    if (this.context.mjData) {
      for (let i = 0; i < this.qposAdrs.length; i++) {
        const adr = this.qposAdrs[i];
        if (adr >= 0) {
          // Keep held positions in absolute qpos coordinates so they match
          // the hold-mode policy observations in `simple.py`.
          this.heldQpos[i] = this.context.mjData.qpos[adr];
        }
      }
    }
  }

  private handlePointerDown = (e: PointerEvent) => {
    // When the mouse/pointer is pressed, release the hold so forces can apply
    this.isHeld = false;
  };

  private handlePointerUp = (e: PointerEvent) => {
    // When the mouse/pointer is released, latch the current positions and hold them
    this.isHeld = true;
    this.latchCurrentPositions();
  };

  private resolveJointQposAdr(mjModel: any, jointName: string): number {
    const namesArray = new Uint8Array(mjModel.names);
    const decoder = new TextDecoder();
    for (let i = 0; i < mjModel.njnt; i++) {
      let start = mjModel.name_jntadr[i];
      let end = start;
      while (end < namesArray.length && namesArray[end] !== 0) {
        end++;
      }
      const name = decoder.decode(namesArray.subarray(start, end));
      if (name === jointName) {
        return mjModel.jnt_qposadr[i];
      }
    }
    return -1;
  }

  getCommand(): Float32Array {
    if (this.isHeld) {
      this.output.set(this.heldQpos);
    } else {
      if (this.context.mjData) {
        for (let i = 0; i < this.qposAdrs.length; i++) {
          const adr = this.qposAdrs[i];
          if (adr >= 0) {
            this.output[i] = this.context.mjData.qpos[adr];
          }
        }
      }
    }
    this.output[this.output.length - 1] = this.isHeld ? 1.0 : 0.0;
    return this.output;
  }

  dispose() {
    window.removeEventListener("pointerdown", this.handlePointerDown);
    window.removeEventListener("pointerup", this.handlePointerUp);
    window.removeEventListener("pointercancel", this.handlePointerUp);
  }
}
