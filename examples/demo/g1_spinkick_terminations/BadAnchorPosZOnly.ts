import { getCommandManager } from '../command';
import { TrackingCommand } from '../command/TrackingCommand';
import { TerminationBase, type TerminationConfig } from './TerminationBase';
import type { PolicyState } from '../policy/types';

export class BadAnchorPosZOnly extends TerminationBase {
  private readonly threshold: number;

  constructor(config: TerminationConfig) {
    super(config);
    this.threshold = (config.params?.threshold as number | undefined) ?? Infinity;
  }

  evaluate(state: PolicyState): boolean {
    const rootPos = state.rootPos;
    const tracking = getCommandManager().getTerm('motion');
    if (!(tracking instanceof TrackingCommand) || !tracking.isReady() || !rootPos || rootPos.length < 3) {
      return false;
    }
    const anchorPos = tracking.getAnchorPos();
    if (!anchorPos || anchorPos.length < 3) {
      return false;
    }
    return Math.abs(anchorPos[2] - rootPos[2]) > this.threshold;
  }
}
