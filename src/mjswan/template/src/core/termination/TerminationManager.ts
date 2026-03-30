import { TerminationBase, type TerminationConfig } from './TerminationBase';
import type { TerminationConstructor } from './terminations';
import type { PolicyState, TerminationConfigEntry } from '../policy/types';

export type TerminationResult = {
  done: boolean;
  terminated: boolean;
  truncated: boolean;
};

export class TerminationManager {
  private terms: { name: string; term: TerminationBase; isTimeOut: boolean }[] = [];

  constructor(
    config: Record<string, TerminationConfigEntry>,
    registry: Record<string, TerminationConstructor>
  ) {
    for (const [name, entry] of Object.entries(config)) {
      const TermClass = registry[entry.name];
      if (!TermClass) {
        console.warn(`[TerminationManager] Unknown termination type: ${entry.name}`);
        continue;
      }
      const termConfig: TerminationConfig = {
        name: entry.name,
        params: entry.params,
        time_out: entry.time_out,
      };
      this.terms.push({
        name,
        term: new TermClass(termConfig),
        isTimeOut: entry.time_out ?? false,
      });
    }
  }

  evaluate(state: PolicyState): TerminationResult {
    let terminated = false;
    let truncated = false;

    for (const { term, isTimeOut } of this.terms) {
      if (term.evaluate(state)) {
        if (isTimeOut) {
          truncated = true;
        } else {
          terminated = true;
        }
      }
    }

    return {
      done: terminated || truncated,
      terminated,
      truncated,
    };
  }

  reset(): void {
    for (const { term } of this.terms) {
      term.reset?.();
    }
  }

  get size(): number {
    return this.terms.length;
  }
}
