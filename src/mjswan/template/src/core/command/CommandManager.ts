import { CustomCommands } from './custom_commands';
import type {
  ButtonCommandConfig,
  CommandConfigEntry,
  CommandDefinition,
  CommandEvent,
  CommandEventListener,
  CommandInputConfig,
  CommandTerm,
  CommandTermConstructor,
  CommandTermContext,
  CommandsConfig,
  SliderCommandConfig,
} from './types';

class UiCommand implements CommandTerm {
  private readonly inputs: CommandInputConfig[];
  private readonly values: Map<string, number>;

  constructor(
    _termName: string,
    config: CommandConfigEntry,
    // _context: CommandTermContext
  ) {
    this.inputs = Array.isArray(config.ui?.inputs) ? config.ui.inputs : [];
    this.values = new Map();
    for (const input of this.inputs) {
      if (input.type === 'slider') {
        this.values.set(input.name, input.default);
      }
    }
  }

  getCommand(): Float32Array {
    const sliderInputs = this.inputs.filter((input): input is SliderCommandConfig => input.type === 'slider');
    const values = new Float32Array(sliderInputs.length);
    for (let i = 0; i < sliderInputs.length; i++) {
      values[i] = this.values.get(sliderInputs[i].name) ?? sliderInputs[i].default ?? 0.0;
    }
    return values;
  }

  getUiConfig() {
    return { inputs: this.inputs };
  }

  reset(): void {
    for (const input of this.inputs) {
      if (input.type === 'slider') {
        this.values.set(input.name, input.default);
      }
    }
  }

  setValue(inputName: string, value: number): number {
    const input = this.inputs.find(
      (entry): entry is SliderCommandConfig => entry.type === 'slider' && entry.name === inputName
    );
    if (!input) {
      return 0.0;
    }
    const clamped = Math.max(input.min, Math.min(input.max, value));
    this.values.set(input.name, clamped);
    return clamped;
  }
}

const BuiltinCommandTerms: Record<string, CommandTermConstructor> = {
  UiCommand,
};

export class CommandManager {
  private terms: Map<string, CommandTerm> = new Map();
  private commands: Map<string, CommandDefinition> = new Map();
  private commandGroups: Map<string, string[]> = new Map();
  private values: Map<string, number> = new Map();
  private listeners: Set<CommandEventListener> = new Set();
  private resetCallback: (() => void) | null = null;

  constructor() {
    this.registerSystemReset();
  }

  initialize(commandsConfig: CommandsConfig, context: CommandTermContext): void {
    this.clear();
    const registry: Record<string, CommandTermConstructor> = {
      ...BuiltinCommandTerms,
      ...CustomCommands,
    };

    for (const [groupName, entry] of Object.entries(commandsConfig)) {
      const Term = registry[entry.name];
      if (!Term) {
        throw new Error(`Unknown command term: ${entry.name}`);
      }
      const term = new Term(groupName, entry, context);
      this.terms.set(groupName, term);
      this.registerUi(groupName, term);
    }
  }

  update(dt: number): void {
    for (const term of this.terms.values()) {
      term.update?.(dt);
    }
  }

  updateDebugVisuals(): void {
    for (const term of this.terms.values()) {
      term.updateDebugVisuals?.();
    }
  }

  resetTerms(): void {
    for (const term of this.terms.values()) {
      term.reset?.();
    }
    this.syncValuesFromTerms();
    this.emit({ type: 'reset', commandId: '*' });
  }

  getCommandGroups(): string[] {
    return Array.from(this.commandGroups.keys()).filter(name => name !== '_system');
  }

  getCommandsInGroup(groupName: string): CommandDefinition[] {
    const ids = this.commandGroups.get(groupName) ?? [];
    return ids.map(id => this.commands.get(id)!).filter(Boolean);
  }

  getCommands(): CommandDefinition[] {
    return Array.from(this.commands.values()).filter(cmd => cmd.groupName !== '_system');
  }

  getAllCommands(): CommandDefinition[] {
    return Array.from(this.commands.values());
  }

  getResetCommand(): CommandDefinition | undefined {
    return this.commands.get('_system:reset');
  }

  getCommandById(id: string): CommandDefinition | undefined {
    return this.commands.get(id);
  }

  getValue(id: string): number {
    return this.values.get(id) ?? 0;
  }

  getValues(): Record<string, number> {
    const result: Record<string, number> = {};
    for (const [id, value] of this.values) {
      result[id] = value;
    }
    return result;
  }

  getCommand(groupName: string): Float32Array {
    const term = this.terms.get(groupName);
    return term ? term.getCommand() : new Float32Array(0);
  }

  getVelocityCommand(): Float32Array {
    if (this.terms.has('velocity')) {
      return this.getCommand('velocity');
    }
    if (this.terms.has('twist')) {
      return this.getCommand('twist');
    }
    return new Float32Array([0.5, 0.0, 0.0]);
  }

  setValue(id: string, value: number): void {
    const command = this.commands.get(id);
    if (!command || command.config.type !== 'slider') {
      return;
    }
    const term = this.terms.get(command.groupName);
    const inputName = command.config.name;
    const clamped = term?.setValue ? term.setValue(inputName, value) : undefined;
    const nextValue = typeof clamped === 'number' ? clamped : value;
    this.values.set(id, nextValue);
    this.emit({
      type: 'change',
      commandId: id,
      groupName: command.groupName,
      value: nextValue,
    });
  }

  triggerButton(id: string): void {
    const command = this.commands.get(id);
    if (!command || command.config.type !== 'button') {
      return;
    }

    if (id === '_system:reset' && this.resetCallback) {
      this.resetCallback();
    } else {
      const term = this.terms.get(command.groupName);
      term?.triggerButton?.(command.config.name);
    }

    this.emit({
      type: 'button',
      commandId: id,
      groupName: command.groupName,
    });
  }

  resetToDefaults(): void {
    this.resetTerms();
  }

  setResetCallback(callback: () => void): void {
    this.resetCallback = callback;
  }

  addEventListener(listener: CommandEventListener): void {
    this.listeners.add(listener);
  }

  removeEventListener(listener: CommandEventListener): void {
    this.listeners.delete(listener);
  }

  clear(): void {
    for (const term of this.terms.values()) {
      term.dispose?.();
    }
    this.terms.clear();
    this.commands.clear();
    this.commandGroups.clear();
    this.values.clear();
    this.registerSystemReset();
    this.emit({ type: 'clear', commandId: '' });
  }

  hasCommands(): boolean {
    return this.commands.size > 1;
  }

  dispose(): void {
    for (const term of this.terms.values()) {
      term.dispose?.();
    }
    this.terms.clear();
    this.commands.clear();
    this.commandGroups.clear();
    this.values.clear();
    this.listeners.clear();
    this.resetCallback = null;
  }

  private registerSystemReset(): void {
    this.commands.set('_system:reset', {
      id: '_system:reset',
      groupName: '_system',
      config: {
        type: 'button',
        name: 'reset',
        label: 'Reset Simulation',
      } satisfies ButtonCommandConfig,
    });
    this.commandGroups.set('_system', ['_system:reset']);
  }

  private registerUi(groupName: string, term: CommandTerm): void {
    const ui = term.getUiConfig?.();
    const inputs = Array.isArray(ui?.inputs) ? ui.inputs : [];
    if (inputs.length === 0) {
      return;
    }
    this.commandGroups.set(groupName, []);
    for (const input of inputs) {
      const id = `${groupName}:${input.name}`;
      this.commands.set(id, { id, groupName, config: input });
      this.commandGroups.get(groupName)!.push(id);
      if (input.type === 'slider') {
        const current = term.getCommand();
        const sliderIndex = inputs
          .filter((entry): entry is SliderCommandConfig => entry.type === 'slider')
          .findIndex(entry => entry.name === input.name);
        this.values.set(id, current[sliderIndex] ?? input.default);
      }
    }
    this.emit({
      type: 'group_registered',
      commandId: groupName,
      groupName,
    });
  }

  private syncValuesFromTerms(): void {
    for (const [id, command] of this.commands) {
      if (command.config.type !== 'slider') {
        continue;
      }
      const term = this.terms.get(command.groupName);
      if (!term) {
        continue;
      }
      const inputs = term.getUiConfig?.()?.inputs ?? [];
      const sliderInputs = inputs.filter(
        (entry): entry is SliderCommandConfig => entry.type === 'slider'
      );
      const index = sliderInputs.findIndex(entry => entry.name === command.config.name);
      if (index >= 0) {
        const current = term.getCommand();
        this.values.set(id, current[index] ?? this.values.get(id) ?? 0.0);
      }
    }
  }

  private emit(event: CommandEvent): void {
    for (const listener of this.listeners) {
      try {
        listener(event);
      } catch (error) {
        console.warn('[CommandManager] Listener error:', error);
      }
    }
  }
}

let globalCommandManager: CommandManager | null = null;

export function getCommandManager(): CommandManager {
  if (!globalCommandManager) {
    globalCommandManager = new CommandManager();
  }
  return globalCommandManager;
}

export function resetCommandManager(): void {
  if (globalCommandManager) {
    globalCommandManager.dispose();
    globalCommandManager = null;
  }
}
