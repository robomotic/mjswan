/**
 * CommandManager - Manages user commands that can be sent to the policy.
 *
 * Commands are user inputs (like velocity targets) that get passed to the
 * ONNX policy as part of the observation. The CommandManager:
 * - Defines available commands with their types and ranges
 * - Stores current command values grouped by command group name
 * - Provides an interface for UI components to update values
 * - Provides observation values for the policy via getCommand(groupName)
 *
 * Command groups are defined in the policy config JSON and loaded when the policy
 * is initialized. Observations access commands by group name (like mjlab pattern).
 */

export type CommandType = 'slider' | 'button';

export interface SliderCommandConfig {
  type: 'slider';
  name: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
}

export interface ButtonCommandConfig {
  type: 'button';
  name: string;
  label: string;
}

export type CommandInputConfig = SliderCommandConfig | ButtonCommandConfig;

/**
 * Command group configuration from policy config JSON
 */
export interface CommandGroupConfig {
  inputs: CommandInputConfig[];
}

/**
 * Commands section from policy config JSON
 */
export type CommandsConfig = Record<string, CommandGroupConfig>;

export interface CommandDefinition {
  id: string;
  groupName: string;
  config: CommandInputConfig;
}

export type CommandEventType = 'change' | 'reset' | 'button' | 'group_registered' | 'clear';

export interface CommandEvent {
  type: CommandEventType;
  commandId: string;
  groupName?: string;
  value?: number;
}

export type CommandEventListener = (event: CommandEvent) => void;

/**
 * Default velocity command configuration (for backwards compatibility)
 */
export const DEFAULT_VELOCITY_COMMANDS: CommandInputConfig[] = [
  {
    type: 'slider',
    name: 'lin_vel_x',
    label: 'Forward Velocity',
    min: -1.0,
    max: 1.0,
    step: 0.05,
    default: 0.5,
  },
  {
    type: 'slider',
    name: 'lin_vel_y',
    label: 'Lateral Velocity',
    min: -0.5,
    max: 0.5,
    step: 0.05,
    default: 0.0,
  },
  {
    type: 'slider',
    name: 'ang_vel_z',
    label: 'Yaw Rate',
    min: -1.0,
    max: 1.0,
    step: 0.05,
    default: 0.0,
  },
];

export class CommandManager {
  private commands: Map<string, CommandDefinition> = new Map();
  private commandGroups: Map<string, string[]> = new Map(); // groupName -> list of command ids
  private values: Map<string, number> = new Map();
  private listeners: Set<CommandEventListener> = new Set();
  private resetCallback: (() => void) | null = null;

  constructor() {
    // Reset command is always available by default
    this.registerCommand('_system', {
      type: 'button',
      name: 'reset',
      label: 'Reset Simulation',
    });
  }

  /**
   * Register a single command input under a group
   */
  registerCommand(groupName: string, config: CommandInputConfig): void {
    const id = `${groupName}:${config.name}`;
    this.commands.set(id, { id, groupName, config });

    // Track command in group
    if (!this.commandGroups.has(groupName)) {
      this.commandGroups.set(groupName, []);
    }
    this.commandGroups.get(groupName)!.push(id);

    // Initialize slider values to default
    if (config.type === 'slider') {
      this.values.set(id, config.default);
    }
  }

  /**
   * Register a command group from policy config
   * This is the main method called when loading a policy config
   */
  registerCommandGroup(groupName: string, groupConfig: CommandGroupConfig): void {
    for (const inputConfig of groupConfig.inputs) {
      this.registerCommand(groupName, inputConfig);
    }

    this.emit({
      type: 'group_registered',
      commandId: groupName,
      groupName,
    });
  }

  /**
   * Register all command groups from a commands config section
   */
  registerCommandsFromConfig(commandsConfig: CommandsConfig): void {
    for (const [groupName, groupConfig] of Object.entries(commandsConfig)) {
      this.registerCommandGroup(groupName, groupConfig);
    }
  }

  /**
   * Get all registered command groups
   */
  getCommandGroups(): string[] {
    return Array.from(this.commandGroups.keys()).filter(name => name !== '_system');
  }

  /**
   * Get all commands in a group
   */
  getCommandsInGroup(groupName: string): CommandDefinition[] {
    const ids = this.commandGroups.get(groupName) ?? [];
    return ids.map(id => this.commands.get(id)!).filter(Boolean);
  }

  /**
   * Get all registered commands (excluding system commands)
   */
  getCommands(): CommandDefinition[] {
    return Array.from(this.commands.values()).filter(cmd => cmd.groupName !== '_system');
  }

  /**
   * Get all commands including system commands (for UI)
   */
  getAllCommands(): CommandDefinition[] {
    return Array.from(this.commands.values());
  }

  /**
   * Get the reset button command
   */
  getResetCommand(): CommandDefinition | undefined {
    return this.commands.get('_system:reset');
  }

  /**
   * Get a specific command by full ID (groupName:name)
   */
  getCommandById(id: string): CommandDefinition | undefined {
    return this.commands.get(id);
  }

  /**
   * Get the current value of a slider command
   */
  getValue(id: string): number {
    return this.values.get(id) ?? 0;
  }

  /**
   * Get all current slider values
   */
  getValues(): Record<string, number> {
    const result: Record<string, number> = {};
    for (const [id, value] of this.values) {
      result[id] = value;
    }
    return result;
  }

  /**
   * Get command values for a group as a Float32Array (for observations)
   * Values are returned in the order they were registered (same as inputs array order)
   *
   * This is the main method used by GeneratedCommandsObservation
   */
  getCommand(groupName: string): Float32Array {
    const ids = this.commandGroups.get(groupName) ?? [];
    const sliderIds = ids.filter(id => {
      const cmd = this.commands.get(id);
      return cmd?.config.type === 'slider';
    });

    const values = new Float32Array(sliderIds.length);
    for (let i = 0; i < sliderIds.length; i++) {
      values[i] = this.values.get(sliderIds[i]) ?? 0;
    }
    return values;
  }

  /**
   * Get velocity command values as an array [lin_vel_x, lin_vel_y, ang_vel_z]
   * For backwards compatibility with existing observations
   */
  getVelocityCommand(): Float32Array {
    // Try to get from 'velocity' group first
    if (this.commandGroups.has('velocity')) {
      return this.getCommand('velocity');
    }

    // Fallback to legacy hardcoded names
    const linVelX = this.values.get('velocity:lin_vel_x') ?? this.values.get('_legacy:vel_x') ?? 0.5;
    const linVelY = this.values.get('velocity:lin_vel_y') ?? this.values.get('_legacy:vel_y') ?? 0.0;
    const angVelZ = this.values.get('velocity:ang_vel_z') ?? this.values.get('_legacy:yaw_rate') ?? 0.0;
    return new Float32Array([linVelX, linVelY, angVelZ]);
  }

  /**
   * Set the value of a slider command
   */
  setValue(id: string, value: number): void {
    const command = this.commands.get(id);
    if (!command || command.config.type !== 'slider') {
      return;
    }

    const config = command.config as SliderCommandConfig;
    const clampedValue = Math.max(config.min, Math.min(config.max, value));
    this.values.set(id, clampedValue);

    this.emit({
      type: 'change',
      commandId: id,
      groupName: command.groupName,
      value: clampedValue,
    });
  }

  /**
   * Trigger a button command
   */
  triggerButton(id: string): void {
    const command = this.commands.get(id);
    if (!command || command.config.type !== 'button') {
      return;
    }

    if (id === '_system:reset' && this.resetCallback) {
      this.resetCallback();
    }

    this.emit({
      type: 'button',
      commandId: id,
      groupName: command.groupName,
    });
  }

  /**
   * Reset all slider values to their defaults
   */
  resetToDefaults(): void {
    for (const [id, command] of this.commands) {
      if (command.config.type === 'slider') {
        const config = command.config as SliderCommandConfig;
        this.values.set(id, config.default);
      }
    }

    this.emit({
      type: 'reset',
      commandId: '*',
    });
  }

  /**
   * Set the reset callback (called when reset button is pressed)
   */
  setResetCallback(callback: () => void): void {
    this.resetCallback = callback;
  }

  /**
   * Add an event listener
   */
  addEventListener(listener: CommandEventListener): void {
    this.listeners.add(listener);
  }

  /**
   * Remove an event listener
   */
  removeEventListener(listener: CommandEventListener): void {
    this.listeners.delete(listener);
  }

  /**
   * Emit an event to all listeners
   */
  private emit(event: CommandEvent): void {
    for (const listener of this.listeners) {
      try {
        listener(event);
      } catch (error) {
        console.warn('[CommandManager] Listener error:', error);
      }
    }
  }

  /**
   * Clear all commands (except reset)
   */
  clear(): void {
    const resetCommand = this.commands.get('_system:reset');
    this.commands.clear();
    this.commandGroups.clear();
    this.values.clear();
    if (resetCommand) {
      this.commands.set('_system:reset', resetCommand);
      this.commandGroups.set('_system', ['_system:reset']);
    }
    // Notify listeners that commands have been cleared
    this.emit({ type: 'clear', commandId: '' });
  }

  /**
   * Check if any commands (besides reset) are registered
   */
  hasCommands(): boolean {
    return this.commands.size > 1; // More than just reset
  }

  /**
   * Dispose of the command manager
   */
  dispose(): void {
    this.commands.clear();
    this.commandGroups.clear();
    this.values.clear();
    this.listeners.clear();
    this.resetCallback = null;
  }
}

// Singleton instance for global access
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
