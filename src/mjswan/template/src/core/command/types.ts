import type * as THREE from 'three';
import type { MainModule, MjData, MjModel } from '@mujoco/mujoco';

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

export interface CommandUiConfig {
  inputs?: CommandInputConfig[];
}

export interface CommandConfigEntry {
  name: string;
  ui?: CommandUiConfig;
  [key: string]: unknown;
}

export type CommandsConfig = Record<string, CommandConfigEntry>;

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

export interface CommandTermContext {
  mujoco: MainModule;
  mjModel: MjModel | null;
  mjData: MjData | null;
  scene: THREE.Scene;
}

export interface CommandTerm {
  getCommand(): Float32Array;
  getUiConfig?(): CommandUiConfig | null;
  reset?(): void;
  update?(dt: number): void;
  updateDebugVisuals?(): void;
  setValue?(inputName: string, value: number): number | void;
  triggerButton?(inputName: string): void;
  dispose?(): void;
}

export type CommandTermConstructor = new (
  termName: string,
  config: CommandConfigEntry,
  context: CommandTermContext
) => CommandTerm;
