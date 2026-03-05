import { useEffect, useState } from 'react';
import { Box, NumberInput, Slider, Text } from '@mantine/core';
import { CommandSection } from './CommandSection';

interface SplatSectionProps {
  scale: number;
  xOffset: number;
  yOffset: number;
  zOffset: number;
  /** Roll in degrees. */
  roll: number;
  /** Pitch in degrees. */
  pitch: number;
  /** Yaw in degrees. */
  yaw: number;
  onCalibrate: (scale: number, xOffset: number, yOffset: number, zOffset: number, roll: number, pitch: number, yaw: number) => void;
}

const SECTION_LABEL_STYLE = { fontSize: '0.875em', fontWeight: 450, lineHeight: '1.375em', letterSpacing: '-0.75px', width: '4.5em', flexShrink: 0 } as const;
const AXIS_LABEL_STYLE = { fontSize: '0.875em', fontWeight: 450, lineHeight: '1.375em', letterSpacing: '-0.75px', flexShrink: 0 } as const;
const SLIDER_STYLES = { root: { padding: '0' }, track: { height: 4 }, thumb: { width: 12, height: 12 } } as const;
const NUMBER_INPUT_STYLES = { input: { height: '1.75em', minHeight: '1.75em', fontSize: '0.8em', textAlign: 'right' as const, padding: '0 0.4em' } } as const;

function ScaleRow({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return (
    <Box pb="0.5em" px="xs" style={{ display: 'flex', alignItems: 'center', gap: '0.5em' }}>
      <Text c="dimmed" style={SECTION_LABEL_STYLE}>Scale</Text>
      <Box style={{ flex: 1 }}>
        <Slider value={value} onChange={onChange} min={0.1} max={5.0} step={0.05} size="xs" label={(val) => val.toFixed(2)} styles={SLIDER_STYLES} />
      </Box>
    </Box>
  );
}

interface AxisInputProps {
  axis: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}

function AxisInput({ axis, value, min, max, step, onChange }: AxisInputProps) {
  return (
    <Box style={{ display: 'flex', alignItems: 'center', gap: '0.25em', flex: 1 }}>
      <Text c="dimmed" style={AXIS_LABEL_STYLE}>{axis}</Text>
      <NumberInput
        value={value}
        onChange={(val) => { if (typeof val === 'number') onChange(val); }}
        min={min} max={max} step={step} decimalScale={2} fixedDecimalScale hideControls size="xs"
        styles={NUMBER_INPUT_STYLES} style={{ flex: 1, minWidth: 0 }}
      />
    </Box>
  );
}

interface VectorRowProps {
  label: string;
  x: number; y: number; z: number;
  min: number; max: number; step: number;
  xLabel?: string; yLabel?: string; zLabel?: string;
  onX: (v: number) => void;
  onY: (v: number) => void;
  onZ: (v: number) => void;
}

function VectorRow({ label, x, y, z, min, max, step, xLabel = 'X', yLabel = 'Y', zLabel = 'Z', onX, onY, onZ }: VectorRowProps) {
  return (
    <Box pb="0.5em" px="xs" style={{ display: 'flex', alignItems: 'center', gap: '0.5em' }}>
      <Text c="dimmed" style={SECTION_LABEL_STYLE}>{label}</Text>
      <AxisInput axis={xLabel} value={x} min={min} max={max} step={step} onChange={onX} />
      <AxisInput axis={yLabel} value={y} min={min} max={max} step={step} onChange={onY} />
      <AxisInput axis={zLabel} value={z} min={min} max={max} step={step} onChange={onZ} />
    </Box>
  );
}

/** Dev-mode calibration controls for a Gaussian Splat background. */
export function SplatSection({ scale: initialScale, xOffset: initialXOffset, yOffset: initialYOffset, zOffset: initialZOffset, roll: initialRoll, pitch: initialPitch, yaw: initialYaw, onCalibrate }: SplatSectionProps) {
  const [scale, setScale] = useState(initialScale);
  const [xOffset, setXOffset] = useState(initialXOffset);
  const [yOffset, setYOffset] = useState(initialYOffset);
  const [zOffset, setZOffset] = useState(initialZOffset);
  const [roll, setRoll] = useState(initialRoll);
  const [pitch, setPitch] = useState(initialPitch);
  const [yaw, setYaw] = useState(initialYaw);

  useEffect(() => { setScale(initialScale); }, [initialScale]);
  useEffect(() => { setXOffset(initialXOffset); }, [initialXOffset]);
  useEffect(() => { setYOffset(initialYOffset); }, [initialYOffset]);
  useEffect(() => { setZOffset(initialZOffset); }, [initialZOffset]);
  useEffect(() => { setRoll(initialRoll); }, [initialRoll]);
  useEffect(() => { setPitch(initialPitch); }, [initialPitch]);
  useEffect(() => { setYaw(initialYaw); }, [initialYaw]);

  const call = (s: number, x: number, y: number, z: number, r: number, p: number, w: number) => onCalibrate(s, x, y, z, r, p, w);

  return (
    <CommandSection label="Control" expandByDefault={true}>
      <ScaleRow value={scale} onChange={(val) => { setScale(val); call(val, xOffset, yOffset, zOffset, roll, pitch, yaw); }} />
      <VectorRow
        label="Position" x={xOffset} y={yOffset} z={zOffset} min={-5.0} max={5.0} step={0.05}
        onX={(val) => { setXOffset(val); call(scale, val, yOffset, zOffset, roll, pitch, yaw); }}
        onY={(val) => { setYOffset(val); call(scale, xOffset, val, zOffset, roll, pitch, yaw); }}
        onZ={(val) => { setZOffset(val); call(scale, xOffset, yOffset, val, roll, pitch, yaw); }}
      />
      <VectorRow
        label="Rotation" x={roll} y={pitch} z={yaw} min={-180} max={180} step={0.5}
        xLabel="R" yLabel="P" zLabel="Y"
        onX={(val) => { setRoll(val);  call(scale, xOffset, yOffset, zOffset, val, pitch, yaw); }}
        onY={(val) => { setPitch(val); call(scale, xOffset, yOffset, zOffset, roll, val, yaw); }}
        onZ={(val) => { setYaw(val);   call(scale, xOffset, yOffset, zOffset, roll, pitch, val); }}
      />
    </CommandSection>
  );
}
