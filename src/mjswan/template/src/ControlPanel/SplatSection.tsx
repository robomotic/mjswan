import { useEffect, useState } from 'react';
import { Box, Slider, Text } from '@mantine/core';
import { CommandSection } from './CommandSection';

interface SplatSectionProps {
  /** Initial scale value from config. */
  scale: number;
  /** Initial X offset value from config. */
  xOffset: number;
  /** Initial Y offset value from config. */
  yOffset: number;
  /** Initial Z offset value from config. */
  zOffset: number;
  /** Called whenever scale or offsets are adjusted. */
  onCalibrate: (scale: number, xOffset: number, yOffset: number, zOffset: number) => void;
}

const SLIDER_TEXT_STYLE = { fontSize: '0.875em', fontWeight: 450, lineHeight: '1.375em', letterSpacing: '-0.75px', width: '50%', flexShrink: 0 } as const;
const SLIDER_STYLES = { root: { padding: '0' }, track: { height: 4 }, thumb: { width: 12, height: 12 } } as const;

/** Dev-mode calibration controls for a Gaussian Splat background. */
export function SplatSection({ scale: initialScale, xOffset: initialXOffset, yOffset: initialYOffset, zOffset: initialZOffset, onCalibrate }: SplatSectionProps) {
  const [scale, setScale] = useState(initialScale);
  const [xOffset, setXOffset] = useState(initialXOffset);
  const [yOffset, setYOffset] = useState(initialYOffset);
  const [zOffset, setZOffset] = useState(initialZOffset);

  // Sync when the config changes (e.g. switching scenes)
  useEffect(() => { setScale(initialScale); }, [initialScale]);
  useEffect(() => { setXOffset(initialXOffset); }, [initialXOffset]);
  useEffect(() => { setYOffset(initialYOffset); }, [initialYOffset]);
  useEffect(() => { setZOffset(initialZOffset); }, [initialZOffset]);

  const call = (s: number, x: number, y: number, z: number) => onCalibrate(s, x, y, z);

  return (
    <CommandSection label="Control" expandByDefault={true}>
      <Box pb="0.5em" px="xs" style={{ display: 'flex', alignItems: 'center' }}>
        <Text c="dimmed" style={SLIDER_TEXT_STYLE}>Scale</Text>
        <Box style={{ width: '50%' }}>
          <Slider value={scale} onChange={(val) => { setScale(val); call(val, xOffset, yOffset, zOffset); }} min={0.1} max={5.0} step={0.05} size="xs" label={(val) => val.toFixed(2)} styles={SLIDER_STYLES} />
        </Box>
      </Box>
      <Box pb="0.5em" px="xs" style={{ display: 'flex', alignItems: 'center' }}>
        <Text c="dimmed" style={SLIDER_TEXT_STYLE}>X Offset</Text>
        <Box style={{ width: '50%' }}>
          <Slider value={xOffset} onChange={(val) => { setXOffset(val); call(scale, val, yOffset, zOffset); }} min={-5.0} max={5.0} step={0.05} size="xs" label={(val) => val.toFixed(2)} styles={SLIDER_STYLES} />
        </Box>
      </Box>
      <Box pb="0.5em" px="xs" style={{ display: 'flex', alignItems: 'center' }}>
        <Text c="dimmed" style={SLIDER_TEXT_STYLE}>Y Offset</Text>
        <Box style={{ width: '50%' }}>
          <Slider value={yOffset} onChange={(val) => { setYOffset(val); call(scale, xOffset, val, zOffset); }} min={-5.0} max={5.0} step={0.05} size="xs" label={(val) => val.toFixed(2)} styles={SLIDER_STYLES} />
        </Box>
      </Box>
      <Box pb="0.5em" px="xs" style={{ display: 'flex', alignItems: 'center' }}>
        <Text c="dimmed" style={SLIDER_TEXT_STYLE}>Z Offset</Text>
        <Box style={{ width: '50%' }}>
          <Slider value={zOffset} onChange={(val) => { setZOffset(val); call(scale, xOffset, yOffset, val); }} min={-5.0} max={5.0} step={0.05} size="xs" label={(val) => val.toFixed(2)} styles={SLIDER_STYLES} />
        </Box>
      </Box>
    </CommandSection>
  );
}
