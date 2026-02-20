import React, { useEffect, useRef } from 'react';
import { mjswanRuntime } from '../core/engine/runtime';
import type { MainModule } from 'mujoco';

type MjswanViewerProps = {
  scenePath: string;
  baseUrl: string;
  policyConfigPath?: string | null;
  onStatusChange?: (status: string) => void;
  onError?: (error: Error) => void;
  onReady?: () => void;
};

const MjswanViewer = ({
  scenePath,
  baseUrl,
  policyConfigPath,
  onStatusChange,
  onError,
  onReady,
}: MjswanViewerProps) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const runtimeRef = useRef<mjswanRuntime | null>(null);
  const mujocoRef = useRef<MainModule | null>(null);

  useEffect(() => {
    let cancelled = false;

    const notify = (status: string) => {
      onStatusChange?.(status);
    };

    const init = async () => {
      notify('Loading MuJoCo...');
      if (!mujocoRef.current) {
        const baseWasmUrl = import.meta.env.BASE_URL || '/';
        const modulePath = `${baseWasmUrl}assets/mujoco/mujoco_wasm.js`.replace(/\/+/g, '/');
        const mujocoModule = await import(/* @vite-ignore */ modulePath);
        mujocoRef.current = await mujocoModule.default();
      }
      if (cancelled) {
        return;
      }

      const container = containerRef.current;
      if (!container) {
        throw new Error('Failed to find viewer container.');
      }

      const mujoco = mujocoRef.current;
      if (!mujoco) {
        throw new Error('MuJoCo not loaded.');
      }

      if (!runtimeRef.current) {
        runtimeRef.current = new mjswanRuntime(mujoco, container, { baseUrl });
      }

      notify('Loading scene assets...');
      await runtimeRef.current.loadEnvironment(scenePath, policyConfigPath ?? null);
      if (cancelled) {
        return;
      }
      notify('Running simulation');
      onReady?.();
    };

    init().catch((error) => {
      if (!cancelled) {
        console.error('Failed to initialize MuJoCo viewer:', error);
        onError?.(error instanceof Error ? error : new Error(String(error)));
        notify('Failed to load scene');
      }
    });

    return () => {
      cancelled = true;
      runtimeRef.current?.dispose();
      runtimeRef.current = null;
    };
  }, [scenePath, baseUrl, policyConfigPath, onStatusChange, onError, onReady]);

  return <div ref={containerRef} className="viewer" />;
};

export default MjswanViewer;
