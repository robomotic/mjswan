import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { vanillaExtractPlugin } from '@vanilla-extract/vite-plugin';
import path from 'path';
import fs from 'fs';

// Extract version from Python package (source of truth)
function getVersionFromPython(): string {
  const initPath = path.resolve(__dirname, '../__init__.py');
  try {
    const content = fs.readFileSync(initPath, 'utf-8');
    const match = content.match(/__version__\s*=\s*["']([^"']+)["']/);
    if (match) {
      return match[1];
    }
  } catch {
    // Fallback to package.json if Python file not found
  }
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const pkg = require('./package.json');
  return pkg.version || '0.0.0';
}

export default defineConfig({
  plugins: [react(), vanillaExtractPlugin()],
  base: process.env.MJSWAN_BASE_PATH || '/',
  define: {
    __APP_VERSION__: JSON.stringify(getVersionFromPython()),
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      'mujoco': path.resolve(__dirname, './src/mujoco/mujoco_wasm'),
    },
  },
  optimizeDeps: {
    exclude: ['mujoco'],
  },
  assetsInclude: ['**/*.wasm'],
  server: {
    port: 8000,
    host: true,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
    chunkSizeWarningLimit: 11000,
    rollupOptions: {
      input: path.resolve(__dirname, 'index.html'),
    },
  },
  worker: {
    format: 'es',
  },
});
