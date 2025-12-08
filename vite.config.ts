import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  root: 'examples',
  appType: 'mpa', // Multi-page app mode - serve HTML files directly
  build: {
    outDir: '../dist-demo',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'examples/index.html'),
        dashboard: resolve(__dirname, 'examples/training-dashboard.html'),
      },
    },
  },
  resolve: {
    alias: {
      'browser-gnn': resolve(__dirname, 'src/index.ts'),
    },
  },
});
