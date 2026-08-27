import { sites } from '@openai/sites-vite-plugin';
import tailwindcss from '@tailwindcss/postcss';
import vinext from 'vinext';
import { defineConfig } from 'vite';

const isCodexSeatbeltSandbox = process.env.CODEX_SANDBOX === 'seatbelt';

export default defineConfig({
  css: { postcss: { plugins: [tailwindcss()] } },
  resolve: {
    conditions: [
      'onnxruntime-web-use-extern-wasm',
      'module',
      'browser',
      'development|production',
    ],
  },
  server: {
    watch: {
      ignored: ['**/public/ort/*.wasm'],
      ...(isCodexSeatbeltSandbox
        ? { useFsEvents: false, usePolling: true }
        : {}),
    },
  },
  plugins: [vinext(), sites()],
});
