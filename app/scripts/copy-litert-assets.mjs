import { cp, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const source = resolve(root, 'node_modules', '@litertjs', 'core', 'wasm');
const target = resolve(root, 'public', 'litert');

await mkdir(target, { recursive: true });

for (const file of [
  'litert_wasm_internal.js',
  'litert_wasm_internal.wasm',
  'litert_wasm_compat_internal.js',
  'litert_wasm_compat_internal.wasm',
]) {
  await cp(resolve(source, file), resolve(target, file));
}

console.log('LiteRT.js Safari runtime assets copied.');
