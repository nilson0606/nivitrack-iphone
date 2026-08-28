import { readFile, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const path = resolve(root, 'public', 'models', 'interactive_segmentation.task');
const expectedSize = 30_525_312;
const expectedHash = '38431BC66B883404E8397F74C3579404315B9B52B04A46C6346FE906A7309B03';
const info = await stat(path);
const bytes = await readFile(path);
const hash = createHash('sha256').update(bytes).digest('hex').toUpperCase();
const zipSignature = bytes.includes(Buffer.from([0x50, 0x4b, 0x03, 0x04]));
const pass = info.size === expectedSize && hash === expectedHash && zipSignature;

console.log(JSON.stringify({
  size: info.size,
  hash,
  zipSignature,
  pass,
}, null, 2));
if (!pass) process.exitCode = 1;
