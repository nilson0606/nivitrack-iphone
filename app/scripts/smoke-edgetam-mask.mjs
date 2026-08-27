import { retainLargestComponent } from '../lib/edgetam-video-segmenter.ts';
import { createHash } from 'node:crypto';
import { readFile, stat } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const modelRoot = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '..',
  'public',
  'models',
  'edgetam-video',
);
const assets = {
  'box_prompt.bin': [4096, 'A97634DB16B23282C15A375794B717EEB7351C9D156AE63657F78BE7145640A7'],
  'decode_box.tflite': [17856560, '3C01AC9CA03FC8129B0C91C7D97E0757AC0912F6515B385B03C4A31EACE76363'],
  'decode.tflite': [17856640, '0F17571EE724FAB5DF208DF3872FF71249EF7D153AF87C795B835049AA2AB54A'],
  'encode.tflite': [10411888, '2E38353F85AD2A24D4FA3F3B9DDDEECA7A3574C414A3491C379A897AD0D58810'],
  'memcond.tflite': [25998528, '280CF7269DCEF2B0BD2B6278F714F960C929C2540E95056C290E8CDF01958C52'],
  'memorize.tflite': [4795808, 'FE2C9D0EB2CFA7C2CAB2BBF65D90AE4AEE3332C9C54D46469667EE4E23C097E6'],
  'mtpe.bin': [1792, 'F736C5694DB176C96DA7DBAFE6B5B36C5375DC52BD4C358C328FA807887F80F9'],
  'no_memory.bin': [1024, 'F974BD1E6436998BA8C372CDCBE768C5AB39EE942D847494C709977DB0592E9D'],
  'no_objptr.bin': [1024, '33988F72D7BEBF7E374449DD2695113144F981D4AA4537A84BB84B6B3F5D467A'],
  'track_sparse.bin': [2048, 'E2034E2F4E1149FE63944E4F8DBC0028925E1D284AEAE8DA58A1B7984840C2B1'],
};

for (const [file, [expectedSize, expectedHash]] of Object.entries(assets)) {
  const path = resolve(modelRoot, file);
  const fileStat = await stat(path);
  if (fileStat.size !== expectedSize) {
    throw new Error(`${file} size mismatch: ${fileStat.size} / ${expectedSize}`);
  }
  const hash = createHash('sha256').update(await readFile(path)).digest('hex').toUpperCase();
  if (hash !== expectedHash) throw new Error(`${file} SHA-256 mismatch`);
}

const logits = new Float32Array(8 * 6).fill(-2);
for (const [x, y] of [[3, 1], [4, 1], [3, 2], [4, 2], [3, 3], [4, 3]]) {
  logits[y * 8 + x] = 2;
}
for (const [x, y] of [[0, 0], [7, 5], [1, 5]]) logits[y * 8 + x] = 3;

const kept = retainLargestComponent(logits, 8, 6);
if (kept !== 6) throw new Error(`expected 6 retained pixels, received ${kept}`);
if (logits[0] > 0 || logits[5 * 8 + 7] > 0 || logits[5 * 8 + 1] > 0) {
  throw new Error('detached background components were not removed');
}
if (logits[2 * 8 + 3] <= 0) throw new Error('main subject component was removed');

console.log('EdgeTAM model assets and largest-component mask smoke test passed.');
