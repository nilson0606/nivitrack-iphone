import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import * as ort from 'onnxruntime-web/wasm';

const width = 320;
const height = 256;
const pixels = width * height;
const frameBytes = pixels * 3;
const root = resolve(import.meta.dirname, '..');

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = pathToFileURL(resolve(root, 'node_modules/onnxruntime-web/dist') + '\\').href;

const model = readFileSync(resolve(root, 'public/models/yolox_n_body_head_hand_256x320.onnx'));
const session = await ort.InferenceSession.create(model, {
  executionProviders: ['wasm'],
  graphOptimizationLevel: 'all',
  executionMode: 'sequential',
});

assert.deepEqual(session.inputNames, ['input']);
assert.deepEqual(session.outputNames, ['batchno_classid_score_x1y1x2y2']);

const rawPath = process.argv[2];
const raw = rawPath ? readFileSync(rawPath) : Buffer.alloc(frameBytes);
assert.equal(raw.length % frameBytes, 0, 'raw BGR input must contain complete 320x256 frames');
const summaries = [];

for (let frame = 0; frame < raw.length / frameBytes; frame += 1) {
  const input = new Float32Array(pixels * 3);
  const base = frame * frameBytes;
  for (let index = 0; index < pixels; index += 1) {
    const pixel = base + index * 3;
    input[index] = raw[pixel];
    input[pixels + index] = raw[pixel + 1];
    input[pixels * 2 + index] = raw[pixel + 2];
  }
  const result = await session.run({
    input: new ort.Tensor('float32', input, [1, 3, height, width]),
  });
  const tensor = result.batchno_classid_score_x1y1x2y2;
  assert.equal(tensor.dims.at(-1), 7);
  const rows = tensor.data;
  let heads = 0;
  let bestScore = 0;
  for (let offset = 0; offset + 6 < rows.length; offset += 7) {
    if (Math.round(rows[offset + 1]) !== 1 || rows[offset + 2] < 0.2) continue;
    heads += 1;
    bestScore = Math.max(bestScore, rows[offset + 2]);
  }
  summaries.push({ frame, heads, bestScore: Number(bestScore.toFixed(3)) });
}

await session.release();
console.log(JSON.stringify({
  frames: summaries.length,
  detectedFrames: summaries.filter((item) => item.heads > 0).length,
  maximumHeads: Math.max(...summaries.map((item) => item.heads)),
  summaries,
}, null, 2));
