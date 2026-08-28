import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import * as ort from 'onnxruntime-web/wasm';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const model = await readFile(resolve(root, 'public', 'models', 'modnet_quantized.onnx'));
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
const session = await ort.InferenceSession.create(model, {
  executionProviders: ['wasm'],
  executionMode: 'sequential',
  graphOptimizationLevel: 'all',
});
const size = 256;
const values = new Float32Array(1 * 3 * size * size);
const input = new ort.Tensor('float32', values, [1, 3, size, size]);
let output;
try {
  const result = await session.run({ [session.inputNames[0]]: input });
  output = result[session.outputNames[0]];
  const data = output.data;
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  let finite = true;
  for (let index = 0; index < data.length; index += 1) {
    const value = Number(data[index]);
    if (!Number.isFinite(value)) finite = false;
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  }
  console.log(JSON.stringify({
    inputs: session.inputNames,
    outputs: session.outputNames,
    dims: output.dims,
    length: data.length,
    finite,
    minimum,
    maximum,
  }, null, 2));
  if (!finite || output.dims.at(-1) !== size || output.dims.at(-2) !== size) {
    process.exitCode = 1;
  }
} finally {
  output?.dispose();
  input.dispose();
  await session.release();
}
