import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import * as ort from 'onnxruntime-web/wasm';

const root = resolve(import.meta.dirname, '..');
const wasmDirectory = resolve(root, 'public', 'ort') + '\\';
const modelPath = resolve(root, 'public', 'models', 'vittrack.onnx');

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = pathToFileURL(wasmDirectory).href;

const model = await readFile(modelPath);
const session = await ort.InferenceSession.create(model, {
  executionProviders: ['wasm'],
  graphOptimizationLevel: 'all',
  executionMode: 'sequential',
});

const template = new ort.Tensor(
  'float32',
  new Float32Array(1 * 3 * 128 * 128),
  [1, 3, 128, 128],
);
const search = new ort.Tensor(
  'float32',
  new Float32Array(1 * 3 * 256 * 256),
  [1, 3, 256, 256],
);
const outputs = await session.run({ template, search });

const result = Object.fromEntries(
  Object.entries(outputs).map(([name, tensor]) => [
    name,
    {
      dims: tensor.dims,
      length: tensor.size,
      finite: Array.from(tensor.data).every((value) => Number.isFinite(Number(value))),
    },
  ]),
);

console.log(JSON.stringify({
  inputs: session.inputNames,
  outputs: session.outputNames,
  tensors: result,
}, null, 2));
