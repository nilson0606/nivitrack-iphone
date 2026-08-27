import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import * as ort from 'onnxruntime-web/wasm';

const root = resolve(import.meta.dirname, '..');
const wasmDirectory = resolve(root, 'public', 'ort') + '\\';
const modelDirectory = resolve(root, 'public', 'models', 'matanyone2');
const stages = [
  'encoder',
  'uncert',
  'memory',
  'readout',
  'decoder',
  'maskencoder',
  'objsummary',
];

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = pathToFileURL(wasmDirectory).href;

const sessions = new Map();
for (const stage of stages) {
  const bytes = await readFile(resolve(modelDirectory, `${stage}.onnx`));
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
    executionMode: 'sequential',
  });
  sessions.set(stage, session);
}

const encoder = sessions.get('encoder');
const image = new ort.Tensor(
  'float16',
  new Uint16Array(1 * 3 * 512 * 288),
  [1, 3, 512, 288],
);
const encoderOutputs = await encoder.run({ image });
const tensors = Object.fromEntries(
  Object.entries(encoderOutputs).map(([name, tensor]) => [
    name,
    {
      type: tensor.type,
      dims: tensor.dims,
      length: tensor.size,
      bytes: tensor.data.byteLength,
    },
  ]),
);

const halfOne = 0x3c00;
const maskBits = new Uint16Array(512 * 288);
for (let y = 96; y < 448; y += 1) {
  maskBits.fill(halfOne, y * 288 + 72, y * 288 + 216);
}
const mask = new ort.Tensor('float16', maskBits, [1, 1, 512, 288]);
const sensory = new ort.Tensor(
  'float16',
  new Uint16Array(1 * 1 * 256 * 32 * 18),
  [1, 1, 256, 32, 18],
);
const maskOutputs = await sessions.get('maskencoder').run({
  image,
  pix_feat: encoderOutputs.pix_feat,
  sensory,
  masks: mask,
});
const summaryOutputs = await sessions.get('objsummary').run({
  masks: mask,
  mask_value: maskOutputs.mask_value,
});

const featurePixels = 32 * 18;
const maxTokens = 5 * featurePixels;
const memoryKeyBits = new Uint16Array(64 * maxTokens);
const memoryShrinkageBits = new Uint16Array(maxTokens);
const memoryValueBits = new Uint16Array(256 * maxTokens);
const keyBits = encoderOutputs.key.data;
const shrinkageBits = encoderOutputs.shrinkage.data;
const valueBits = maskOutputs.mask_value.data;
for (let channel = 0; channel < 64; channel += 1) {
  memoryKeyBits.set(
    keyBits.subarray(channel * featurePixels, (channel + 1) * featurePixels),
    channel * maxTokens,
  );
}
memoryShrinkageBits.set(shrinkageBits);
for (let channel = 0; channel < 256; channel += 1) {
  memoryValueBits.set(
    valueBits.subarray(channel * featurePixels, (channel + 1) * featurePixels),
    channel * maxTokens,
  );
}
const validBits = new Uint16Array(maxTokens);
validBits.fill(halfOne, 0, featurePixels);
const memoryKey = new ort.Tensor('float16', memoryKeyBits, [1, 64, maxTokens]);
const memoryShrinkage = new ort.Tensor(
  'float16',
  memoryShrinkageBits,
  [1, 1, maxTokens],
);
const memoryValue = new ort.Tensor(
  'float16',
  memoryValueBits,
  [1, 256, maxTokens],
);
const validMask = new ort.Tensor('float16', validBits, [1, maxTokens, 1]);
const queryKey = new ort.Tensor('float16', keyBits, [1, 64, featurePixels]);
const querySelection = new ort.Tensor(
  'float16',
  encoderOutputs.selection.data,
  [1, 64, featurePixels],
);
const memoryOutputs = await sessions.get('memory').run({
  memory_key: memoryKey,
  memory_shrinkage: memoryShrinkage,
  memory_value: memoryValue,
  valid_mask: validMask,
  query_key: queryKey,
  query_selection: querySelection,
});
const difference = new ort.Tensor(
  'float16',
  new Uint16Array(1 * 256 * 32 * 18),
  [1, 256, 32, 18],
);
const uncertaintyOutputs = await sessions.get('uncert').run({
  last_pix_feat: encoderOutputs.pix_feat,
  cur_pix_feat: encoderOutputs.pix_feat,
  last_mask: mask,
  mem_val_diff: difference,
});
const pixel = new ort.Tensor(
  'float16',
  memoryOutputs.visual_readout.data,
  [1, 1, 256, 32, 18],
);
const objectMemory = new ort.Tensor(
  'float16',
  summaryOutputs.obj_summaries.data,
  [1, 1, 1, 16, 257],
);
const readoutOutputs = await sessions.get('readout').run({
  pix_feat: encoderOutputs.pix_feat,
  pixel,
  sensory: maskOutputs.new_sensory,
  last_mask: mask,
  obj_memory: objectMemory,
});
const decoderOutputs = await sessions.get('decoder').run({
  f8: encoderOutputs.f8,
  f4: encoderOutputs.f4,
  f2: encoderOutputs.f2,
  f1: encoderOutputs.f1,
  memory_readout: readoutOutputs.mem_readout,
  sensory: maskOutputs.new_sensory,
});

const fullPipeline = {
  maskValue: maskOutputs.mask_value.dims,
  objectSummary: summaryOutputs.obj_summaries.dims,
  memoryReadout: memoryOutputs.visual_readout.dims,
  uncertainty: uncertaintyOutputs.prob.dims,
  decoderLogits: decoderOutputs.logits.dims,
  decoderSensory: decoderOutputs.new_sensory.dims,
};

const tensorsToDispose = new Set([
  image,
  mask,
  sensory,
  memoryKey,
  memoryShrinkage,
  memoryValue,
  validMask,
  queryKey,
  querySelection,
  difference,
  pixel,
  objectMemory,
  ...Object.values(encoderOutputs),
  ...Object.values(maskOutputs),
  ...Object.values(summaryOutputs),
  ...Object.values(memoryOutputs),
  ...Object.values(uncertaintyOutputs),
  ...Object.values(readoutOutputs),
  ...Object.values(decoderOutputs),
]);
for (const tensor of tensorsToDispose) tensor.dispose();
await Promise.all(Array.from(sessions.values(), (session) => session.release()));

console.log(JSON.stringify({
  runtime: 'onnxruntime-web/wasm',
  loadedStages: Array.from(sessions.keys()),
  encoderOutputs: tensors,
  fullPipeline,
}, null, 2));
