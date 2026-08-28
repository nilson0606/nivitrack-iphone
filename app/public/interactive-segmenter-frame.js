import {
  FilesetResolver,
  InteractiveSegmenter,
} from './mediapipe/vision_bundle.mjs';

const CHANNEL = 'nivitrack-instance-segmenter-v1';
let segmenter = null;

function reply(requestId, payload, transfer = []) {
  parent.postMessage({ channel: CHANNEL, requestId, ...payload }, location.origin, transfer);
}

async function initialize(requestId, wasmRoot, modelUrl) {
  const fileset = await FilesetResolver.forVisionTasks(wasmRoot);
  const canvas = document.createElement('canvas');
  canvas.width = 1;
  canvas.height = 1;
  let delegate = 'CPU';
  try {
    segmenter = await InteractiveSegmenter.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: modelUrl, delegate: 'CPU' },
      canvas,
    });
  } catch (cpuError) {
    delegate = 'GPU';
    try {
      segmenter = await InteractiveSegmenter.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: modelUrl, delegate: 'GPU' },
        canvas,
      });
    } catch (gpuError) {
      const cpuMessage = cpuError instanceof Error ? cpuError.message : String(cpuError);
      const gpuMessage = gpuError instanceof Error ? gpuError.message : String(gpuError);
      throw new Error('CPU：' + cpuMessage + '；GPU：' + gpuMessage);
    }
  }
  reply(requestId, { type: 'ready', delegate });
}

function segmentFrame(
  requestId,
  width,
  height,
  buffer,
  positivePoints,
  negativePoints,
) {
  if (!segmenter) throw new Error('單人實例分割模型尚未初始化');
  const pixels = new Uint8ClampedArray(buffer);
  segmenter.setImage({ data: pixels, width, height });
  const strokes = [
    {
      brushMode: 1,
      point: positivePoints,
      isCompleted: true,
    },
    ...negativePoints.map((point) => ({
      brushMode: 2,
      point: [point],
      isCompleted: true,
    })),
  ];
  const started = performance.now();
  const mask = segmenter.segment(strokes);
  try {
    const copied = new Float32Array(mask.getAsFloat32Array());
    reply(requestId, {
      type: 'result',
      width: mask.width,
      height: mask.height,
      inferenceMs: performance.now() - started,
      buffer: copied.buffer,
    }, [copied.buffer]);
  } finally {
    mask.close();
  }
}

addEventListener('message', (event) => {
  if (
    event.source !== parent
    || event.origin !== location.origin
    || event.data?.channel !== CHANNEL
  ) return;
  const request = event.data;
  if (request.type === 'close') {
    segmenter?.close();
    segmenter = null;
    return;
  }
  Promise.resolve()
    .then(() => request.type === 'init'
      ? initialize(request.requestId, request.wasmRoot, request.modelUrl)
      : segmentFrame(
          request.requestId,
          request.width,
          request.height,
          request.buffer,
          request.positivePoints,
          request.negativePoints,
        ))
    .catch((error) => {
      reply(request.requestId, {
        type: 'error',
        message: error instanceof Error ? error.message : String(error),
      });
    });
});
