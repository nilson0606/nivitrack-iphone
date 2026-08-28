/// <reference lib="webworker" />

import {
  FilesetResolver,
  InteractiveSegmenter,
  type Stroke,
} from '@mediapipe/tasks-vision';

type Point = { x: number; y: number };

type WorkerRequest =
  | { type: 'init'; requestId: number; wasmRoot: string; modelUrl: string }
  | {
      type: 'segment';
      requestId: number;
      bitmap: ImageBitmap;
      positivePoints: Point[];
      negativePoints: Point[];
    }
  | { type: 'close' };

const workerScope = self as DedicatedWorkerGlobalScope;
let segmenter: InteractiveSegmenter | null = null;
type StrokeBrushMode = Stroke['brushMode'];
const POSITIVE_BRUSH_MODE = 1 as StrokeBrushMode;
const NEGATIVE_BRUSH_MODE = 2 as StrokeBrushMode;

function respond(requestId: number, payload: Record<string, unknown>, transfer: Transferable[] = []) {
  workerScope.postMessage({ requestId, ...payload }, transfer);
}

async function initialize(requestId: number, wasmRoot: string, modelUrl: string) {
  const fileset = await FilesetResolver.forVisionTasks(wasmRoot);
  const canvas = new OffscreenCanvas(1, 1);
  let delegate: 'CPU' | 'GPU' = 'CPU';
  try {
    segmenter = await InteractiveSegmenter.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: modelUrl, delegate: 'CPU' },
      canvas,
    });
  } catch {
    delegate = 'GPU';
    segmenter = await InteractiveSegmenter.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: modelUrl, delegate: 'GPU' },
      canvas,
    });
  }
  respond(requestId, { type: 'ready', delegate });
}

function segmentFrame(
  requestId: number,
  bitmap: ImageBitmap,
  positivePoints: Point[],
  negativePoints: Point[],
) {
  if (!segmenter) throw new Error('單人實例分割模型尚未初始化');
  try {
    segmenter.setImage(bitmap);
  } finally {
    bitmap.close();
  }
  const strokes: Stroke[] = [
    {
      brushMode: POSITIVE_BRUSH_MODE,
      point: positivePoints,
      isCompleted: true,
    },
    ...negativePoints.map((point) => ({
      brushMode: NEGATIVE_BRUSH_MODE,
      point: [point],
      isCompleted: true,
    })),
  ];
  const started = performance.now();
  const mask = segmenter.segment(strokes);
  try {
    const copied = new Float32Array(mask.getAsFloat32Array());
    respond(requestId, {
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

workerScope.addEventListener('message', (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  if (request.type === 'close') {
    segmenter?.close();
    segmenter = null;
    workerScope.close();
    return;
  }
  Promise.resolve()
    .then(() => request.type === 'init'
      ? initialize(request.requestId, request.wasmRoot, request.modelUrl)
      : segmentFrame(
          request.requestId,
          request.bitmap,
          request.positivePoints,
          request.negativePoints,
        ))
    .catch((error) => {
      respond(request.requestId, {
        type: 'error',
        message: error instanceof Error ? error.message : String(error),
      });
    });
});
