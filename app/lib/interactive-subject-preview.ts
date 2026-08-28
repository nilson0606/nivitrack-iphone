import type { Box } from './vit-tracker';
import {
  ModnetPreviewTimeline,
  type ModnetPreviewFrame,
} from './modnet-background-preview';
import { trackedSubjectRegion } from './person-background-removal';
import { interpolateBox, type TrackPoint } from './video-export';

const INPUT_SIZE = 256;
const MASK_SIZE = 256;
const EXPERIMENT_FPS = 5;

type Point = { x: number; y: number };

type PrepareOptions = {
  startTime: number;
  endTime: number;
  seekTo: (time: number) => Promise<void>;
  isCancelled: () => boolean;
  onProgress: (progress: number, frame: number, total: number) => void;
};

const FRAME_CHANNEL = 'nivitrack-instance-segmenter-v1';

type SegmenterReply = {
  channel: typeof FRAME_CHANNEL;
  requestId: number;
  type: 'ready' | 'result' | 'error' | 'bootstrap-error';
  delegate?: 'CPU' | 'GPU';
  width?: number;
  height?: number;
  inferenceMs?: number;
  buffer?: ArrayBuffer;
  message?: string;
};

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function frameRequest(
  frame: HTMLIFrameElement,
  requestId: number,
  message: object,
  transfer: Transferable[],
  isCancelled: () => boolean,
  timeoutMs: number,
) {
  return new Promise<SegmenterReply>((resolve, reject) => {
    const target = frame.contentWindow;
    if (!target) {
      reject(new Error('單人實例分割隔離頁面尚未就緒'));
      return;
    }
    let finished = false;
    const cleanup = () => {
      if (finished) return;
      finished = true;
      window.clearInterval(cancelPoll);
      window.clearTimeout(timeout);
      window.removeEventListener('message', onMessage);
      frame.removeEventListener('error', onError);
    };
    const fail = (error: Error) => {
      cleanup();
      reject(error);
    };
    const onMessage = (event: MessageEvent<SegmenterReply>) => {
      if (event.source !== target || event.data?.channel !== FRAME_CHANNEL) return;
      if (event.data.type === 'bootstrap-error') {
        fail(new Error(event.data.message ?? '單人實例分割隔離頁面啟動失敗'));
        return;
      }
      if (event.data.requestId !== requestId) return;
      if (event.data.type === 'error') {
        fail(new Error(event.data.message ?? '單人實例分割失敗'));
        return;
      }
      cleanup();
      resolve(event.data);
    };
    const onError = () => fail(new Error('單人實例分割隔離頁面載入失敗'));
    const cancelPoll = window.setInterval(() => {
      if (isCancelled()) fail(new Error('使用者已取消單人實例分割'));
    }, 100);
    const timeout = window.setTimeout(
      () => fail(new Error('單人實例分割運算逾時')),
      timeoutMs,
    );
    window.addEventListener('message', onMessage);
    frame.addEventListener('error', onError);
    try {
      target.postMessage(
        { channel: FRAME_CHANNEL, requestId, ...message },
        window.location.origin,
        transfer,
      );
    } catch (error) {
      fail(error instanceof Error ? error : new Error(String(error)));
    }
  });
}

function createSegmenterFrame(isCancelled: () => boolean) {
  return new Promise<HTMLIFrameElement>((resolve, reject) => {
    const frame = document.createElement('iframe');
    frame.setAttribute('aria-hidden', 'true');
    frame.tabIndex = -1;
    frame.style.cssText = 'position:fixed;width:1px;height:1px;left:-10px;top:-10px;opacity:0;pointer-events:none;border:0';
    frame.src = new URL('interactive-segmenter-frame.html?v=1', document.baseURI).href;
    let finished = false;
    const cleanup = () => {
      if (finished) return;
      finished = true;
      window.clearInterval(cancelPoll);
      window.clearTimeout(timeout);
      frame.removeEventListener('load', onLoad);
      frame.removeEventListener('error', onError);
    };
    const fail = (error: Error) => {
      cleanup();
      frame.remove();
      reject(error);
    };
    const onLoad = () => {
      cleanup();
      resolve(frame);
    };
    const onError = () => fail(new Error('單人實例分割隔離頁面載入失敗'));
    const cancelPoll = window.setInterval(() => {
      if (isCancelled()) fail(new Error('使用者已取消單人實例分割'));
    }, 100);
    const timeout = window.setTimeout(
      () => fail(new Error('單人實例分割隔離頁面載入逾時')),
      30000,
    );
    frame.addEventListener('load', onLoad);
    frame.addEventListener('error', onError);
    document.body.appendChild(frame);
  });
}

function promptsForBox(box: Box, sourceWidth: number, sourceHeight: number) {
  const left = clamp(box[0] / sourceWidth, 0.02, 0.98);
  const right = clamp((box[0] + box[2]) / sourceWidth, 0.02, 0.98);
  const top = clamp(box[1] / sourceHeight, 0.02, 0.98);
  const bottom = clamp((box[1] + box[3]) / sourceHeight, 0.02, 0.98);
  const centerX = (left + right) / 2;
  const positivePoints: Point[] = [
    { x: centerX, y: top + (bottom - top) * 0.34 },
    { x: centerX, y: top + (bottom - top) * 0.48 },
    { x: centerX, y: top + (bottom - top) * 0.62 },
  ];
  const marginX = Math.max(0.025, (right - left) * 0.12);
  const marginY = Math.max(0.025, (bottom - top) * 0.08);
  const outerLeft = clamp(left - marginX, 0.01, 0.99);
  const outerRight = clamp(right + marginX, 0.01, 0.99);
  const outerTop = clamp(top - marginY, 0.01, 0.99);
  const outerBottom = clamp(bottom + marginY, 0.01, 0.99);
  const negativePoints: Point[] = [
    { x: outerLeft, y: outerTop },
    { x: centerX, y: outerTop },
    { x: outerRight, y: outerTop },
    { x: outerLeft, y: (top + bottom) / 2 },
    { x: outerRight, y: (top + bottom) / 2 },
    { x: outerLeft, y: outerBottom },
    { x: centerX, y: outerBottom },
    { x: outerRight, y: outerBottom },
  ];
  return { positivePoints, negativePoints };
}

function resizeMask(source: Float32Array, sourceWidth: number, sourceHeight: number) {
  if (source.length !== sourceWidth * sourceHeight) {
    throw new Error('單人實例分割遮罩尺寸不正確');
  }
  const alpha = new Uint8ClampedArray(MASK_SIZE * MASK_SIZE);
  for (let y = 0; y < MASK_SIZE; y += 1) {
    const sourceY = clamp(Math.round(((y + 0.5) / MASK_SIZE) * sourceHeight - 0.5), 0, sourceHeight - 1);
    for (let x = 0; x < MASK_SIZE; x += 1) {
      const sourceX = clamp(Math.round(((x + 0.5) / MASK_SIZE) * sourceWidth - 0.5), 0, sourceWidth - 1);
      alpha[y * MASK_SIZE + x] = Math.round(
        clamp(source[sourceY * sourceWidth + sourceX], 0, 1) * 255,
      );
    }
  }
  return alpha;
}

export async function prepareInteractiveSubjectPreview(
  video: HTMLVideoElement,
  path: TrackPoint[],
  options: PrepareOptions,
) {
  if (!video.videoWidth || !video.videoHeight || path.length < 2) {
    throw new Error('影片或追蹤路徑尚未準備好');
  }
  if (typeof HTMLIFrameElement === 'undefined') {
    throw new Error('此 Safari 不支援隔離式單人分割頁面');
  }
  const startTime = clamp(options.startTime, 0, video.duration);
  const endTime = clamp(options.endTime, startTime, Math.min(video.duration, startTime + 3));
  const steps = Math.max(1, Math.ceil((endTime - startTime) * EXPERIMENT_FPS));
  const times = Array.from(
    { length: steps + 1 },
    (_, index) => startTime + ((endTime - startTime) * index) / steps,
  );
  const inputCanvas = document.createElement('canvas');
  inputCanvas.width = INPUT_SIZE;
  inputCanvas.height = INPUT_SIZE;
  const context = inputCanvas.getContext('2d', { alpha: false, willReadFrequently: true });
  if (!context) throw new Error('Safari 無法建立單人分割輸入畫布');
  const frames: ModnetPreviewFrame[] = [];
  let frame: HTMLIFrameElement | null = null;
  let requestId = 1;
  let inferenceTotal = 0;
  const started = performance.now();
  try {
    frame = await createSegmenterFrame(options.isCancelled);
    await frameRequest(frame, requestId++, {
      type: 'init',
      wasmRoot: new URL('mediapipe/', document.baseURI).href,
      modelUrl: new URL('models/interactive_segmentation.task', document.baseURI).href,
      }, [], options.isCancelled, 90000);
    for (let index = 0; index < times.length; index += 1) {
      if (options.isCancelled()) throw new Error('使用者已取消單人實例分割');
      const time = times[index];
      await options.seekTo(time);
      const box = interpolateBox(path, time);
      const [regionX, regionY, regionWidth, regionHeight] = trackedSubjectRegion(
        box,
        video.videoWidth,
        video.videoHeight,
      );
      context.drawImage(
        video,
        regionX,
        regionY,
        regionWidth,
        regionHeight,
        0,
        0,
        INPUT_SIZE,
        INPUT_SIZE,
      );
      const relativeBox: Box = [
        box[0] - regionX,
        box[1] - regionY,
        box[2],
        box[3],
      ];
      const prompts = promptsForBox(relativeBox, regionWidth, regionHeight);
      const pixels = context.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
      const buffer = pixels.buffer as ArrayBuffer;
      const reply = await frameRequest(frame, requestId++, {
        type: 'segment',
        width: INPUT_SIZE,
        height: INPUT_SIZE,
        buffer,
        ...prompts,
      }, [buffer], options.isCancelled, 60000);
      if (!reply.buffer || !reply.width || !reply.height) {
        throw new Error('單人實例分割沒有回傳遮罩');
      }
      frames.push({
        time,
        alpha: resizeMask(new Float32Array(reply.buffer), reply.width, reply.height),
      });
      inferenceTotal += reply.inferenceMs ?? 0;
      options.onProgress((index + 1) / times.length, index + 1, times.length);
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    }
  } finally {
    frame?.contentWindow?.postMessage(
      { channel: FRAME_CHANNEL, type: 'close' },
      window.location.origin,
    );
    frame?.remove();
    inputCanvas.width = 1;
    inputCanvas.height = 1;
  }
  return {
    timeline: new ModnetPreviewTimeline(frames),
    stats: {
      frames: times.length,
      elapsedMs: performance.now() - started,
      averageInferenceMs: inferenceTotal / Math.max(1, times.length),
    },
  };
}
