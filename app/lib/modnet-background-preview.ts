import * as ort from 'onnxruntime-web/wasm';

import {
  recoverTrackedSubjectAlpha,
  selectModnetTrackedAlpha,
  trackedSubjectRegion,
} from './person-background-removal';
import type { Box } from './vit-tracker';
import { interpolateBox, type TrackPoint } from './video-export';

const INFERENCE_SIZE = 384;
const MASK_SIZE = 256;
const PREVIEW_FPS = 10;

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function downsampleAlpha(source: Float32Array) {
  const alpha = new Uint8ClampedArray(MASK_SIZE * MASK_SIZE);
  const scale = INFERENCE_SIZE / MASK_SIZE;
  for (let y = 0; y < MASK_SIZE; y += 1) {
    const sourceY = (y + 0.5) * scale - 0.5;
    const top = clamp(Math.floor(sourceY), 0, INFERENCE_SIZE - 1);
    const bottom = Math.min(INFERENCE_SIZE - 1, top + 1);
    const mixY = clamp(sourceY - top, 0, 1);
    for (let x = 0; x < MASK_SIZE; x += 1) {
      const sourceX = (x + 0.5) * scale - 0.5;
      const left = clamp(Math.floor(sourceX), 0, INFERENCE_SIZE - 1);
      const right = Math.min(INFERENCE_SIZE - 1, left + 1);
      const mixX = clamp(sourceX - left, 0, 1);
      const topValue = source[top * INFERENCE_SIZE + left] * (1 - mixX)
        + source[top * INFERENCE_SIZE + right] * mixX;
      const bottomValue = source[bottom * INFERENCE_SIZE + left] * (1 - mixX)
        + source[bottom * INFERENCE_SIZE + right] * mixX;
      alpha[y * MASK_SIZE + x] = Math.round(
        clamp(topValue * (1 - mixY) + bottomValue * mixY, 0, 1) * 255,
      );
    }
  }
  return alpha;
}

type ModnetPreviewFrame = {
  time: number;
  alpha: Uint8ClampedArray;
};

export type ModnetPreviewStats = {
  frames: number;
  elapsedMs: number;
  averageInferenceMs: number;
};

type PrepareOptions = {
  startTime: number;
  endTime: number;
  maxFrames?: number;
  seekTo: (time: number) => Promise<void>;
  isCancelled: () => boolean;
  onProgress: (progress: number, frame: number, total: number) => void;
};

async function waitForDecodedFrame(video: HTMLVideoElement) {
  await new Promise<void>((resolve) => {
    let finished = false;
    let frameCallback = 0;
    const finish = () => {
      if (finished) return;
      finished = true;
      window.clearTimeout(timeout);
      if (frameCallback && typeof video.cancelVideoFrameCallback === 'function') {
        video.cancelVideoFrameCallback(frameCallback);
      }
      resolve();
    };
    const timeout = window.setTimeout(finish, 180);
    if (typeof video.requestVideoFrameCallback === 'function') {
      frameCallback = video.requestVideoFrameCallback(() => finish());
    } else {
      requestAnimationFrame(() => requestAnimationFrame(finish));
    }
  });
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
}

class ModnetPreviewGenerator {
  private readonly session: ort.InferenceSession;
  private readonly inputCanvas = document.createElement('canvas');
  private readonly inputData = new Float32Array(1 * 3 * INFERENCE_SIZE * INFERENCE_SIZE);
  private previousAlpha: Float32Array | null = null;
  private missedFrames = 0;

  private constructor(session: ort.InferenceSession) {
    this.session = session;
    this.inputCanvas.width = INFERENCE_SIZE;
    this.inputCanvas.height = INFERENCE_SIZE;
  }

  static async create() {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    ort.env.wasm.wasmPaths = new URL('ort/', document.baseURI).href;
    const modelUrl = new URL('models/modnet_quantized.onnx', document.baseURI).href;
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      executionMode: 'sequential',
    });
    return new ModnetPreviewGenerator(session);
  }

  async infer(video: HTMLVideoElement, box: Box) {
    const [regionX, regionY, regionWidth, regionHeight] = trackedSubjectRegion(
      box,
      video.videoWidth,
      video.videoHeight,
    );
    const context = this.inputCanvas.getContext('2d', {
      alpha: false,
      willReadFrequently: true,
    });
    if (!context) throw new Error('Safari 無法建立 MODNet 輸入畫布');
    context.drawImage(
      video,
      regionX,
      regionY,
      regionWidth,
      regionHeight,
      0,
      0,
      INFERENCE_SIZE,
      INFERENCE_SIZE,
    );
    const rgba = context.getImageData(0, 0, INFERENCE_SIZE, INFERENCE_SIZE).data;
    const plane = INFERENCE_SIZE * INFERENCE_SIZE;
    for (let index = 0; index < plane; index += 1) {
      const pixel = index * 4;
      this.inputData[index] = rgba[pixel] / 127.5 - 1;
      this.inputData[plane + index] = rgba[pixel + 1] / 127.5 - 1;
      this.inputData[plane * 2 + index] = rgba[pixel + 2] / 127.5 - 1;
    }

    const input = new ort.Tensor('float32', this.inputData, [1, 3, INFERENCE_SIZE, INFERENCE_SIZE]);
    const started = performance.now();
    try {
      const outputs = await this.session.run({ [this.session.inputNames[0]]: input });
      try {
        const inferenceMs = performance.now() - started;
        const output = outputs[this.session.outputNames[0]];
        if (!output || output.dims.at(-1) !== INFERENCE_SIZE || output.dims.at(-2) !== INFERENCE_SIZE) {
          throw new Error('MODNet 輸出尺寸不正確');
        }
        const confidence = output.data as Float32Array;
        const relativeBox: Box = [
          box[0] - regionX,
          box[1] - regionY,
          box[2],
          box[3],
        ];
        const selected = selectModnetTrackedAlpha(
          confidence,
          INFERENCE_SIZE,
          INFERENCE_SIZE,
          relativeBox,
          regionWidth,
          regionHeight,
        );
        const recovered = recoverTrackedSubjectAlpha(
          selected,
          this.previousAlpha,
          plane,
          this.missedFrames,
        );
        this.missedFrames = recovered.missedFrames;
        const current = recovered.alpha;
        if (this.previousAlpha?.length === current.length && recovered.fresh) {
          for (let index = 0; index < current.length; index += 1) {
            current[index] = current[index] * 0.78 + this.previousAlpha[index] * 0.22;
          }
        }
        this.previousAlpha = new Float32Array(current);
        const alpha = downsampleAlpha(current);
        return { alpha, inferenceMs };
      } finally {
        Object.values(outputs).forEach((tensor) => tensor.dispose());
      }
    } finally {
      input.dispose();
    }
  }

  async close() {
    this.previousAlpha = null;
    await this.session.release();
  }
}

export class ModnetPreviewTimeline {
  readonly startTime: number;
  readonly endTime: number;
  private readonly frames: ModnetPreviewFrame[];
  private readonly maskCanvas = document.createElement('canvas');
  private readonly subjectCanvas = document.createElement('canvas');
  private readonly mixedAlpha = new Uint8ClampedArray(MASK_SIZE * MASK_SIZE);
  private readonly imageData: ImageData;

  constructor(frames: ModnetPreviewFrame[]) {
    if (frames.length === 0) throw new Error('MODNet Preview 沒有任何遮罩影格');
    this.frames = frames;
    this.startTime = frames[0].time;
    this.endTime = frames[frames.length - 1].time;
    this.maskCanvas.width = MASK_SIZE;
    this.maskCanvas.height = MASK_SIZE;
    const context = this.maskCanvas.getContext('2d', { alpha: true });
    if (!context) throw new Error('Safari 無法建立 MODNet 遮罩畫布');
    this.imageData = context.createImageData(MASK_SIZE, MASK_SIZE);
    for (let index = 0; index < this.mixedAlpha.length; index += 1) {
      const pixel = index * 4;
      this.imageData.data[pixel] = 255;
      this.imageData.data[pixel + 1] = 255;
      this.imageData.data[pixel + 2] = 255;
    }
  }

  private alphaAt(time: number) {
    if (time <= this.frames[0].time) return this.frames[0].alpha;
    const last = this.frames[this.frames.length - 1];
    if (time >= last.time) return last.alpha;
    let low = 0;
    let high = this.frames.length - 1;
    while (low + 1 < high) {
      const middle = Math.floor((low + high) / 2);
      if (this.frames[middle].time <= time) low = middle;
      else high = middle;
    }
    const before = this.frames[low];
    const after = this.frames[high];
    const amount = (time - before.time) / Math.max(0.0001, after.time - before.time);
    for (let index = 0; index < this.mixedAlpha.length; index += 1) {
      this.mixedAlpha[index] = Math.round(
        before.alpha[index] * (1 - amount) + after.alpha[index] * amount,
      );
    }
    return this.mixedAlpha;
  }

  draw(
    video: HTMLVideoElement,
    outputCanvas: HTMLCanvasElement,
    box: Box,
    suppression = 0.62,
    crop: Box = [0, 0, video.videoWidth, video.videoHeight],
  ) {
    const alpha = this.alphaAt(video.currentTime);
    const exponent = 0.65 + clamp(suppression, 0, 1) * 0.55;
    const maskContext = this.maskCanvas.getContext('2d', { alpha: true });
    if (!maskContext) throw new Error('Safari 無法更新 MODNet 遮罩');
    for (let index = 0; index < alpha.length; index += 1) {
      this.imageData.data[index * 4 + 3] = Math.round(
        Math.pow(alpha[index] / 255, exponent) * 255,
      );
    }
    maskContext.putImageData(this.imageData, 0, 0);

    const [regionX, regionY, regionWidth, regionHeight] = trackedSubjectRegion(
      box,
      video.videoWidth,
      video.videoHeight,
    );
    const [cropX, cropY, cropWidth, cropHeight] = crop;
    const outputScaleX = outputCanvas.width / Math.max(1, cropWidth);
    const outputScaleY = outputCanvas.height / Math.max(1, cropHeight);
    const destinationX = (regionX - cropX) * outputScaleX;
    const destinationY = (regionY - cropY) * outputScaleY;
    const destinationWidth = regionWidth * outputScaleX;
    const destinationHeight = regionHeight * outputScaleY;
    const subjectWidth = Math.max(1, Math.ceil(destinationWidth));
    const subjectHeight = Math.max(1, Math.ceil(destinationHeight));
    if (this.subjectCanvas.width !== subjectWidth || this.subjectCanvas.height !== subjectHeight) {
      this.subjectCanvas.width = subjectWidth;
      this.subjectCanvas.height = subjectHeight;
    }
    const subjectContext = this.subjectCanvas.getContext('2d', { alpha: true });
    const outputContext = outputCanvas.getContext('2d', { alpha: false });
    if (!subjectContext || !outputContext) throw new Error('Safari 無法建立 MODNet 合成畫布');

    subjectContext.globalCompositeOperation = 'source-over';
    subjectContext.clearRect(0, 0, subjectWidth, subjectHeight);
    subjectContext.drawImage(
      video,
      regionX,
      regionY,
      regionWidth,
      regionHeight,
      0,
      0,
      subjectWidth,
      subjectHeight,
    );
    subjectContext.globalCompositeOperation = 'destination-in';
    subjectContext.drawImage(
      this.maskCanvas,
      0,
      0,
      subjectWidth,
      subjectHeight,
    );
    subjectContext.globalCompositeOperation = 'source-over';
    outputContext.fillStyle = '#000';
    outputContext.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
    outputContext.drawImage(
      this.subjectCanvas,
      destinationX,
      destinationY,
      destinationWidth,
      destinationHeight,
    );
  }

  close() {
    this.frames.length = 0;
    this.maskCanvas.width = 1;
    this.maskCanvas.height = 1;
    this.subjectCanvas.width = 1;
    this.subjectCanvas.height = 1;
  }
}

export async function prepareModnetPreview(
  video: HTMLVideoElement,
  path: TrackPoint[],
  options: PrepareOptions,
) {
  if (!video.videoWidth || !video.videoHeight || path.length < 2) {
    throw new Error('影片或追蹤路徑尚未準備好');
  }
  const startTime = clamp(options.startTime, 0, video.duration);
  const endTime = clamp(options.endTime, startTime, video.duration);
  const duration = Math.max(0, endTime - startTime);
  const maximumFrames = Math.max(2, Math.floor(options.maxFrames ?? Number.MAX_SAFE_INTEGER));
  const steps = Math.max(1, Math.min(maximumFrames - 1, Math.ceil(duration * PREVIEW_FPS)));
  const times: number[] = [];
  for (let step = 0; step <= steps; step += 1) {
    times.push(startTime + (duration * step) / steps);
  }
  const frames: ModnetPreviewFrame[] = [];
  let inferenceTotal = 0;
  const started = performance.now();
  let generator: ModnetPreviewGenerator | null = null;
  try {
    generator = await ModnetPreviewGenerator.create();
    for (let index = 0; index < times.length; index += 1) {
      if (options.isCancelled()) throw new Error('使用者已取消 MODNet 去背');
      const time = times[index];
      await options.seekTo(time);
      await waitForDecodedFrame(video);
      const result = await generator.infer(video, interpolateBox(path, time));
      inferenceTotal += result.inferenceMs;
      frames.push({ time, alpha: result.alpha });
      options.onProgress((index + 1) / times.length, index + 1, times.length);
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    }
  } finally {
    await generator?.close();
  }
  return {
    timeline: new ModnetPreviewTimeline(frames),
    stats: {
      frames: frames.length,
      elapsedMs: performance.now() - started,
      averageInferenceMs: inferenceTotal / Math.max(1, frames.length),
    } satisfies ModnetPreviewStats,
  };
}
