import type { Box } from './vit-tracker';
import type { EdgeTamMask, EdgeTamOnnxSegmenter } from './edgetam-onnx-segmenter';
import { interpolateBox, type FrameMatteRenderer, type TrackPoint } from './video-export';

const MASK_FPS = 15;

export type BackgroundMaskFrame = {
  time: number;
  alpha: Uint8ClampedArray;
  foregroundPixels: number;
};

export type BackgroundRemovalStats = {
  frames: number;
  elapsedMs: number;
  averageInferenceMs: number;
};

export type PrepareBackgroundRemovalOptions = {
  startTime: number;
  endTime: number;
  anchorTime: number;
  reuseAnchor?: boolean;
  onProgress: (progress: number, frame: number, total: number) => void;
  isCancelled: () => boolean;
  seekTo: (time: number) => Promise<void>;
};

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function normalizedPromptBox(box: Box, width: number, height: number): [number, number, number, number] {
  const left = clamp(box[0] / width, 0, 1);
  const top = clamp(box[1] / height, 0, 1);
  const boxWidth = clamp(box[2] / width, 0, 1 - left);
  const boxHeight = clamp(box[3] / height, 0, 1 - top);
  // The ViT box is deliberately generous so limbs cannot leave the camera crop.
  // EdgeTAM only needs the inner object core to identify which object to follow;
  // its output mask can still grow beyond this prompt to cover the full subject.
  const insetX = boxWidth * 0.125;
  const insetY = boxHeight * 0.025;
  return [
    left + insetX,
    top + insetY,
    Math.max(0.001, boxWidth - insetX * 2),
    Math.max(0.001, boxHeight - insetY * 2),
  ];
}

export class BackgroundMaskTimeline implements FrameMatteRenderer {
  readonly frames: BackgroundMaskFrame[];
  readonly startTime: number;
  readonly endTime: number;
  private readonly maskCanvas = document.createElement('canvas');
  private readonly subjectCanvas = document.createElement('canvas');
  private readonly maskContext: CanvasRenderingContext2D;
  private readonly subjectContext: CanvasRenderingContext2D;
  private readonly imageData: ImageData;
  private readonly mixedAlpha = new Uint8ClampedArray(256 * 256);

  constructor(frames: BackgroundMaskFrame[]) {
    if (frames.length === 0) throw new Error('去背時間軸沒有任何影格');
    this.frames = frames;
    this.startTime = frames[0].time;
    this.endTime = frames[frames.length - 1].time;
    this.maskCanvas.width = 256;
    this.maskCanvas.height = 256;
    const maskContext = this.maskCanvas.getContext('2d', { alpha: true });
    const subjectContext = this.subjectCanvas.getContext('2d', { alpha: true });
    if (!maskContext || !subjectContext) throw new Error('Safari 無法建立去背合成畫布');
    this.maskContext = maskContext;
    this.subjectContext = subjectContext;
    this.imageData = maskContext.createImageData(256, 256);
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
    const mix = (time - before.time) / Math.max(0.000001, after.time - before.time);
    for (let index = 0; index < this.mixedAlpha.length; index += 1) {
      this.mixedAlpha[index] = Math.round(before.alpha[index] * (1 - mix) + after.alpha[index] * mix);
    }
    return this.mixedAlpha;
  }

  draw(
    video: HTMLVideoElement,
    outputContext: CanvasRenderingContext2D,
    cropX: number,
    cropY: number,
    cropWidth: number,
    cropHeight: number,
    outputWidth: number,
    outputHeight: number,
    time: number,
  ) {
    const alpha = this.alphaAt(time);
    for (let index = 0; index < alpha.length; index += 1) {
      this.imageData.data[index * 4 + 3] = alpha[index];
    }
    this.maskContext.putImageData(this.imageData, 0, 0);

    if (this.subjectCanvas.width !== outputWidth || this.subjectCanvas.height !== outputHeight) {
      this.subjectCanvas.width = outputWidth;
      this.subjectCanvas.height = outputHeight;
    }
    this.subjectContext.globalCompositeOperation = 'source-over';
    this.subjectContext.clearRect(0, 0, outputWidth, outputHeight);
    this.subjectContext.drawImage(
      video,
      cropX,
      cropY,
      cropWidth,
      cropHeight,
      0,
      0,
      outputWidth,
      outputHeight,
    );
    this.subjectContext.globalCompositeOperation = 'destination-in';
    this.subjectContext.drawImage(
      this.maskCanvas,
      (cropX / video.videoWidth) * 256,
      (cropY / video.videoHeight) * 256,
      (cropWidth / video.videoWidth) * 256,
      (cropHeight / video.videoHeight) * 256,
      0,
      0,
      outputWidth,
      outputHeight,
    );
    this.subjectContext.globalCompositeOperation = 'source-over';

    outputContext.fillStyle = '#000';
    outputContext.fillRect(0, 0, outputWidth, outputHeight);
    outputContext.drawImage(this.subjectCanvas, 0, 0);
  }
}

function storeMask(time: number, mask: EdgeTamMask): BackgroundMaskFrame {
  return {
    time,
    alpha: new Uint8ClampedArray(mask.alpha),
    foregroundPixels: mask.foregroundPixels,
  };
}

export async function prepareBackgroundRemoval(
  video: HTMLVideoElement,
  path: TrackPoint[],
  segmenter: EdgeTamOnnxSegmenter,
  options: PrepareBackgroundRemovalOptions,
) {
  if (!video.videoWidth || !video.videoHeight || !Number.isFinite(video.duration)) {
    throw new Error('影片尚未準備好，無法去背');
  }
  const startTime = clamp(options.startTime, 0, video.duration);
  const endTime = clamp(options.endTime, startTime, video.duration);
  const anchorTime = clamp(options.anchorTime, startTime, endTime);
  const times: number[] = [];
  for (let time = startTime; time <= endTime + 0.0001; time += 1 / MASK_FPS) {
    times.push(Math.min(time, endTime));
  }
  if (times.length === 0 || endTime - times[times.length - 1] > 0.001) times.push(endTime);
  if (!times.some((time) => Math.abs(time - anchorTime) < 0.001)) times.push(anchorTime);
  times.sort((left, right) => left - right);
  const forwardTimes = times.filter((time) => time > anchorTime + 0.001);
  const backwardTimes = times.filter((time) => time < anchorTime - 0.001).reverse();
  const totalWork = 1 + forwardTimes.length + backwardTimes.length;
  const frames: BackgroundMaskFrame[] = [];
  let inferenceTotal = 0;
  let inferenceCount = 0;
  let completedWork = 0;
  const started = performance.now();

  if (options.isCancelled()) throw new Error('使用者已取消去背');
  const firstBox = interpolateBox(path, anchorTime);
  const reuseAnchor = Boolean(options.reuseAnchor && segmenter.hasAnchor());
  let mask: EdgeTamMask;
  if (reuseAnchor) {
    mask = segmenter.resetToAnchor();
  } else {
    await options.seekTo(anchorTime);
    mask = await segmenter.start(
      video,
      normalizedPromptBox(firstBox, video.videoWidth, video.videoHeight),
    );
    inferenceTotal += mask.inferenceMs;
    inferenceCount += 1;
  }
  frames.push(storeMask(anchorTime, mask));
  completedWork += 1;
  options.onProgress(completedWork / totalWork, frames.length, times.length);

  for (const time of forwardTimes) {
    if (options.isCancelled()) throw new Error('使用者已取消去背');
    await options.seekTo(time);
    mask = await segmenter.track(video);
    frames.push(storeMask(time, mask));
    inferenceTotal += mask.inferenceMs;
    inferenceCount += 1;
    completedWork += 1;
    options.onProgress(completedWork / totalWork, frames.length, times.length);
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  }

  if (backwardTimes.length > 0) {
    if (options.isCancelled()) throw new Error('使用者已取消去背');
    mask = segmenter.resetToAnchor();
    for (const time of backwardTimes) {
      if (options.isCancelled()) throw new Error('使用者已取消去背');
      await options.seekTo(time);
      mask = await segmenter.track(video);
      frames.push(storeMask(time, mask));
      inferenceTotal += mask.inferenceMs;
      inferenceCount += 1;
      completedWork += 1;
      options.onProgress(completedWork / totalWork, frames.length, times.length);
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    }
  }

  frames.sort((left, right) => left.time - right.time);

  return {
    timeline: new BackgroundMaskTimeline(frames),
    stats: {
      frames: frames.length,
      elapsedMs: performance.now() - started,
      averageInferenceMs: inferenceTotal / Math.max(1, inferenceCount),
    } satisfies BackgroundRemovalStats,
  };
}
