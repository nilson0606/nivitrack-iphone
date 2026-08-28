import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';

import type { Box } from './vit-tracker';

const LOW_CONFIDENCE_LIMIT = 0.06;
const MAX_MISSED_FRAMES = 12;
const INFERENCE_SIZE = 256;

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function smoothstep(edge0: number, edge1: number, value: number) {
  const amount = clamp((value - edge0) / Math.max(0.0001, edge1 - edge0), 0, 1);
  return amount * amount * (3 - 2 * amount);
}

function findSeed(
  confidence: Float32Array,
  maskWidth: number,
  maskHeight: number,
  box: Box,
  sourceWidth: number,
  sourceHeight: number,
) {
  const [x, y, width, height] = box;
  const left = clamp(Math.floor(((x + width * 0.04) / sourceWidth) * maskWidth), 0, maskWidth - 1);
  const right = clamp(Math.ceil(((x + width * 0.96) / sourceWidth) * maskWidth), left + 1, maskWidth);
  const top = clamp(Math.floor(((y + height * 0.02) / sourceHeight) * maskHeight), 0, maskHeight - 1);
  const bottom = clamp(Math.ceil(((y + height * 0.98) / sourceHeight) * maskHeight), top + 1, maskHeight);
  let bestIndex = top * maskWidth + left;
  let bestConfidence = -1;
  for (let maskY = top; maskY < bottom; maskY += 1) {
    const row = maskY * maskWidth;
    for (let maskX = left; maskX < right; maskX += 1) {
      const index = row + maskX;
      if (confidence[index] > bestConfidence) {
        bestConfidence = confidence[index];
        bestIndex = index;
      }
    }
  }
  return { index: bestIndex, confidence: bestConfidence };
}

export function selectTrackedSubjectAlpha(
  confidence: Float32Array,
  maskWidth: number,
  maskHeight: number,
  box: Box,
  sourceWidth: number,
  sourceHeight: number,
) {
  const pixelCount = maskWidth * maskHeight;
  if (confidence.length !== pixelCount) throw new Error('MediaPipe 人物遮罩尺寸不正確');
  const seed = findSeed(confidence, maskWidth, maskHeight, box, sourceWidth, sourceHeight);
  if (seed.confidence < LOW_CONFIDENCE_LIMIT) return null;

  // A low, peak-relative threshold keeps fast-moving arms and legs connected,
  // while the tracked seed prevents separate people from being retained.
  const threshold = clamp(seed.confidence * 0.34, 0.06, 0.38);
  const solidConfidence = Math.max(threshold + 0.02, Math.min(0.82, seed.confidence * 0.82));
  const [boxX, boxY, boxWidth, boxHeight] = box;
  const gateCenterX = ((boxX + boxWidth / 2) / sourceWidth) * maskWidth;
  const gateCenterY = ((boxY + boxHeight / 2) / sourceHeight) * maskHeight;
  const gateRadiusX = Math.max(2, (boxWidth / sourceWidth) * maskWidth * 0.62);
  const gateRadiusY = Math.max(2, (boxHeight / sourceHeight) * maskHeight * 0.62);
  const isInsideTrackedDancer = (index: number) => {
    const x = index % maskWidth;
    const y = Math.floor(index / maskWidth);
    const normalizedX = (x - gateCenterX) / gateRadiusX;
    const normalizedY = (y - gateCenterY) / gateRadiusY;
    return Math.pow(Math.abs(normalizedX), 4) + Math.pow(Math.abs(normalizedY), 4) <= 1;
  };
  const selected = new Uint8Array(pixelCount);
  const queue = new Int32Array(pixelCount);
  let head = 0;
  let tail = 0;
  selected[seed.index] = 1;
  queue[tail++] = seed.index;

  while (head < tail) {
    const current = queue[head++];
    const currentX = current % maskWidth;
    const currentY = Math.floor(current / maskWidth);
    for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
      const nextY = currentY + offsetY;
      if (nextY < 0 || nextY >= maskHeight) continue;
      for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
        if (offsetX === 0 && offsetY === 0) continue;
        const nextX = currentX + offsetX;
        if (nextX < 0 || nextX >= maskWidth) continue;
        const next = nextY * maskWidth + nextX;
        if (selected[next] || !isInsideTrackedDancer(next) || confidence[next] < threshold) continue;
        selected[next] = 1;
        queue[tail++] = next;
      }
    }
  }

  const alpha = new Float32Array(pixelCount);
  for (let index = 0; index < pixelCount; index += 1) {
    if (selected[index] && isInsideTrackedDancer(index)) {
      alpha[index] = smoothstep(threshold, solidConfidence, confidence[index]);
    }
  }
  return alpha;
}

export function trackedSubjectRegion(box: Box, sourceWidth: number, sourceHeight: number): Box {
  const safeWidth = Math.max(2, box[2]);
  const safeHeight = Math.max(2, box[3]);
  const left = clamp(box[0] - safeWidth * 0.34, 0, Math.max(0, sourceWidth - 2));
  const top = clamp(box[1] - safeHeight * 0.18, 0, Math.max(0, sourceHeight - 2));
  const right = clamp(box[0] + safeWidth * 1.34, left + 2, sourceWidth);
  const bottom = clamp(box[1] + safeHeight * 1.18, top + 2, sourceHeight);
  return [left, top, right - left, bottom - top];
}

export function recoverTrackedSubjectAlpha(
  currentAlpha: Float32Array | null,
  previousAlpha: Float32Array | null,
  pixelCount: number,
  missedFrames: number,
) {
  if (currentAlpha) return { alpha: currentAlpha, missedFrames: 0, fresh: true };
  const nextMissedFrames = missedFrames + 1;
  const alpha = new Float32Array(pixelCount);
  if (previousAlpha?.length === pixelCount && nextMissedFrames <= MAX_MISSED_FRAMES) {
    const retention = 1 - nextMissedFrames / (MAX_MISSED_FRAMES + 4);
    for (let index = 0; index < pixelCount; index += 1) {
      alpha[index] = previousAlpha[index] * retention;
    }
  }
  return { alpha, missedFrames: nextMissedFrames, fresh: false };
}

export class PersonBackgroundRenderer {
  private readonly segmenter: ImageSegmenter;
  private readonly inferenceCanvas = document.createElement('canvas');
  private readonly maskCanvas = document.createElement('canvas');
  private readonly subjectCanvas = document.createElement('canvas');
  private previousAlpha: Float32Array | null = null;
  private timestamp = 0;
  private missedFrames = 0;

  private constructor(segmenter: ImageSegmenter) {
    this.segmenter = segmenter;
  }

  static async create() {
    const wasmRoot = new URL('mediapipe/', document.baseURI).href;
    const modelUrl = new URL('models/selfie_segmenter.tflite', document.baseURI).href;
    const fileset = await FilesetResolver.forVisionTasks(wasmRoot);
    const canvas = document.createElement('canvas');
    const common = {
      baseOptions: { modelAssetPath: modelUrl, delegate: 'GPU' as const },
      runningMode: 'VIDEO' as const,
      outputConfidenceMasks: true,
      outputCategoryMask: false,
      canvas,
    };
    try {
      return new PersonBackgroundRenderer(await ImageSegmenter.createFromOptions(fileset, common));
    } catch {
      return new PersonBackgroundRenderer(await ImageSegmenter.createFromOptions(fileset, {
        ...common,
        baseOptions: { modelAssetPath: modelUrl, delegate: 'CPU' },
      }));
    }
  }

  render(video: HTMLVideoElement, outputCanvas: HTMLCanvasElement, box: Box) {
    if (!video.videoWidth || !video.videoHeight) return;
    const [regionX, regionY, regionWidth, regionHeight] = trackedSubjectRegion(
      box,
      video.videoWidth,
      video.videoHeight,
    );
    const inferenceWidth = INFERENCE_SIZE;
    const inferenceHeight = INFERENCE_SIZE;
    if (this.inferenceCanvas.width !== inferenceWidth || this.inferenceCanvas.height !== inferenceHeight) {
      this.inferenceCanvas.width = inferenceWidth;
      this.inferenceCanvas.height = inferenceHeight;
    }
    const inferenceContext = this.inferenceCanvas.getContext('2d', { alpha: false });
    if (!inferenceContext) throw new Error('Safari 無法建立人物分割畫布');
    inferenceContext.drawImage(
      video,
      regionX,
      regionY,
      regionWidth,
      regionHeight,
      0,
      0,
      inferenceWidth,
      inferenceHeight,
    );
    this.timestamp += 1;
    let currentAlpha: Float32Array | null = null;
    let maskWidth = inferenceWidth;
    let maskHeight = inferenceHeight;
    const relativeBox: Box = [
      box[0] - regionX,
      box[1] - regionY,
      box[2],
      box[3],
    ];

    this.segmenter.segmentForVideo(this.inferenceCanvas, this.timestamp, (result) => {
      const mask = result.confidenceMasks?.[0];
      if (!mask) return;
      maskWidth = mask.width;
      maskHeight = mask.height;
      currentAlpha = selectTrackedSubjectAlpha(
        mask.getAsFloat32Array(),
        maskWidth,
        maskHeight,
        relativeBox,
        regionWidth,
        regionHeight,
      );
    });

    const recovered = recoverTrackedSubjectAlpha(
      currentAlpha,
      this.previousAlpha,
      maskWidth * maskHeight,
      this.missedFrames,
    );
    currentAlpha = recovered.alpha;
    this.missedFrames = recovered.missedFrames;

    if (recovered.fresh && this.previousAlpha?.length === currentAlpha.length && this.previousAlpha !== currentAlpha) {
      for (let index = 0; index < currentAlpha.length; index += 1) {
        // A light one-frame blend reduces edge shimmer without leaving long trails
        // behind a dancer's fast-moving hands and feet.
        currentAlpha[index] = currentAlpha[index] * 0.86 + this.previousAlpha[index] * 0.14;
      }
    }
    this.previousAlpha = new Float32Array(currentAlpha);

    if (this.maskCanvas.width !== maskWidth || this.maskCanvas.height !== maskHeight) {
      this.maskCanvas.width = maskWidth;
      this.maskCanvas.height = maskHeight;
    }
    const maskContext = this.maskCanvas.getContext('2d');
    if (!maskContext) throw new Error('Safari 無法建立人物遮罩畫布');
    const image = maskContext.createImageData(maskWidth, maskHeight);
    for (let index = 0; index < currentAlpha.length; index += 1) {
      const pixel = index * 4;
      image.data[pixel] = 255;
      image.data[pixel + 1] = 255;
      image.data[pixel + 2] = 255;
      image.data[pixel + 3] = Math.round(clamp(currentAlpha[index], 0, 1) * 255);
    }
    maskContext.putImageData(image, 0, 0);

    if (this.subjectCanvas.width !== outputCanvas.width || this.subjectCanvas.height !== outputCanvas.height) {
      this.subjectCanvas.width = outputCanvas.width;
      this.subjectCanvas.height = outputCanvas.height;
    }
    const subjectContext = this.subjectCanvas.getContext('2d', { alpha: true });
    const outputContext = outputCanvas.getContext('2d', { alpha: false });
    if (!subjectContext || !outputContext) throw new Error('Safari 無法建立人物去背合成畫布');

    subjectContext.globalCompositeOperation = 'source-over';
    subjectContext.clearRect(0, 0, this.subjectCanvas.width, this.subjectCanvas.height);
    const outputScaleX = outputCanvas.width / video.videoWidth;
    const outputScaleY = outputCanvas.height / video.videoHeight;
    const destinationX = regionX * outputScaleX;
    const destinationY = regionY * outputScaleY;
    const destinationWidth = regionWidth * outputScaleX;
    const destinationHeight = regionHeight * outputScaleY;
    subjectContext.drawImage(
      video,
      regionX,
      regionY,
      regionWidth,
      regionHeight,
      destinationX,
      destinationY,
      destinationWidth,
      destinationHeight,
    );
    subjectContext.globalCompositeOperation = 'destination-in';
    subjectContext.drawImage(
      this.maskCanvas,
      destinationX,
      destinationY,
      destinationWidth,
      destinationHeight,
    );
    subjectContext.globalCompositeOperation = 'source-over';

    outputContext.fillStyle = '#000';
    outputContext.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
    outputContext.drawImage(this.subjectCanvas, 0, 0);
  }

  close() {
    this.segmenter.close();
    this.previousAlpha = null;
  }

  reset() {
    this.previousAlpha = null;
    this.missedFrames = 0;
  }
}
