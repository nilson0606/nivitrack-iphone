import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';

import type { Box } from './vit-tracker';

const LOW_CONFIDENCE_LIMIT = 0.12;
const MAX_MISSED_FRAMES = 12;

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
  const left = clamp(Math.floor(((x + width * 0.16) / sourceWidth) * maskWidth), 0, maskWidth - 1);
  const right = clamp(Math.ceil(((x + width * 0.84) / sourceWidth) * maskWidth), left + 1, maskWidth);
  const top = clamp(Math.floor(((y + height * 0.12) / sourceHeight) * maskHeight), 0, maskHeight - 1);
  const bottom = clamp(Math.ceil(((y + height * 0.88) / sourceHeight) * maskHeight), top + 1, maskHeight);
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
  const threshold = clamp(seed.confidence * 0.34, 0.20, 0.38);
  const [boxX, boxY, boxWidth, boxHeight] = box;
  const gateCenterX = ((boxX + boxWidth / 2) / sourceWidth) * maskWidth;
  const gateCenterY = ((boxY + boxHeight / 2) / sourceHeight) * maskHeight;
  const gateRadiusX = Math.max(2, (boxWidth / sourceWidth) * maskWidth * 0.68);
  const gateRadiusY = Math.max(2, (boxHeight / sourceHeight) * maskHeight * 0.68);
  const isInsideTrackedDancer = (index: number) => {
    const x = index % maskWidth;
    const y = Math.floor(index / maskWidth);
    const normalizedX = (x - gateCenterX) / gateRadiusX;
    const normalizedY = (y - gateCenterY) / gateRadiusY;
    return normalizedX * normalizedX + normalizedY * normalizedY <= 1;
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
      alpha[index] = smoothstep(threshold, Math.min(0.82, threshold + 0.42), confidence[index]);
    }
  }
  return alpha;
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
    const inferenceScale = Math.min(1, 256 / Math.max(video.videoWidth, video.videoHeight));
    const inferenceWidth = Math.max(2, Math.round(video.videoWidth * inferenceScale));
    const inferenceHeight = Math.max(2, Math.round(video.videoHeight * inferenceScale));
    if (this.inferenceCanvas.width !== inferenceWidth || this.inferenceCanvas.height !== inferenceHeight) {
      this.inferenceCanvas.width = inferenceWidth;
      this.inferenceCanvas.height = inferenceHeight;
    }
    const inferenceContext = this.inferenceCanvas.getContext('2d', { alpha: false });
    if (!inferenceContext) throw new Error('Safari 無法建立人物分割畫布');
    inferenceContext.drawImage(video, 0, 0, inferenceWidth, inferenceHeight);
    this.timestamp += 1;
    let currentAlpha: Float32Array | null = null;
    let maskWidth = 0;
    let maskHeight = 0;

    this.segmenter.segmentForVideo(this.inferenceCanvas, this.timestamp, (result) => {
      const mask = result.confidenceMasks?.[0];
      if (!mask) return;
      maskWidth = mask.width;
      maskHeight = mask.height;
      currentAlpha = selectTrackedSubjectAlpha(
        mask.getAsFloat32Array(),
        maskWidth,
        maskHeight,
        box,
        video.videoWidth,
        video.videoHeight,
      );
    });

    if (!currentAlpha) {
      this.missedFrames += 1;
      if (!this.previousAlpha || this.missedFrames > MAX_MISSED_FRAMES) {
        throw new Error('複雜背景中暫時找不到已選舞者；請換到人物較清楚的畫面重新框選');
      }
      currentAlpha = this.previousAlpha;
    } else {
      this.missedFrames = 0;
    }

    if (this.previousAlpha?.length === currentAlpha.length && this.previousAlpha !== currentAlpha) {
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
    subjectContext.drawImage(video, 0, 0, this.subjectCanvas.width, this.subjectCanvas.height);
    subjectContext.globalCompositeOperation = 'destination-in';
    subjectContext.drawImage(this.maskCanvas, 0, 0, this.subjectCanvas.width, this.subjectCanvas.height);
    subjectContext.globalCompositeOperation = 'source-over';

    outputContext.fillStyle = '#000';
    outputContext.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
    outputContext.drawImage(this.subjectCanvas, 0, 0);
  }

  close() {
    this.segmenter.close();
    this.previousAlpha = null;
  }
}
