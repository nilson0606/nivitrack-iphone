import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';

import type { Box } from './vit-tracker';

const LOW_CONFIDENCE_LIMIT = 0.12;
const MAX_MISSED_FRAMES = 12;
const INFERENCE_SIZE = 256;
const TRAIL_CAPTURE_INTERVAL = 3;
const MAX_TRAIL_SNAPSHOT_SIZE = 480;

export type BackgroundFillMode = 'color' | 'blur';
export type CloneLayout = 'trail' | 'row';
export type CloneStyle = 'subject' | 'outline';

export type PersonBackgroundEffects = {
  backgroundMode: BackgroundFillMode;
  backgroundColor: string;
  backgroundBlur: number;
  outlineColor: string;
  outlineWidth: number;
  cloneCount: number;
  cloneLayout: CloneLayout;
  cloneStyle: CloneStyle;
};

export const DEFAULT_PERSON_BACKGROUND_EFFECTS: PersonBackgroundEffects = {
  backgroundMode: 'color',
  backgroundColor: '#000000',
  backgroundBlur: 18,
  outlineColor: '#d9f06f',
  outlineWidth: 0,
  cloneCount: 0,
  cloneLayout: 'trail',
  cloneStyle: 'subject',
};

type TrailFrame = {
  canvas: HTMLCanvasElement;
  x: number;
  y: number;
  width: number;
  height: number;
};

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

export function equalRowCloneFrames(
  outputWidth: number,
  outputHeight: number,
  subjectWidth: number,
  subjectHeight: number,
  cloneCount: number,
): Box[] {
  const total = Math.round(clamp(cloneCount, 0, 4)) + 1;
  const gap = outputWidth * 0.025;
  let destinationHeight = outputHeight * 0.82;
  let destinationWidth = destinationHeight * (subjectWidth / Math.max(2, subjectHeight));
  const groupWidth = destinationWidth * total + gap * (total - 1);
  if (groupWidth > outputWidth * 0.94) {
    const fitScale = (outputWidth * 0.94 - gap * (total - 1)) / Math.max(2, destinationWidth * total);
    destinationWidth *= fitScale;
    destinationHeight *= fitScale;
  }
  const firstX = (outputWidth - (destinationWidth * total + gap * (total - 1))) / 2;
  const destinationY = (outputHeight - destinationHeight) / 2;
  return Array.from({ length: total }, (_, index) => [
    firstX + index * (destinationWidth + gap),
    destinationY,
    destinationWidth,
    destinationHeight,
  ]);
}

function smoothstep(edge0: number, edge1: number, value: number) {
  const amount = clamp((value - edge0) / Math.max(0.0001, edge1 - edge0), 0, 1);
  return amount * amount * (3 - 2 * amount);
}

export function tightenTrackedSubjectEdges(alpha: Float32Array, width: number, height: number) {
  const tightened = new Float32Array(alpha.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      const current = alpha[index];
      if (current <= 0) continue;
      const left = x > 0 ? alpha[index - 1] : 0;
      const right = x + 1 < width ? alpha[index + 1] : 0;
      const top = y > 0 ? alpha[index - width] : 0;
      const bottom = y + 1 < height ? alpha[index + width] : 0;
      const eroded = Math.min(current, left, right, top, bottom);
      const next = current * 0.18 + eroded * 0.82;
      tightened[index] = next < 0.055 ? 0 : next;
    }
  }
  return tightened;
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
  const threshold = Math.min(seed.confidence * 0.78, clamp(seed.confidence * 0.38, 0.14, 0.36));
  const solidConfidence = Math.max(threshold + 0.08, Math.min(0.86, seed.confidence * 0.82));
  const [boxX, boxY, boxWidth, boxHeight] = box;
  const gateCenterX = ((boxX + boxWidth / 2) / sourceWidth) * maskWidth;
  const gateCenterY = ((boxY + boxHeight / 2) / sourceHeight) * maskHeight;
  const gateRadiusX = Math.max(2, (boxWidth / sourceWidth) * maskWidth * 0.56);
  const gateRadiusY = Math.max(2, (boxHeight / sourceHeight) * maskHeight * 0.56);
  const isInsideTrackedDancer = (index: number) => {
    const x = index % maskWidth;
    const y = Math.floor(index / maskWidth);
    const normalizedX = (x - gateCenterX) / gateRadiusX;
    const normalizedY = (y - gateCenterY) / gateRadiusY;
    return Math.pow(Math.abs(normalizedX), 8) + Math.pow(Math.abs(normalizedY), 8) <= 1;
  };
  const eligible = new Uint8Array(pixelCount);
  for (let index = 0; index < pixelCount; index += 1) {
    if (isInsideTrackedDancer(index) && confidence[index] >= threshold) eligible[index] = 1;
  }

  // Erode one pixel before selecting the connected component. This breaks the
  // narrow bridge that can appear when a dancer touches a background object or
  // another person. We restore one edge pixel
  // after selecting the tracked core so hands and feet are not over-trimmed.
  const core = new Uint8Array(pixelCount);
  for (let y = 1; y + 1 < maskHeight; y += 1) {
    for (let x = 1; x + 1 < maskWidth; x += 1) {
      const index = y * maskWidth + x;
      if (eligible[index]
        && eligible[index - 1]
        && eligible[index + 1]
        && eligible[index - maskWidth]
        && eligible[index + maskWidth]) {
        core[index] = 1;
      }
    }
  }

  const bodyCenterX = gateCenterX;
  const bodyCenterY = ((boxY + boxHeight * 0.42) / sourceHeight) * maskHeight;
  let coreSeed = -1;
  let bestCoreScore = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < pixelCount; index += 1) {
    if (!core[index]) continue;
    const x = index % maskWidth;
    const y = Math.floor(index / maskWidth);
    const distanceX = (x - bodyCenterX) / Math.max(2, gateRadiusX);
    const distanceY = (y - bodyCenterY) / Math.max(2, gateRadiusY);
    const score = confidence[index] - (distanceX * distanceX + distanceY * distanceY) * 0.18;
    if (score > bestCoreScore) {
      bestCoreScore = score;
      coreSeed = index;
    }
  }

  const walkable = coreSeed >= 0 ? core : eligible;
  const startIndex = coreSeed >= 0 ? coreSeed : seed.index;
  const selectedCore = new Uint8Array(pixelCount);
  const queue = new Int32Array(pixelCount);
  let head = 0;
  let tail = 0;
  selectedCore[startIndex] = 1;
  queue[tail++] = startIndex;

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
        if (selectedCore[next] || !walkable[next]) continue;
        selectedCore[next] = 1;
        queue[tail++] = next;
      }
    }
  }

  const selected = new Uint8Array(selectedCore);
  if (coreSeed >= 0) {
    for (let index = 0; index < pixelCount; index += 1) {
      if (!selectedCore[index]) continue;
      const currentX = index % maskWidth;
      const currentY = Math.floor(index / maskWidth);
      for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
        const nextY = currentY + offsetY;
        if (nextY < 0 || nextY >= maskHeight) continue;
        for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
          const nextX = currentX + offsetX;
          if (nextX < 0 || nextX >= maskWidth) continue;
          const next = nextY * maskWidth + nextX;
          if (eligible[next]) selected[next] = 1;
        }
      }
    }
  }

  const alpha = new Float32Array(pixelCount);
  for (let index = 0; index < pixelCount; index += 1) {
    if (selected[index] && isInsideTrackedDancer(index)) {
      alpha[index] = smoothstep(threshold, solidConfidence, confidence[index]);
    }
  }
  return tightenTrackedSubjectEdges(alpha, maskWidth, maskHeight);
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

export function stabilizeTrackedSubjectAlpha(
  currentAlpha: Float32Array,
  previousAlpha: Float32Array | null,
  pendingAlpha: Float32Array | null,
) {
  if (previousAlpha?.length !== currentAlpha.length) return new Float32Array(currentAlpha);
  const stabilized = new Float32Array(currentAlpha.length);
  const hasPending = pendingAlpha?.length === currentAlpha.length;
  for (let index = 0; index < currentAlpha.length; index += 1) {
    const current = currentAlpha[index];
    const previous = previousAlpha[index];
    const pending = hasPending ? pendingAlpha![index] : 0;
    if (previous < 0.06 && (current < 0.48 || pending < 0.32)) {
      stabilized[index] = 0;
      continue;
    }
    const blended = current >= previous
      ? previous * 0.34 + current * 0.66
      : previous * 0.55 + current * 0.45;
    stabilized[index] = blended < 0.025 ? 0 : blended;
  }
  return stabilized;
}

export class PersonBackgroundRenderer {
  private readonly segmenter: ImageSegmenter;
  private readonly inferenceCanvas = document.createElement('canvas');
  private readonly maskCanvas = document.createElement('canvas');
  private readonly subjectCanvas = document.createElement('canvas');
  private readonly backgroundCanvas = document.createElement('canvas');
  private readonly outlineCloneCanvas = document.createElement('canvas');
  private readonly rowSubjectCanvas = document.createElement('canvas');
  private previousAlpha: Float32Array | null = null;
  private pendingAlpha: Float32Array | null = null;
  private timestamp = 0;
  private missedFrames = 0;
  private trailCaptureTick = 0;
  private trailFrames: TrailFrame[] = [];
  private lastCloneLayout: CloneLayout = 'trail';

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

  private clearTrailFrames() {
    for (const frame of this.trailFrames) {
      frame.canvas.width = 1;
      frame.canvas.height = 1;
    }
    this.trailFrames = [];
    this.trailCaptureTick = 0;
  }

  private drawBackdrop(
    video: HTMLVideoElement,
    context: CanvasRenderingContext2D,
    canvas: HTMLCanvasElement,
    crop: Box,
    effects: PersonBackgroundEffects,
  ) {
    context.save();
    context.globalCompositeOperation = 'source-over';
    context.filter = 'none';
    context.globalAlpha = 1;
    context.fillStyle = effects.backgroundColor;
    context.fillRect(0, 0, canvas.width, canvas.height);
    if (effects.backgroundMode === 'blur') {
      const blur = clamp(effects.backgroundBlur, 4, 36);
      const maxSide = Math.round(320 - ((blur - 4) / 32) * 240);
      const scale = Math.min(1, maxSide / Math.max(canvas.width, canvas.height));
      const blurredWidth = Math.max(24, Math.round(canvas.width * scale));
      const blurredHeight = Math.max(24, Math.round(canvas.height * scale));
      if (this.backgroundCanvas.width !== blurredWidth || this.backgroundCanvas.height !== blurredHeight) {
        this.backgroundCanvas.width = blurredWidth;
        this.backgroundCanvas.height = blurredHeight;
      }
      const backgroundContext = this.backgroundCanvas.getContext('2d', { alpha: false });
      if (!backgroundContext) throw new Error('Safari 無法建立模糊背景畫布');
      backgroundContext.imageSmoothingEnabled = true;
      backgroundContext.drawImage(
        video,
        crop[0],
        crop[1],
        crop[2],
        crop[3],
        0,
        0,
        blurredWidth,
        blurredHeight,
      );
      context.imageSmoothingEnabled = true;
      context.drawImage(this.backgroundCanvas, 0, 0, blurredWidth, blurredHeight, 0, 0, canvas.width, canvas.height);
      context.fillStyle = 'rgba(0,0,0,.24)';
      context.fillRect(0, 0, canvas.width, canvas.height);
    }
    context.restore();
  }

  private drawStyledSubject(
    context: CanvasRenderingContext2D,
    source: CanvasImageSource,
    x: number,
    y: number,
    width: number,
    height: number,
    effects: PersonBackgroundEffects,
    opacity = 1,
  ) {
    context.save();
    context.globalCompositeOperation = 'source-over';
    context.globalAlpha = clamp(opacity, 0, 1);
    context.filter = 'none';
    context.imageSmoothingEnabled = true;
    const outlineWidth = clamp(effects.outlineWidth, 0, 48);
    if (outlineWidth > 0) {
      context.shadowColor = effects.outlineColor;
      if (outlineWidth > 16) {
        const radius = outlineWidth * 0.68;
        context.shadowBlur = outlineWidth * 0.46;
        for (const [directionX, directionY] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
          context.shadowOffsetX = directionX * radius;
          context.shadowOffsetY = directionY * radius;
          context.drawImage(source, x, y, width, height);
        }
      } else {
        context.shadowBlur = outlineWidth * 1.55;
        context.drawImage(source, x, y, width, height);
        context.shadowBlur = Math.max(1, outlineWidth * 0.7);
        context.drawImage(source, x, y, width, height);
      }
      context.shadowColor = 'transparent';
      context.shadowBlur = 0;
      context.shadowOffsetX = 0;
      context.shadowOffsetY = 0;
    }
    context.drawImage(source, x, y, width, height);
    context.restore();
  }

  private drawOutlineClone(
    context: CanvasRenderingContext2D,
    source: CanvasImageSource,
    x: number,
    y: number,
    width: number,
    height: number,
    effects: PersonBackgroundEffects,
    opacity: number,
  ) {
    const outlineWidth = Math.max(8, clamp(effects.outlineWidth, 0, 48));
    const padding = Math.ceil(outlineWidth * 2);
    const canvasWidth = Math.max(2, Math.ceil(width + padding * 2));
    const canvasHeight = Math.max(2, Math.ceil(height + padding * 2));
    if (this.outlineCloneCanvas.width !== canvasWidth || this.outlineCloneCanvas.height !== canvasHeight) {
      this.outlineCloneCanvas.width = canvasWidth;
      this.outlineCloneCanvas.height = canvasHeight;
    }
    const outlineContext = this.outlineCloneCanvas.getContext('2d', { alpha: true });
    if (!outlineContext) throw new Error('Safari 無法建立線框分身畫布');
    outlineContext.globalCompositeOperation = 'source-over';
    outlineContext.globalAlpha = 1;
    outlineContext.filter = 'none';
    outlineContext.clearRect(0, 0, canvasWidth, canvasHeight);
    outlineContext.imageSmoothingEnabled = true;
    outlineContext.shadowColor = effects.outlineColor;
    outlineContext.shadowBlur = outlineWidth * 0.48;
    const radius = outlineWidth * 0.72;
    const diagonal = radius * Math.SQRT1_2;
    const directions = [
      [-radius, 0],
      [radius, 0],
      [0, -radius],
      [0, radius],
      [-diagonal, -diagonal],
      [diagonal, -diagonal],
      [-diagonal, diagonal],
      [diagonal, diagonal],
    ];
    for (const [offsetX, offsetY] of directions) {
      outlineContext.shadowOffsetX = offsetX;
      outlineContext.shadowOffsetY = offsetY;
      outlineContext.drawImage(source, padding, padding, width, height);
    }
    outlineContext.shadowColor = 'transparent';
    outlineContext.shadowBlur = 0;
    outlineContext.shadowOffsetX = 0;
    outlineContext.shadowOffsetY = 0;
    outlineContext.globalCompositeOperation = 'destination-out';
    outlineContext.drawImage(source, padding, padding, width, height);
    outlineContext.globalCompositeOperation = 'source-over';

    context.save();
    context.globalCompositeOperation = 'source-over';
    context.globalAlpha = clamp(opacity, 0, 1);
    context.filter = 'none';
    context.drawImage(this.outlineCloneCanvas, x - padding, y - padding);
    context.restore();
  }

  private drawCloneSubject(
    context: CanvasRenderingContext2D,
    source: CanvasImageSource,
    x: number,
    y: number,
    width: number,
    height: number,
    effects: PersonBackgroundEffects,
    opacity: number,
  ) {
    if (effects.cloneStyle === 'outline') {
      this.drawOutlineClone(context, source, x, y, width, height, effects, opacity);
      return;
    }
    this.drawStyledSubject(context, source, x, y, width, height, effects, opacity);
  }

  private captureTrailFrame(x: number, y: number, width: number, height: number) {
    this.trailCaptureTick += 1;
    if (this.trailCaptureTick % TRAIL_CAPTURE_INTERVAL !== 0) return;
    const left = clamp(Math.floor(x), 0, this.subjectCanvas.width);
    const top = clamp(Math.floor(y), 0, this.subjectCanvas.height);
    const right = clamp(Math.ceil(x + width), left, this.subjectCanvas.width);
    const bottom = clamp(Math.ceil(y + height), top, this.subjectCanvas.height);
    const clippedWidth = right - left;
    const clippedHeight = bottom - top;
    if (clippedWidth < 2 || clippedHeight < 2) return;

    const scale = Math.min(1, MAX_TRAIL_SNAPSHOT_SIZE / Math.max(clippedWidth, clippedHeight));
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(2, Math.round(clippedWidth * scale));
    canvas.height = Math.max(2, Math.round(clippedHeight * scale));
    const context = canvas.getContext('2d', { alpha: true });
    if (!context) return;
    context.drawImage(
      this.subjectCanvas,
      left,
      top,
      clippedWidth,
      clippedHeight,
      0,
      0,
      canvas.width,
      canvas.height,
    );
    this.trailFrames.unshift({ canvas, x: left, y: top, width: clippedWidth, height: clippedHeight });
    while (this.trailFrames.length > 4) {
      const removed = this.trailFrames.pop();
      if (removed) {
        removed.canvas.width = 1;
        removed.canvas.height = 1;
      }
    }
  }

  private prepareRowSubject(x: number, y: number, width: number, height: number) {
    const left = clamp(Math.floor(x), 0, this.subjectCanvas.width);
    const top = clamp(Math.floor(y), 0, this.subjectCanvas.height);
    const right = clamp(Math.ceil(x + width), left, this.subjectCanvas.width);
    const bottom = clamp(Math.ceil(y + height), top, this.subjectCanvas.height);
    const clippedWidth = right - left;
    const clippedHeight = bottom - top;
    if (clippedWidth < 2 || clippedHeight < 2) return null;
    if (this.rowSubjectCanvas.width !== clippedWidth || this.rowSubjectCanvas.height !== clippedHeight) {
      this.rowSubjectCanvas.width = clippedWidth;
      this.rowSubjectCanvas.height = clippedHeight;
    }
    const context = this.rowSubjectCanvas.getContext('2d', { alpha: true });
    if (!context) throw new Error('Safari 無法建立並排分身畫布');
    context.globalCompositeOperation = 'source-over';
    context.clearRect(0, 0, clippedWidth, clippedHeight);
    context.drawImage(
      this.subjectCanvas,
      left,
      top,
      clippedWidth,
      clippedHeight,
      0,
      0,
      clippedWidth,
      clippedHeight,
    );
    return this.rowSubjectCanvas;
  }

  private compositeEffects(
    video: HTMLVideoElement,
    outputCanvas: HTMLCanvasElement,
    crop: Box,
    subjectBounds: Box,
    effects: PersonBackgroundEffects,
  ) {
    const outputContext = outputCanvas.getContext('2d', { alpha: false });
    if (!outputContext) throw new Error('Safari 無法建立人物特效合成畫布');
    this.drawBackdrop(video, outputContext, outputCanvas, crop, effects);

    const cloneCount = Math.round(clamp(effects.cloneCount, 0, 4));
    if (effects.cloneLayout !== this.lastCloneLayout) {
      this.clearTrailFrames();
      this.lastCloneLayout = effects.cloneLayout;
    }

    if (effects.cloneLayout === 'row' && cloneCount > 0) {
      this.clearTrailFrames();
      const total = cloneCount + 1;
      const mainIndex = Math.floor(total / 2);
      const rowSubject = this.prepareRowSubject(
        subjectBounds[0],
        subjectBounds[1],
        subjectBounds[2],
        subjectBounds[3],
      );
      if (!rowSubject) return;
      const frames = equalRowCloneFrames(
        outputCanvas.width,
        outputCanvas.height,
        rowSubject.width,
        rowSubject.height,
        cloneCount,
      );
      for (let index = 0; index < frames.length; index += 1) {
        const [destinationX, destinationY, destinationWidth, destinationHeight] = frames[index];
        if (effects.cloneStyle === 'outline' && index !== mainIndex) {
          this.drawOutlineClone(
            outputContext,
            rowSubject,
            destinationX,
            destinationY,
            destinationWidth,
            destinationHeight,
            effects,
            0.9,
          );
        } else {
          this.drawStyledSubject(
            outputContext,
            rowSubject,
            destinationX,
            destinationY,
            destinationWidth,
            destinationHeight,
            effects,
            1,
          );
        }
      }
      return;
    }

    if (effects.cloneLayout === 'trail' && cloneCount > 0) {
      const frames = this.trailFrames.slice(0, cloneCount).reverse();
      for (let index = 0; index < frames.length; index += 1) {
        const frame = frames[index];
        const depth = frames.length - index;
        const scale = subjectBounds[3] / Math.max(2, frame.height);
        const width = frame.width * scale;
        const height = frame.height * scale;
        const x = subjectBounds[0] + (subjectBounds[2] - width) / 2 - depth * outputCanvas.width * 0.018;
        const y = subjectBounds[1] + (subjectBounds[3] - height) / 2 + depth * outputCanvas.height * 0.007;
        const opacity = 0.22 + (index / Math.max(1, frames.length)) * 0.48;
        this.drawCloneSubject(outputContext, frame.canvas, x, y, width, height, effects, opacity);
      }
    } else if (cloneCount === 0) {
      this.clearTrailFrames();
    }

    this.drawStyledSubject(
      outputContext,
      this.subjectCanvas,
      0,
      0,
      outputCanvas.width,
      outputCanvas.height,
      effects,
    );
    if (effects.cloneLayout === 'trail' && cloneCount > 0) {
      this.captureTrailFrame(subjectBounds[0], subjectBounds[1], subjectBounds[2], subjectBounds[3]);
    }
  }

  render(
    video: HTMLVideoElement,
    outputCanvas: HTMLCanvasElement,
    box: Box,
    crop?: Box,
    effects: PersonBackgroundEffects = DEFAULT_PERSON_BACKGROUND_EFFECTS,
  ) {
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

    if (recovered.fresh) {
      const freshAlpha = currentAlpha;
      currentAlpha = stabilizeTrackedSubjectAlpha(freshAlpha, this.previousAlpha, this.pendingAlpha);
      this.pendingAlpha = new Float32Array(freshAlpha);
    } else {
      this.pendingAlpha = null;
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
      this.clearTrailFrames();
      this.subjectCanvas.width = outputCanvas.width;
      this.subjectCanvas.height = outputCanvas.height;
    }
    const subjectContext = this.subjectCanvas.getContext('2d', { alpha: true });
    if (!subjectContext) throw new Error('Safari 無法建立人物去背合成畫布');

    subjectContext.globalCompositeOperation = 'source-over';
    subjectContext.clearRect(0, 0, this.subjectCanvas.width, this.subjectCanvas.height);
    const [cropX, cropY, cropWidth, cropHeight] = crop ?? [0, 0, video.videoWidth, video.videoHeight];
    const outputScaleX = outputCanvas.width / cropWidth;
    const outputScaleY = outputCanvas.height / cropHeight;
    const destinationX = (regionX - cropX) * outputScaleX;
    const destinationY = (regionY - cropY) * outputScaleY;
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

    this.compositeEffects(
      video,
      outputCanvas,
      [cropX, cropY, cropWidth, cropHeight],
      [destinationX, destinationY, destinationWidth, destinationHeight],
      effects,
    );
  }

  close() {
    this.segmenter.close();
    this.previousAlpha = null;
    this.pendingAlpha = null;
    this.outlineCloneCanvas.width = 1;
    this.outlineCloneCanvas.height = 1;
    this.rowSubjectCanvas.width = 1;
    this.rowSubjectCanvas.height = 1;
    this.clearTrailFrames();
  }

  reset() {
    this.previousAlpha = null;
    this.pendingAlpha = null;
    this.missedFrames = 0;
    this.clearTrailFrames();
  }
}
