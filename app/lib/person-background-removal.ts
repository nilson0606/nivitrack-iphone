import { FilesetResolver, ImageSegmenter, PoseLandmarker } from '@mediapipe/tasks-vision';

import type { Box } from './vit-tracker';

const LOW_CONFIDENCE_LIMIT = 0.12;
const MAX_MISSED_FRAMES = 12;
const INFERENCE_SIZE = 256;

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function smoothstep(edge0: number, edge1: number, value: number) {
  const amount = clamp((value - edge0) / Math.max(0.0001, edge1 - edge0), 0, 1);
  return amount * amount * (3 - 2 * amount);
}

export function constrainSubjectConfidenceToPose(
  confidence: Float32Array,
  width: number,
  height: number,
  poseConfidence: Float32Array,
  poseWidth: number,
  poseHeight: number,
) {
  if (confidence.length !== width * height || poseConfidence.length !== poseWidth * poseHeight) {
    throw new Error('MediaPipe 人體姿態遮罩尺寸不正確');
  }

  // The pose mask is a body prior, not the final matte. Expand it slightly so
  // fast limbs, hair and loose clothing still use the finer selfie mask edge.
  const radius = Math.max(2, Math.round(Math.max(poseWidth, poseHeight) * 0.012));
  const horizontal = new Float32Array(poseConfidence.length);
  const expanded = new Float32Array(poseConfidence.length);
  for (let y = 0; y < poseHeight; y += 1) {
    const row = y * poseWidth;
    for (let x = 0; x < poseWidth; x += 1) {
      let maximum = 0;
      for (let offset = -radius; offset <= radius; offset += 1) {
        const sampleX = clamp(x + offset, 0, poseWidth - 1);
        maximum = Math.max(maximum, poseConfidence[row + sampleX]);
      }
      horizontal[row + x] = maximum;
    }
  }
  for (let y = 0; y < poseHeight; y += 1) {
    for (let x = 0; x < poseWidth; x += 1) {
      let maximum = 0;
      for (let offset = -radius; offset <= radius; offset += 1) {
        const sampleY = clamp(y + offset, 0, poseHeight - 1);
        maximum = Math.max(maximum, horizontal[sampleY * poseWidth + x]);
      }
      expanded[y * poseWidth + x] = maximum;
    }
  }

  const constrained = new Float32Array(confidence.length);
  for (let y = 0; y < height; y += 1) {
    const poseY = clamp(Math.floor(((y + 0.5) / height) * poseHeight), 0, poseHeight - 1);
    for (let x = 0; x < width; x += 1) {
      const poseX = clamp(Math.floor(((x + 0.5) / width) * poseWidth), 0, poseWidth - 1);
      const bodyPrior = smoothstep(0.08, 0.5, expanded[poseY * poseWidth + poseX]);
      // A very small floor keeps anti-aliased body edges recoverable, but pushes
      // confident non-body regions below the existing subject threshold.
      constrained[y * width + x] = confidence[y * width + x] * (0.035 + bodyPrior * 0.965);
    }
  }
  return constrained;
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
  private readonly poseLandmarker: PoseLandmarker;
  private readonly inferenceCanvas = document.createElement('canvas');
  private readonly maskCanvas = document.createElement('canvas');
  private readonly subjectCanvas = document.createElement('canvas');
  private previousAlpha: Float32Array | null = null;
  private pendingAlpha: Float32Array | null = null;
  private timestamp = 0;
  private missedFrames = 0;

  private constructor(segmenter: ImageSegmenter, poseLandmarker: PoseLandmarker) {
    this.segmenter = segmenter;
    this.poseLandmarker = poseLandmarker;
  }

  static async create() {
    const wasmRoot = new URL('mediapipe/', document.baseURI).href;
    const modelUrl = new URL('models/selfie_segmenter.tflite', document.baseURI).href;
    const poseModelUrl = new URL('models/pose_landmarker_lite.task', document.baseURI).href;
    const fileset = await FilesetResolver.forVisionTasks(wasmRoot);
    const segmenterCanvas = document.createElement('canvas');
    const segmenterOptions = {
      baseOptions: { modelAssetPath: modelUrl, delegate: 'GPU' as const },
      runningMode: 'VIDEO' as const,
      outputConfidenceMasks: true,
      outputCategoryMask: false,
      canvas: segmenterCanvas,
    };
    let segmenter: ImageSegmenter;
    try {
      segmenter = await ImageSegmenter.createFromOptions(fileset, segmenterOptions);
    } catch {
      segmenter = await ImageSegmenter.createFromOptions(fileset, {
        ...segmenterOptions,
        baseOptions: { modelAssetPath: modelUrl, delegate: 'CPU' },
      });
    }

    const poseCanvas = document.createElement('canvas');
    const poseOptions = {
      baseOptions: { modelAssetPath: poseModelUrl, delegate: 'GPU' as const },
      runningMode: 'VIDEO' as const,
      numPoses: 1,
      minPoseDetectionConfidence: 0.35,
      minPosePresenceConfidence: 0.35,
      minTrackingConfidence: 0.35,
      outputSegmentationMasks: true,
      canvas: poseCanvas,
    };
    let poseLandmarker: PoseLandmarker;
    try {
      poseLandmarker = await PoseLandmarker.createFromOptions(fileset, poseOptions);
    } catch {
      poseLandmarker = await PoseLandmarker.createFromOptions(fileset, {
        ...poseOptions,
        baseOptions: { modelAssetPath: poseModelUrl, delegate: 'CPU' },
      });
    }
    return new PersonBackgroundRenderer(segmenter, poseLandmarker);
  }

  render(video: HTMLVideoElement, outputCanvas: HTMLCanvasElement, box: Box, crop?: Box) {
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
    let personConfidence: Float32Array | null = null;
    let poseConfidence: Float32Array | null = null;
    let poseWidth = inferenceWidth;
    let poseHeight = inferenceHeight;
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
      personConfidence = new Float32Array(mask.getAsFloat32Array());
    });

    this.poseLandmarker.detectForVideo(this.inferenceCanvas, this.timestamp, (result) => {
      const mask = result.segmentationMasks?.[0];
      if (!mask) return;
      poseWidth = mask.width;
      poseHeight = mask.height;
      poseConfidence = new Float32Array(mask.getAsFloat32Array());
    });

    if (personConfidence) {
      const constrainedConfidence = poseConfidence
        ? constrainSubjectConfidenceToPose(
          personConfidence,
          maskWidth,
          maskHeight,
          poseConfidence,
          poseWidth,
          poseHeight,
        )
        : personConfidence;
      currentAlpha = selectTrackedSubjectAlpha(
        constrainedConfidence,
        maskWidth,
        maskHeight,
        relativeBox,
        regionWidth,
        regionHeight,
      );
    }

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
      this.subjectCanvas.width = outputCanvas.width;
      this.subjectCanvas.height = outputCanvas.height;
    }
    const subjectContext = this.subjectCanvas.getContext('2d', { alpha: true });
    const outputContext = outputCanvas.getContext('2d', { alpha: false });
    if (!subjectContext || !outputContext) throw new Error('Safari 無法建立人物去背合成畫布');

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

    outputContext.fillStyle = '#000';
    outputContext.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
    outputContext.drawImage(this.subjectCanvas, 0, 0);
  }

  close() {
    this.segmenter.close();
    this.poseLandmarker.close();
    this.previousAlpha = null;
    this.pendingAlpha = null;
  }

  reset() {
    this.previousAlpha = null;
    this.pendingAlpha = null;
    this.missedFrames = 0;
  }
}
