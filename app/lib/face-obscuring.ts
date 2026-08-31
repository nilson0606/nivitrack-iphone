import { FaceDetector, FilesetResolver } from '@mediapipe/tasks-vision';

import type { Box } from './vit-tracker';

export type FaceMaskStyle =
  | 'soft-blur'
  | 'strong-blur'
  | 'pixelate'
  | 'black-oval'
  | 'emoji'
  | 'sticker';

export type FaceMaskEffects = {
  style: FaceMaskStyle;
  strength: number;
  scale: number;
  emoji: string;
  stickerUrl?: string;
  privacyFirst: boolean;
};

export type HeadDetectionFrame = {
  time: number;
  heads: Box[];
};

export const DEFAULT_FACE_MASK_EFFECTS: FaceMaskEffects = {
  style: 'strong-blur',
  strength: 0.72,
  scale: 1.38,
  emoji: '😎',
  privacyFirst: true,
};

type FaceTrack = {
  id: number;
  box: Box;
  missedFrames: number;
};

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function boxCenter(box: Box): [number, number] {
  return [box[0] + box[2] / 2, box[1] + box[3] / 2];
}

function boxIou(left: Box, right: Box) {
  const overlapLeft = Math.max(left[0], right[0]);
  const overlapTop = Math.max(left[1], right[1]);
  const overlapRight = Math.min(left[0] + left[2], right[0] + right[2]);
  const overlapBottom = Math.min(left[1] + left[3], right[1] + right[3]);
  const overlap = Math.max(0, overlapRight - overlapLeft) * Math.max(0, overlapBottom - overlapTop);
  const union = left[2] * left[3] + right[2] * right[3] - overlap;
  return union > 0 ? overlap / union : 0;
}

function faceTrackDistance(left: Box, right: Box) {
  const [leftX, leftY] = boxCenter(left);
  const [rightX, rightY] = boxCenter(right);
  const scale = Math.max(4, Math.max(left[2], left[3], right[2], right[3]));
  return Math.hypot(rightX - leftX, rightY - leftY) / scale;
}

function unionBox(left: Box, right: Box): Box {
  const x = Math.min(left[0], right[0]);
  const y = Math.min(left[1], right[1]);
  const rightEdge = Math.max(left[0] + left[2], right[0] + right[2]);
  const bottomEdge = Math.max(left[1] + left[3], right[1] + right[3]);
  return [x, y, rightEdge - x, bottomEdge - y];
}

export function personBoxToHeadBox(
  person: Box,
  sourceWidth: number,
  sourceHeight: number,
): Box {
  const [x, y, width, height] = person;
  const headWidth = clamp(width * 0.5, height * 0.14, height * 0.24);
  const headHeight = clamp(Math.max(headWidth * 1.08, height * 0.18), headWidth, height * 0.28);
  const centerX = x + width / 2;
  return [
    clamp(centerX - headWidth / 2, 0, Math.max(0, sourceWidth - headWidth)),
    clamp(y + height * 0.01, 0, Math.max(0, sourceHeight - headHeight)),
    Math.min(headWidth, sourceWidth),
    Math.min(headHeight, sourceHeight),
  ];
}

export function selectMainPersonIndex(people: Box[], subject: Box) {
  if (people.length === 0) return null;
  const [subjectX, subjectY] = boxCenter(subject);
  const subjectScale = Math.max(8, Math.hypot(subject[2], subject[3]));
  let bestIndex: number | null = null;
  let bestScore = Number.NEGATIVE_INFINITY;
  people.forEach((person, index) => {
    const [personX, personY] = boxCenter(person);
    const distance = Math.hypot(personX - subjectX, personY - subjectY) / subjectScale;
    const overlap = boxIou(person, subject);
    const containsSubjectCenter = subjectX >= person[0]
      && subjectX <= person[0] + person[2]
      && subjectY >= person[1]
      && subjectY <= person[1] + person[3];
    const score = overlap * 2.4 - distance + (containsSubjectCenter ? 0.55 : 0);
    if (score > bestScore) {
      bestScore = score;
      bestIndex = index;
    }
  });
  return bestScore >= -0.15 ? bestIndex : null;
}

export function bystanderHeadBoxes(
  people: Box[],
  subject: Box,
  sourceWidth: number,
  sourceHeight: number,
) {
  const mainIndex = selectMainPersonIndex(people, subject);
  return people
    .filter((_, index) => index !== mainIndex)
    .map((person) => personBoxToHeadBox(person, sourceWidth, sourceHeight));
}

export function headBoxesAt(frames: HeadDetectionFrame[], time: number) {
  if (frames.length === 0) return [];
  let low = 0;
  let high = frames.length - 1;
  while (low + 1 < high) {
    const middle = Math.floor((low + high) / 2);
    if (frames[middle].time <= time) low = middle;
    else high = middle;
  }
  const candidate = Math.abs(frames[high].time - time) < Math.abs(frames[low].time - time)
    ? frames[high]
    : frames[low];
  return candidate.heads.map((head) => [...head] as Box);
}

export function mergeHeadBoxes(boxes: Box[]) {
  const merged: Box[] = [];
  for (const box of boxes) {
    const duplicateIndex = merged.findIndex((existing) =>
      boxIou(existing, box) >= 0.08 || faceTrackDistance(existing, box) <= 0.42,
    );
    if (duplicateIndex >= 0) merged[duplicateIndex] = unionBox(merged[duplicateIndex], box);
    else merged.push([...box] as Box);
  }
  return merged;
}

export function expandFaceBox(
  box: Box,
  scale: number,
  sourceWidth: number,
  sourceHeight: number,
): Box {
  const safeScale = clamp(scale, 1, 1.8);
  const centerX = box[0] + box[2] / 2;
  const centerY = box[1] + box[3] * 0.5;
  const width = Math.min(sourceWidth, Math.max(2, box[2] * safeScale));
  const height = Math.min(sourceHeight, Math.max(2, box[3] * safeScale * 1.06));
  const x = clamp(centerX - width / 2, 0, Math.max(0, sourceWidth - width));
  const y = clamp(centerY - height / 2, 0, Math.max(0, sourceHeight - height));
  return [x, y, width, height];
}

function mainFaceScore(face: Box, subject: Box, previousMainFace: Box | null) {
  const [faceX, faceY] = boxCenter(face);
  const headLeft = subject[0] + subject[2] * 0.04;
  const headRight = subject[0] + subject[2] * 0.96;
  const headTop = subject[1] - subject[3] * 0.08;
  const headBottom = subject[1] + subject[3] * 0.43;
  if (faceX < headLeft || faceX > headRight || faceY < headTop || faceY > headBottom) {
    return Number.POSITIVE_INFINITY;
  }
  const expectedX = subject[0] + subject[2] * 0.5;
  const expectedY = subject[1] + subject[3] * 0.17;
  const subjectDistance = Math.hypot(
    (faceX - expectedX) / Math.max(8, subject[2] * 0.55),
    (faceY - expectedY) / Math.max(8, subject[3] * 0.3),
  );
  const previousDistance = previousMainFace
    ? faceTrackDistance(face, previousMainFace)
    : 0;
  const previousOverlap = previousMainFace ? boxIou(face, previousMainFace) : 0;
  return subjectDistance
    + previousDistance * (previousMainFace ? 0.72 : 0)
    - previousOverlap * 0.35;
}

export function selectMainFaceIndex(
  faces: Box[],
  subject: Box,
  previousMainFace: Box | null,
  privacyFirst = true,
) {
  const ranked = faces
    .map((face, index) => ({ index, score: mainFaceScore(face, subject, previousMainFace) }))
    .filter((item) => Number.isFinite(item.score))
    .sort((left, right) => left.score - right.score);
  if (ranked.length === 0 || ranked[0].score > (previousMainFace ? 1.7 : 1.15)) return null;
  if (
    privacyFirst
    && !previousMainFace
    && ranked.length > 1
    && ranked[1].score - ranked[0].score < 0.2
  ) {
    return null;
  }
  return ranked[0].index;
}

export function smoothFaceBox(previous: Box, current: Box, missedFrames = 0): Box {
  const currentWeight = missedFrames > 0 ? 0 : 0.68;
  return previous.map(
    (value, index) => value * (1 - currentWeight) + current[index] * currentWeight,
  ) as Box;
}

function loadStickerImage(url?: string) {
  if (!url) return Promise.resolve<HTMLImageElement | null>(null);
  return new Promise<HTMLImageElement | null>((resolve) => {
    const image = new Image();
    image.decoding = 'async';
    image.addEventListener('load', () => resolve(image), { once: true });
    image.addEventListener('error', () => resolve(null), { once: true });
    image.src = url;
  });
}

export class FaceObscuringRenderer {
  private readonly detector: FaceDetector;
  private readonly pixelCanvas = document.createElement('canvas');
  private readonly stickerImage: HTMLImageElement | null;
  private faceTracks: FaceTrack[] = [];
  private maskTracks: FaceTrack[] = [];
  private previousMainFace: Box | null = null;
  private nextTrackId = 1;
  private timestamp = 0;

  private constructor(detector: FaceDetector, stickerImage: HTMLImageElement | null) {
    this.detector = detector;
    this.stickerImage = stickerImage;
  }

  static async create(stickerUrl?: string) {
    const wasmRoot = new URL('mediapipe/', document.baseURI).href;
    const modelUrl = new URL('models/blaze_face_short_range.tflite', document.baseURI).href;
    const fileset = await FilesetResolver.forVisionTasks(wasmRoot);
    const common = {
      baseOptions: { modelAssetPath: modelUrl, delegate: 'CPU' as const },
      runningMode: 'VIDEO' as const,
      minDetectionConfidence: 0.32,
      minSuppressionThreshold: 0.3,
      canvas: document.createElement('canvas'),
    };
    let detector: FaceDetector;
    try {
      detector = await FaceDetector.createFromOptions(fileset, common);
    } catch {
      detector = await FaceDetector.createFromOptions(fileset, {
        ...common,
        baseOptions: { modelAssetPath: modelUrl, delegate: 'GPU' },
      });
    }
    return new FaceObscuringRenderer(detector, await loadStickerImage(stickerUrl));
  }

  private updateTracks(existingTracks: FaceTrack[], detections: Box[], maximumMissedFrames: number) {
    const unused = new Set(detections.map((_, index) => index));
    const nextTracks: FaceTrack[] = [];
    for (const track of existingTracks) {
      let bestIndex = -1;
      let bestCost = Number.POSITIVE_INFINITY;
      for (const index of unused) {
        const candidate = detections[index];
        const distance = faceTrackDistance(track.box, candidate);
        const overlap = boxIou(track.box, candidate);
        const cost = distance - overlap * 0.7;
        if ((distance <= 1.15 || overlap >= 0.04) && cost < bestCost) {
          bestCost = cost;
          bestIndex = index;
        }
      }
      if (bestIndex >= 0) {
        unused.delete(bestIndex);
        nextTracks.push({
          id: track.id,
          box: smoothFaceBox(track.box, detections[bestIndex]),
          missedFrames: 0,
        });
      } else if (track.missedFrames < maximumMissedFrames) {
        nextTracks.push({ ...track, missedFrames: track.missedFrames + 1 });
      }
    }
    for (const index of unused) {
      nextTracks.push({ id: this.nextTrackId++, box: [...detections[index]] as Box, missedFrames: 0 });
    }
    return nextTracks;
  }

  private drawBlur(
    video: HTMLVideoElement,
    context: CanvasRenderingContext2D,
    sourceBox: Box,
    destinationBox: Box,
    radius: number,
    opacity: number,
  ) {
    const [sourceX, sourceY, sourceWidth, sourceHeight] = sourceBox;
    const [destinationX, destinationY, destinationWidth, destinationHeight] = destinationBox;
    const sourcePaddingX = sourceWidth * 0.36;
    const sourcePaddingY = sourceHeight * 0.36;
    const paddedSourceX = clamp(sourceX - sourcePaddingX, 0, video.videoWidth);
    const paddedSourceY = clamp(sourceY - sourcePaddingY, 0, video.videoHeight);
    const paddedSourceRight = clamp(sourceX + sourceWidth + sourcePaddingX, 0, video.videoWidth);
    const paddedSourceBottom = clamp(sourceY + sourceHeight + sourcePaddingY, 0, video.videoHeight);
    const scaleX = destinationWidth / Math.max(2, sourceWidth);
    const scaleY = destinationHeight / Math.max(2, sourceHeight);
    context.save();
    context.globalAlpha = opacity;
    context.beginPath();
    context.ellipse(
      destinationX + destinationWidth / 2,
      destinationY + destinationHeight / 2,
      destinationWidth / 2,
      destinationHeight / 2,
      0,
      0,
      Math.PI * 2,
    );
    context.clip();
    context.filter = `blur(${radius}px)`;
    context.drawImage(
      video,
      paddedSourceX,
      paddedSourceY,
      paddedSourceRight - paddedSourceX,
      paddedSourceBottom - paddedSourceY,
      destinationX - (sourceX - paddedSourceX) * scaleX,
      destinationY - (sourceY - paddedSourceY) * scaleY,
      (paddedSourceRight - paddedSourceX) * scaleX,
      (paddedSourceBottom - paddedSourceY) * scaleY,
    );
    context.restore();
  }

  private drawPixelated(
    video: HTMLVideoElement,
    context: CanvasRenderingContext2D,
    sourceBox: Box,
    destinationBox: Box,
    strength: number,
    opacity: number,
  ) {
    const blocks = Math.round(7 + clamp(strength, 0, 1) * 17);
    const tinyWidth = Math.max(3, Math.round(destinationBox[2] / blocks));
    const tinyHeight = Math.max(3, Math.round(destinationBox[3] / blocks));
    if (this.pixelCanvas.width !== tinyWidth || this.pixelCanvas.height !== tinyHeight) {
      this.pixelCanvas.width = tinyWidth;
      this.pixelCanvas.height = tinyHeight;
    }
    const pixelContext = this.pixelCanvas.getContext('2d', { alpha: false });
    if (!pixelContext) throw new Error('Safari 無法建立人臉馬賽克畫布');
    pixelContext.imageSmoothingEnabled = true;
    pixelContext.drawImage(video, ...sourceBox, 0, 0, tinyWidth, tinyHeight);
    context.save();
    context.globalAlpha = opacity;
    context.beginPath();
    context.ellipse(
      destinationBox[0] + destinationBox[2] / 2,
      destinationBox[1] + destinationBox[3] / 2,
      destinationBox[2] / 2,
      destinationBox[3] / 2,
      0,
      0,
      Math.PI * 2,
    );
    context.clip();
    context.imageSmoothingEnabled = false;
    context.drawImage(this.pixelCanvas, ...destinationBox);
    context.restore();
  }

  private drawCover(
    context: CanvasRenderingContext2D,
    box: Box,
    effects: FaceMaskEffects,
    opacity: number,
  ) {
    const [x, y, width, height] = box;
    context.save();
    context.globalAlpha = opacity;
    if (effects.style === 'black-oval') {
      context.fillStyle = '#050806';
      context.beginPath();
      context.ellipse(x + width / 2, y + height / 2, width / 2, height / 2, 0, 0, Math.PI * 2);
      context.fill();
    } else if (effects.style === 'sticker' && this.stickerImage) {
      context.drawImage(this.stickerImage, x, y, width, height);
    } else {
      context.font = `${Math.round(height * 0.82)}px "Apple Color Emoji", "Segoe UI Emoji", sans-serif`;
      context.textAlign = 'center';
      context.textBaseline = 'middle';
      context.fillText(effects.style === 'sticker' ? '⭐' : effects.emoji, x + width / 2, y + height / 2);
    }
    context.restore();
  }

  render(
    video: HTMLVideoElement,
    outputCanvas: HTMLCanvasElement,
    subjectBox: Box,
    effects: FaceMaskEffects = DEFAULT_FACE_MASK_EFFECTS,
    detectedBystanderHeads: Box[] = [],
  ) {
    if (!video.videoWidth || !video.videoHeight) return;
    const context = outputCanvas.getContext('2d', { alpha: false });
    if (!context) throw new Error('Safari 無法建立旁人人臉遮罩畫布');
    context.globalCompositeOperation = 'source-over';
    context.globalAlpha = 1;
    context.filter = 'none';
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = 'high';
    context.drawImage(video, 0, 0, outputCanvas.width, outputCanvas.height);

    this.timestamp += 1;
    const result = this.detector.detectForVideo(video, this.timestamp);
    const detections: Box[] = result.detections.flatMap((detection) => {
      const bounds = detection.boundingBox;
      if (!bounds || bounds.width < 4 || bounds.height < 4) return [];
      return [[bounds.originX, bounds.originY, bounds.width, bounds.height] as Box];
    });
    this.faceTracks = this.updateTracks(this.faceTracks, detections, 8);
    const visibleBoxes = this.faceTracks.map((track) => track.box);
    const mainIndex = selectMainFaceIndex(
      visibleBoxes,
      subjectBox,
      this.previousMainFace,
      effects.privacyFirst,
    );
    if (mainIndex !== null) this.previousMainFace = [...visibleBoxes[mainIndex]] as Box;

    const faceMasks = this.faceTracks
      .filter((_, index) => index !== mainIndex)
      .map((track) => track.box);
    const maskCandidates = mergeHeadBoxes([
      ...detectedBystanderHeads,
      ...faceMasks,
    ]);
    this.maskTracks = this.updateTracks(this.maskTracks, maskCandidates, 10);

    const outputScaleX = outputCanvas.width / video.videoWidth;
    const outputScaleY = outputCanvas.height / video.videoHeight;
    for (const track of this.maskTracks) {
      const sourceBox = expandFaceBox(
        track.box,
        effects.scale,
        video.videoWidth,
        video.videoHeight,
      );
      const destinationBox: Box = [
        sourceBox[0] * outputScaleX,
        sourceBox[1] * outputScaleY,
        sourceBox[2] * outputScaleX,
        sourceBox[3] * outputScaleY,
      ];
      const opacity = clamp(1 - track.missedFrames / 12, 0.42, 1);
      if (effects.style === 'soft-blur' || effects.style === 'strong-blur') {
        const base = effects.style === 'strong-blur' ? 22 : 9;
        const range = effects.style === 'strong-blur' ? 42 : 22;
        this.drawBlur(video, context, sourceBox, destinationBox, base + range * effects.strength, opacity);
      } else if (effects.style === 'pixelate') {
        this.drawPixelated(video, context, sourceBox, destinationBox, effects.strength, opacity);
      } else {
        this.drawCover(context, destinationBox, effects, opacity);
      }
    }
  }

  reset() {
    this.faceTracks = [];
    this.maskTracks = [];
    this.previousMainFace = null;
  }

  close() {
    this.detector.close();
    this.pixelCanvas.width = 1;
    this.pixelCanvas.height = 1;
    this.faceTracks = [];
    this.maskTracks = [];
    this.previousMainFace = null;
  }
}
