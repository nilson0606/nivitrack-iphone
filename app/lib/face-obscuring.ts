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

type HeadFrameTrack = {
  id: number;
  box: Box;
  velocity: Box;
  lastFrameTime: number;
  lastSeenTime: number;
  missedFrames: number;
  age: number;
};

type TrackOptions = {
  maximumMissedFrames: number;
  maximumDistance: number;
  minimumOverlap: number;
  maximumAreaRatio: number;
  replacementDistance: number;
};

const FACE_TRACK_MISSED_FRAMES = 4;
const MASK_TRACK_MISSED_FRAMES = 5;

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function boxCenter(box: Box): [number, number] {
  return [box[0] + box[2] / 2, box[1] + box[3] / 2];
}

function boxArea(box: Box) {
  return Math.max(0, box[2]) * Math.max(0, box[3]);
}

function boxAreaRatio(left: Box, right: Box) {
  const smaller = Math.max(1, Math.min(boxArea(left), boxArea(right)));
  return Math.max(boxArea(left), boxArea(right)) / smaller;
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

export function personBoxToHeadBox(
  person: Box,
  sourceWidth: number,
  sourceHeight: number,
): Box {
  const [x, y, width, height] = person;
  const bodyAspect = width / Math.max(1, height);
  const armExpansion = clamp((bodyAspect - 0.34) / 0.5, 0, 1);
  const headWidth = clamp(
    width * (0.44 - armExpansion * 0.12),
    height * 0.115,
    height * 0.19,
  );
  const headHeight = clamp(Math.max(headWidth * 1.08, height * 0.15), headWidth, height * 0.21);
  const centerX = x + width / 2;
  // COCO person boxes include raised hands. A wider-than-standing box therefore
  // needs a lower head anchor; otherwise the highest hand becomes a floating face.
  const headInset = height * (0.035 + armExpansion * 0.22);
  return [
    clamp(centerX - headWidth / 2, 0, Math.max(0, sourceWidth - headWidth)),
    clamp(y + headInset, 0, Math.max(0, sourceHeight - headHeight)),
    Math.min(headWidth, sourceWidth),
    Math.min(headHeight, sourceHeight),
  ];
}

export function subjectHeadProtectionBox(
  subject: Box,
  sourceWidth: number,
  sourceHeight: number,
): Box {
  const inferredHead = personBoxToHeadBox(subject, sourceWidth, sourceHeight);
  const protectionWidth = Math.min(sourceWidth, Math.max(inferredHead[2] * 1.5, subject[2] * 0.62));
  const protectionHeight = Math.min(sourceHeight, Math.max(inferredHead[3] * 1.72, subject[3] * 0.25));
  const centerX = inferredHead[0] + inferredHead[2] / 2;
  return [
    clamp(centerX - protectionWidth / 2, 0, Math.max(0, sourceWidth - protectionWidth)),
    clamp(inferredHead[1] - inferredHead[3] * 0.22, 0, Math.max(0, sourceHeight - protectionHeight)),
    protectionWidth,
    protectionHeight,
  ];
}

export function isProtectedMainHead(
  head: Box,
  subject: Box,
  sourceWidth: number,
  sourceHeight: number,
) {
  const protection = subjectHeadProtectionBox(subject, sourceWidth, sourceHeight);
  const [centerX, centerY] = boxCenter(head);
  const centerInside = centerX >= protection[0]
    && centerX <= protection[0] + protection[2]
    && centerY >= protection[1]
    && centerY <= protection[1] + protection[3];
  return centerInside || boxIou(head, protection) >= 0.12;
}

function isPlausibleHeadBox(
  head: Box,
  sourceWidth: number,
  sourceHeight: number,
  referenceHead?: Box,
  maximumReferenceScale = 2.35,
) {
  const [x, y, width, height] = head;
  if (![x, y, width, height].every(Number.isFinite) || width < 4 || height < 4) return false;
  if (x < -1 || y < -1 || x + width > sourceWidth + 1 || y + height > sourceHeight + 1) return false;
  const aspect = width / Math.max(1, height);
  if (aspect < 0.42 || aspect > 1.75) return false;
  if (width > sourceWidth * 0.3 || height > sourceHeight * 0.42) return false;
  if (!referenceHead) return true;
  return width <= Math.max(18, referenceHead[2] * maximumReferenceScale)
    && height <= Math.max(20, referenceHead[3] * maximumReferenceScale);
}

export function selectMainPersonIndex(people: Box[], subject: Box) {
  if (people.length === 0) return null;
  const [subjectX, subjectY] = boxCenter(subject);
  const subjectScale = Math.max(8, Math.hypot(subject[2], subject[3]));
  let bestIndex: number | null = null;
  let bestScore = Number.NEGATIVE_INFINITY;
  let bestIsMainLike = false;
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
      bestIsMainLike = containsSubjectCenter || overlap >= 0.12 || distance <= 0.38;
    }
  });
  return bestIsMainLike && bestScore >= -0.15 ? bestIndex : null;
}

export function bystanderHeadBoxes(
  people: Box[],
  subject: Box,
  sourceWidth: number,
  sourceHeight: number,
) {
  const mainIndex = selectMainPersonIndex(people, subject);
  const referenceHead = personBoxToHeadBox(subject, sourceWidth, sourceHeight);
  return people
    .filter((_, index) => index !== mainIndex)
    .filter((person) => person[2] < sourceWidth * 0.72 && person[3] < sourceHeight * 0.96)
    .map((person) => personBoxToHeadBox(person, sourceWidth, sourceHeight))
    .filter((head) => isPlausibleHeadBox(head, sourceWidth, sourceHeight, referenceHead, 1.9))
    .filter((head) => !isProtectedMainHead(head, subject, sourceWidth, sourceHeight));
}

function constrainHeadFrameBox(previous: Box, detection: Box, elapsed: number): Box {
  const timeScale = clamp(elapsed / 0.2, 0.5, 2.25);
  const [previousX, previousY] = boxCenter(previous);
  const [detectionX, detectionY] = boxCenter(detection);
  const referenceWidth = Math.max(4, previous[2], detection[2]);
  const referenceHeight = Math.max(4, previous[3], detection[3]);
  const maximumHorizontalMove = referenceWidth * 1.15 * timeScale;
  const maximumUpwardMove = referenceHeight * 0.42 * timeScale;
  const maximumDownwardMove = referenceHeight * 1.05 * timeScale;
  const centerX = previousX + clamp(
    detectionX - previousX,
    -maximumHorizontalMove,
    maximumHorizontalMove,
  );
  const centerY = previousY + clamp(
    detectionY - previousY,
    -maximumUpwardMove,
    maximumDownwardMove,
  );
  const maximumScale = Math.pow(1.32, timeScale);
  const constrainedWidth = clamp(detection[2], previous[2] / maximumScale, previous[2] * maximumScale);
  const constrainedHeight = clamp(detection[3], previous[3] / maximumScale, previous[3] * maximumScale);
  const currentWeight = 0.64;
  const width = previous[2] * (1 - currentWeight) + constrainedWidth * currentWeight;
  const height = previous[3] * (1 - currentWeight) + constrainedHeight * currentWeight;
  const smoothedX = previousX * (1 - currentWeight) + centerX * currentWeight;
  const smoothedY = previousY * (1 - currentWeight) + centerY * currentWeight;
  return [
    Math.max(0, smoothedX - width / 2),
    Math.max(0, smoothedY - height / 2),
    width,
    height,
  ];
}

function predictHeadFrameBox(track: HeadFrameTrack, time: number): Box {
  const elapsed = clamp(time - track.lastFrameTime, 0, 0.3);
  const damping = Math.pow(0.42, track.missedFrames + 1);
  return track.box.map((value, index) =>
    index < 2 ? Math.max(0, value + track.velocity[index] * elapsed * damping) : value,
  ) as Box;
}

/**
 * Turns the 5 fps person-derived head samples into short, continuous tracks.
 * This is intentionally separate from the live face detector: inferred heads
 * only bridge brief missed faces and cannot grow by unioning nearby dancers.
 */
export function stabilizeHeadDetectionFrames(frames: HeadDetectionFrame[]) {
  if (frames.length === 0) return [];
  const ordered: HeadDetectionFrame[] = [];
  for (const frame of [...frames].sort((left, right) => left.time - right.time)) {
    const previous = ordered.at(-1);
    if (previous && Math.abs(previous.time - frame.time) < 0.001) {
      previous.heads = mergeHeadBoxes([...previous.heads, ...frame.heads]);
    } else {
      ordered.push({
        time: frame.time,
        heads: mergeHeadBoxes(frame.heads).map((head) => [...head] as Box),
      });
    }
  }

  let nextTrackId = 1;
  let tracks: HeadFrameTrack[] = [];
  return ordered.map((frame) => {
    const predicted = tracks.map((track) => predictHeadFrameBox(track, frame.time));
    const pairs: Array<{ trackIndex: number; detectionIndex: number; cost: number }> = [];
    tracks.forEach((track, trackIndex) => {
      frame.heads.forEach((detection, detectionIndex) => {
        const distance = faceTrackDistance(predicted[trackIndex], detection);
        const overlap = boxIou(predicted[trackIndex], detection);
        const areaRatio = boxAreaRatio(predicted[trackIndex], detection);
        if (areaRatio <= 3.2 && (distance <= 1.55 || overlap >= 0.025)) {
          pairs.push({
            trackIndex,
            detectionIndex,
            cost: distance + Math.abs(Math.log(areaRatio)) * 0.32 - overlap * 0.72,
          });
        }
      });
    });
    pairs.sort((left, right) => left.cost - right.cost);

    const usedTracks = new Set<number>();
    const usedDetections = new Set<number>();
    const nextTracks: HeadFrameTrack[] = [];
    for (const pair of pairs) {
      if (usedTracks.has(pair.trackIndex) || usedDetections.has(pair.detectionIndex)) continue;
      const track = tracks[pair.trackIndex];
      const elapsed = clamp(frame.time - track.lastFrameTime, 0.04, 0.6);
      const box = constrainHeadFrameBox(track.box, frame.heads[pair.detectionIndex], elapsed);
      const measuredVelocity = box.map((value, index) =>
        index < 2 ? (value - track.box[index]) / elapsed : 0,
      ) as Box;
      nextTracks.push({
        ...track,
        box,
        velocity: track.velocity.map((value, index) =>
          index < 2 ? value * 0.35 + measuredVelocity[index] * 0.65 : 0,
        ) as Box,
        lastFrameTime: frame.time,
        lastSeenTime: frame.time,
        missedFrames: 0,
        age: track.age + 1,
      });
      usedTracks.add(pair.trackIndex);
      usedDetections.add(pair.detectionIndex);
    }

    tracks.forEach((track, trackIndex) => {
      if (usedTracks.has(trackIndex)) return;
      const missingFor = frame.time - track.lastSeenTime;
      if (track.age < 2 || missingFor > 0.46) return;
      nextTracks.push({
        ...track,
        box: predicted[trackIndex],
        velocity: track.velocity.map((value, index) => index < 2 ? value * 0.58 : 0) as Box,
        lastFrameTime: frame.time,
        missedFrames: track.missedFrames + 1,
      });
    });

    frame.heads.forEach((detection, detectionIndex) => {
      if (usedDetections.has(detectionIndex)) return;
      const duplicate = nextTracks.some((track) =>
        boxAreaRatio(track.box, detection) <= 3.5
        && (boxIou(track.box, detection) >= 0.08 || faceTrackDistance(track.box, detection) <= 0.72),
      );
      if (duplicate) return;
      nextTracks.push({
        id: nextTrackId++,
        box: [...detection] as Box,
        velocity: [0, 0, 0, 0],
        lastFrameTime: frame.time,
        lastSeenTime: frame.time,
        missedFrames: 0,
        age: 1,
      });
    });

    tracks = nextTracks;
    return {
      time: frame.time,
      heads: mergeHeadBoxes(tracks.map((track) => track.box)),
    };
  });
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
      boxAreaRatio(existing, box) <= 3
      && (boxIou(existing, box) >= 0.12 || faceTrackDistance(existing, box) <= 0.42),
    );
    // The first box is the higher-priority source. Never union nearby heads:
    // a union can chain across a dance group and grow into one giant mask.
    if (duplicateIndex < 0) merged.push([...box] as Box);
  }
  return merged;
}

export function suppressFaceSupportedFallbacks(faceBoxes: Box[], inferredHeads: Box[]) {
  return inferredHeads.filter((inferred) => {
    const [inferredX, inferredY] = boxCenter(inferred);
    return !faceBoxes.some((face) => {
      if (boxAreaRatio(face, inferred) > 4.5) return false;
      const [faceX, faceY] = boxCenter(face);
      const scale = Math.max(4, face[2], face[3], inferred[2], inferred[3]);
      const horizontalDistance = Math.abs(inferredX - faceX) / scale;
      const verticalOffset = (inferredY - faceY) / scale;
      // A person-box fallback just above an already detected face is normally
      // the same dancer's raised hand. Keep the real face and discard the ghost.
      return horizontalDistance <= 0.78
        && verticalOffset >= -2.45
        && verticalOffset <= 0.62;
    });
  });
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
    const modelUrl = new URL('models/blaze_face_full_range.tflite', document.baseURI).href;
    const fileset = await FilesetResolver.forVisionTasks(wasmRoot);
    const common = {
      baseOptions: { modelAssetPath: modelUrl, delegate: 'CPU' as const },
      runningMode: 'VIDEO' as const,
      minDetectionConfidence: 0.24,
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

  private updateTracks(existingTracks: FaceTrack[], detections: Box[], options: TrackOptions) {
    const unused = new Set(detections.map((_, index) => index));
    const nextTracks: FaceTrack[] = [];
    for (const track of existingTracks) {
      let bestIndex = -1;
      let bestCost = Number.POSITIVE_INFINITY;
      for (const index of unused) {
        const candidate = detections[index];
        const distance = faceTrackDistance(track.box, candidate);
        const overlap = boxIou(track.box, candidate);
        const areaRatio = boxAreaRatio(track.box, candidate);
        const cost = distance - overlap * 0.7;
        if (
          areaRatio <= options.maximumAreaRatio
          && (distance <= options.maximumDistance || overlap >= options.minimumOverlap)
          && cost < bestCost
        ) {
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
      } else if (track.missedFrames < options.maximumMissedFrames) {
        const nearbyReplacement = [...unused].some((index) => {
          const candidate = detections[index];
          return boxAreaRatio(track.box, candidate) <= options.maximumAreaRatio * 1.25
            && faceTrackDistance(track.box, candidate) <= options.replacementDistance;
        });
        // A farther jump starts a fresh track instead of keeping both old and
        // new masks on screen, which created the visible double-star trail.
        if (!nearbyReplacement) {
          nextTracks.push({ ...track, missedFrames: track.missedFrames + 1 });
        }
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
    const referenceHead = personBoxToHeadBox(
      subjectBox,
      video.videoWidth,
      video.videoHeight,
    );
    const plausibleFaces = detections.filter((face) =>
      isPlausibleHeadBox(face, video.videoWidth, video.videoHeight, referenceHead, 2.6),
    );
    this.faceTracks = this.updateTracks(this.faceTracks, plausibleFaces, {
      maximumMissedFrames: FACE_TRACK_MISSED_FRAMES,
      maximumDistance: 0.78,
      minimumOverlap: 0.06,
      maximumAreaRatio: 3,
      replacementDistance: 1.35,
    });
    const visibleBoxes = this.faceTracks.map((track) => track.box);
    const mainIndex = selectMainFaceIndex(
      visibleBoxes,
      subjectBox,
      this.previousMainFace,
      effects.privacyFirst,
    );
    if (mainIndex !== null) this.previousMainFace = [...visibleBoxes[mainIndex]] as Box;

    const faceMasks = this.faceTracks
      .filter((track, index) => track.missedFrames === 0
        && index !== mainIndex
        && !isProtectedMainHead(
          track.box,
          subjectBox,
          video.videoWidth,
          video.videoHeight,
        ))
      .map((track) => track.box);
    const inferredFallbacks = suppressFaceSupportedFallbacks(faceMasks, detectedBystanderHeads);
    const maskCandidates = mergeHeadBoxes([
      ...faceMasks,
      ...inferredFallbacks,
    ]).filter((head) =>
      isPlausibleHeadBox(head, video.videoWidth, video.videoHeight, referenceHead, 2.6)
      && !isProtectedMainHead(
        head,
        subjectBox,
        video.videoWidth,
        video.videoHeight,
      ),
    );
    this.maskTracks = this.updateTracks(this.maskTracks, maskCandidates, {
      maximumMissedFrames: MASK_TRACK_MISSED_FRAMES,
      maximumDistance: 0.64,
      minimumOverlap: 0.08,
      maximumAreaRatio: 2.5,
      replacementDistance: 1.3,
    }).filter((track) => !isProtectedMainHead(
      track.box,
      subjectBox,
      video.videoWidth,
      video.videoHeight,
    ));

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
      const opacity = clamp(1 - track.missedFrames / (MASK_TRACK_MISSED_FRAMES + 1), 0, 1);
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
