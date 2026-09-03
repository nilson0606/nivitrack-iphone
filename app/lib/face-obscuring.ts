import * as ort from 'onnxruntime-web/wasm';

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
  /** 0-100，50 等於 V35 行為。越高越靈敏，抓到更多人頭。追蹤階段生效，改了要重跑追蹤。 */
  detectSensitivity?: number;
  /** 0-100，50 等於 V35 行為。越高主角頭部周圍的保護範圍越大。繪製階段生效。 */
  subjectProtection?: number;
  /** 0-100，50 等於 V35 行為。越高遮罩在偵測中斷後撐越久。繪製階段生效。 */
  maskPersistence?: number;
};

export type HeadDetectionFrame = {
  time: number;
  heads: Box[];
};

export type MainHeadTrackingContext = {
  trackerBox: Box;
  relativeHeadBox: Box;
};

export type FaceHeadDetection = {
  box: Box;
  score: number;
};

export type FaceHeadDetectionScene = {
  heads: FaceHeadDetection[];
  bodies: FaceHeadDetection[];
};

export const DEFAULT_FACE_MASK_EFFECTS: FaceMaskEffects = {
  style: 'strong-blur',
  strength: 0.72,
  scale: 1.38,
  emoji: '😎',
  privacyFirst: true,
  detectSensitivity: 50,
  subjectProtection: 50,
  maskPersistence: 50,
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

const MASK_TRACK_MISSED_FRAMES = 5;
const HEAD_MODEL_WIDTH = 320;
const HEAD_MODEL_HEIGHT = 256;
const BODY_CLASS_ID = 0;
const HEAD_CLASS_ID = 1;
const HEAD_MIN_SCORE = 0.2;
const BODY_MIN_SCORE = 0.2;

// 三個取捨參數由使用者在介面上決定，50 一律等於改動前的寫死值。
export function headDetectionMinScore(sensitivity?: number) {
  const ratio = clamp((sensitivity ?? 50) / 100, 0, 1);
  return 0.3 - ratio * 0.2;
}

export function subjectProtectionScale(protection?: number) {
  const ratio = clamp((protection ?? 50) / 100, 0, 1);
  return 0.6 + ratio * 0.8;
}

export function maskPersistenceFrames(persistence?: number) {
  const ratio = clamp((persistence ?? 50) / 100, 0, 1);
  const minimum = Math.max(1, MASK_TRACK_MISSED_FRAMES - 3);
  const maximum = MASK_TRACK_MISSED_FRAMES + 3;
  return Math.round(minimum + ratio * (maximum - minimum));
}

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

export function normalizedFaceMaskCrop(
  crop: Box | undefined,
  sourceWidth: number,
  sourceHeight: number,
): Box {
  if (!crop) return [0, 0, sourceWidth, sourceHeight];
  const width = clamp(crop[2], 2, sourceWidth);
  const height = clamp(crop[3], 2, sourceHeight);
  const x = clamp(crop[0], 0, Math.max(0, sourceWidth - width));
  const y = clamp(crop[1], 0, Math.max(0, sourceHeight - height));
  return [x, y, width, height];
}

export function defaultFaceMaskCrop(
  mainHead: Box,
  sourceWidth: number,
  sourceHeight: number,
): Box {
  const width = Math.min(
    sourceWidth,
    Math.max(sourceWidth * 0.5, mainHead[2] * 14),
  );
  const height = Math.min(
    sourceHeight,
    Math.max(sourceHeight * 0.72, mainHead[3] * 12),
  );
  const centerX = mainHead[0] + mainHead[2] / 2;
  const centerY = mainHead[1] + mainHead[3] * 3.1;
  return normalizedFaceMaskCrop([
    centerX - width / 2,
    centerY - height / 2,
    width,
    height,
  ], sourceWidth, sourceHeight);
}

export function faceMaskDestinationBox(
  sourceBox: Box,
  crop: Box,
  outputWidth: number,
  outputHeight: number,
): Box {
  const scaleX = outputWidth / Math.max(2, crop[2]);
  const scaleY = outputHeight / Math.max(2, crop[3]);
  return [
    (sourceBox[0] - crop[0]) * scaleX,
    (sourceBox[1] - crop[1]) * scaleY,
    sourceBox[2] * scaleX,
    sourceBox[3] * scaleY,
  ];
}

function isHeadLikeSelection(subject: Box) {
  const aspect = subject[2] / Math.max(1, subject[3]);
  return aspect >= 0.5 && aspect <= 1.65;
}

function referenceHeadForSubject(
  subject: Box,
  sourceWidth: number,
  sourceHeight: number,
) {
  if (isHeadLikeSelection(subject)) {
    return [
      clamp(subject[0], 0, Math.max(0, sourceWidth - subject[2])),
      clamp(subject[1], 0, Math.max(0, sourceHeight - subject[3])),
      Math.min(subject[2], sourceWidth),
      Math.min(subject[3], sourceHeight),
    ] as Box;
  }
  return personBoxToHeadBox(subject, sourceWidth, sourceHeight);
}

export function selectMainBodyForHead(bodies: Box[], head: Box) {
  const [headCenterX, headCenterY] = boxCenter(head);
  const ranked = bodies
    .map((body, index) => {
      const widthRatio = body[2] / Math.max(1, head[2]);
      const heightRatio = body[3] / Math.max(1, head[3]);
      const horizontalInset = body[2] * 0.14;
      const containsHead = headCenterX >= body[0] - horizontalInset
        && headCenterX <= body[0] + body[2] + horizontalInset
        && headCenterY >= body[1] - head[3] * 0.55
        && headCenterY <= body[1] + body[3] * 0.46;
      if (
        !containsHead
        || widthRatio < 1.15
        || widthRatio > 8
        || heightRatio < 2.5
        || heightRatio > 13
      ) {
        return { index, score: Number.POSITIVE_INFINITY };
      }
      const bodyCenterX = body[0] + body[2] / 2;
      const expectedHeadY = body[1] + body[3] * 0.1;
      return {
        index,
        score: Math.abs(headCenterX - bodyCenterX) / Math.max(8, body[2])
          + Math.abs(headCenterY - expectedHeadY) / Math.max(8, body[3]) * 0.72,
      };
    })
    .filter((item) => Number.isFinite(item.score))
    .sort((left, right) => left.score - right.score);
  return ranked[0]?.index ?? null;
}

export function createMainHeadTrackingContext(
  head: Box,
  sourceWidth: number,
  sourceHeight: number,
  bodies: Box[] = [],
): MainHeadTrackingContext {
  const headCenterX = head[0] + head[2] / 2;
  const bodyIndex = selectMainBodyForHead(bodies, head);
  const body = bodyIndex === null ? null : bodies[bodyIndex];
  const desiredWidth = Math.min(
    sourceWidth,
    body ? Math.max(head[2], body[2] * 1.06) : Math.max(head[2], head[2] * 3.05),
  );
  const desiredHeight = Math.min(
    sourceHeight,
    body ? Math.max(head[3], body[3] * 1.04) : Math.max(head[3], head[3] * 7.1),
  );
  const trackerX = body
    ? clamp(
      body[0] + body[2] / 2 - desiredWidth / 2,
      0,
      Math.max(0, sourceWidth - desiredWidth),
    )
    : clamp(
      headCenterX - desiredWidth / 2,
      0,
      Math.max(0, sourceWidth - desiredWidth),
    );
  const trackerY = body
    ? clamp(
      body[1] - body[3] * 0.02,
      0,
      Math.max(0, sourceHeight - desiredHeight),
    )
    : clamp(
      head[1] - head[3] * 0.34,
      0,
      Math.max(0, sourceHeight - desiredHeight),
    );
  const trackerBox: Box = [trackerX, trackerY, desiredWidth, desiredHeight];
  return {
    trackerBox,
    relativeHeadBox: [
      (head[0] - trackerX) / Math.max(1, desiredWidth),
      (head[1] - trackerY) / Math.max(1, desiredHeight),
      head[2] / Math.max(1, desiredWidth),
      head[3] / Math.max(1, desiredHeight),
    ],
  };
}

export function headBoxFromTrackingContext(
  trackedContextBox: Box,
  context: MainHeadTrackingContext,
  sourceWidth: number,
  sourceHeight: number,
): Box {
  const width = Math.min(sourceWidth, Math.max(2, trackedContextBox[2] * context.relativeHeadBox[2]));
  const height = Math.min(sourceHeight, Math.max(2, trackedContextBox[3] * context.relativeHeadBox[3]));
  return [
    clamp(
      trackedContextBox[0] + trackedContextBox[2] * context.relativeHeadBox[0],
      0,
      Math.max(0, sourceWidth - width),
    ),
    clamp(
      trackedContextBox[1] + trackedContextBox[3] * context.relativeHeadBox[1],
      0,
      Math.max(0, sourceHeight - height),
    ),
    width,
    height,
  ];
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
  protectionScale = 1,
): Box {
  const inferredHead = referenceHeadForSubject(subject, sourceWidth, sourceHeight);
  const protectionWidth = Math.min(sourceWidth, Math.max(inferredHead[2] * 1.5, subject[2] * 0.62) * protectionScale);
  const protectionHeight = Math.min(sourceHeight, Math.max(inferredHead[3] * 1.72, subject[3] * 0.25) * protectionScale);
  const centerX = inferredHead[0] + inferredHead[2] / 2;
  return [
    clamp(centerX - protectionWidth / 2, 0, Math.max(0, sourceWidth - protectionWidth)),
    clamp(inferredHead[1] - inferredHead[3] * 0.22, 0, Math.max(0, sourceHeight - protectionHeight)),
    protectionWidth,
    protectionHeight,
  ];
}

export function bystanderDetectedHeadBoxes(
  heads: Box[],
  subjectHead: Box,
  sourceWidth: number,
  sourceHeight: number,
  protectionScale = 1,
) {
  const mainIndex = selectMainFaceIndex(heads, subjectHead, null, false);
  const referenceHead = referenceHeadForSubject(subjectHead, sourceWidth, sourceHeight);
  return heads
    .filter((_, index) => index !== mainIndex)
    .filter((head) => isPlausibleHeadBox(head, sourceWidth, sourceHeight, referenceHead, 3.2))
    .filter((head) => !isProtectedMainHead(head, subjectHead, sourceWidth, sourceHeight, protectionScale));
}

// Kept for older smoke tests and restore points. New code uses the precise name.
export const bystanderFaceBoxes = bystanderDetectedHeadBoxes;

export function plausibleDetectedHeadBoxes(
  heads: Box[],
  subjectHead: Box,
  sourceWidth: number,
  sourceHeight: number,
) {
  const referenceHead = referenceHeadForSubject(subjectHead, sourceWidth, sourceHeight);
  return heads.filter((head) =>
    isPlausibleHeadBox(head, sourceWidth, sourceHeight, referenceHead, 3.2),
  );
}

export function isProtectedMainHead(
  head: Box,
  subject: Box,
  sourceWidth: number,
  sourceHeight: number,
  protectionScale = 1,
) {
  const protection = subjectHeadProtectionBox(subject, sourceWidth, sourceHeight, protectionScale);
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
  if (isHeadLikeSelection(subject)) {
    const [subjectX, subjectY] = boxCenter(subject);
    const distance = Math.hypot(
      (faceX - subjectX) / Math.max(8, subject[2] * 0.82),
      (faceY - subjectY) / Math.max(8, subject[3] * 0.82),
    );
    if (distance > 1.55) return Number.POSITIVE_INFINITY;
    const previousDistance = previousMainFace ? faceTrackDistance(face, previousMainFace) : 0;
    const overlap = boxIou(face, subject);
    const previousOverlap = previousMainFace ? boxIou(face, previousMainFace) : 0;
    return distance - overlap * 0.62
      + previousDistance * (previousMainFace ? 0.68 : 0)
      - previousOverlap * 0.35;
  }
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

export function privacyEffectRasterSize(
  style: 'soft-blur' | 'strong-blur' | 'pixelate',
  width: number,
  height: number,
  strength: number,
): [number, number] {
  const safeStrength = clamp(strength, 0, 1);
  const samplesAcross = style === 'soft-blur'
    ? Math.round(14 - safeStrength * 6)
    : style === 'strong-blur'
      ? Math.round(8 - safeStrength * 5)
      : Math.round(9 - safeStrength * 6);
  const minimum = style === 'soft-blur' ? 6 : 3;
  const maximum = style === 'soft-blur' ? 14 : style === 'strong-blur' ? 8 : 9;
  const rasterWidth = Math.round(clamp(samplesAcross, minimum, maximum));
  const rasterHeight = Math.max(
    minimum,
    Math.round(rasterWidth * Math.max(1, height) / Math.max(1, width)),
  );
  return [rasterWidth, rasterHeight];
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

export class FaceHeadDetector {
  private readonly session: ort.InferenceSession;
  private readonly inputCanvas = document.createElement('canvas');
  private readonly inputContext: CanvasRenderingContext2D;

  private constructor(session: ort.InferenceSession) {
    this.session = session;
    this.inputCanvas.width = HEAD_MODEL_WIDTH;
    this.inputCanvas.height = HEAD_MODEL_HEIGHT;
    const context = this.inputCanvas.getContext('2d', {
      alpha: false,
      willReadFrequently: true,
    });
    if (!context) throw new Error('Safari 無法建立人頭辨識畫布');
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = 'medium';
    this.inputContext = context;
  }

  static async create() {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    ort.env.wasm.wasmPaths = new URL('ort/', document.baseURI).href;
    const modelUrl = new URL('models/yolox_n_body_head_hand_256x320.onnx', document.baseURI).href;
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      executionMode: 'sequential',
    });
    return new FaceHeadDetector(session);
  }

  async detectScene(
    video: HTMLVideoElement,
    sourceCrop?: Box,
    minScore?: number,
  ): Promise<FaceHeadDetectionScene> {
    if (!video.videoWidth || !video.videoHeight) return { heads: [], bodies: [] };
    const crop = normalizedFaceMaskCrop(sourceCrop, video.videoWidth, video.videoHeight);
    this.inputContext.drawImage(
      video,
      crop[0],
      crop[1],
      crop[2],
      crop[3],
      0,
      0,
      HEAD_MODEL_WIDTH,
      HEAD_MODEL_HEIGHT,
    );
    const rgba = this.inputContext.getImageData(0, 0, HEAD_MODEL_WIDTH, HEAD_MODEL_HEIGHT).data;
    const plane = HEAD_MODEL_WIDTH * HEAD_MODEL_HEIGHT;
    const bgr = new Float32Array(plane * 3);
    for (let index = 0; index < plane; index += 1) {
      const pixel = index * 4;
      bgr[index] = rgba[pixel + 2];
      bgr[plane + index] = rgba[pixel + 1];
      bgr[plane * 2 + index] = rgba[pixel];
    }

    const result = await this.session.run({
      [this.session.inputNames[0]]: new ort.Tensor(
        'float32',
        bgr,
        [1, 3, HEAD_MODEL_HEIGHT, HEAD_MODEL_WIDTH],
      ),
    });
    const output = result[this.session.outputNames[0]]?.data as Float32Array | undefined;
    if (!output) return { heads: [], bodies: [] };
    const scaleX = crop[2] / HEAD_MODEL_WIDTH;
    const scaleY = crop[3] / HEAD_MODEL_HEIGHT;
    const heads: FaceHeadDetection[] = [];
    const bodies: FaceHeadDetection[] = [];
    for (let offset = 0; offset + 6 < output.length; offset += 7) {
      const classId = Math.round(output[offset + 1]);
      const score = output[offset + 2];
      if (
        (classId !== HEAD_CLASS_ID && classId !== BODY_CLASS_ID)
        || (classId === HEAD_CLASS_ID && score < (minScore ?? HEAD_MIN_SCORE))
        || (classId === BODY_CLASS_ID && score < BODY_MIN_SCORE)
      ) continue;
      const x1 = crop[0] + clamp(output[offset + 3], 0, HEAD_MODEL_WIDTH) * scaleX;
      const y1 = crop[1] + clamp(output[offset + 4], 0, HEAD_MODEL_HEIGHT) * scaleY;
      const x2 = crop[0] + clamp(output[offset + 5], 0, HEAD_MODEL_WIDTH) * scaleX;
      const y2 = crop[1] + clamp(output[offset + 6], 0, HEAD_MODEL_HEIGHT) * scaleY;
      if (x2 - x1 < 4 || y2 - y1 < 4) continue;
      const detection = { box: [x1, y1, x2 - x1, y2 - y1] as Box, score };
      if (classId === HEAD_CLASS_ID) heads.push(detection);
      else bodies.push(detection);
    }
    return { heads, bodies };
  }

  async detect(video: HTMLVideoElement, sourceCrop?: Box, minScore?: number): Promise<FaceHeadDetection[]> {
    return (await this.detectScene(video, sourceCrop, minScore)).heads;
  }

  close() {
    this.inputCanvas.width = 1;
    this.inputCanvas.height = 1;
    void this.session.release();
  }
}

export class FaceObscuringRenderer {
  private readonly blurCanvas = document.createElement('canvas');
  private readonly pixelCanvas = document.createElement('canvas');
  private readonly stickerImage: HTMLImageElement | null;
  private maskTracks: FaceTrack[] = [];
  private nextTrackId = 1;

  private constructor(stickerImage: HTMLImageElement | null) {
    this.stickerImage = stickerImage;
  }

  static async create(stickerUrl?: string) {
    return new FaceObscuringRenderer(await loadStickerImage(stickerUrl));
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
    style: 'soft-blur' | 'strong-blur',
    strength: number,
    opacity: number,
  ) {
    const [rasterWidth, rasterHeight] = privacyEffectRasterSize(
      style,
      destinationBox[2],
      destinationBox[3],
      strength,
    );
    if (this.blurCanvas.width !== rasterWidth || this.blurCanvas.height !== rasterHeight) {
      this.blurCanvas.width = rasterWidth;
      this.blurCanvas.height = rasterHeight;
    }
    const blurContext = this.blurCanvas.getContext('2d', { alpha: false });
    if (!blurContext) throw new Error('Safari 無法建立人臉模糊畫布');
    blurContext.imageSmoothingEnabled = true;
    blurContext.imageSmoothingQuality = 'high';
    blurContext.drawImage(video, ...sourceBox, 0, 0, rasterWidth, rasterHeight);

    const [destinationX, destinationY, destinationWidth, destinationHeight] = destinationBox;
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
    // Safari's CanvasRenderingContext2D.filter can silently behave like
    // "none" during video recording. A tiny raster reconstructed with high
    // quality smoothing produces a deterministic, smooth privacy blur.
    context.filter = 'none';
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = 'high';
    context.drawImage(this.blurCanvas, destinationX, destinationY, destinationWidth, destinationHeight);
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
    const [tinyWidth, tinyHeight] = privacyEffectRasterSize(
      'pixelate',
      destinationBox[2],
      destinationBox[3],
      strength,
    );
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
    sourceCrop?: Box,
  ) {
    if (!video.videoWidth || !video.videoHeight) return;
    const context = outputCanvas.getContext('2d', { alpha: false });
    if (!context) throw new Error('Safari 無法建立旁人人臉遮罩畫布');
    const protectionScale = subjectProtectionScale(effects.subjectProtection);
    const persistenceFrames = maskPersistenceFrames(effects.maskPersistence);
    context.globalCompositeOperation = 'source-over';
    context.globalAlpha = 1;
    context.filter = 'none';
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = 'high';
    const crop = normalizedFaceMaskCrop(sourceCrop, video.videoWidth, video.videoHeight);
    context.drawImage(
      video,
      crop[0],
      crop[1],
      crop[2],
      crop[3],
      0,
      0,
      outputCanvas.width,
      outputCanvas.height,
    );

    const referenceHead = referenceHeadForSubject(
      subjectBox,
      video.videoWidth,
      video.videoHeight,
    );
    const plausibleHeads = mergeHeadBoxes(detectedBystanderHeads).filter((head) =>
      isPlausibleHeadBox(head, video.videoWidth, video.videoHeight, referenceHead, 2.6),
    );
    // ViT already supplies the current main-head path. Do not grant a second
    // moving detection "main" status: when two dancers cross, that stale
    // exemption can jump to the bystander and simultaneously mask the real main.
    const maskCandidates = plausibleHeads.filter((head) =>
      !isProtectedMainHead(
        head,
        subjectBox,
        video.videoWidth,
        video.videoHeight,
        protectionScale,
      ),
    );
    this.maskTracks = this.updateTracks(this.maskTracks, maskCandidates, {
      maximumMissedFrames: persistenceFrames,
      maximumDistance: 0.64,
      minimumOverlap: 0.08,
      maximumAreaRatio: 2.5,
      replacementDistance: 1.3,
    }).filter((track) => !isProtectedMainHead(
      track.box,
      subjectBox,
      video.videoWidth,
      video.videoHeight,
      protectionScale,
    ));

    for (const track of this.maskTracks) {
      const sourceBox = expandFaceBox(
        track.box,
        effects.scale,
        video.videoWidth,
        video.videoHeight,
      );
      const destinationBox = faceMaskDestinationBox(
        sourceBox,
        crop,
        outputCanvas.width,
        outputCanvas.height,
      );
      const opacity = effects.privacyFirst
        ? 1
        : clamp(1 - track.missedFrames / (persistenceFrames + 1), 0, 1);
      if (effects.style === 'soft-blur' || effects.style === 'strong-blur') {
        this.drawBlur(
          video,
          context,
          sourceBox,
          destinationBox,
          effects.style,
          effects.strength,
          opacity,
        );
      } else if (effects.style === 'pixelate') {
        this.drawPixelated(video, context, sourceBox, destinationBox, effects.strength, opacity);
      } else {
        this.drawCover(context, destinationBox, effects, opacity);
      }
    }
  }

  reset() {
    this.maskTracks = [];
  }

  close() {
    this.blurCanvas.width = 1;
    this.blurCanvas.height = 1;
    this.pixelCanvas.width = 1;
    this.pixelCanvas.height = 1;
    this.maskTracks = [];
  }
}
