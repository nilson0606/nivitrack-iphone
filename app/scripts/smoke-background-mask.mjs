import {
  constrainAlphaToBodyEnvelope,
  constrainSubjectConfidenceToPose,
  createPoseBodyEnvelope,
  preferUsablePoseAlpha,
  preserveSuddenSubjectLoss,
  recoverTrackedSubjectAlpha,
  selectModnetTrackedAlpha,
  selectTrackedSubjectAlpha,
  solidifyAndInsetAlpha,
  stabilizeTrackedSubjectAlpha,
  tightenTrackedSubjectEdges,
  trackedSubjectRegion,
} from '../lib/person-background-removal.ts';
import { trackedFrameCrop } from '../lib/video-export.ts';

const width = 20;
const height = 10;
const trackedBox = [5, 5, 30, 40];

function countSides(alpha) {
  let selected = 0;
  let leaked = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if ((alpha?.[y * width + x] ?? 0) <= 0.1) continue;
      if (x <= 8) selected += 1;
      else leaked += 1;
    }
  }
  return { selected, leaked };
}

function testSeparatedDancers() {
  const mask = new Float32Array(width * height);
  for (let y = 2; y <= 7; y += 1) {
    for (let x = 2; x <= 6; x += 1) mask[y * width + x] = 0.9;
    for (let x = 12; x <= 16; x += 1) mask[y * width + x] = 0.95;
  }
  return countSides(selectTrackedSubjectAlpha(mask, width, height, trackedBox, 100, 50));
}

function testTouchingDancers() {
  const mask = new Float32Array(width * height);
  for (let y = 2; y <= 7; y += 1) {
    for (let x = 2; x <= 16; x += 1) mask[y * width + x] = 0.9;
  }
  return countSides(selectTrackedSubjectAlpha(mask, width, height, trackedBox, 100, 50));
}

function testNarrowBackgroundBridge() {
  const mask = new Float32Array(width * height);
  for (let y = 1; y <= 8; y += 1) {
    for (let x = 3; x <= 7; x += 1) mask[y * width + x] = 0.92;
  }
  for (let y = 2; y <= 5; y += 1) {
    for (let x = 11; x <= 14; x += 1) mask[y * width + x] = 0.76;
  }
  for (let y = 5; y <= 8; y += 1) {
    for (let x = 12; x <= 13; x += 1) mask[y * width + x] = 0.76;
  }
  for (let x = 8; x <= 10; x += 1) mask[4 * width + x] = 0.7;

  const alpha = selectTrackedSubjectAlpha(mask, width, height, [10, 0, 70, 50], 100, 50);
  let dancer = 0;
  let backgroundObject = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if ((alpha?.[y * width + x] ?? 0) <= 0.1) continue;
      if (x <= 8) dancer += 1;
      if (x >= 10) backgroundObject += 1;
    }
  }
  return { dancer, backgroundObject };
}

function testBroadBackgroundOverlap() {
  const personMask = new Float32Array(width * height);
  const poseMask = new Float32Array(width * height);
  for (let y = 1; y <= 8; y += 1) {
    for (let x = 3; x <= 7; x += 1) {
      personMask[y * width + x] = 0.92;
      poseMask[y * width + x] = 0.96;
    }
  }
  // Simulate a broad, high-confidence background region that directly overlaps
  // the subject matte and therefore cannot be removed by connectivity alone.
  for (let y = 1; y <= 7; y += 1) {
    for (let x = 8; x <= 13; x += 1) personMask[y * width + x] = 0.88;
  }
  const constrained = constrainSubjectConfidenceToPose(
    personMask,
    width,
    height,
    poseMask,
    width,
    height,
  );
  const alpha = selectTrackedSubjectAlpha(constrained, width, height, [10, 0, 70, 50], 100, 50);
  let dancer = 0;
  let backgroundObject = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if ((alpha?.[y * width + x] ?? 0) <= 0.1) continue;
      if (x <= 7) dancer += 1;
      if (x >= 10) backgroundObject += 1;
    }
  }
  const fallback = selectTrackedSubjectAlpha(
    personMask,
    width,
    height,
    [10, 0, 70, 50],
    100,
    50,
  );
  const preferred = preferUsablePoseAlpha(alpha, fallback);
  return {
    dancer,
    backgroundObject,
    usableConstraintPreferred: preferred === alpha,
  };
}

function testPoseBodyEnvelope() {
  const landmarks = Array.from({ length: 33 }, () => ({
    x: 0.5,
    y: 0.5,
    visibility: 0,
  }));
  const set = (index, x, y) => {
    landmarks[index] = { x, y, visibility: 1 };
  };
  set(0, 0.5, 0.14);
  set(7, 0.45, 0.17);
  set(8, 0.55, 0.17);
  set(11, 0.4, 0.3);
  set(12, 0.6, 0.3);
  set(13, 0.31, 0.45);
  set(14, 0.69, 0.45);
  set(15, 0.25, 0.6);
  set(16, 0.75, 0.6);
  set(19, 0.23, 0.64);
  set(20, 0.77, 0.64);
  set(23, 0.43, 0.55);
  set(24, 0.57, 0.55);
  set(25, 0.4, 0.75);
  set(26, 0.6, 0.75);
  set(27, 0.38, 0.92);
  set(28, 0.62, 0.92);
  set(31, 0.34, 0.96);
  set(32, 0.66, 0.96);

  const loose = createPoseBodyEnvelope(landmarks, 100, 100, 0);
  const tight = createPoseBodyEnvelope(landmarks, 100, 100, 1);
  const fallback = new Float32Array(10000).fill(1);
  const constrained = tight ? constrainAlphaToBodyEnvelope(fallback, tight) : null;
  const loosePixels = loose?.filter((value) => value > 0.1).length ?? 0;
  const tightPixels = tight?.filter((value) => value > 0.1).length ?? 0;
  return {
    torsoRetained: (constrained?.[45 * 100 + 50] ?? 0) > 0.8,
    distantBackgroundRemoved: (constrained?.[45 * 100 + 90] ?? 1) === 0,
    tightnessShrinksEnvelope: loosePixels > tightPixels && tightPixels > 0,
    usableEnvelopePreferred: preferUsablePoseAlpha(constrained, fallback) === constrained,
    invalidPoseRejected: createPoseBodyEnvelope([], 100, 100, 0.5) === null,
  };
}

function testInvalidPoseFallback() {
  const fallback = new Float32Array(100).fill(0.8);
  const emptyPose = new Float32Array(100);
  const tinyPose = new Float32Array(100);
  tinyPose[50] = 1;
  return {
    emptyFallsBack: preferUsablePoseAlpha(emptyPose, fallback) === fallback,
    tinyFallsBack: preferUsablePoseAlpha(tinyPose, fallback) === fallback,
    missingFallsBack: preferUsablePoseAlpha(null, fallback) === fallback,
  };
}

function testLowConfidenceDancer() {
  const mask = new Float32Array(width * height);
  for (let y = 2; y <= 7; y += 1) {
    for (let x = 2; x <= 6; x += 1) mask[y * width + x] = 0.18;
    for (let x = 12; x <= 16; x += 1) mask[y * width + x] = 0.19;
  }
  return countSides(selectTrackedSubjectAlpha(mask, width, height, trackedBox, 100, 50));
}

function testSoftModnetDancer() {
  const mask = new Float32Array(width * height).fill(0.003);
  for (let y = 1; y <= 8; y += 1) {
    for (let x = 3; x <= 7; x += 1) mask[y * width + x] = 0.055 + y * 0.004;
  }
  for (let y = 2; y <= 7; y += 1) {
    for (let x = 14; x <= 18; x += 1) mask[y * width + x] = 0.12;
  }
  const alpha = selectModnetTrackedAlpha(mask, width, height, [10, 0, 35, 50], 100, 50);
  let dancer = 0;
  let otherDancer = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if ((alpha?.[y * width + x] ?? 0) <= 0.1) continue;
      if (x >= 3 && x <= 7) dancer += 1;
      if (x >= 14) otherDancer += 1;
    }
  }
  return { dancer, otherDancer };
}

function testMissingFrameRecovery() {
  const previous = new Float32Array(width * height);
  previous[23] = 1;
  const firstMissing = recoverTrackedSubjectAlpha(null, null, previous.length, 0);
  const temporaryMissing = recoverTrackedSubjectAlpha(null, previous, previous.length, 0);
  let expired = { alpha: previous, missedFrames: 0, fresh: true };
  for (let index = 0; index < 13; index += 1) {
    expired = recoverTrackedSubjectAlpha(null, expired.alpha, previous.length, expired.missedFrames);
  }
  return {
    firstMissingIsBlack: firstMissing.alpha.every((value) => value === 0),
    temporaryMissingRetained: temporaryMissing.alpha[23] > 0,
    expiredIsBlack: expired.alpha.every((value) => value === 0),
  };
}

function testPartialSubjectLossRecovery() {
  const maskWidth = 12;
  const maskHeight = 12;
  const full = new Float32Array(maskWidth * maskHeight);
  for (let y = 1; y <= 10; y += 1) {
    for (let x = 4; x <= 7; x += 1) full[y * maskWidth + x] = 0.9;
  }
  const missingTop = new Float32Array(full);
  for (let y = 1; y <= 4; y += 1) {
    for (let x = 4; x <= 7; x += 1) missingTop[y * maskWidth + x] = 0;
  }
  const initialState = {
    referenceAlpha: new Float32Array(full),
    incompleteBandFrames: [0, 0, 0],
  };
  const recovered = preserveSuddenSubjectLoss(
    missingTop,
    maskWidth,
    maskHeight,
    [3, 1, 6, 10],
    maskWidth,
    maskHeight,
    initialState,
  );
  const expired = preserveSuddenSubjectLoss(
    missingTop,
    maskWidth,
    maskHeight,
    [3, 1, 6, 10],
    maskWidth,
    maskHeight,
    recovered.state,
  );
  return {
    topTemporarilyPreserved: recovered.alpha[2 * maskWidth + 5] > 0.7,
    lowerBodyUnchanged: recovered.alpha[9 * maskWidth + 5] === missingTop[9 * maskWidth + 5],
    persistentLossEventuallyAccepted: expired.alpha[2 * maskWidth + 5] === 0,
  };
}

function testTrackedRegion() {
  const box = [30, 20, 40, 80];
  const region = trackedSubjectRegion(box, 100, 200);
  return {
    region,
    containsBox: region[0] <= box[0]
      && region[1] <= box[1]
      && region[0] + region[2] >= box[0] + box[2]
      && region[1] + region[3] >= box[1] + box[3],
    cropped: region[2] < 100 && region[3] < 200,
  };
}

function testTemporalStability() {
  const current = new Float32Array([0.9]);
  const empty = new Float32Array([0]);
  const firstAppearance = stabilizeTrackedSubjectAlpha(current, empty, empty);
  const confirmedAppearance = stabilizeTrackedSubjectAlpha(current, empty, current);
  const persistentSubject = stabilizeTrackedSubjectAlpha(new Float32Array([0.7]), new Float32Array([0.8]), current);
  return {
    oneFrameLeakRejected: firstAppearance[0] === 0,
    confirmedSubjectAccepted: confirmedAppearance[0] > 0.5,
    persistentSubjectRetained: persistentSubject[0] > 0.6,
  };
}

function testEdgeTightening() {
  const alpha = new Float32Array(25).fill(1);
  const tightened = tightenTrackedSubjectEdges(alpha, 5, 5);
  return {
    centerPreserved: tightened[12] === 1,
    edgeFeathered: tightened[2] > 0 && tightened[2] < 0.25,
  };
}

function testBlackSafetyEdge() {
  const maskWidth = 20;
  const maskHeight = 10;
  const alpha = new Uint8ClampedArray(maskWidth * maskHeight);
  for (let y = 2; y <= 7; y += 1) {
    for (let x = 3; x <= 9; x += 1) alpha[y * maskWidth + x] = 90;
  }
  alpha[5 * maskWidth + 15] = 20;
  solidifyAndInsetAlpha(alpha, maskWidth, maskHeight, 1, new Uint16Array(alpha.length));
  return {
    subjectInteriorSolidified: alpha[5 * maskWidth + 6] > 120,
    subjectEdgeInsetToBlack: alpha[5 * maskWidth + 3] === 0,
    weakDistantLeakRemoved: alpha[5 * maskWidth + 15] === 0,
  };
}

function testAdjustableFraming() {
  const box = [400, 300, 200, 600];
  const smallerSubject = trackedFrameCrop(1080, 1920, box, 9 / 16, 0.25);
  const largerSubject = trackedFrameCrop(1080, 1920, box, 9 / 16, 0.8);
  return {
    largerSubjectUsesTighterCrop: largerSubject[2] < smallerSubject[2]
      && largerSubject[3] < smallerSubject[3],
    aspectPreserved: Math.abs(largerSubject[2] / largerSubject[3] - 9 / 16) < 0.0001,
  };
}

const separated = testSeparatedDancers();
const touching = testTouchingDancers();
const narrowBackgroundBridge = testNarrowBackgroundBridge();
const broadBackgroundOverlap = testBroadBackgroundOverlap();
const poseBodyEnvelope = testPoseBodyEnvelope();
const invalidPoseFallback = testInvalidPoseFallback();
const lowConfidence = testLowConfidenceDancer();
const softModnet = testSoftModnetDancer();
const recovery = testMissingFrameRecovery();
const partialLoss = testPartialSubjectLossRecovery();
const region = testTrackedRegion();
const temporal = testTemporalStability();
const edges = testEdgeTightening();
const blackSafetyEdge = testBlackSafetyEdge();
const framing = testAdjustableFraming();
const pass = separated.selected > 0 && separated.leaked === 0
  && touching.selected > 0 && touching.leaked === 0
  && narrowBackgroundBridge.dancer > 0 && narrowBackgroundBridge.backgroundObject === 0
  && broadBackgroundOverlap.dancer > 0
  && broadBackgroundOverlap.backgroundObject === 0
  && broadBackgroundOverlap.usableConstraintPreferred
  && poseBodyEnvelope.torsoRetained
  && poseBodyEnvelope.distantBackgroundRemoved
  && poseBodyEnvelope.tightnessShrinksEnvelope
  && poseBodyEnvelope.usableEnvelopePreferred
  && poseBodyEnvelope.invalidPoseRejected
  && invalidPoseFallback.emptyFallsBack
  && invalidPoseFallback.tinyFallsBack
  && invalidPoseFallback.missingFallsBack
  && lowConfidence.selected > 0 && lowConfidence.leaked === 0
  && softModnet.dancer > 20 && softModnet.otherDancer === 0
  && recovery.firstMissingIsBlack
  && recovery.temporaryMissingRetained
  && recovery.expiredIsBlack
  && partialLoss.topTemporarilyPreserved
  && partialLoss.lowerBodyUnchanged
  && partialLoss.persistentLossEventuallyAccepted
  && region.containsBox
  && region.cropped
  && temporal.oneFrameLeakRejected
  && temporal.confirmedSubjectAccepted
  && temporal.persistentSubjectRetained
  && edges.centerPreserved
  && edges.edgeFeathered
  && blackSafetyEdge.subjectInteriorSolidified
  && blackSafetyEdge.subjectEdgeInsetToBlack
  && blackSafetyEdge.weakDistantLeakRemoved
  && framing.largerSubjectUsesTighterCrop
  && framing.aspectPreserved;

console.log(JSON.stringify({
  separated,
  touching,
  narrowBackgroundBridge,
  broadBackgroundOverlap,
  poseBodyEnvelope,
  invalidPoseFallback,
  lowConfidence,
  softModnet,
  recovery,
  partialLoss,
  region,
  temporal,
  edges,
  blackSafetyEdge,
  framing,
  pass,
}, null, 2));
if (!pass) process.exitCode = 1;
