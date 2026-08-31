import {
  equalRowCloneFrames,
  lockTrackedSubjectIdentity,
  recoverTrackedSubjectAlpha,
  selectTrackedSubjectAlpha,
  smoothBackdropParameters,
  stabilizeTrackedPromptAlpha,
  stabilizeTrackedSubjectAlpha,
  tightenTrackedSubjectEdges,
  trackedBackdropPatch,
  trackedMaskOverlap,
  trackedPromptAlpha,
  trackedPromptPoint,
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

function testLowConfidenceDancer() {
  const mask = new Float32Array(width * height);
  for (let y = 2; y <= 7; y += 1) {
    for (let x = 2; x <= 6; x += 1) mask[y * width + x] = 0.18;
    for (let x = 12; x <= 16; x += 1) mask[y * width + x] = 0.19;
  }
  return countSides(selectTrackedSubjectAlpha(mask, width, height, trackedBox, 100, 50));
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

function testPointPromptedObjectIdentity() {
  const promptWidth = 20;
  const promptHeight = 20;
  const box = [5, 4, 10, 12];
  const fallback = trackedPromptPoint(null, promptWidth, promptHeight, box, 20, 20);
  const previous = new Float32Array(promptWidth * promptHeight);
  for (let y = 5; y <= 14; y += 1) {
    for (let x = 6; x <= 11; x += 1) previous[y * promptWidth + x] = 0.9;
  }
  const continued = trackedPromptPoint(previous, promptWidth, promptHeight, box, 20, 20);
  const confidence = new Float32Array(promptWidth * promptHeight);
  for (let y = 5; y <= 14; y += 1) {
    for (let x = 6; x <= 11; x += 1) confidence[y * promptWidth + x] = 0.95;
  }
  for (let x = 12; x <= 15; x += 1) confidence[7 * promptWidth + x] = 0.9;
  for (let y = 5; y <= 14; y += 1) {
    for (let x = 17; x <= 19; x += 1) confidence[y * promptWidth + x] = 0.98;
  }
  const alpha = trackedPromptAlpha(confidence, promptWidth, promptHeight, box, 20, 20);
  const unrelated = new Float32Array(promptWidth * promptHeight);
  for (let y = 5; y <= 14; y += 1) {
    for (let x = 14; x <= 18; x += 1) unrelated[y * promptWidth + x] = 0.95;
  }
  const stabilized = stabilizeTrackedPromptAlpha(alpha, previous);
  return {
    fallbackUsesTorso: Math.abs(fallback.x - 0.5) < 0.001
      && Math.abs(fallback.y - 0.452) < 0.001,
    promptFollowsPreviousMask: continued.x < 0.5 && continued.y > 0.4,
    newLimbAcceptedImmediately: stabilized[7 * promptWidth + 14] > 0.5,
    farObjectExcludedByTrackedRegion: alpha[7 * promptWidth + 19] === 0,
    unrelatedMaskRejectedByOverlap: trackedMaskOverlap(unrelated, previous) < 0.12,
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

function testStrictSubjectIdentityLock() {
  const lockWidth = 32;
  const lockHeight = 16;
  const previous = new Float32Array(lockWidth * lockHeight);
  for (let y = 3; y <= 12; y += 1) {
    for (let x = 4; x <= 10; x += 1) previous[y * lockWidth + x] = 0.92;
  }
  const withPasserBy = new Float32Array(previous);
  for (let y = 3; y <= 12; y += 1) {
    for (let x = 11; x <= 25; x += 1) withPasserBy[y * lockWidth + x] = 0.9;
  }
  let locked = lockTrackedSubjectIdentity(
    withPasserBy,
    previous,
    lockWidth,
    lockHeight,
    70,
    0,
  );
  let farPasserPixels = 0;
  let mainPixels = 0;
  for (let y = 0; y < lockHeight; y += 1) {
    for (let x = 0; x < lockWidth; x += 1) {
      if (locked.alpha[y * lockWidth + x] < 0.035) continue;
      if (x <= 10) mainPixels += 1;
      if (x >= 16) farPasserPixels += 1;
    }
  }
  for (let frame = 0; frame < 6; frame += 1) {
    locked = lockTrackedSubjectIdentity(
      withPasserBy,
      locked.alpha,
      lockWidth,
      lockHeight,
      locked.referenceArea,
      0,
    );
  }
  let sustainedFarPasserPixels = 0;
  for (let y = 0; y < lockHeight; y += 1) {
    for (let x = 16; x < lockWidth; x += 1) {
      if (locked.alpha[y * lockWidth + x] >= 0.035) sustainedFarPasserPixels += 1;
    }
  }

  const movedSubject = new Float32Array(lockWidth * lockHeight);
  for (let y = 3; y <= 12; y += 1) {
    for (let x = 7; x <= 13; x += 1) movedSubject[y * lockWidth + x] = 0.9;
  }
  const moved = lockTrackedSubjectIdentity(
    movedSubject,
    previous,
    lockWidth,
    lockHeight,
    70,
    0.5,
  );
  const extendedLimb = new Float32Array(previous);
  for (let y = 4; y <= 5; y += 1) {
    for (let x = 11; x <= 22; x += 1) extendedLimb[y * lockWidth + x] = 0.9;
  }
  const limb = lockTrackedSubjectIdentity(
    extendedLimb,
    previous,
    lockWidth,
    lockHeight,
    70,
    0,
  );
  const weakFrame = lockTrackedSubjectIdentity(
    new Float32Array(lockWidth * lockHeight),
    previous,
    lockWidth,
    lockHeight,
    70,
    0,
  );
  return {
    mainSubjectRetained: mainPixels === 70,
    farPasserRejected: farPasserPixels === 0,
    sustainedPasserCannotCreepIn: sustainedFarPasserPixels === 0,
    suddenGrowthWasTrimmed: locked.rejectedPixels > 0,
    normalMotionRetained: moved.retainedPixels === 70,
    attachedThinLimbRetained: limb.alpha[5 * lockWidth + 22] > 0,
    weakFrameKeepsIdentityBaseline: weakFrame.referenceArea === 70,
  };
}

function testEqualRowCloneSizes() {
  const frames = equalRowCloneFrames(1080, 1920, 420, 980, 4);
  const oneClone = equalRowCloneFrames(1080, 1920, 420, 980, 1);
  return {
    totalIncludesMain: frames.length === 5,
    equalWidths: frames.every((frame) => Math.abs(frame[2] - frames[0][2]) < 0.0001),
    equalHeights: frames.every((frame) => Math.abs(frame[3] - frames[0][3]) < 0.0001),
    insideCanvas: frames.every((frame) => frame[0] >= 0 && frame[0] + frame[2] <= 1080),
    plusOneMeansTwoPeople: oneClone.length === 2,
    plusOneKeepsLargeSubjects: oneClone.every((frame) => frame[3] >= 1920 * 0.7),
  };
}

function testSmoothBackdrop() {
  const minimum = smoothBackdropParameters(12, 1080, 1920);
  const normal = smoothBackdropParameters(36, 1080, 1920);
  const maximum = smoothBackdropParameters(64, 1080, 1920);
  return {
    keepsSmoothResolution: minimum.height === 420 && maximum.height === 420,
    avoidsTinyPixelBuffer: maximum.width >= 200,
    reducesToColorField: maximum.fieldHeight <= 80
      && maximum.fieldWidth < maximum.width,
    strongerUsesLowerFrequency: maximum.fieldHeight < minimum.fieldHeight,
    strengthIncreasesRadius: minimum.filterRadius < normal.filterRadius
      && normal.filterRadius < maximum.filterRadius,
    padsBlurredEdges: maximum.paddingX > maximum.filterRadius,
  };
}

function testBackdropSubjectRemovalRegion() {
  const subject = [400, 300, 200, 600];
  const patch = trackedBackdropPatch(subject, 1080, 1920, 236, 420);
  const scaledSubject = [
    subject[0] * 236 / 1080,
    subject[1] * 420 / 1920,
    subject[2] * 236 / 1080,
    subject[3] * 420 / 1920,
  ];
  return {
    patch,
    containsSubject: patch[0] <= scaledSubject[0]
      && patch[1] <= scaledSubject[1]
      && patch[0] + patch[2] >= scaledSubject[0] + scaledSubject[2]
      && patch[1] + patch[3] >= scaledSubject[1] + scaledSubject[3],
    staysInsideBackdrop: patch[0] >= 0
      && patch[1] >= 0
      && patch[0] + patch[2] <= 236
      && patch[1] + patch[3] <= 420,
  };
}

const separated = testSeparatedDancers();
const touching = testTouchingDancers();
const narrowBackgroundBridge = testNarrowBackgroundBridge();
const lowConfidence = testLowConfidenceDancer();
const recovery = testMissingFrameRecovery();
const region = testTrackedRegion();
const temporal = testTemporalStability();
const promptedIdentity = testPointPromptedObjectIdentity();
const strictIdentity = testStrictSubjectIdentityLock();
const edges = testEdgeTightening();
const framing = testAdjustableFraming();
const rowClones = testEqualRowCloneSizes();
const smoothBackdrop = testSmoothBackdrop();
const backdropSubjectRemoval = testBackdropSubjectRemovalRegion();
const pass = separated.selected > 0 && separated.leaked === 0
  && touching.selected > 0 && touching.leaked === 0
  && narrowBackgroundBridge.dancer > 0 && narrowBackgroundBridge.backgroundObject === 0
  && lowConfidence.selected > 0 && lowConfidence.leaked === 0
  && recovery.firstMissingIsBlack
  && recovery.temporaryMissingRetained
  && recovery.expiredIsBlack
  && region.containsBox
  && region.cropped
  && temporal.oneFrameLeakRejected
  && temporal.confirmedSubjectAccepted
  && temporal.persistentSubjectRetained
  && promptedIdentity.fallbackUsesTorso
  && promptedIdentity.promptFollowsPreviousMask
  && promptedIdentity.newLimbAcceptedImmediately
  && promptedIdentity.farObjectExcludedByTrackedRegion
  && promptedIdentity.unrelatedMaskRejectedByOverlap
  && strictIdentity.mainSubjectRetained
  && strictIdentity.farPasserRejected
  && strictIdentity.sustainedPasserCannotCreepIn
  && strictIdentity.suddenGrowthWasTrimmed
  && strictIdentity.normalMotionRetained
  && strictIdentity.attachedThinLimbRetained
  && strictIdentity.weakFrameKeepsIdentityBaseline
  && edges.centerPreserved
  && edges.edgeFeathered
  && framing.largerSubjectUsesTighterCrop
  && framing.aspectPreserved
  && rowClones.totalIncludesMain
  && rowClones.equalWidths
  && rowClones.equalHeights
  && rowClones.insideCanvas
  && rowClones.plusOneMeansTwoPeople
  && rowClones.plusOneKeepsLargeSubjects
  && smoothBackdrop.keepsSmoothResolution
  && smoothBackdrop.avoidsTinyPixelBuffer
  && smoothBackdrop.reducesToColorField
  && smoothBackdrop.strongerUsesLowerFrequency
  && smoothBackdrop.strengthIncreasesRadius
  && smoothBackdrop.padsBlurredEdges
  && backdropSubjectRemoval.containsSubject
  && backdropSubjectRemoval.staysInsideBackdrop;

console.log(JSON.stringify({
  separated,
  touching,
  narrowBackgroundBridge,
  lowConfidence,
  recovery,
  region,
  temporal,
  promptedIdentity,
  strictIdentity,
  edges,
  framing,
  rowClones,
  smoothBackdrop,
  backdropSubjectRemoval,
  pass,
}, null, 2));
if (!pass) process.exitCode = 1;
