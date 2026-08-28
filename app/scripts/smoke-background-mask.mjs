import {
  recoverTrackedSubjectAlpha,
  selectTrackedSubjectAlpha,
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

const separated = testSeparatedDancers();
const touching = testTouchingDancers();
const lowConfidence = testLowConfidenceDancer();
const recovery = testMissingFrameRecovery();
const region = testTrackedRegion();
const temporal = testTemporalStability();
const edges = testEdgeTightening();
const framing = testAdjustableFraming();
const pass = separated.selected > 0 && separated.leaked === 0
  && touching.selected > 0 && touching.leaked === 0
  && lowConfidence.selected > 0 && lowConfidence.leaked === 0
  && recovery.firstMissingIsBlack
  && recovery.temporaryMissingRetained
  && recovery.expiredIsBlack
  && region.containsBox
  && region.cropped
  && temporal.oneFrameLeakRejected
  && temporal.confirmedSubjectAccepted
  && temporal.persistentSubjectRetained
  && edges.centerPreserved
  && edges.edgeFeathered
  && framing.largerSubjectUsesTighterCrop
  && framing.aspectPreserved;

console.log(JSON.stringify({
  separated,
  touching,
  lowConfidence,
  recovery,
  region,
  temporal,
  edges,
  framing,
  pass,
}, null, 2));
if (!pass) process.exitCode = 1;
