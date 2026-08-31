import assert from 'node:assert/strict';

import {
  bystanderFaceBoxes,
  bystanderHeadBoxes,
  createMainHeadTrackingContext,
  expandFaceBox,
  headBoxFromTrackingContext,
  headBoxesAt,
  isProtectedMainHead,
  mergeHeadBoxes,
  personBoxToHeadBox,
  selectMainFaceIndex,
  selectMainPersonIndex,
  smoothFaceBox,
  stabilizeHeadDetectionFrames,
  suppressFaceSupportedFallbacks,
  subjectHeadProtectionBox,
} from '../lib/face-obscuring.ts';

const subject = [100, 100, 100, 300];
const mainFace = [135, 115, 30, 30];
const bystanderFace = [260, 105, 32, 32];

assert.equal(
  selectMainFaceIndex([mainFace, bystanderFace], subject, null, true),
  0,
  'the face in the tracked subject head zone must be preserved',
);

const selectedMainHead = [134, 108, 34, 38];
const mainHeadTrackingContext = createMainHeadTrackingContext(selectedMainHead, 500, 500);
assert.ok(
  mainHeadTrackingContext.trackerBox[2] > selectedMainHead[2] * 2,
  'feature 13 must give ViT shoulder context instead of tracking only a tiny interchangeable head',
);
assert.ok(
  mainHeadTrackingContext.trackerBox[3] > selectedMainHead[3] * 4,
  'feature 13 identity tracking must include enough upper-body appearance to survive crossings',
);
assert.ok(
  headBoxFromTrackingContext(
    mainHeadTrackingContext.trackerBox,
    mainHeadTrackingContext,
    500,
    500,
  ).every((value, index) => Math.abs(value - selectedMainHead[index]) < 1e-6),
  'the hidden identity context must map back to the exact user-selected main head',
);
const movedMainHead = headBoxFromTrackingContext(
  [
    mainHeadTrackingContext.trackerBox[0] + 40,
    mainHeadTrackingContext.trackerBox[1] + 15,
    mainHeadTrackingContext.trackerBox[2],
    mainHeadTrackingContext.trackerBox[3],
  ],
  mainHeadTrackingContext,
  500,
  500,
);
assert.ok(
  movedMainHead.every(
    (value, index) =>
      Math.abs(value - [
        selectedMainHead[0] + 40,
        selectedMainHead[1] + 15,
        selectedMainHead[2],
        selectedMainHead[3],
      ][index]) < 1e-6,
  ),
  'only the tracked identity context may move the protected main-head path',
);
assert.equal(
  selectMainFaceIndex([[136, 110, 30, 34], bystanderFace], selectedMainHead, null, false),
  0,
  'feature 13 must identify the main face from a head-sized ViT selection',
);
assert.deepEqual(
  bystanderFaceBoxes(
    [[136, 110, 30, 34], bystanderFace],
    selectedMainHead,
    500,
    500,
  ),
  [bystanderFace],
  'feature 13 must keep direct bystander face boxes and exclude only the selected main face',
);
assert.equal(
  isProtectedMainHead([188, 110, 30, 34], selectedMainHead, 500, 500),
  false,
  'a head-sized main selection must not protect the adjacent bystander',
);

assert.equal(
  selectMainFaceIndex([bystanderFace], subject, null, true),
  null,
  'a face outside the tracked subject must never be treated as the main face',
);

assert.equal(
  selectMainFaceIndex(
    [[170, 120, 30, 30], [195, 120, 30, 30]],
    [100, 100, 200, 300],
    null,
    true,
  ),
  null,
  'privacy-first mode must mask all faces when the main face is ambiguous',
);

assert.equal(
  selectMainFaceIndex(
    [[144, 119, 30, 30], [180, 120, 30, 30]],
    [100, 100, 160, 300],
    [140, 118, 30, 30],
    true,
  ),
  0,
  'the previous main face must keep continuity while people cross',
);

assert.deepEqual(
  expandFaceBox([2, 3, 40, 50], 1.8, 100, 100),
  [0, 0, 72, 95.4],
  'expanded masks must cover the face and remain clamped to the source frame',
);

const smoothed = smoothFaceBox([0, 0, 20, 20], [10, 20, 30, 40]);
assert.ok(smoothed[0] > 6 && smoothed[0] < 8, 'face tracking should move toward the new detection');
assert.ok(smoothed[1] > 12 && smoothed[1] < 15, 'face tracking should smooth vertical movement');
assert.deepEqual(
  smoothFaceBox([1, 2, 3, 4], [40, 50, 60, 70], 1),
  [1, 2, 3, 4],
  'a short missed detection must retain the previous mask instead of flickering',
);

const people = [
  [100, 100, 100, 300],
  [300, 90, 90, 310],
];
assert.equal(
  selectMainPersonIndex(people, [105, 105, 95, 295]),
  0,
  'the person matching the ViT subject must be excluded from bystander masks',
);
assert.deepEqual(
  personBoxToHeadBox(people[0], 500, 500),
  [128, 110.5, 44, 47.52],
  'a full person box must produce a stable upper-body head region even without a visible face',
);
const protectedMainHead = subjectHeadProtectionBox(subject, 500, 500);
assert.ok(
  protectedMainHead.every(
    (value, index) => Math.abs(value - [117, 100.0456, 66, 81.7344][index]) < 1e-6,
  ),
  'the protected main-head zone must cover the selected head without sheltering nearby bystanders',
);
assert.equal(
  isProtectedMainHead([130, 110, 35, 35], subject, 500, 500),
  true,
  'a mask in the tracked subject head zone must be rejected',
);
assert.equal(
  isProtectedMainHead([300, 100, 35, 35], subject, 500, 500),
  false,
  'a bystander head away from the subject must remain maskable',
);
assert.equal(
  isProtectedMainHead([188, 110, 35, 35], subject, 500, 500),
  false,
  'a head beside the main subject must no longer be excluded by an oversized protection zone',
);
const bystanderHeads = bystanderHeadBoxes(people, [105, 105, 95, 295], 500, 500);
assert.equal(bystanderHeads.length, 1, 'only non-main people should produce inferred head masks');
assert.ok(bystanderHeads[0][0] > 300, 'the remaining inferred head must belong to the bystander');
assert.equal(
  selectMainPersonIndex([[330, 90, 80, 300]], [100, 100, 100, 300]),
  null,
  'a nearby bystander must not be mistaken for a missing main-person detection',
);
assert.equal(
  bystanderHeadBoxes([
    [100, 100, 100, 300],
    [20, 10, 420, 470],
  ], [100, 100, 100, 300], 500, 500).length,
  0,
  'a group-sized person box must not create a giant head mask',
);

assert.deepEqual(
  headBoxesAt([
    { time: 0, heads: [[10, 10, 20, 20]] },
    { time: 0.2, heads: [[20, 10, 20, 20]] },
  ], 0.11),
  [[20, 10, 20, 20]],
  'preview and export must use the nearest precomputed head sample',
);
assert.deepEqual(
  mergeHeadBoxes([[10, 10, 30, 30], [14, 12, 26, 28]]),
  [[10, 10, 30, 30]],
  'duplicate detections must keep the higher-priority box without growing it',
);
assert.equal(
  mergeHeadBoxes([
    [10, 10, 30, 30],
    [14, 12, 26, 28],
    [43, 10, 30, 30],
  ]).length,
  2,
  'deduplication must not chain adjacent dancers into one giant mask',
);
assert.deepEqual(
  suppressFaceSupportedFallbacks(
    [[100, 100, 30, 30]],
    [[93, 49, 44, 50], [250, 100, 30, 30]],
  ),
  [[250, 100, 30, 30]],
  'a raised-hand fallback above a detected face must be removed without affecting another person',
);

const stabilizedMissingFrame = stabilizeHeadDetectionFrames([
  { time: 0, heads: [[10, 10, 20, 20]] },
  { time: 0.2, heads: [[13, 10, 20, 20]] },
  { time: 0.4, heads: [] },
  { time: 0.6, heads: [[19, 10, 20, 20]] },
]);
assert.equal(
  stabilizedMissingFrame[2].heads.length,
  1,
  'a confirmed small head must survive one missed 5 fps sample without flickering',
);
assert.ok(
  stabilizedMissingFrame[3].heads[0][0] > stabilizedMissingFrame[1].heads[0][0],
  'a recovered moving head must continue along its temporal track',
);

const raisedArmSpike = stabilizeHeadDetectionFrames([
  { time: 0, heads: [[100, 100, 30, 34]] },
  { time: 0.2, heads: [[102, 102, 30, 34]] },
  { time: 0.4, heads: [[103, 48, 48, 54]] },
]);
assert.equal(raisedArmSpike[2].heads.length, 1, 'a raised-arm spike must not create a second floating mask');
assert.ok(
  raisedArmSpike[2].heads[0][1] > 80,
  'person-box fallback must limit sudden upward jumps caused by raised arms',
);

const expiredTrack = stabilizeHeadDetectionFrames([
  { time: 0, heads: [[10, 10, 20, 20]] },
  { time: 0.2, heads: [[10, 10, 20, 20]] },
  { time: 0.4, heads: [] },
  { time: 0.8, heads: [] },
]);
assert.equal(expiredTrack[3].heads.length, 0, 'a missing person must expire instead of leaving a floating mask');

const adjacentTracks = stabilizeHeadDetectionFrames([
  { time: 0, heads: [[10, 10, 20, 20], [42, 10, 20, 20]] },
  { time: 0.2, heads: [[13, 10, 20, 20], [39, 10, 20, 20]] },
  { time: 0.4, heads: [[16, 10, 20, 20], [36, 10, 20, 20]] },
]);
assert.equal(adjacentTracks[2].heads.length, 2, 'nearby dancers must keep separate head tracks');

console.log('Face-obscuring helper smoke tests passed.');
