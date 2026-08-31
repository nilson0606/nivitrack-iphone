import assert from 'node:assert/strict';

import {
  bystanderHeadBoxes,
  expandFaceBox,
  headBoxesAt,
  isProtectedMainHead,
  mergeHeadBoxes,
  personBoxToHeadBox,
  selectMainFaceIndex,
  selectMainPersonIndex,
  smoothFaceBox,
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
  [125, 112, 50, 54],
  'a full person box must produce a stable upper-body head region even without a visible face',
);
const protectedMainHead = subjectHeadProtectionBox(subject, 500, 500);
assert.ok(
  protectedMainHead.every((value, index) => Math.abs(value - [92, 70, 116, 156][index]) < 1e-6),
  'the protected main-head zone must cover the tracked subject upper body',
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

console.log('Face-obscuring helper smoke tests passed.');
