import assert from 'node:assert/strict';

import {
  bystanderHeadBoxes,
  expandFaceBox,
  headBoxesAt,
  mergeHeadBoxes,
  personBoxToHeadBox,
  selectMainFaceIndex,
  selectMainPersonIndex,
  smoothFaceBox,
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
  [125, 103, 50, 54],
  'a full person box must produce a stable upper-body head region even without a visible face',
);
const bystanderHeads = bystanderHeadBoxes(people, [105, 105, 95, 295], 500, 500);
assert.equal(bystanderHeads.length, 1, 'only non-main people should produce inferred head masks');
assert.ok(bystanderHeads[0][0] > 300, 'the remaining inferred head must belong to the bystander');

assert.deepEqual(
  headBoxesAt([
    { time: 0, heads: [[10, 10, 20, 20]] },
    { time: 0.2, heads: [[20, 10, 20, 20]] },
  ], 0.11),
  [[20, 10, 20, 20]],
  'preview and export must use the nearest precomputed head sample',
);
assert.equal(
  mergeHeadBoxes([[10, 10, 30, 30], [14, 12, 26, 28]]).length,
  1,
  'face and person-derived head detections must merge instead of drawing duplicate masks',
);

console.log('Face-obscuring helper smoke tests passed.');
