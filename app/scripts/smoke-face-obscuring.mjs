import assert from 'node:assert/strict';

import {
  expandFaceBox,
  selectMainFaceIndex,
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

console.log('Face-obscuring helper smoke tests passed.');
