import assert from 'node:assert/strict';
import { TemporalTargetMaskStabilizer } from '../lib/temporal-target-mask.ts';

const width = 24;
const height = 16;
const region = { x: 0, y: 0, width, height };
const trackedBox = [7, 3, 8, 10];

function rectangle(left, top, right, bottom, value = 255) {
  const alpha = new Uint8ClampedArray(width * height);
  for (let y = top; y <= bottom; y += 1) {
    for (let x = left; x <= right; x += 1) alpha[y * width + x] = value;
  }
  return alpha;
}

function count(alpha, left, top, right, bottom, threshold = 24) {
  let total = 0;
  for (let y = top; y <= bottom; y += 1) {
    for (let x = left; x <= right; x += 1) total += alpha[y * width + x] >= threshold ? 1 : 0;
  }
  return total;
}

const stabilizer = new TemporalTargetMaskStabilizer();
const seed = stabilizer.stabilize(rectangle(8, 4, 13, 12), width, height, region, trackedBox);
assert.equal(count(seed, 8, 4, 13, 12), 54, '乾淨選角幀應完整建立主角');

const attachedObject = rectangle(8, 4, 22, 12);
const guarded = stabilizer.stabilize(attachedObject, width, height, region, trackedBox);
assert.equal(count(guarded, 19, 4, 22, 12), 0, '突然黏到主角的大型物體不可整塊進入');
assert.ok(count(guarded, 8, 4, 13, 12) >= 54, '主角本體必須保留');

const missedFrame = stabilizer.stabilize(new Uint8ClampedArray(width * height), width, height, region, trackedBox);
assert.ok(count(missedFrame, 8, 4, 13, 12) >= 54, '模型漏掉一格時主角不可整體閃退');

stabilizer.reset(seed);
const movingHand = new Uint8ClampedArray(seed);
movingHand[7 * width + 14] = 255;
movingHand[7 * width + 15] = 255;
const grown = stabilizer.stabilize(movingHand, width, height, region, trackedBox);
assert.ok(grown[7 * width + 14] > 0 && grown[7 * width + 15] > 0, '鄰近肢體動作仍可延伸');

console.log(JSON.stringify({
  subjectPixels: count(guarded, 8, 4, 13, 12),
  rejectedFarObjectPixels: count(guarded, 19, 4, 22, 12),
  missedFrameSubjectPixels: count(missedFrame, 8, 4, 13, 12),
}, null, 2));
