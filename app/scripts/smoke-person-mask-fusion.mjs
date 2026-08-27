import assert from 'node:assert/strict';
import { fuseSubjectAndPersonMasks } from '../lib/person-mask-fusion.ts';

const width = 8;
const height = 4;
const subject = new Float32Array(width * height);
const person = new Float32Array(width * height);

for (let y = 0; y < height; y += 1) {
  for (let x = 0; x < width; x += 1) {
    const index = y * width + x;
    if (x <= 2) {
      subject[index] = 0.95;
      person[index] = 0.9;
    }
    if (x === 3 || x === 4) {
      subject[index] = 0.03;
      person[index] = 0.95;
    }
    if (x >= 5) {
      subject[index] = 0.9;
      person[index] = 0.03;
    }
  }
}

const fused = fuseSubjectAndPersonMasks(subject, width, height, person, width, height);
const personPixels = fused.filter((value, index) => index % width <= 2);
const bystanderPixels = fused.filter((value, index) => index % width === 3 || index % width === 4);
const fanPixels = fused.filter((value, index) => index % width >= 5);
assert.ok(personPixels.every((value) => value > 0.9), '人體應保留');
assert.ok(bystanderPixels.every((value) => value < 0.04), '非指定的其他人物應由主角遮罩排除');
assert.ok(fanPixels.every((value) => value === 0), '電風扇等非人體物件應排除');

const resized = fuseSubjectAndPersonMasks(
  new Float32Array([1, 1, 1, 1]),
  4,
  1,
  new Float32Array([1, 0]),
  2,
  1,
);
assert.deepEqual(Array.from(resized), [1, 1, 0, 0], '不同遮罩尺寸應正確對齊');

console.log('personKept', Math.min(...personPixels).toFixed(3));
console.log('bystanderRemoved', Math.max(...bystanderPixels).toFixed(3));
console.log('fanRemoved', Math.max(...fanPixels).toFixed(3));
console.log('resample', Array.from(resized).join(','));
