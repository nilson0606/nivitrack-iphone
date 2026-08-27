import assert from 'node:assert/strict';
import { TrackedPersonMaskSelector } from '../lib/tracked-person-mask.ts';

const width = 16;
const height = 8;
const values = new Float32Array(width * height);

for (let y = 2; y <= 5; y += 1) {
  for (let x = 1; x <= 4; x += 1) values[y * width + x] = 0.95;
  for (let x = 11; x <= 14; x += 1) values[y * width + x] = 0.95;
}

function count(gate, fromX, toX) {
  let selected = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = fromX; x <= toX; x += 1) selected += gate[y * width + x] ? 1 : 0;
  }
  return selected;
}

const selector = new TrackedPersonMaskSelector();
const region = { x: 0, y: 0, width, height };
let gate = selector.select(values, width, height, region, [0, 1, 6, 6]);
const result = {
  boxLeftSelectLeft: count(gate, 0, 7),
  boxLeftRejectRight: count(gate, 9, 15),
  boxRightRejectLeft: 0,
  boxRightSelectRight: 0,
};

gate = selector.select(values, width, height, region, [10, 1, 6, 6]);
result.boxRightRejectLeft = count(gate, 0, 6);
result.boxRightSelectRight = count(gate, 8, 15);

assert.ok(result.boxLeftSelectLeft > 0);
assert.equal(result.boxLeftRejectRight, 0);
assert.equal(result.boxRightRejectLeft, 0);
assert.ok(result.boxRightSelectRight > 0);
console.log(JSON.stringify(result, null, 2));
