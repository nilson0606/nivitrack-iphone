import assert from 'node:assert/strict';
import { createBodyCoreSupport } from '../lib/magic-pose-matte.ts';

const points = Array.from(
  { length: 33 },
  () => ({ x: 0.5, y: 0.5, visibility: 0, presence: 0 }),
);
function point(index, x, y) {
  points[index] = { x, y, visibility: 1, presence: 1 };
}

point(0, 0.5, 0.16);
point(7, 0.46, 0.17);
point(8, 0.54, 0.17);
point(11, 0.4, 0.3);
point(12, 0.6, 0.3);
point(13, 0.34, 0.43);
point(14, 0.66, 0.43);
point(15, 0.28, 0.55);
point(16, 0.72, 0.55);
point(17, 0.27, 0.57);
point(18, 0.73, 0.57);
point(19, 0.28, 0.57);
point(20, 0.72, 0.57);
point(21, 0.29, 0.56);
point(22, 0.71, 0.56);
point(23, 0.44, 0.56);
point(24, 0.56, 0.56);
point(25, 0.43, 0.73);
point(26, 0.57, 0.73);
point(27, 0.42, 0.9);
point(28, 0.58, 0.9);
point(29, 0.41, 0.93);
point(30, 0.59, 0.93);
point(31, 0.44, 0.95);
point(32, 0.56, 0.95);

const width = 128;
const height = 128;
const support = createBodyCoreSupport(points, width, height);
assert.ok(support);
const sample = (x, y) => support[y * width + x];
console.log(
  'head', sample(64, 22),
  'torso', sample(64, 55),
  'hand', sample(36, 71),
  'foot', sample(56, 120),
  'outside', sample(116, 54),
);
assert.ok(sample(64, 22) > 180);
assert.ok(sample(64, 55) > 220);
assert.ok(sample(36, 71) > 100);
assert.ok(sample(56, 120) > 100);
assert.equal(sample(116, 54), 0);
