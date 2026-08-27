import assert from 'node:assert/strict';
import { estimateMotion, stabilizeAlpha } from '../lib/temporal-mask.ts';

const width = 64;
const height = 64;
const shift = 3;
const previousLuma = new Uint8Array(width * height).fill(24);
const currentLuma = new Uint8Array(width * height).fill(24);
const previousAlpha = new Uint8ClampedArray(width * height);
const currentAlpha = new Uint8ClampedArray(width * height);

for (let y = 14; y <= 49; y += 1) {
  for (let x = 18; x <= 43; x += 1) {
    const texture = 92 + ((x * 17 + y * 29 + x * y) % 142);
    previousLuma[y * width + x] = texture;
    previousAlpha[y * width + x] = 255;
    currentLuma[y * width + x + shift] = texture;
    currentAlpha[y * width + x + shift] = 255;
  }
}

for (let y = 26; y <= 34; y += 1) {
  currentAlpha[y * width + 33] = 0;
}

const motion = estimateMotion(previousLuma, currentLuma);
assert.ok(motion, '應建立光流欄位');
const centerCell = 4 * motion.columns + 4;
assert.ok(motion.dx[centerCell] <= -2 && motion.dx[centerCell] >= -4, '人物紋理應向前一格對齊');
assert.ok(motion.confidence[centerCell] > 0.35, '人物紋理區應有足夠光流信心');

const stable = stabilizeAlpha(
  currentAlpha,
  previousAlpha,
  currentLuma,
  previousLuma,
  width,
  height,
  0.58,
);

assert.ok(stable[30 * width + 33] > 45, '短暫遮罩破洞應由對齊後前幀補強');
assert.ok(stable[5 * width + 5] < 8, '遠離人物的背景不可被前幀污染');
assert.ok(stable[30 * width + 18] < 90, '人物移動後的舊邊緣不可形成明顯拖影');

const missingFirstFrame = new Uint8ClampedArray(width * height);
const backwardRecovered = stabilizeAlpha(
  missingFirstFrame,
  previousAlpha,
  previousLuma,
  previousLuma,
  width,
  height,
  0.44,
);
assert.ok(
  backwardRecovered[30 * width + 30] > 70,
  '後一格已有可信人物時，反向穩定應補回開頭失敗幀',
);
assert.ok(backwardRecovered[5 * width + 5] < 8, '反向補幀不可污染遠端背景');

console.log('motionDx', motion.dx[centerCell].toFixed(1));
console.log('motionConfidence', motion.confidence[centerCell].toFixed(3));
console.log('filledGap', stable[30 * width + 33]);
console.log('oldEdge', stable[30 * width + 18]);
console.log('backwardRecovered', backwardRecovered[30 * width + 30]);
