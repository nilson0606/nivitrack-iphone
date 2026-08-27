import assert from 'node:assert/strict';
import {
  applyOriginalBackgroundSuppression,
  buildOriginalBackgroundModel,
  createOriginalBackgroundSuppression,
} from '../lib/original-background.ts';

const width = 14;
const height = 8;
const samples = [];

function frame(subjectX, unstableValue) {
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const pixel = (y * width + x) * 4;
      rgba[pixel] = 42;
      rgba[pixel + 1] = 48;
      rgba[pixel + 2] = 52;
      rgba[pixel + 3] = 255;
    }
  }
  // Generic fixed background prop.
  for (let y = 2; y <= 5; y += 1) {
    for (let x = 11; x <= 12; x += 1) {
      const pixel = (y * width + x) * 4;
      rgba[pixel] = 28;
      rgba[pixel + 1] = 112;
      rgba[pixel + 2] = 168;
    }
  }
  // Moving selected subject.
  for (let y = 2; y <= 6; y += 1) {
    for (let x = subjectX; x <= subjectX + 1; x += 1) {
      const pixel = (y * width + x) * 4;
      rgba[pixel] = 205;
      rgba[pixel + 1] = 62;
      rgba[pixel + 2] = 54;
    }
  }
  const unstablePixel = (1 * width + 9) * 4;
  rgba[unstablePixel] = unstableValue;
  rgba[unstablePixel + 1] = 255 - unstableValue;
  rgba[unstablePixel + 2] = (unstableValue * 3) % 255;
  return rgba;
}

for (let index = 0; index < 12; index += 1) {
  const subjectX = 1 + (index % 7);
  samples.push({
    rgba: frame(subjectX, (index * 71) % 255),
    excluded: {
      x: Math.max(0, (subjectX - 0.5) / width),
      y: 1.5 / height,
      width: 3 / width,
      height: 6 / height,
    },
  });
}

const model = buildOriginalBackgroundModel(samples, width, height);
assert.ok(model);
const current = frame(7, 39);
const suppression = createOriginalBackgroundSuppression(model, current);
const alpha = new Uint8ClampedArray(width * height);
for (let y = 2; y <= 6; y += 1) {
  for (let x = 7; x <= 8; x += 1) alpha[y * width + x] = 255;
}
for (let y = 2; y <= 5; y += 1) {
  for (let x = 11; x <= 12; x += 1) alpha[y * width + x] = 255;
}
const bodyCore = new Uint8Array(width * height);
for (let y = 2; y <= 6; y += 1) {
  for (let x = 7; x <= 8; x += 1) bodyCore[y * width + x] = 255;
}
const unprotectedAlpha = new Uint8ClampedArray(alpha);

const removed = applyOriginalBackgroundSuppression(
  alpha,
  width,
  height,
  { x: 0, y: 0, width, height },
  width,
  height,
  suppression,
  width,
  height,
  bodyCore,
  width,
  height,
);
const subject = alpha[4 * width + 7];
const fixedObject = alpha[4 * width + 11];
const unstableConfidence = model.confidence[1 * width + 9];
applyOriginalBackgroundSuppression(
  unprotectedAlpha,
  width,
  height,
  { x: 0, y: 0, width, height },
  width,
  height,
  suppression,
  width,
  height,
  null,
  width,
  height,
);
const distinctMovingSubject = unprotectedAlpha[4 * width + 7];

const cropWidth = 7;
const cropHeight = 7;
const cropAlpha = new Uint8ClampedArray(cropWidth * cropHeight);
const cropCore = new Uint8Array(cropWidth * cropHeight);
for (let y = 1; y <= 5; y += 1) {
  for (let x = 1; x <= 2; x += 1) {
    cropAlpha[y * cropWidth + x] = 255;
    cropCore[y * cropWidth + x] = 255;
  }
}
for (let y = 1; y <= 4; y += 1) {
  for (let x = 5; x <= 6; x += 1) cropAlpha[y * cropWidth + x] = 255;
}
applyOriginalBackgroundSuppression(
  cropAlpha,
  cropWidth,
  cropHeight,
  { x: 6, y: 1, width: 7, height: 7 },
  width,
  height,
  suppression,
  width,
  height,
  cropCore,
  cropWidth,
  cropHeight,
);
const croppedSubject = cropAlpha[3 * cropWidth + 1];
const croppedFixedObject = cropAlpha[3 * cropWidth + 5];

console.log({
  subject,
  distinctMovingSubject,
  fixedObject,
  croppedSubject,
  croppedFixedObject,
  unstableConfidence,
  removed,
});
assert.ok(subject > 245, 'selected subject core must be preserved');
assert.ok(distinctMovingSubject > 245, 'moving foreground that differs from the plate must remain');
assert.ok(fixedObject < 24, 'stable background object must be removed');
assert.ok(croppedSubject > 245, 'subject must survive a moving source-region mapping');
assert.ok(croppedFixedObject < 24, 'fixed object must be removed through source-region mapping');
assert.ok(unstableConfidence < 96, 'unstable pixels must not become trusted background');
