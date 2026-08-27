import assert from 'node:assert/strict';
import { excludePersistentBackground } from '../lib/temporal-background.ts';

const FLOW_WIDTH = 64;
const FLOW_HEIGHT = 64;
const WIDTH = 128;
const HEIGHT = 128;
const FRAME_COUNT = 28;

function texture(x, y) {
  const first = ((x * 29 + y * 17) % 91 + 91) % 91;
  const second = ((x * 7 - y * 13) % 47 + 47) % 47;
  return 54 + first + second;
}

function paintRectangle(values, width, left, top, right, bottom, value) {
  for (let y = Math.max(0, top); y < Math.min(values.length / width, bottom); y += 1) {
    for (let x = Math.max(0, left); x < Math.min(width, right); x += 1) {
      values[y * width + x] = value;
    }
  }
}

function paintCircle(values, width, height, centerX, centerY, radius, value) {
  const left = Math.max(0, Math.floor(centerX - radius));
  const right = Math.min(width - 1, Math.ceil(centerX + radius));
  const top = Math.max(0, Math.floor(centerY - radius));
  const bottom = Math.min(height - 1, Math.ceil(centerY + radius));
  for (let y = top; y <= bottom; y += 1) {
    for (let x = left; x <= right; x += 1) {
      if (Math.hypot(x - centerX, y - centerY) <= radius) values[y * width + x] = value;
    }
  }
}

function createScenario(trackedOutput) {
  const frames = [];
  const fanPositions = [];
  const subjectPositions = [];
  for (let frame = 0; frame < FRAME_COUNT; frame += 1) {
    const backgroundShift = trackedOutput ? Math.floor(frame / 2) : 0;
    const subjectX = trackedOutput ? 45 : 30 + frame;
    const fanX = trackedOutput ? 98 - backgroundShift * 2 : 98;
    const alpha = new Uint8ClampedArray(WIDTH * HEIGHT);
    const bodyCore = new Uint8Array(FLOW_WIDTH * FLOW_HEIGHT);
    const flowLuma = new Uint8Array(FLOW_WIDTH * FLOW_HEIGHT);

    paintRectangle(alpha, WIDTH, subjectX, 24, subjectX + 28, 108, 255);
    paintCircle(alpha, WIDTH, HEIGHT, fanX, 56, 17, 235);
    paintRectangle(
      bodyCore,
      FLOW_WIDTH,
      Math.floor(subjectX / 2) - 2,
      9,
      Math.ceil((subjectX + 28) / 2) + 2,
      57,
      255,
    );

    for (let y = 0; y < FLOW_HEIGHT; y += 1) {
      for (let x = 0; x < FLOW_WIDTH; x += 1) {
        const sourceX = x + backgroundShift;
        let value = texture(sourceX, y);
        const fullX = x * 2;
        const fullY = y * 2;
        if (
          fullX >= subjectX
          && fullX < subjectX + 28
          && fullY >= 24
          && fullY < 108
        ) {
          value = 185 + ((x + y + frame) % 24);
        }
        flowLuma[y * FLOW_WIDTH + x] = value;
      }
    }
    frames.push({ alpha, width: WIDTH, height: HEIGHT, flowLuma, bodyCore });
    fanPositions.push(fanX);
    subjectPositions.push(subjectX);
  }
  return { frames, fanPositions, subjectPositions };
}

function averageArea(frames, positions, kind) {
  let sum = 0;
  let samples = 0;
  for (let frame = 0; frame < frames.length; frame += 1) {
    if (kind === 'subject') {
      const left = positions[frame] + 8;
      for (let y = 35; y < 98; y += 1) {
        for (let x = left; x < left + 12; x += 1) {
          sum += frames[frame].alpha[y * WIDTH + x] / 255;
          samples += 1;
        }
      }
    } else {
      const centerX = positions[frame];
      for (let y = 45; y <= 67; y += 1) {
        for (let x = centerX - 11; x <= centerX + 11; x += 1) {
          if (Math.hypot(x - centerX, y - 56) > 11) continue;
          const coreX = Math.min(FLOW_WIDTH - 1, Math.floor(x / 2));
          const coreY = Math.min(FLOW_HEIGHT - 1, Math.floor(y / 2));
          if (frames[frame].bodyCore[coreY * FLOW_WIDTH + coreX] > 24) continue;
          sum += frames[frame].alpha[y * WIDTH + x] / 255;
          samples += 1;
        }
      }
    }
  }
  return sum / samples;
}

for (const trackedOutput of [false, true]) {
  const scenario = createScenario(trackedOutput);
  const removed = excludePersistentBackground(scenario.frames);
  const subject = averageArea(scenario.frames, scenario.subjectPositions, 'subject');
  const fan = averageArea(scenario.frames, scenario.fanPositions, 'fan');
  const label = trackedOutput ? 'moving-background' : 'fixed-background';
  console.log(label, 'subject', subject.toFixed(3), 'background-object', fan.toFixed(3), 'removed', removed);
  assert.ok(subject > 0.9, label + ' must preserve the protected subject core');
  assert.ok(fan < 0.25, label + ' must suppress a persistent object outside the body core');
}
