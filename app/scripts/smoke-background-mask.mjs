import { selectTrackedSubjectAlpha } from '../lib/person-background-removal.ts';

const width = 20;
const height = 10;
const trackedBox = [5, 5, 30, 40];

function countSides(alpha) {
  let selected = 0;
  let leaked = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if ((alpha?.[y * width + x] ?? 0) <= 0.1) continue;
      if (x <= 8) selected += 1;
      else leaked += 1;
    }
  }
  return { selected, leaked };
}

function testSeparatedDancers() {
  const mask = new Float32Array(width * height);
  for (let y = 2; y <= 7; y += 1) {
    for (let x = 2; x <= 6; x += 1) mask[y * width + x] = 0.9;
    for (let x = 12; x <= 16; x += 1) mask[y * width + x] = 0.95;
  }
  return countSides(selectTrackedSubjectAlpha(mask, width, height, trackedBox, 100, 50));
}

function testTouchingDancers() {
  const mask = new Float32Array(width * height);
  for (let y = 2; y <= 7; y += 1) {
    for (let x = 2; x <= 16; x += 1) mask[y * width + x] = 0.9;
  }
  return countSides(selectTrackedSubjectAlpha(mask, width, height, trackedBox, 100, 50));
}

const separated = testSeparatedDancers();
const touching = testTouchingDancers();
const pass = separated.selected > 0 && separated.leaked === 0
  && touching.selected > 0 && touching.leaked === 0;

console.log(JSON.stringify({ separated, touching, pass }, null, 2));
if (!pass) process.exitCode = 1;
