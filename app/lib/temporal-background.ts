import {
  FLOW_HEIGHT,
  FLOW_WIDTH,
  estimateMotion,
  sampleMotion,
} from './temporal-mask.ts';

export type TemporalBackgroundFrame = {
  alpha: Uint8ClampedArray;
  width: number;
  height: number;
  flowLuma: Uint8Array;
  bodyCore: Uint8Array | null;
};

const GRID_SIZE = FLOW_WIDTH * FLOW_HEIGHT;

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function smoothStep(value: number, minimum: number, maximum: number) {
  const normalized = clamp((value - minimum) / Math.max(0.000001, maximum - minimum), 0, 1);
  return normalized * normalized * (3 - 2 * normalized);
}

function sampleGrid(
  values: Float32Array | Uint8Array,
  x: number,
  y: number,
) {
  const atX = clamp(x, 0, FLOW_WIDTH - 1);
  const atY = clamp(y, 0, FLOW_HEIGHT - 1);
  const left = Math.floor(atX);
  const top = Math.floor(atY);
  const right = Math.min(FLOW_WIDTH - 1, left + 1);
  const bottom = Math.min(FLOW_HEIGHT - 1, top + 1);
  const mixX = atX - left;
  const mixY = atY - top;
  const above = values[top * FLOW_WIDTH + left] * (1 - mixX)
    + values[top * FLOW_WIDTH + right] * mixX;
  const below = values[bottom * FLOW_WIDTH + left] * (1 - mixX)
    + values[bottom * FLOW_WIDTH + right] * mixX;
  return above * (1 - mixY) + below * mixY;
}

function sampleMaskAtGrid(
  values: Uint8ClampedArray,
  width: number,
  height: number,
  gridX: number,
  gridY: number,
) {
  const x = Math.min(width - 1, Math.floor(((gridX + 0.5) / FLOW_WIDTH) * width));
  const y = Math.min(height - 1, Math.floor(((gridY + 0.5) / FLOW_HEIGHT) * height));
  return values[y * width + x] / 255;
}

function scanEvidence(frames: TemporalBackgroundFrame[], order: number[]) {
  const scores = frames.map(() => new Uint8Array(GRID_SIZE));
  let previousEvidence = new Float32Array(GRID_SIZE);
  let previousLuma: Uint8Array | null = null;

  for (const frameIndex of order) {
    const frame = frames[frameIndex];
    const score = scores[frameIndex];
    if (
      frame.flowLuma.length !== GRID_SIZE
      || !frame.bodyCore
      || frame.bodyCore.length !== GRID_SIZE
    ) {
      previousEvidence = new Float32Array(GRID_SIZE);
      previousLuma = frame.flowLuma.length === GRID_SIZE ? frame.flowLuma : null;
      continue;
    }

    const motion = previousLuma ? estimateMotion(previousLuma, frame.flowLuma) : null;
    const currentEvidence = new Float32Array(GRID_SIZE);
    for (let y = 0; y < FLOW_HEIGHT; y += 1) {
      for (let x = 0; x < FLOW_WIDTH; x += 1) {
        const index = y * FLOW_WIDTH + x;
        const core = frame.bodyCore[index] / 255;
        const alpha = sampleMaskAtGrid(frame.alpha, frame.width, frame.height, x, y);
        const flow = motion
          ? sampleMotion(motion, x + 0.5, y + 0.5)
          : { dx: 0, dy: 0, confidence: 0 };
        const previousX = x + flow.dx;
        const previousY = y + flow.dy;
        const carried = motion
          ? sampleGrid(previousEvidence, previousX, previousY)
          : 0;
        const lumaDifference = previousLuma
          ? Math.abs(frame.flowLuma[index] - sampleGrid(previousLuma, previousX, previousY))
          : 255;
        const appearanceContinuity = 1 - smoothStep(lumaDifference, 5, 30);
        const motionConfidence = motion ? 0.3 + flow.confidence * 0.7 : 0;
        const alphaEvidence = smoothStep(alpha, 0.08, 0.5);
        const outsideCore = 1 - smoothStep(core, 0.12, 0.62);
        const candidate = appearanceContinuity
          * motionConfidence
          * alphaEvidence
          * outsideCore;

        // Background evidence must outlive brief segmentation dropouts. Otherwise
        // a persistent prop can disappear from the matte for a frame, lose its
        // history, then visibly stick to the subject when it is detected again.
        let evidence = carried * (0.94 + flow.confidence * 0.045);
        if (candidate > 0.06) {
          evidence += 0.11 * candidate;
        } else {
          evidence -= core > 0.2 ? 0.2 : 0.025;
        }
        if (core > 0.58) evidence *= 0.32;
        currentEvidence[index] = clamp(evidence, 0, 1);
        score[index] = Math.round(currentEvidence[index] * 255);
      }
    }
    previousEvidence = currentEvidence;
    previousLuma = frame.flowLuma;
  }
  return scores;
}

type UpsampleMap = {
  left: Uint16Array;
  right: Uint16Array;
  mixX: Float32Array;
  top: Uint16Array;
  bottom: Uint16Array;
  mixY: Float32Array;
};

function createUpsampleMap(width: number, height: number): UpsampleMap {
  const left = new Uint16Array(width);
  const right = new Uint16Array(width);
  const mixX = new Float32Array(width);
  for (let x = 0; x < width; x += 1) {
    const gridX = clamp(((x + 0.5) / width) * FLOW_WIDTH - 0.5, 0, FLOW_WIDTH - 1);
    left[x] = Math.floor(gridX);
    right[x] = Math.min(FLOW_WIDTH - 1, left[x] + 1);
    mixX[x] = gridX - left[x];
  }
  const top = new Uint16Array(height);
  const bottom = new Uint16Array(height);
  const mixY = new Float32Array(height);
  for (let y = 0; y < height; y += 1) {
    const gridY = clamp(((y + 0.5) / height) * FLOW_HEIGHT - 0.5, 0, FLOW_HEIGHT - 1);
    top[y] = Math.floor(gridY);
    bottom[y] = Math.min(FLOW_HEIGHT - 1, top[y] + 1);
    mixY[y] = gridY - top[y];
  }
  return { left, right, mixX, top, bottom, mixY };
}

function applySuppression(
  alpha: Uint8ClampedArray,
  width: number,
  height: number,
  suppression: Uint8Array,
  mapping: UpsampleMap,
) {
  let removed = 0;
  for (let y = 0; y < height; y += 1) {
    const topRow = mapping.top[y] * FLOW_WIDTH;
    const bottomRow = mapping.bottom[y] * FLOW_WIDTH;
    const mixY = mapping.mixY[y];
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      const before = alpha[index];
      if (before === 0) continue;
      const mixX = mapping.mixX[x];
      const above = suppression[topRow + mapping.left[x]] * (1 - mixX)
        + suppression[topRow + mapping.right[x]] * mixX;
      const below = suppression[bottomRow + mapping.left[x]] * (1 - mixX)
        + suppression[bottomRow + mapping.right[x]] * mixX;
      const amount = (above * (1 - mixY) + below * mixY) / 255;
      alpha[index] = Math.round(before * (1 - amount));
      if (alpha[index] + 24 < before) removed += 1;
    }
  }
  return removed;
}

export function excludePersistentBackground(frames: TemporalBackgroundFrame[]) {
  if (frames.length < 4) return 0;
  const forwardOrder = frames.map((_, index) => index);
  const backwardOrder = [...forwardOrder].reverse();
  const forward = scanEvidence(frames, forwardOrder);
  const backward = scanEvidence(frames, backwardOrder);
  const upsampleMaps = new Map<string, UpsampleMap>();
  let removed = 0;

  for (let frameIndex = 0; frameIndex < frames.length; frameIndex += 1) {
    const frame = frames[frameIndex];
    const bodyCore = frame.bodyCore;
    if (!bodyCore || bodyCore.length !== GRID_SIZE) continue;
    const suppression = new Uint8Array(GRID_SIZE);
    for (let index = 0; index < GRID_SIZE; index += 1) {
        const forwardScore = forward[frameIndex][index] / 255;
        const backwardScore = backward[frameIndex][index] / 255;
        const persistence = Math.max(forwardScore, backwardScore) * 0.72
          + Math.min(forwardScore, backwardScore) * 0.28;
        const core = bodyCore[index] / 255;
        const outsideCore = 1 - smoothStep(core, 0.1, 0.68);
        const amount = smoothStep(persistence, 0.34, 0.76) * outsideCore * 0.97;
        suppression[index] = Math.round(amount * 255);
    }
    const mapKey = frame.width + 'x' + frame.height;
    let mapping = upsampleMaps.get(mapKey);
    if (!mapping) {
      mapping = createUpsampleMap(frame.width, frame.height);
      upsampleMaps.set(mapKey, mapping);
    }
    removed += applySuppression(
      frame.alpha,
      frame.width,
      frame.height,
      suppression,
      mapping,
    );
  }
  return removed;
}
