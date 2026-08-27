export const FLOW_WIDTH = 64;
export const FLOW_HEIGHT = 64;

const FLOW_CELL = 8;
const FLOW_PATCH_RADIUS = 2;
const FLOW_SEARCH_RADIUS = 4;

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

export type MotionField = {
  columns: number;
  rows: number;
  dx: Float32Array;
  dy: Float32Array;
  confidence: Float32Array;
};

export function estimateMotion(previous: Uint8Array, current: Uint8Array): MotionField | null {
  if (previous.length !== FLOW_WIDTH * FLOW_HEIGHT || current.length !== previous.length) return null;
  const columns = Math.ceil(FLOW_WIDTH / FLOW_CELL);
  const rows = Math.ceil(FLOW_HEIGHT / FLOW_CELL);
  const dx = new Float32Array(columns * rows);
  const dy = new Float32Array(columns * rows);
  const confidence = new Float32Array(columns * rows);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const centerX = Math.min(FLOW_WIDTH - 1, column * FLOW_CELL + Math.floor(FLOW_CELL / 2));
      const centerY = Math.min(FLOW_HEIGHT - 1, row * FLOW_CELL + Math.floor(FLOW_CELL / 2));
      let patchMinimum = 255;
      let patchMaximum = 0;
      for (let patchY = -FLOW_PATCH_RADIUS; patchY <= FLOW_PATCH_RADIUS; patchY += 1) {
        const atY = centerY + patchY;
        if (atY < 0 || atY >= FLOW_HEIGHT) continue;
        for (let patchX = -FLOW_PATCH_RADIUS; patchX <= FLOW_PATCH_RADIUS; patchX += 1) {
          const atX = centerX + patchX;
          if (atX < 0 || atX >= FLOW_WIDTH) continue;
          const value = current[atY * FLOW_WIDTH + atX];
          patchMinimum = Math.min(patchMinimum, value);
          patchMaximum = Math.max(patchMaximum, value);
        }
      }
      let bestScore = Number.POSITIVE_INFINITY;
      let bestX = 0;
      let bestY = 0;
      for (let moveY = -FLOW_SEARCH_RADIUS; moveY <= FLOW_SEARCH_RADIUS; moveY += 1) {
        for (let moveX = -FLOW_SEARCH_RADIUS; moveX <= FLOW_SEARCH_RADIUS; moveX += 1) {
          let difference = 0;
          let samples = 0;
          for (let patchY = -FLOW_PATCH_RADIUS; patchY <= FLOW_PATCH_RADIUS; patchY += 1) {
            const currentY = centerY + patchY;
            const previousY = currentY + moveY;
            if (currentY < 0 || currentY >= FLOW_HEIGHT || previousY < 0 || previousY >= FLOW_HEIGHT) continue;
            for (let patchX = -FLOW_PATCH_RADIUS; patchX <= FLOW_PATCH_RADIUS; patchX += 1) {
              const currentX = centerX + patchX;
              const previousX = currentX + moveX;
              if (currentX < 0 || currentX >= FLOW_WIDTH || previousX < 0 || previousX >= FLOW_WIDTH) continue;
              difference += Math.abs(
                current[currentY * FLOW_WIDTH + currentX]
                - previous[previousY * FLOW_WIDTH + previousX],
              );
              samples += 1;
            }
          }
          if (samples === 0) continue;
          const score = difference / samples + (Math.abs(moveX) + Math.abs(moveY)) * 0.3;
          if (score < bestScore) {
            bestScore = score;
            bestX = moveX;
            bestY = moveY;
          }
        }
      }
      const index = row * columns + column;
      dx[index] = bestX;
      dy[index] = bestY;
      const matchConfidence = clamp((34 - bestScore) / 28, 0, 1);
      const textureConfidence = clamp((patchMaximum - patchMinimum) / 42, 0, 1);
      confidence[index] = matchConfidence * textureConfidence;
    }
  }
  return { columns, rows, dx, dy, confidence };
}

function sampleField(values: Float32Array, field: MotionField, x: number, y: number) {
  const gridX = clamp(x / FLOW_CELL - 0.5, 0, field.columns - 1);
  const gridY = clamp(y / FLOW_CELL - 0.5, 0, field.rows - 1);
  const left = Math.floor(gridX);
  const top = Math.floor(gridY);
  const right = Math.min(field.columns - 1, left + 1);
  const bottom = Math.min(field.rows - 1, top + 1);
  const mixX = gridX - left;
  const mixY = gridY - top;
  const topValue = values[top * field.columns + left] * (1 - mixX)
    + values[top * field.columns + right] * mixX;
  const bottomValue = values[bottom * field.columns + left] * (1 - mixX)
    + values[bottom * field.columns + right] * mixX;
  return topValue * (1 - mixY) + bottomValue * mixY;
}

export function sampleMotion(field: MotionField, x: number, y: number) {
  return {
    dx: sampleField(field.dx, field, x, y),
    dy: sampleField(field.dy, field, x, y),
    confidence: sampleField(field.confidence, field, x, y),
  };
}

function sampleAlpha(alpha: Uint8ClampedArray, width: number, height: number, x: number, y: number) {
  const atX = clamp(x, 0, width - 1);
  const atY = clamp(y, 0, height - 1);
  const left = Math.floor(atX);
  const top = Math.floor(atY);
  const right = Math.min(width - 1, left + 1);
  const bottom = Math.min(height - 1, top + 1);
  const mixX = atX - left;
  const mixY = atY - top;
  const topValue = alpha[top * width + left] * (1 - mixX) + alpha[top * width + right] * mixX;
  const bottomValue = alpha[bottom * width + left] * (1 - mixX) + alpha[bottom * width + right] * mixX;
  return (topValue * (1 - mixY) + bottomValue * mixY) / 255;
}

export function stabilizeAlpha(
  current: Uint8ClampedArray,
  previous: Uint8ClampedArray | null,
  currentLuma: Uint8Array,
  previousLuma: Uint8Array | null,
  width: number,
  height: number,
  previousWeight: number,
) {
  const stable = new Uint8ClampedArray(current.length);
  if (!previous || previous.length !== current.length || !previousLuma || previousWeight <= 0) {
    stable.set(current);
    return stable;
  }
  const motion = estimateMotion(previousLuma, currentLuma);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      const next = current[index] / 255;
      const flowX = ((x + 0.5) / width) * FLOW_WIDTH;
      const flowY = ((y + 0.5) / height) * FLOW_HEIGHT;
      const moveX = motion ? sampleField(motion.dx, motion, flowX, flowY) : 0;
      const moveY = motion ? sampleField(motion.dy, motion, flowX, flowY) : 0;
      const match = motion ? sampleField(motion.confidence, motion, flowX, flowY) : 0.25;
      const before = sampleAlpha(
        previous,
        width,
        height,
        x + moveX * width / FLOW_WIDTH,
        y + moveY * height / FLOW_HEIGHT,
      );
      const directionWeight = next < before ? 0.88 : 0.34;
      const weight = previousWeight * directionWeight * (0.08 + match * 0.92);
      let value = next * (1 - weight) + before * weight;
      if (before > 0.68 && next > 0.12 && match > 0.3) value = Math.max(value, before * 0.84);
      if (before < 0.08 && next < 0.38) value = Math.min(value, next * 0.9);
      stable[index] = Math.round(clamp(value, 0, 1) * 255);
    }
  }
  return stable;
}
