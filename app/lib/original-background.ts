export type NormalizedBackgroundRect = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type OriginalBackgroundSample = {
  rgba: Uint8ClampedArray;
  excluded: NormalizedBackgroundRect | null;
};

export type OriginalBackgroundModel = {
  width: number;
  height: number;
  rgb: Uint8Array;
  confidence: Uint8Array;
};

export type SourceRegion = {
  x: number;
  y: number;
  width: number;
  height: number;
};

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function smoothStep(value: number, minimum: number, maximum: number) {
  const normalized = clamp(
    (value - minimum) / Math.max(0.000001, maximum - minimum),
    0,
    1,
  );
  return normalized * normalized * (3 - 2 * normalized);
}

function median(values: number[]) {
  values.sort((first, second) => first - second);
  const middle = Math.floor(values.length / 2);
  return values.length % 2 === 0
    ? (values[middle - 1] + values[middle]) / 2
    : values[middle];
}

function isExcluded(
  rect: NormalizedBackgroundRect | null,
  x: number,
  y: number,
) {
  if (!rect) return false;
  return (
    x >= rect.x
    && x <= rect.x + rect.width
    && y >= rect.y
    && y <= rect.y + rect.height
  );
}

export function buildOriginalBackgroundModel(
  samples: OriginalBackgroundSample[],
  width: number,
  height: number,
): OriginalBackgroundModel | null {
  if (samples.length < 4 || width <= 0 || height <= 0) return null;
  const pixelCount = width * height;
  if (samples.some((sample) => sample.rgba.length !== pixelCount * 4)) return null;

  const rgb = new Uint8Array(pixelCount * 3);
  const confidence = new Uint8Array(pixelCount);
  const requiredObservations = Math.max(4, Math.ceil(samples.length * 0.2));
  const reds: number[] = [];
  const greens: number[] = [];
  const blues: number[] = [];

  for (let y = 0; y < height; y += 1) {
    const normalizedY = (y + 0.5) / height;
    for (let x = 0; x < width; x += 1) {
      const normalizedX = (x + 0.5) / width;
      reds.length = 0;
      greens.length = 0;
      blues.length = 0;
      for (const sample of samples) {
        if (isExcluded(sample.excluded, normalizedX, normalizedY)) continue;
        const pixel = (y * width + x) * 4;
        reds.push(sample.rgba[pixel]);
        greens.push(sample.rgba[pixel + 1]);
        blues.push(sample.rgba[pixel + 2]);
      }
      if (reds.length < requiredObservations) continue;

      const red = median(reds);
      const green = median(greens);
      const blue = median(blues);
      const index = y * width + x;
      const rgbIndex = index * 3;
      rgb[rgbIndex] = Math.round(red);
      rgb[rgbIndex + 1] = Math.round(green);
      rgb[rgbIndex + 2] = Math.round(blue);

      let deviation = 0;
      for (let sampleIndex = 0; sampleIndex < reds.length; sampleIndex += 1) {
        deviation += (
          Math.abs(reds[sampleIndex] - red)
          + Math.abs(greens[sampleIndex] - green)
          + Math.abs(blues[sampleIndex] - blue)
        ) / 3;
      }
      deviation /= reds.length;
      const stability = 1 - smoothStep(deviation, 4, 24);
      const observationStrength = 0.72 + 0.28 * smoothStep(
        reds.length,
        requiredObservations,
        Math.max(requiredObservations + 1, samples.length * 0.55),
      );
      confidence[index] = Math.round(stability * observationStrength * 255);
    }
  }
  return { width, height, rgb, confidence };
}

export function createOriginalBackgroundSuppression(
  model: OriginalBackgroundModel,
  currentRgba: Uint8ClampedArray,
) {
  const pixelCount = model.width * model.height;
  const suppression = new Uint8Array(pixelCount);
  if (currentRgba.length !== pixelCount * 4) return suppression;
  for (let index = 0; index < pixelCount; index += 1) {
    const rgbaIndex = index * 4;
    const rgbIndex = index * 3;
    const difference = (
      Math.abs(currentRgba[rgbaIndex] - model.rgb[rgbIndex])
      + Math.abs(currentRgba[rgbaIndex + 1] - model.rgb[rgbIndex + 1])
      + Math.abs(currentRgba[rgbaIndex + 2] - model.rgb[rgbIndex + 2])
    ) / 3;
    const appearanceMatch = 1 - smoothStep(difference, 7, 38);
    const evidence = (model.confidence[index] / 255) * appearanceMatch;
    suppression[index] = Math.round(smoothStep(evidence, 0.42, 0.82) * 250);
  }
  return suppression;
}

function sampleGrid(
  values: Uint8Array,
  width: number,
  height: number,
  x: number,
  y: number,
) {
  const atX = clamp(x, 0, width - 1);
  const atY = clamp(y, 0, height - 1);
  const left = Math.floor(atX);
  const top = Math.floor(atY);
  const right = Math.min(width - 1, left + 1);
  const bottom = Math.min(height - 1, top + 1);
  const mixX = atX - left;
  const mixY = atY - top;
  const above = values[top * width + left] * (1 - mixX)
    + values[top * width + right] * mixX;
  const below = values[bottom * width + left] * (1 - mixX)
    + values[bottom * width + right] * mixX;
  return above * (1 - mixY) + below * mixY;
}

export function applyOriginalBackgroundSuppression(
  alpha: Uint8ClampedArray,
  alphaWidth: number,
  alphaHeight: number,
  region: SourceRegion,
  sourceWidth: number,
  sourceHeight: number,
  suppression: Uint8Array,
  suppressionWidth: number,
  suppressionHeight: number,
  bodyCore: Uint8Array | null,
  bodyCoreWidth: number,
  bodyCoreHeight: number,
) {
  if (
    alpha.length !== alphaWidth * alphaHeight
    || suppression.length !== suppressionWidth * suppressionHeight
  ) {
    return 0;
  }
  const hasBodyCore = Boolean(
    bodyCore
    && bodyCore.length === bodyCoreWidth * bodyCoreHeight,
  );
  let removed = 0;

  for (let y = 0; y < alphaHeight; y += 1) {
    const sourceY = region.y + ((y + 0.5) / alphaHeight) * region.height;
    const suppressionY = (sourceY / sourceHeight) * suppressionHeight - 0.5;
    const coreY = ((y + 0.5) / alphaHeight) * bodyCoreHeight - 0.5;
    for (let x = 0; x < alphaWidth; x += 1) {
      const index = y * alphaWidth + x;
      const before = alpha[index];
      if (before === 0) continue;
      const sourceX = region.x + ((x + 0.5) / alphaWidth) * region.width;
      const suppressionX = (sourceX / sourceWidth) * suppressionWidth - 0.5;
      const background = sampleGrid(
        suppression,
        suppressionWidth,
        suppressionHeight,
        suppressionX,
        suppressionY,
      ) / 255;
      const coreX = ((x + 0.5) / alphaWidth) * bodyCoreWidth - 0.5;
      const core = hasBodyCore && bodyCore
        ? sampleGrid(bodyCore, bodyCoreWidth, bodyCoreHeight, coreX, coreY) / 255
        : 0;
      const outsideCore = 1 - smoothStep(core, 0.12, 0.62);
      const amount = background * outsideCore;
      alpha[index] = Math.round(before * (1 - amount));
      if (alpha[index] + 24 < before) removed += 1;
    }
  }
  return removed;
}
