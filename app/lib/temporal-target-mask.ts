import type { Box } from './vit-tracker';

type Region = { x: number; y: number; width: number; height: number };

const PREVIOUS_VISIBLE_ALPHA = 48;
const MAX_GROW_DISTANCE = 3;
const DISTANCE_INFINITY = 0xffff;

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

/**
 * Keeps a prompted target stable between adjacent video frames.
 *
 * The target box is the trusted core. Outside that core, foreground is only
 * allowed to grow a few pixels from the previously accepted silhouette. This
 * prevents a newly touching prop or bystander from entering as one large
 * connected component while still allowing hands, hair and clothing to move.
 */
export class TemporalTargetMaskStabilizer {
  private previous = new Uint8ClampedArray(0);
  private distance = new Uint16Array(0);

  reset(seed?: Uint8ClampedArray) {
    this.previous = seed ? new Uint8ClampedArray(seed) : new Uint8ClampedArray(0);
  }

  stabilize(
    raw: Uint8ClampedArray,
    width: number,
    height: number,
    region: Region,
    trackedBox: Box,
  ) {
    if (raw.length !== width * height) throw new Error('時間遮罩尺寸不正確');
    if (this.previous.length !== raw.length) {
      this.previous = new Uint8ClampedArray(raw);
      return new Uint8ClampedArray(raw);
    }

    if (this.distance.length !== raw.length) this.distance = new Uint16Array(raw.length);
    for (let index = 0; index < raw.length; index += 1) {
      this.distance[index] = this.previous[index] >= PREVIOUS_VISIBLE_ALPHA
        ? 0
        : DISTANCE_INFINITY;
    }

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const index = y * width + x;
        let next = this.distance[index];
        if (x > 0) next = Math.min(next, this.distance[index - 1] + 1);
        if (y > 0) next = Math.min(next, this.distance[index - width] + 1);
        this.distance[index] = next;
      }
    }
    for (let y = height - 1; y >= 0; y -= 1) {
      for (let x = width - 1; x >= 0; x -= 1) {
        const index = y * width + x;
        let next = this.distance[index];
        if (x + 1 < width) next = Math.min(next, this.distance[index + 1] + 1);
        if (y + 1 < height) next = Math.min(next, this.distance[index + width] + 1);
        this.distance[index] = next;
      }
    }

    const [boxX, boxY, boxWidth, boxHeight] = trackedBox;
    const left = ((boxX - region.x) / region.width) * width;
    const right = ((boxX + boxWidth - region.x) / region.width) * width;
    const top = ((boxY - region.y) / region.height) * height;
    const bottom = ((boxY + boxHeight - region.y) / region.height) * height;
    const marginX = Math.max(3, (right - left) * 0.16);
    const marginY = Math.max(3, (bottom - top) * 0.1);
    const trustedLeft = clamp(left - marginX, 0, width - 1);
    const trustedRight = clamp(right + marginX, 0, width - 1);
    const trustedTop = clamp(top - marginY, 0, height - 1);
    const trustedBottom = clamp(bottom + marginY, 0, height - 1);

    const output = new Uint8ClampedArray(raw.length);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const index = y * width + x;
        const previous = this.previous[index];
        const insideTrustedTarget = x >= trustedLeft && x <= trustedRight
          && y >= trustedTop && y <= trustedBottom;
        const supportedByTargetHistory = this.distance[index] <= MAX_GROW_DISTANCE;
        const current = insideTrustedTarget || supportedByTargetHistory ? raw[index] : 0;

        // Enter foreground quickly, leave it slowly. A one-frame model miss no
        // longer makes the person flash, but unsupported objects cannot appear
        // far away from the selected target in a single frame.
        const mixed = current >= previous
          ? current * 0.72 + previous * 0.28
          : current * 0.34 + previous * 0.66;
        output[index] = mixed < 5 ? 0 : Math.round(mixed);
      }
    }

    this.previous.set(output);
    return output;
  }
}
