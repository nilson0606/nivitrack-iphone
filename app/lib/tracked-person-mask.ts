import type { Box } from './vit-tracker';

type Region = { x: number; y: number; width: number; height: number };

const COMPONENT_THRESHOLD = 0.32;
const COMPONENT_EDGE_RADIUS = 2;

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

export class TrackedPersonMaskSelector {
  private labels = new Int32Array(0);
  private queue = new Int32Array(0);
  private gate = new Uint8Array(0);
  private scratch = new Uint8Array(0);

  select(values: Float32Array, width: number, height: number, region: Region, trackedBox: Box) {
    const total = width * height;
    if (this.labels.length !== total) {
      this.labels = new Int32Array(total);
      this.queue = new Int32Array(total);
      this.gate = new Uint8Array(total);
      this.scratch = new Uint8Array(total);
    }
    this.labels.fill(0);
    this.gate.fill(0);

    const [boxX, boxY, boxWidth, boxHeight] = trackedBox;
    const left = clamp(((boxX - region.x) / region.width) * width, 0, width - 1);
    const right = clamp(((boxX + boxWidth - region.x) / region.width) * width, 0, width - 1);
    const top = clamp(((boxY - region.y) / region.height) * height, 0, height - 1);
    const bottom = clamp(((boxY + boxHeight - region.y) / region.height) * height, 0, height - 1);
    const centerX = (left + right) / 2;
    const centerY = (top + bottom) / 2;
    const halfWidth = Math.max(1, (right - left) / 2);
    const halfHeight = Math.max(1, (bottom - top) / 2);
    const allowedLeft = Math.max(0, Math.floor(left - (right - left) * 0.14));
    const allowedRight = Math.min(width - 1, Math.ceil(right + (right - left) * 0.14));
    const allowedTop = Math.max(0, Math.floor(top - (bottom - top) * 0.06));
    const allowedBottom = Math.min(height - 1, Math.ceil(bottom + (bottom - top) * 0.08));

    let nextLabel = 0;
    let bestLabel = 0;
    let bestAnchor = -1;
    let bestDistance = Number.POSITIVE_INFINITY;
    let bestConfidence = -1;

    for (let start = 0; start < total; start += 1) {
      const startX = start % width;
      const startY = Math.floor(start / width);
      if (
        this.labels[start] !== 0
        || values[start] < COMPONENT_THRESHOLD
        || startX < allowedLeft
        || startX > allowedRight
        || startY < allowedTop
        || startY > allowedBottom
      ) continue;
      nextLabel += 1;
      let head = 0;
      let tail = 1;
      this.queue[0] = start;
      this.labels[start] = nextLabel;
      let anchor = 0;
      let closestDistance = Number.POSITIVE_INFINITY;
      let confidence = 0;

      while (head < tail) {
        const index = this.queue[head];
        head += 1;
        const x = index % width;
        const y = Math.floor(index / width);
        const value = values[index];
        confidence += value;
        const normalizedX = (x - centerX) / halfWidth;
        const normalizedY = (y - centerY) / halfHeight;
        const distance = normalizedX * normalizedX + normalizedY * normalizedY;
        closestDistance = Math.min(closestDistance, distance);
        if (x >= left && x <= right && y >= top && y <= bottom) {
          anchor += value * (1 + Math.max(0, 1 - Math.sqrt(distance)));
        }

        let neighbor = index - 1;
        if (x > allowedLeft && this.labels[neighbor] === 0 && values[neighbor] >= COMPONENT_THRESHOLD) {
          this.labels[neighbor] = nextLabel;
          this.queue[tail] = neighbor;
          tail += 1;
        }
        neighbor = index + 1;
        if (x < allowedRight && this.labels[neighbor] === 0 && values[neighbor] >= COMPONENT_THRESHOLD) {
          this.labels[neighbor] = nextLabel;
          this.queue[tail] = neighbor;
          tail += 1;
        }
        neighbor = index - width;
        if (y > allowedTop && this.labels[neighbor] === 0 && values[neighbor] >= COMPONENT_THRESHOLD) {
          this.labels[neighbor] = nextLabel;
          this.queue[tail] = neighbor;
          tail += 1;
        }
        neighbor = index + width;
        if (y < allowedBottom && this.labels[neighbor] === 0 && values[neighbor] >= COMPONENT_THRESHOLD) {
          this.labels[neighbor] = nextLabel;
          this.queue[tail] = neighbor;
          tail += 1;
        }
      }

      const anchored = anchor > 0;
      const bestAnchored = bestAnchor > 0;
      const equalAnchor = Math.abs(anchor - bestAnchor) <= 0.001;
      const isBetter = bestLabel === 0
        || (anchored && !bestAnchored)
        || (anchored === bestAnchored && (
          (anchored && anchor > bestAnchor + 0.001)
          || (equalAnchor && closestDistance < bestDistance)
          || (equalAnchor && closestDistance === bestDistance && confidence > bestConfidence)
        ));
      if (isBetter) {
        bestLabel = nextLabel;
        bestAnchor = anchor;
        bestDistance = closestDistance;
        bestConfidence = confidence;
      }
    }

    if (bestLabel === 0) {
      for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
          if (x >= left && x <= right && y >= top && y <= bottom) this.gate[y * width + x] = 1;
        }
      }
      return this.gate;
    }

    for (let index = 0; index < total; index += 1) {
      if (this.labels[index] === bestLabel) this.gate[index] = 1;
    }
    for (let radius = 0; radius < COMPONENT_EDGE_RADIUS; radius += 1) {
      this.scratch.set(this.gate);
      for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
          const index = y * width + x;
          if (this.scratch[index] || this.labels[index] !== 0) continue;
          const minY = Math.max(0, y - 1);
          const maxY = Math.min(height - 1, y + 1);
          const minX = Math.max(0, x - 1);
          const maxX = Math.min(width - 1, x + 1);
          for (let neighborY = minY; neighborY <= maxY && !this.gate[index]; neighborY += 1) {
            for (let neighborX = minX; neighborX <= maxX; neighborX += 1) {
              if (this.scratch[neighborY * width + neighborX]) {
                this.gate[index] = 1;
                break;
              }
            }
          }
        }
      }
    }
    return this.gate;
  }
}
