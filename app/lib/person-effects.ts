import { InteractiveSubjectSegmenter } from './interactive-subject-segmenter';
import type { Box } from './vit-tracker';
import { TrackedPersonMaskSelector } from './tracked-person-mask';
import type { TrackPoint } from './video-export';

export type BackgroundEffect = 'original' | 'black' | 'blur';
export type SubjectEffect = 'original' | 'blue' | 'black';
export type OutlineEffect = 'white' | 'neon' | 'none';
export type CloneLayout = 'trail' | 'lineup';

export type PersonEffectOptions = {
  enabled: boolean;
  background: BackgroundEffect;
  subject: SubjectEffect;
  outline: OutlineEffect;
  cloneLayout: CloneLayout;
  cloneCount: number;
  cloneDelay: number;
  cloneOpacity: number;
  cloneColor: string;
};

export const DEFAULT_PERSON_EFFECTS: PersonEffectOptions = {
  enabled: false,
  background: 'black',
  subject: 'original',
  outline: 'white',
  cloneLayout: 'trail',
  cloneCount: 1,
  cloneDelay: 0.4,
  cloneOpacity: 0.72,
  cloneColor: '#165dff',
};

type Rect = { x: number; y: number; width: number; height: number };
type Snapshot = { time: number; region: Rect; canvas: HTMLCanvasElement };
type PreparedMask = {
  time: number;
  region: Rect;
  width: number;
  height: number;
  alpha: Uint8ClampedArray;
};

export type PersonMaskPreparationOptions = {
  startTime: number;
  endTime: number;
  onProgress: (progress: number) => void;
  isCancelled: () => boolean;
};

const SPRITE_SIZE = 256;
const PREPARE_INTERVAL = 1 / 30;
const SNAPSHOT_INTERVAL = 1 / 15;

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function createCanvas(width = SPRITE_SIZE, height = SPRITE_SIZE) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

async function seekVideo(video: HTMLVideoElement, time: number) {
  if (Math.abs(video.currentTime - time) < 0.001) return;
  await new Promise<void>((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error('逐格人物去背定位逾時'));
    }, 8000);
    const cleanup = () => {
      window.clearTimeout(timeout);
      video.removeEventListener('seeked', done);
      video.removeEventListener('error', failed);
    };
    const done = () => {
      cleanup();
      requestAnimationFrame(() => resolve());
    };
    const failed = () => {
      cleanup();
      reject(new Error('逐格人物去背定位失敗'));
    };
    video.addEventListener('seeked', done);
    video.addEventListener('error', failed);
    video.currentTime = time;
  });
}

function interpolateBox(path: TrackPoint[], time: number): Box {
  if (path.length === 0) return [0, 0, 1, 1];
  if (time <= path[0].time) return [...path[0].box] as Box;
  const last = path[path.length - 1];
  if (time >= last.time) return [...last.box] as Box;

  let low = 0;
  let high = path.length - 1;
  while (low + 1 < high) {
    const middle = Math.floor((low + high) / 2);
    if (path[middle].time <= time) low = middle;
    else high = middle;
  }
  const before = path[low];
  const after = path[high];
  const mix = (time - before.time) / Math.max(0.000001, after.time - before.time);
  return before.box.map((value, index) => value + (after.box[index] - value) * mix) as Box;
}

function cameraCrop(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  path: TrackPoint[],
  time: number,
  subjectScale: number,
): Rect {
  const [x, y, width, height] = interpolateBox(path, time);
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  const outputAspect = canvas.width / canvas.height;
  const scale = clamp(subjectScale, 0.25, 0.8);
  let cropHeight = Math.max(height / scale, width / (outputAspect * Math.min(0.9, scale + 0.12)));
  cropHeight = Math.min(cropHeight, sourceHeight, sourceWidth / outputAspect);
  const cropWidth = cropHeight * outputAspect;
  const centerX = x + width / 2;
  const centerY = y + height / 2;
  return {
    x: clamp(centerX - cropWidth / 2, 0, Math.max(0, sourceWidth - cropWidth)),
    y: clamp(centerY - cropHeight / 2, 0, Math.max(0, sourceHeight - cropHeight)),
    width: cropWidth,
    height: cropHeight,
  };
}

function subjectRegion(box: Box, sourceWidth: number, sourceHeight: number): Rect {
  const [x, y, width, height] = box;
  const size = Math.min(
    Math.max(width * 2.2, height * 1.7, 96),
    sourceWidth,
    sourceHeight,
  );
  return {
    x: clamp(x + width / 2 - size / 2, 0, Math.max(0, sourceWidth - size)),
    y: clamp(y + height / 2 - size / 2, 0, Math.max(0, sourceHeight - size)),
    width: size,
    height: size,
  };
}

function outputRect(region: Rect, crop: Rect, canvas: HTMLCanvasElement): Rect {
  return {
    x: ((region.x - crop.x) / crop.width) * canvas.width,
    y: ((region.y - crop.y) / crop.height) * canvas.height,
    width: (region.width / crop.width) * canvas.width,
    height: (region.height / crop.height) * canvas.height,
  };
}

function smoothAlpha(confidence: number) {
  const normalized = clamp((confidence - 0.16) / 0.58, 0, 1);
  return normalized * normalized * (3 - 2 * normalized);
}

export class PersonEffectRenderer {
  private readonly segmenter: InteractiveSubjectSegmenter;
  private readonly inputCanvas = createCanvas();
  private readonly maskCanvas = createCanvas();
  private readonly spriteCanvas = createCanvas();
  private readonly tintCanvas = createCanvas();
  private readonly history: Snapshot[] = [];
  private readonly preparedMasks: PreparedMask[] = [];
  private readonly maskSelector = new TrackedPersonMaskSelector();
  private previousMaskAlpha = new Uint8ClampedArray(0);
  private maskReady = false;
  private lastRegion: Rect | null = null;
  private preparedMaskIndex = 0;
  private lastSnapshotTime = Number.NEGATIVE_INFINITY;
  private lastMediaTime = Number.NEGATIVE_INFINITY;

  private constructor(segmenter: InteractiveSubjectSegmenter) {
    this.segmenter = segmenter;
  }

  static async create(wasmBaseUrl: string, subjectModelUrl: string, personModelUrl: string) {
    const segmenter = await InteractiveSubjectSegmenter.create(
      wasmBaseUrl,
      subjectModelUrl,
      personModelUrl,
    );
    return new PersonEffectRenderer(segmenter);
  }

  resetPlayback() {
    this.history.length = 0;
    this.maskReady = false;
    this.lastRegion = null;
    this.preparedMaskIndex = 0;
    this.lastSnapshotTime = Number.NEGATIVE_INFINITY;
    this.lastMediaTime = Number.NEGATIVE_INFINITY;
  }

  clearPrepared() {
    this.preparedMasks.length = 0;
    this.previousMaskAlpha.fill(0);
    this.resetPlayback();
  }

  reset() {
    this.clearPrepared();
  }

  private segmentMask(video: HTMLVideoElement, region: Rect, trackedBox: Box): PreparedMask {
    const input = this.inputCanvas.getContext('2d', { alpha: false });
    if (!input) throw new Error('Safari 無法建立人物去背 Canvas');
    input.drawImage(
      video,
      region.x,
      region.y,
      region.width,
      region.height,
      0,
      0,
      SPRITE_SIZE,
      SPRITE_SIZE,
    );
    const [boxX, boxY, boxWidth, boxHeight] = trackedBox;
    const centerX = clamp((boxX + boxWidth / 2 - region.x) / region.width, 0.02, 0.98);
    const centerY = clamp((boxY + boxHeight / 2 - region.y) / region.height, 0.02, 0.98);
    const promptHalfHeight = clamp((boxHeight / region.height) * 0.08, 0.015, 0.055);
    const result = this.segmenter.segment(this.inputCanvas, {
      scribble: [-1, -0.5, 0, 0.5, 1].map((offset) => ({
        x: centerX,
        y: clamp(centerY + offset * promptHalfHeight, 0.02, 0.98),
      })),
    });
    const values = result.values;
    const componentGate = this.maskSelector.select(
      values,
      result.width,
      result.height,
      region,
      trackedBox,
    );
    if (this.previousMaskAlpha.length !== values.length) {
      this.previousMaskAlpha = new Uint8ClampedArray(values.length);
    }
    const alpha = new Uint8ClampedArray(values.length);
    const hasPrevious = this.previousMaskAlpha.some((value) => value > 0);
    for (let index = 0; index < values.length; index += 1) {
      const rawAlpha = componentGate[index] ? Math.round(smoothAlpha(values[index]) * 255) : 0;
      alpha[index] = hasPrevious
        ? Math.round(rawAlpha * 0.9 + this.previousMaskAlpha[index] * 0.1)
        : rawAlpha;
    }
    this.previousMaskAlpha.set(alpha);
    return {
      time: video.currentTime,
      region: { ...region },
      width: result.width,
      height: result.height,
      alpha,
    };
  }

  private applyMask(prepared: PreparedMask) {
    this.maskCanvas.width = prepared.width;
    this.maskCanvas.height = prepared.height;
    const mask = this.maskCanvas.getContext('2d');
    if (!mask) throw new Error('Safari 無法建立人物遮罩 Canvas');
    const image = mask.createImageData(prepared.width, prepared.height);
    for (let index = 0; index < prepared.alpha.length; index += 1) {
      const pixel = index * 4;
      image.data[pixel] = 255;
      image.data[pixel + 1] = 255;
      image.data[pixel + 2] = 255;
      image.data[pixel + 3] = prepared.alpha[index];
    }
    mask.putImageData(image, 0, 0);
    this.maskReady = true;
    this.lastRegion = prepared.region;
  }

  private preparedMaskAt(time: number) {
    if (this.preparedMasks.length === 0) return null;
    while (
      this.preparedMaskIndex + 1 < this.preparedMasks.length
      && this.preparedMasks[this.preparedMaskIndex + 1].time <= time + PREPARE_INTERVAL / 2
    ) {
      this.preparedMaskIndex += 1;
    }
    while (
      this.preparedMaskIndex > 0
      && this.preparedMasks[this.preparedMaskIndex].time > time + PREPARE_INTERVAL / 2
    ) {
      this.preparedMaskIndex -= 1;
    }
    const prepared = this.preparedMasks[this.preparedMaskIndex];
    if (time < this.preparedMasks[0].time - PREPARE_INTERVAL) return null;
    if (time > this.preparedMasks[this.preparedMasks.length - 1].time + PREPARE_INTERVAL) return null;
    return prepared;
  }

  async prepare(
    video: HTMLVideoElement,
    path: TrackPoint[],
    options: PersonMaskPreparationOptions,
  ) {
    if (path.length < 2) throw new Error('尚未建立完整追蹤路徑');
    const sourceDuration = Number.isFinite(video.duration) ? video.duration : 0;
    const startTime = clamp(options.startTime, 0, sourceDuration);
    const endTime = clamp(options.endTime, startTime, sourceDuration);
    if (endTime - startTime < 0.03) throw new Error('人物去背測試片段太短');

    const originalTime = video.currentTime;
    const totalFrames = Math.max(1, Math.ceil((endTime - startTime) / PREPARE_INTERVAL));
    this.clearPrepared();
    video.pause();
    try {
      for (let frame = 0; frame <= totalFrames; frame += 1) {
        if (options.isCancelled()) throw new Error('使用者已取消人物去背分析');
        const at = Math.min(
          Math.max(startTime, endTime - 0.001),
          startTime + frame * PREPARE_INTERVAL,
        );
        await seekVideo(video, at);
        const trackedBox = interpolateBox(path, at);
        const region = subjectRegion(trackedBox, video.videoWidth, video.videoHeight);
        const prepared = this.segmentMask(video, region, trackedBox);
        prepared.time = at;
        this.preparedMasks.push(prepared);
        options.onProgress(frame / totalFrames);
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      }
      options.onProgress(1);
    } catch (error) {
      this.clearPrepared();
      throw error;
    } finally {
      video.pause();
      await seekVideo(video, Math.min(originalTime, sourceDuration)).catch(() => undefined);
      this.resetPlayback();
    }
  }

  private updateSprite(video: HTMLVideoElement) {
    const region = this.lastRegion;
    const context = this.spriteCanvas.getContext('2d');
    if (!region || !context || !this.maskReady) return;
    context.clearRect(0, 0, SPRITE_SIZE, SPRITE_SIZE);
    context.globalCompositeOperation = 'source-over';
    context.drawImage(
      video,
      region.x,
      region.y,
      region.width,
      region.height,
      0,
      0,
      SPRITE_SIZE,
      SPRITE_SIZE,
    );
    context.globalCompositeOperation = 'destination-in';
    context.drawImage(this.maskCanvas, 0, 0, SPRITE_SIZE, SPRITE_SIZE);
    context.globalCompositeOperation = 'source-over';
  }

  private snapshot(time: number) {
    if (!this.lastRegion) return;
    const canvas = createCanvas();
    canvas.getContext('2d')!.drawImage(this.spriteCanvas, 0, 0);
    this.history.push({ time, region: { ...this.lastRegion }, canvas });
  }

  private tint(source: CanvasImageSource, color: string) {
    const context = this.tintCanvas.getContext('2d')!;
    context.clearRect(0, 0, SPRITE_SIZE, SPRITE_SIZE);
    context.globalCompositeOperation = 'source-over';
    context.drawImage(source, 0, 0, SPRITE_SIZE, SPRITE_SIZE);
    context.globalCompositeOperation = 'source-in';
    context.fillStyle = color;
    context.fillRect(0, 0, SPRITE_SIZE, SPRITE_SIZE);
    context.globalCompositeOperation = 'source-over';
    return this.tintCanvas;
  }

  private drawBackground(
    context: CanvasRenderingContext2D,
    video: HTMLVideoElement,
    canvas: HTMLCanvasElement,
    crop: Rect,
    effect: BackgroundEffect,
  ) {
    context.save();
    context.globalAlpha = 1;
    context.globalCompositeOperation = 'source-over';
    context.filter = 'none';
    context.fillStyle = '#000';
    context.fillRect(0, 0, canvas.width, canvas.height);
    if (effect !== 'black') {
      if (effect === 'blur') context.filter = 'blur(22px) saturate(0.82)';
      const margin = effect === 'blur' ? 24 : 0;
      context.drawImage(
        video,
        crop.x,
        crop.y,
        crop.width,
        crop.height,
        -margin,
        -margin,
        canvas.width + margin * 2,
        canvas.height + margin * 2,
      );
    }
    context.restore();
  }

  private drawMain(
    context: CanvasRenderingContext2D,
    destination: Rect,
    options: PersonEffectOptions,
  ) {
    if (options.outline !== 'none') {
      const outlineColor = options.outline === 'white' ? '#ffffff' : '#58ffd0';
      const outline = this.tint(this.spriteCanvas, outlineColor);
      const radius = Math.max(4, Math.min(10, destination.width / 44));
      context.save();
      if (options.outline === 'neon') {
        context.shadowColor = '#35d292';
        context.shadowBlur = radius * 3.5;
      }
      for (let index = 0; index < 12; index += 1) {
        const angle = (index / 12) * Math.PI * 2;
        context.drawImage(
          outline,
          destination.x + Math.cos(angle) * radius,
          destination.y + Math.sin(angle) * radius,
          destination.width,
          destination.height,
        );
      }
      context.restore();
    }

    if (options.subject === 'original') {
      context.drawImage(this.spriteCanvas, destination.x, destination.y, destination.width, destination.height);
    } else {
      const color = options.subject === 'blue' ? '#155cff' : '#020403';
      context.drawImage(this.tint(this.spriteCanvas, color), destination.x, destination.y, destination.width, destination.height);
    }
  }

  private nearestSnapshot(targetTime: number) {
    for (let index = this.history.length - 1; index >= 0; index -= 1) {
      if (this.history[index].time <= targetTime + SNAPSHOT_INTERVAL / 2) return this.history[index];
    }
    return null;
  }

  render(
    video: HTMLVideoElement,
    canvas: HTMLCanvasElement,
    path: TrackPoint[],
    time: number,
    subjectScale: number,
    options: PersonEffectOptions,
  ) {
    const context = canvas.getContext('2d', { alpha: false });
    if (!context || !video.videoWidth || !video.videoHeight) return;
    if (time + 0.01 < this.lastMediaTime) this.resetPlayback();
    this.lastMediaTime = time;

    const trackedBox = interpolateBox(path, time);
    const crop = cameraCrop(video, canvas, path, time, subjectScale);
    this.drawBackground(context, video, canvas, crop, options.background);

    const prepared = this.preparedMaskAt(time);
    if (prepared) {
      this.applyMask(prepared);
    } else {
      const region = subjectRegion(trackedBox, video.videoWidth, video.videoHeight);
      this.applyMask(this.segmentMask(video, region, trackedBox));
    }
    this.updateSprite(video);
    if (time - this.lastSnapshotTime >= SNAPSHOT_INTERVAL) {
      this.snapshot(time);
      this.lastSnapshotTime = time;
    }

    const keepSeconds = Math.max(0.5, options.cloneCount * options.cloneDelay + 0.35);
    while (this.history.length > 0 && this.history[0].time < time - keepSeconds) this.history.shift();

    for (let clone = options.cloneCount; clone >= 1; clone -= 1) {
      const snapshot = this.nearestSnapshot(time - clone * options.cloneDelay);
      if (!snapshot) continue;
      const trailDestination = outputRect(snapshot.region, crop, canvas);
      const currentDestination = this.lastRegion
        ? outputRect(this.lastRegion, crop, canvas)
        : trailDestination;
      const trackedDestination = outputRect({
        x: trackedBox[0],
        y: trackedBox[1],
        width: trackedBox[2],
        height: trackedBox[3],
      }, crop, canvas);
      const side = clone % 2 === 1 ? 1 : -1;
      const rank = Math.ceil(clone / 2);
      const destination = options.cloneLayout === 'lineup'
        ? {
            ...currentDestination,
            x: currentDestination.x + side * rank * trackedDestination.width * 1.08,
          }
        : trailDestination;
      context.save();
      context.globalAlpha = clamp(options.cloneOpacity, 0.1, 1) * (1 - (clone - 1) * 0.1);
      context.drawImage(
        this.tint(snapshot.canvas, options.cloneColor),
        destination.x,
        destination.y,
        destination.width,
        destination.height,
      );
      context.restore();
    }

    if (this.lastRegion) this.drawMain(context, outputRect(this.lastRegion, crop, canvas), options);
  }

  close() {
    this.history.length = 0;
    this.segmenter.close();
  }
}
