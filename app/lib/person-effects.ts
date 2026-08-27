import { MagicPoseMatte } from './magic-pose-matte';
import type { Box } from './vit-tracker';
import type { TrackPoint } from './video-export';
import { excludePersistentBackground } from './temporal-background';
import { FLOW_HEIGHT, FLOW_WIDTH, stabilizeAlpha } from './temporal-mask';

export type BackgroundEffect = 'original' | 'black' | 'blur';
export type SubjectEffect = 'original' | 'blue' | 'black';
export type OutlineEffect = 'white' | 'neon' | 'none';
export type CloneLayout = 'trail' | 'lineup';
export type MaskCorrectionMode = 'keep' | 'remove';

export type PersonEffectOptions = {
  enabled: boolean;
  preserveFraming: boolean;
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
  preserveFraming: false,
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
  flowLuma: Uint8Array;
  bodyCore: Uint8Array | null;
  baseAlpha: Uint8ClampedArray | null;
  alpha: Uint8ClampedArray;
};

type MaskCorrectionStroke = {
  group: number;
  time: number;
  x: number;
  y: number;
  radius: number;
  mode: MaskCorrectionMode;
};

export type PersonMaskPreparationOptions = {
  startTime: number;
  endTime: number;
  preserveFraming: boolean;
  retainSourceForCorrections?: boolean;
  onProgress: (progress: number) => void;
  isCancelled: () => boolean;
};

const SPRITE_SIZE = 256;
const MATTE_PIXEL_BUDGET = 360 * 640;
const MATTE_MAX_EDGE = 640;
const PREPARE_INTERVAL = 1 / 30;
const SNAPSHOT_INTERVAL = 1 / 15;
const CORRECTION_BACKWARD_SECONDS = 0.12;
const CORRECTION_FORWARD_SECONDS = 1.2;

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function paintAlpha(
  alpha: Uint8ClampedArray,
  width: number,
  height: number,
  stroke: MaskCorrectionStroke,
) {
  const centerX = stroke.x * Math.max(0, width - 1);
  const centerY = stroke.y * Math.max(0, height - 1);
  const radius = Math.max(2, stroke.radius * Math.min(width, height));
  const left = Math.max(0, Math.floor(centerX - radius));
  const right = Math.min(width - 1, Math.ceil(centerX + radius));
  const top = Math.max(0, Math.floor(centerY - radius));
  const bottom = Math.min(height - 1, Math.ceil(centerY + radius));
  for (let y = top; y <= bottom; y += 1) {
    for (let x = left; x <= right; x += 1) {
      const distance = Math.hypot(x - centerX, y - centerY);
      if (distance > radius) continue;
      const feather = 1 - clamp((distance / radius - 0.72) / 0.28, 0, 1);
      const index = y * width + x;
      if (stroke.mode === 'keep') {
        alpha[index] = Math.max(alpha[index], Math.round(feather * 255));
      } else {
        alpha[index] = Math.round(alpha[index] * (1 - feather));
      }
    }
  }
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

function framingCrop(video: HTMLVideoElement, canvas: HTMLCanvasElement): Rect {
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  const sourceAspect = sourceWidth / sourceHeight;
  const outputAspect = canvas.width / canvas.height;
  if (sourceAspect > outputAspect) {
    const width = sourceHeight * outputAspect;
    return { x: (sourceWidth - width) / 2, y: 0, width, height: sourceHeight };
  }
  const height = sourceWidth / outputAspect;
  return { x: 0, y: (sourceHeight - height) / 2, width: sourceWidth, height };
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

export class PersonEffectRenderer {
  private readonly segmenter: MagicPoseMatte;
  private readonly inputCanvas = createCanvas();
  private readonly flowCanvas = createCanvas(FLOW_WIDTH, FLOW_HEIGHT);
  private readonly maskCanvas = createCanvas();
  private readonly spriteCanvas = createCanvas();
  private readonly tintCanvas = createCanvas();
  private readonly history: Snapshot[] = [];
  private readonly preparedMasks: PreparedMask[] = [];
  private readonly corrections: MaskCorrectionStroke[] = [];
  private maskReady = false;
  private lastRegion: Rect | null = null;
  private preparedMaskIndex = 0;
  private lastSnapshotTime = Number.NEGATIVE_INFINITY;
  private lastMediaTime = Number.NEGATIVE_INFINITY;
  private nextCorrectionGroup = 0;
  private activeCorrectionGroup = 0;
  private preparedPreserveFraming = false;

  private constructor(segmenter: MagicPoseMatte) {
    this.segmenter = segmenter;
  }

  static async create(wasmBaseUrl: string, subjectModelUrl: string, poseModelUrl: string) {
    const segmenter = await MagicPoseMatte.create(
      wasmBaseUrl,
      subjectModelUrl,
      poseModelUrl,
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
    this.segmenter.beginSequence();
    this.resetPlayback();
  }

  reset() {
    this.clearPrepared();
    this.corrections.length = 0;
    this.activeCorrectionGroup = 0;
    this.nextCorrectionGroup = 0;
  }

  get correctionCount() {
    return new Set(this.corrections.map((stroke) => stroke.group)).size;
  }

  beginCorrectionStroke() {
    this.nextCorrectionGroup += 1;
    this.activeCorrectionGroup = this.nextCorrectionGroup;
  }

  endCorrectionStroke() {
    this.activeCorrectionGroup = 0;
  }

  private applyCorrections(prepared: PreparedMask) {
    for (const stroke of this.corrections) {
      const delta = prepared.time - stroke.time;
      if (delta < -CORRECTION_BACKWARD_SECONDS || delta > CORRECTION_FORWARD_SECONDS) continue;
      paintAlpha(prepared.alpha, prepared.width, prepared.height, stroke);
    }
  }

  private stabilizePreparedBackward() {
    let following: PreparedMask | null = null;
    const followingWeight = this.preparedPreserveFraming ? 0.44 : 0.36;
    for (let index = this.preparedMasks.length - 1; index >= 0; index -= 1) {
      const prepared = this.preparedMasks[index];
      prepared.alpha = stabilizeAlpha(
        prepared.alpha,
        following?.alpha ?? null,
        prepared.flowLuma,
        following?.flowLuma ?? null,
        prepared.width,
        prepared.height,
        followingWeight,
      );
      following = prepared;
    }
  }

  private fillShortLeadingBodyCoreGaps() {
    let followingCore: Uint8Array | null = null;
    let missing = 0;
    for (let index = this.preparedMasks.length - 1; index >= 0; index -= 1) {
      const prepared = this.preparedMasks[index];
      if (prepared.bodyCore) {
        followingCore = prepared.bodyCore;
        missing = 0;
      } else if (followingCore && missing < 2) {
        prepared.bodyCore = new Uint8Array(followingCore);
        missing += 1;
      } else {
        followingCore = null;
        missing = 0;
      }
    }
  }

  private rebuildPreparedMasks() {
    let previous: PreparedMask | null = null;
    const previousWeight = this.preparedPreserveFraming ? 0.42 : 0.58;
    for (const prepared of this.preparedMasks) {
      const sourceAlpha = prepared.baseAlpha ?? prepared.alpha;
      prepared.alpha = stabilizeAlpha(
        sourceAlpha,
        previous?.alpha ?? null,
        prepared.flowLuma,
        previous?.flowLuma ?? null,
        prepared.width,
        prepared.height,
        previousWeight,
      );
      previous = prepared;
    }
    this.fillShortLeadingBodyCoreGaps();
    this.stabilizePreparedBackward();
    excludePersistentBackground(this.preparedMasks);
    for (const prepared of this.preparedMasks) this.applyCorrections(prepared);
    this.resetPlayback();
  }

  undoCorrection() {
    if (this.corrections.length === 0) return 0;
    const group = this.corrections[this.corrections.length - 1].group;
    for (let index = this.corrections.length - 1; index >= 0; index -= 1) {
      if (this.corrections[index].group === group) this.corrections.splice(index, 1);
    }
    this.rebuildPreparedMasks();
    return this.correctionCount;
  }

  clearCorrections() {
    this.corrections.length = 0;
    this.activeCorrectionGroup = 0;
    this.rebuildPreparedMasks();
  }

  paintCorrection(
    time: number,
    normalizedCanvasX: number,
    normalizedCanvasY: number,
    brushRadiusPixels: number,
    mode: MaskCorrectionMode,
    video: HTMLVideoElement,
    canvas: HTMLCanvasElement,
    path: TrackPoint[],
    subjectScale: number,
    options: PersonEffectOptions,
  ) {
    const prepared = this.preparedMaskAt(time);
    if (!prepared || canvas.width <= 0 || canvas.height <= 0) return this.correctionCount;
    if (this.activeCorrectionGroup === 0) this.beginCorrectionStroke();

    const crop = options.preserveFraming
      ? framingCrop(video, canvas)
      : cameraCrop(video, canvas, path, time, subjectScale);
    const sourceX = crop.x + clamp(normalizedCanvasX, 0, 1) * crop.width;
    const sourceY = crop.y + clamp(normalizedCanvasY, 0, 1) * crop.height;
    const x = (sourceX - prepared.region.x) / prepared.region.width;
    const y = (sourceY - prepared.region.y) / prepared.region.height;
    if (x < 0 || x > 1 || y < 0 || y > 1) return this.correctionCount;

    const sourceRadius = Math.max(2, brushRadiusPixels) * crop.width / canvas.width;
    const stroke: MaskCorrectionStroke = {
      group: this.activeCorrectionGroup,
      time: prepared.time,
      x,
      y,
      radius: clamp(sourceRadius / Math.min(prepared.region.width, prepared.region.height), 0.008, 0.2),
      mode,
    };
    this.corrections.push(stroke);
    for (const candidate of this.preparedMasks) {
      const delta = candidate.time - stroke.time;
      if (delta < -CORRECTION_BACKWARD_SECONDS || delta > CORRECTION_FORWARD_SECONDS) continue;
      paintAlpha(candidate.alpha, candidate.width, candidate.height, stroke);
    }
    return this.correctionCount;
  }

  private segmentMask(video: HTMLVideoElement, region: Rect): PreparedMask {
    const input = this.inputCanvas.getContext('2d', { alpha: false });
    if (!input) throw new Error('Safari 無法建立人物去背 Canvas');
    const inputScale = Math.min(
      Math.sqrt(MATTE_PIXEL_BUDGET / Math.max(1, region.width * region.height)),
      MATTE_MAX_EDGE / Math.max(1, region.width, region.height),
    );
    const inputWidth = Math.max(1, Math.round(region.width * inputScale));
    const inputHeight = Math.max(1, Math.round(region.height * inputScale));
    if (this.inputCanvas.width !== inputWidth || this.inputCanvas.height !== inputHeight) {
      this.inputCanvas.width = inputWidth;
      this.inputCanvas.height = inputHeight;
    }
    input.drawImage(
      video,
      region.x,
      region.y,
      region.width,
      region.height,
      0,
      0,
      inputWidth,
      inputHeight,
    );
    const result = this.segmenter.segment(this.inputCanvas);
    const flow = this.flowCanvas.getContext('2d', { alpha: false, willReadFrequently: true });
    if (!flow) throw new Error('Safari 無法建立光流分析 Canvas');
    flow.drawImage(this.inputCanvas, 0, 0, FLOW_WIDTH, FLOW_HEIGHT);
    const pixels = flow.getImageData(0, 0, FLOW_WIDTH, FLOW_HEIGHT).data;
    const flowLuma = new Uint8Array(FLOW_WIDTH * FLOW_HEIGHT);
    for (let index = 0; index < flowLuma.length; index += 1) {
      const pixel = index * 4;
      flowLuma[index] = Math.round(
        pixels[pixel] * 0.299 + pixels[pixel + 1] * 0.587 + pixels[pixel + 2] * 0.114,
      );
    }
    let bodyCore: Uint8Array | null = null;
    if (result.bodySupport?.length === result.width * result.height) {
      bodyCore = new Uint8Array(FLOW_WIDTH * FLOW_HEIGHT);
      for (let y = 0; y < result.height; y += 1) {
        const targetY = Math.min(
          FLOW_HEIGHT - 1,
          Math.floor(((y + 0.5) / result.height) * FLOW_HEIGHT),
        );
        for (let x = 0; x < result.width; x += 1) {
          const targetX = Math.min(
            FLOW_WIDTH - 1,
            Math.floor(((x + 0.5) / result.width) * FLOW_WIDTH),
          );
          const targetIndex = targetY * FLOW_WIDTH + targetX;
          bodyCore[targetIndex] = Math.max(
            bodyCore[targetIndex],
            result.bodySupport[y * result.width + x],
          );
        }
      }
    }
    return {
      time: video.currentTime,
      region: { ...region },
      width: result.width,
      height: result.height,
      flowLuma,
      bodyCore,
      baseAlpha: null,
      alpha: result.alpha,
    };
  }

  private applyMask(prepared: PreparedMask) {
    this.maskCanvas.width = prepared.width;
    this.maskCanvas.height = prepared.height;
    if (this.spriteCanvas.width !== prepared.width || this.spriteCanvas.height !== prepared.height) {
      this.spriteCanvas.width = prepared.width;
      this.spriteCanvas.height = prepared.height;
      this.tintCanvas.width = prepared.width;
      this.tintCanvas.height = prepared.height;
    }
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
    this.preparedPreserveFraming = options.preserveFraming;
    const previousWeight = options.preserveFraming ? 0.42 : 0.58;
    let previousPrepared: PreparedMask | null = null;
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
        const region = options.preserveFraming
          ? { x: 0, y: 0, width: video.videoWidth, height: video.videoHeight }
          : subjectRegion(trackedBox, video.videoWidth, video.videoHeight);
        const prepared = this.segmentMask(video, region);
        prepared.time = at;
        const sourceAlpha = prepared.alpha;
        if (options.retainSourceForCorrections) {
          prepared.baseAlpha = new Uint8ClampedArray(sourceAlpha);
        }
        prepared.alpha = stabilizeAlpha(
          sourceAlpha,
          previousPrepared?.alpha ?? null,
          prepared.flowLuma,
          previousPrepared?.flowLuma ?? null,
          prepared.width,
          prepared.height,
          previousWeight,
        );
        this.preparedMasks.push(prepared);
        previousPrepared = prepared;
        options.onProgress(frame / totalFrames);
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      }
      this.fillShortLeadingBodyCoreGaps();
      this.stabilizePreparedBackward();
      excludePersistentBackground(this.preparedMasks);
      for (const prepared of this.preparedMasks) this.applyCorrections(prepared);
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
    context.clearRect(0, 0, this.spriteCanvas.width, this.spriteCanvas.height);
    context.globalCompositeOperation = 'source-over';
    context.drawImage(
      video,
      region.x,
      region.y,
      region.width,
      region.height,
      0,
      0,
      this.spriteCanvas.width,
      this.spriteCanvas.height,
    );
    context.globalCompositeOperation = 'destination-in';
    context.drawImage(this.maskCanvas, 0, 0, this.spriteCanvas.width, this.spriteCanvas.height);
    context.globalCompositeOperation = 'source-over';
  }

  private snapshot(time: number) {
    if (!this.lastRegion) return;
    const canvas = createCanvas(this.spriteCanvas.width, this.spriteCanvas.height);
    canvas.getContext('2d')!.drawImage(this.spriteCanvas, 0, 0);
    this.history.push({ time, region: { ...this.lastRegion }, canvas });
  }

  private tint(source: HTMLCanvasElement, color: string) {
    const context = this.tintCanvas.getContext('2d')!;
    if (this.tintCanvas.width !== source.width || this.tintCanvas.height !== source.height) {
      this.tintCanvas.width = source.width;
      this.tintCanvas.height = source.height;
    }
    context.clearRect(0, 0, source.width, source.height);
    context.globalCompositeOperation = 'source-over';
    context.drawImage(source, 0, 0, source.width, source.height);
    context.globalCompositeOperation = 'source-in';
    context.fillStyle = color;
    context.fillRect(0, 0, source.width, source.height);
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
    const crop = options.preserveFraming
      ? framingCrop(video, canvas)
      : cameraCrop(video, canvas, path, time, subjectScale);
    this.drawBackground(context, video, canvas, crop, options.background);

    const prepared = this.preparedMaskAt(time);
    if (prepared) {
      this.applyMask(prepared);
    } else {
      const region = options.preserveFraming
        ? { x: 0, y: 0, width: video.videoWidth, height: video.videoHeight }
        : subjectRegion(trackedBox, video.videoWidth, video.videoHeight);
      this.applyMask(this.segmentMask(video, region));
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
