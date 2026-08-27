import type { ImageSegmenter } from '@mediapipe/tasks-vision';
import type { Box } from './vit-tracker';
import type { TrackPoint } from './video-export';

export type BackgroundEffect = 'original' | 'black' | 'blur';
export type SubjectEffect = 'original' | 'blue' | 'black';
export type OutlineEffect = 'white' | 'neon' | 'none';

export type PersonEffectOptions = {
  enabled: boolean;
  background: BackgroundEffect;
  subject: SubjectEffect;
  outline: OutlineEffect;
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
  cloneCount: 1,
  cloneDelay: 0.4,
  cloneOpacity: 0.72,
  cloneColor: '#165dff',
};

type Rect = { x: number; y: number; width: number; height: number };
type Snapshot = { time: number; region: Rect; canvas: HTMLCanvasElement };

const SPRITE_SIZE = 256;
const SEGMENT_INTERVAL = 1 / 12;

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function createCanvas(width = SPRITE_SIZE, height = SPRITE_SIZE) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
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
    Math.max(width * 1.55, height * 1.16, 64),
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
  private readonly segmenter: ImageSegmenter;
  private readonly personMaskIndex: number;
  private readonly inputCanvas = createCanvas();
  private readonly maskCanvas = createCanvas();
  private readonly spriteCanvas = createCanvas();
  private readonly tintCanvas = createCanvas();
  private readonly history: Snapshot[] = [];
  private maskReady = false;
  private lastRegion: Rect | null = null;
  private lastSegmentTime = Number.NEGATIVE_INFINITY;
  private lastMediaTime = Number.NEGATIVE_INFINITY;
  private lastTaskTimestamp = 0;

  private constructor(segmenter: ImageSegmenter, personMaskIndex: number) {
    this.segmenter = segmenter;
    this.personMaskIndex = personMaskIndex;
  }

  static async create(wasmBaseUrl: string, modelUrl: string) {
    const { FilesetResolver, ImageSegmenter } = await import('@mediapipe/tasks-vision');
    // MediaPipe injects the loader through a classic <script> element on the
    // browser main thread. Use its UMD loader here; the ESM loader leaves
    // ModuleFactory unset when launched from an iOS home-screen web app.
    const fileset = await FilesetResolver.forVisionTasks(wasmBaseUrl);
    const gpuCanvas = createCanvas(1, 1);
    let segmenter: ImageSegmenter;
    try {
      segmenter = await ImageSegmenter.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: modelUrl, delegate: 'GPU' },
        canvas: gpuCanvas,
        runningMode: 'VIDEO',
        outputCategoryMask: false,
        outputConfidenceMasks: true,
      });
    } catch {
      segmenter = await ImageSegmenter.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: modelUrl, delegate: 'CPU' },
        runningMode: 'VIDEO',
        outputCategoryMask: false,
        outputConfidenceMasks: true,
      });
    }
    const labels = segmenter.getLabels().map((label) => label.toLowerCase());
    const personIndex = labels.findIndex((label) => label.includes('person'));
    return new PersonEffectRenderer(segmenter, personIndex >= 0 ? personIndex : Math.max(0, labels.length - 1));
  }

  reset() {
    this.history.length = 0;
    this.maskReady = false;
    this.lastRegion = null;
    this.lastSegmentTime = Number.NEGATIVE_INFINITY;
    this.lastMediaTime = Number.NEGATIVE_INFINITY;
  }

  private updateMask(video: HTMLVideoElement, region: Rect) {
    const input = this.inputCanvas.getContext('2d', { alpha: false });
    const mask = this.maskCanvas.getContext('2d');
    if (!input || !mask) throw new Error('Safari 無法建立人物去背 Canvas');
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
    const timestamp = Math.max(performance.now(), this.lastTaskTimestamp + 1);
    this.lastTaskTimestamp = timestamp;
    const result = this.segmenter.segmentForVideo(this.inputCanvas, timestamp);
    try {
      const masks = result.confidenceMasks ?? [];
      const selected = masks[Math.min(this.personMaskIndex, Math.max(0, masks.length - 1))];
      if (!selected) throw new Error('人物去背模型沒有回傳遮罩');
      const values = selected.getAsFloat32Array();
      const image = mask.createImageData(selected.width, selected.height);
      for (let index = 0; index < values.length; index += 1) {
        const alpha = Math.round(smoothAlpha(values[index]) * 255);
        const pixel = index * 4;
        image.data[pixel] = 255;
        image.data[pixel + 1] = 255;
        image.data[pixel + 2] = 255;
        image.data[pixel + 3] = alpha;
      }
      this.maskCanvas.width = selected.width;
      this.maskCanvas.height = selected.height;
      this.maskCanvas.getContext('2d')!.putImageData(image, 0, 0);
      this.maskReady = true;
      this.lastRegion = region;
    } finally {
      result.close();
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
      if (this.history[index].time <= targetTime + SEGMENT_INTERVAL / 2) return this.history[index];
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
    if (time + 0.01 < this.lastMediaTime) this.reset();
    this.lastMediaTime = time;

    const crop = cameraCrop(video, canvas, path, time, subjectScale);
    this.drawBackground(context, video, canvas, crop, options.background);

    const shouldSegment = !this.maskReady || time - this.lastSegmentTime >= SEGMENT_INTERVAL;
    if (shouldSegment) {
      const region = subjectRegion(interpolateBox(path, time), video.videoWidth, video.videoHeight);
      this.updateMask(video, region);
      this.lastSegmentTime = time;
    }
    this.updateSprite(video);
    if (shouldSegment) this.snapshot(time);

    const keepSeconds = Math.max(0.5, options.cloneCount * options.cloneDelay + 0.35);
    while (this.history.length > 0 && this.history[0].time < time - keepSeconds) this.history.shift();

    for (let clone = options.cloneCount; clone >= 1; clone -= 1) {
      const snapshot = this.nearestSnapshot(time - clone * options.cloneDelay);
      if (!snapshot) continue;
      const destination = outputRect(snapshot.region, crop, canvas);
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
