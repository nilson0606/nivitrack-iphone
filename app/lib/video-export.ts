import type { Box } from './vit-tracker';

export type AspectPreset = '9:16' | '1:1' | '16:9';

export type FilterPreset = 'vivid' | 'soft' | 'cinematic' | 'warm' | 'mono' | 'vintage';

export type ExportOperation =
  | {
      kind: 'filter';
      preset: FilterPreset;
      strength: number;
    }
  | {
      kind: 'crop';
      aspect: AspectPreset | 'source';
      centerX: number;
      centerY: number;
      zoom: number;
    }
  | {
      kind: 'track';
      aspect: AspectPreset;
      subjectScale: number;
      smoothness: number;
    };

export type TrackPoint = {
  time: number;
  box: Box;
  score: number;
  accepted: boolean;
};

export type RecorderSupport = {
  h264: string | null;
  hevc: string | null;
};

export type ExportOptions = {
  operation: ExportOperation;
  codec: 'h264' | 'hevc';
  onProgress: (progress: number) => void;
  isCancelled: () => boolean;
};

export type ExportResult = {
  blob: Blob;
  mimeType: string;
  width: number;
  height: number;
  duration: number;
};

const H264_TYPES = [
  'video/mp4;codecs=avc1.42E01E,mp4a.40.2',
  'video/mp4;codecs=avc1.42E01E',
  'video/mp4',
];

const HEVC_TYPES = [
  'video/mp4;codecs=hvc1.1.6.L93.B0,mp4a.40.2',
  'video/mp4;codecs=hvc1,mp4a.40.2',
  'video/mp4;codecs=hev1,mp4a.40.2',
];

const OUTPUT_SIZES: Record<AspectPreset, [number, number]> = {
  '9:16': [720, 1280],
  '1:1': [720, 720],
  '16:9': [1280, 720],
};

function firstSupported(types: string[]) {
  if (typeof MediaRecorder === 'undefined') return null;
  if (typeof MediaRecorder.isTypeSupported !== 'function') return types.at(-1) ?? null;
  return types.find((type) => MediaRecorder.isTypeSupported(type)) ?? null;
}

export function getRecorderSupport(): RecorderSupport {
  return {
    h264: firstSupported(H264_TYPES),
    hevc: firstSupported(HEVC_TYPES),
  };
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function mix(from: number, to: number, strength: number) {
  return from + (to - from) * clamp(strength, 0, 1);
}

export function getFilterCss(preset: FilterPreset, strength: number) {
  const amount = clamp(strength, 0, 1);
  switch (preset) {
    case 'vivid':
      return `brightness(${mix(1, 1.03, amount)}) contrast(${mix(1, 1.12, amount)}) saturate(${mix(1, 1.5, amount)})`;
    case 'soft':
      return `brightness(${mix(1, 1.08, amount)}) contrast(${mix(1, 0.9, amount)}) saturate(${mix(1, 0.9, amount)})`;
    case 'cinematic':
      return `brightness(${mix(1, 0.95, amount)}) contrast(${mix(1, 1.2, amount)}) saturate(${mix(1, 0.76, amount)})`;
    case 'warm':
      return `brightness(${mix(1, 1.03, amount)}) contrast(${mix(1, 1.05, amount)}) saturate(${mix(1, 1.2, amount)}) sepia(${mix(0, 0.28, amount)})`;
    case 'mono':
      return `grayscale(${amount}) contrast(${mix(1, 1.25, amount)})`;
    case 'vintage':
      return `brightness(${mix(1, 0.96, amount)}) contrast(${mix(1, 1.08, amount)}) saturate(${mix(1, 0.82, amount)}) sepia(${mix(0, 0.46, amount)})`;
  }
}

function even(value: number) {
  const rounded = Math.max(2, Math.round(value));
  return rounded % 2 === 0 ? rounded : rounded - 1;
}

function sourceOutputSize(video: HTMLVideoElement): [number, number] {
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  const scale = Math.min(1, 1280 / Math.max(sourceWidth, sourceHeight));
  return [even(sourceWidth * scale), even(sourceHeight * scale)];
}

function configureOutputCanvas(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  operation: ExportOperation,
) {
  if (operation.kind === 'filter' || operation.aspect === 'source') {
    [canvas.width, canvas.height] = sourceOutputSize(video);
    return;
  }
  [canvas.width, canvas.height] = OUTPUT_SIZES[operation.aspect];
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

export function smoothTrackPath(path: TrackPoint[], smoothness: number) {
  if (path.length < 2) return path.map((point) => ({ ...point, box: [...point.box] as Box }));
  const alpha = 1 - clamp(smoothness, 0, 1) * 0.88;
  const forward: TrackPoint[] = [];
  let previous = [...path[0].box] as Box;
  for (const point of path) {
    const next = point.box.map((value, index) => previous[index] + (value - previous[index]) * alpha) as Box;
    forward.push({ ...point, box: next });
    previous = next;
  }

  const result = forward.map((point) => ({ ...point, box: [...point.box] as Box }));
  previous = [...result[result.length - 1].box] as Box;
  for (let index = result.length - 1; index >= 0; index -= 1) {
    const next = result[index].box.map(
      (value, axis) => (value + previous[axis] + (value - previous[axis]) * alpha) / 2,
    ) as Box;
    result[index].box = next;
    previous = next;
  }
  return result;
}

function drawTrackedFrame(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  path: TrackPoint[],
  time: number,
  subjectScale: number,
) {
  const context = canvas.getContext('2d', { alpha: false });
  if (!context || !video.videoWidth || !video.videoHeight) return;
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
  const cropX = clamp(centerX - cropWidth / 2, 0, Math.max(0, sourceWidth - cropWidth));
  const cropY = clamp(centerY - cropHeight / 2, 0, Math.max(0, sourceHeight - cropHeight));

  context.fillStyle = '#000';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.filter = 'none';
  context.drawImage(
    video,
    cropX,
    cropY,
    cropWidth,
    cropHeight,
    0,
    0,
    canvas.width,
    canvas.height,
  );
}

function drawCroppedFrame(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  operation: Extract<ExportOperation, { kind: 'crop' }>,
) {
  const context = canvas.getContext('2d', { alpha: false });
  if (!context || !video.videoWidth || !video.videoHeight) return;
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  const outputAspect = canvas.width / canvas.height;
  const sourceAspect = sourceWidth / sourceHeight;
  let cropWidth = sourceWidth;
  let cropHeight = sourceHeight;
  if (sourceAspect > outputAspect) cropWidth = sourceHeight * outputAspect;
  else cropHeight = sourceWidth / outputAspect;

  const zoom = clamp(operation.zoom, 1, 3);
  cropWidth /= zoom;
  cropHeight /= zoom;
  const centerX = clamp(operation.centerX, 0, 1) * sourceWidth;
  const centerY = clamp(operation.centerY, 0, 1) * sourceHeight;
  const cropX = clamp(centerX - cropWidth / 2, 0, Math.max(0, sourceWidth - cropWidth));
  const cropY = clamp(centerY - cropHeight / 2, 0, Math.max(0, sourceHeight - cropHeight));

  context.filter = 'none';
  context.fillStyle = '#000';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(video, cropX, cropY, cropWidth, cropHeight, 0, 0, canvas.width, canvas.height);
}

function drawFilteredFrame(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  operation: Extract<ExportOperation, { kind: 'filter' }>,
) {
  const context = canvas.getContext('2d', { alpha: false });
  if (!context || !video.videoWidth || !video.videoHeight) return;
  context.fillStyle = '#000';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.filter = getFilterCss(operation.preset, operation.strength);
  context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, canvas.width, canvas.height);
  context.filter = 'none';
}

function drawOutputFrame(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  path: TrackPoint[],
  time: number,
  operation: ExportOperation,
) {
  if (operation.kind === 'track') {
    drawTrackedFrame(video, canvas, path, time, operation.subjectScale);
    return;
  }
  if (operation.kind === 'crop') {
    drawCroppedFrame(video, canvas, operation);
    return;
  }
  drawFilteredFrame(video, canvas, operation);
}

async function seek(video: HTMLVideoElement, time: number) {
  if (Math.abs(video.currentTime - time) < 0.002) return;
  await new Promise<void>((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error('輸出前影片定位逾時'));
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
      reject(new Error('輸出前影片定位失敗'));
    };
    video.addEventListener('seeked', done);
    video.addEventListener('error', failed);
    video.currentTime = time;
  });
}

export class RealtimeVideoExporter {
  private readonly video: HTMLVideoElement;
  private audioContext: AudioContext | null = null;
  private source: MediaElementAudioSourceNode | null = null;
  private destination: MediaStreamAudioDestinationNode | null = null;
  private monitor: GainNode | null = null;

  constructor(video: HTMLVideoElement) {
    this.video = video;
  }

  private async prepareAudio() {
    if (!this.audioContext) {
      this.audioContext = new AudioContext();
      this.source = this.audioContext.createMediaElementSource(this.video);
      this.destination = this.audioContext.createMediaStreamDestination();
      this.monitor = this.audioContext.createGain();
      this.source.connect(this.destination);
      this.source.connect(this.monitor);
      this.monitor.connect(this.audioContext.destination);
    }
    this.monitor!.gain.value = 0;
    await this.audioContext.resume();
    return this.destination!.stream.getAudioTracks();
  }

  async export(
    path: TrackPoint[],
    canvas: HTMLCanvasElement,
    options: ExportOptions,
  ): Promise<ExportResult> {
    if (options.operation.kind === 'track' && path.length < 2) {
      throw new Error('尚未建立完整追蹤路徑');
    }
    const support = getRecorderSupport();
    const requestedType = options.codec === 'hevc' ? support.hevc : support.h264;
    if (!requestedType) {
      throw new Error(options.codec === 'hevc' ? '這台 iPhone 的 Safari 不支援 HEVC 網頁輸出' : '這台 iPhone 的 Safari 不支援 H.264 MP4 網頁輸出');
    }

    configureOutputCanvas(this.video, canvas, options.operation);
    const width = canvas.width;
    const height = canvas.height;
    const smoothedPath = options.operation.kind === 'track'
      ? smoothTrackPath(path, options.operation.smoothness)
      : path;
    const originalTime = this.video.currentTime;
    const originalRate = this.video.playbackRate;
    const audioTracks = await this.prepareAudio();
    const canvasStream = canvas.captureStream(30);
    const stream = new MediaStream([
      ...canvasStream.getVideoTracks(),
      ...audioTracks,
    ]);
    const chunks: Blob[] = [];
    const recorder = new MediaRecorder(stream, {
      mimeType: requestedType,
      videoBitsPerSecond: width > 720 ? 8_000_000 : 5_000_000,
      audioBitsPerSecond: 160_000,
    });
    recorder.addEventListener('dataavailable', (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    });
    const stopped = new Promise<void>((resolve, reject) => {
      recorder.addEventListener('stop', () => resolve(), { once: true });
      recorder.addEventListener('error', () => reject(new Error('Safari 影片編碼失敗')), { once: true });
    });

    let frameCallback = 0;
    let animationFrame = 0;
    let playbackError: Error | null = null;
    let recorderStarted = false;
    try {
      this.video.pause();
      await seek(this.video, 0);
      this.video.playbackRate = 1;
      drawOutputFrame(this.video, canvas, smoothedPath, 0, options.operation);
      recorder.start(1000);
      recorderStarted = true;

      await new Promise<void>((resolve, reject) => {
        let finished = false;
        const cleanup = () => {
          this.video.removeEventListener('ended', ended);
          this.video.removeEventListener('error', failed);
          if (frameCallback && 'cancelVideoFrameCallback' in this.video) {
            this.video.cancelVideoFrameCallback(frameCallback);
          }
          if (animationFrame) cancelAnimationFrame(animationFrame);
        };
        const settle = (error?: Error) => {
          if (finished) return;
          finished = true;
          cleanup();
          if (error) reject(error);
          else resolve();
        };
        const render = (mediaTime: number) => {
          if (options.isCancelled()) {
            this.video.pause();
            settle(new Error('使用者已取消輸出'));
            return false;
          }
          drawOutputFrame(this.video, canvas, smoothedPath, mediaTime, options.operation);
          options.onProgress(clamp(mediaTime / Math.max(0.001, this.video.duration), 0, 1));
          return true;
        };
        const videoFrame = (_now: number, metadata: VideoFrameCallbackMetadata) => {
          if (!render(metadata.mediaTime)) return;
          frameCallback = this.video.requestVideoFrameCallback(videoFrame);
        };
        const animation = () => {
          if (!render(this.video.currentTime)) return;
          animationFrame = requestAnimationFrame(animation);
        };
        const ended = () => {
          render(this.video.duration);
          settle();
        };
        const failed = () => settle(new Error('輸出播放來源影片時發生錯誤'));
        this.video.addEventListener('ended', ended);
        this.video.addEventListener('error', failed);
        if ('requestVideoFrameCallback' in this.video) {
          frameCallback = this.video.requestVideoFrameCallback(videoFrame);
        } else {
          animationFrame = requestAnimationFrame(animation);
        }
        this.video.play().catch((error) => settle(error instanceof Error ? error : new Error(String(error))));
      });
    } catch (error) {
      playbackError = error instanceof Error ? error : new Error(String(error));
    } finally {
      this.video.pause();
      if (recorder.state !== 'inactive') recorder.stop();
      if (recorderStarted) {
        await stopped.catch((error) => {
          playbackError ??= error instanceof Error ? error : new Error(String(error));
        });
      }
      canvasStream.getTracks().forEach((track) => track.stop());
      this.monitor!.gain.value = 1;
      this.video.playbackRate = originalRate;
      await seek(this.video, Math.min(originalTime, this.video.duration)).catch(() => undefined);
    }

    if (playbackError) throw playbackError;
    if (chunks.length === 0) throw new Error('Safari 沒有產生任何影片資料');
    options.onProgress(1);
    const mimeType = recorder.mimeType || requestedType;
    return {
      blob: new Blob(chunks, { type: mimeType }),
      mimeType,
      width,
      height,
      duration: this.video.duration,
    };
  }

  async dispose() {
    await this.audioContext?.close().catch(() => undefined);
    this.audioContext = null;
    this.source = null;
    this.destination = null;
    this.monitor = null;
  }
}
