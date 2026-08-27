import type { Box } from './vit-tracker';
import type { PersonEffectOptions, PersonEffectRenderer } from './person-effects';

export type AspectPreset = '9:16' | '1:1' | '16:9';

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
  aspect: AspectPreset;
  subjectScale: number;
  smoothness: number;
  codec: 'h264' | 'hevc';
  effects?: PersonEffectOptions;
  effectRenderer?: PersonEffectRenderer;
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

export function configureOutputCanvas(canvas: HTMLCanvasElement, aspect: AspectPreset) {
  const [width, height] = OUTPUT_SIZES[aspect];
  canvas.width = width;
  canvas.height = height;
  return { width, height };
}

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

function drawOutputFrame(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  path: TrackPoint[],
  time: number,
  subjectScale: number,
  effects?: PersonEffectOptions,
  effectRenderer?: PersonEffectRenderer,
) {
  if (effects?.enabled) {
    if (!effectRenderer) throw new Error('人物特效模型尚未載入');
    effectRenderer.render(video, canvas, path, time, subjectScale, effects);
    return;
  }
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
    if (path.length < 2) throw new Error('尚未建立完整追蹤路徑');
    const support = getRecorderSupport();
    const requestedType = options.codec === 'hevc' ? support.hevc : support.h264;
    if (!requestedType) {
      throw new Error(options.codec === 'hevc' ? '這台 iPhone 的 Safari 不支援 HEVC 網頁輸出' : '這台 iPhone 的 Safari 不支援 H.264 MP4 網頁輸出');
    }

    const { width, height } = configureOutputCanvas(canvas, options.aspect);
    const smoothedPath = smoothTrackPath(path, options.smoothness);
    options.effectRenderer?.reset();
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
      drawOutputFrame(
        this.video,
        canvas,
        smoothedPath,
        0,
        options.subjectScale,
        options.effects,
        options.effectRenderer,
      );
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
          try {
            drawOutputFrame(
              this.video,
              canvas,
              smoothedPath,
              mediaTime,
              options.subjectScale,
              options.effects,
              options.effectRenderer,
            );
            options.onProgress(clamp(mediaTime / Math.max(0.001, this.video.duration), 0, 1));
            return true;
          } catch (error) {
            this.video.pause();
            settle(error instanceof Error ? error : new Error(String(error)));
            return false;
          }
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
