import type { Box } from './vit-tracker';
import type { PersonBackgroundRenderer } from './person-background-removal';
import type { ModnetPreviewTimeline } from './modnet-background-preview';

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
      selectionBox?: Box;
    }
  | {
      kind: 'track';
      aspect: AspectPreset;
      subjectScale: number;
      smoothness: number;
    }
  | {
      kind: 'remove-background';
      aspect: AspectPreset;
      subjectScale: number;
      smoothness: number;
      bodyTightness: number;
      blackOutline: number;
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
  backgroundTimeline?: ModnetPreviewTimeline;
  onStage?: (stage: 'audio' | 'recorder' | 'seek' | 'playback') => void;
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

export const OUTPUT_SIZES: Record<AspectPreset, [number, number]> = {
  '9:16': [720, 1280],
  '1:1': [720, 720],
  '16:9': [1280, 720],
};

const FILTER_INDEX: Record<FilterPreset, number> = {
  vivid: 0,
  soft: 1,
  cinematic: 2,
  warm: 3,
  mono: 4,
  vintage: 5,
};

const FILTER_VERTEX_SHADER = `
attribute vec2 a_position;
attribute vec2 a_texCoord;
varying vec2 v_texCoord;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
  v_texCoord = a_texCoord;
}`;

const FILTER_FRAGMENT_SHADER = `
precision mediump float;
varying vec2 v_texCoord;
uniform sampler2D u_image;
uniform float u_strength;
uniform int u_preset;

vec3 adjustSaturation(vec3 color, float amount) {
  float gray = dot(color, vec3(0.2126, 0.7152, 0.0722));
  return mix(vec3(gray), color, amount);
}

vec3 adjustContrast(vec3 color, float amount) {
  return (color - 0.5) * amount + 0.5;
}

vec3 sepiaColor(vec3 color) {
  return vec3(
    dot(color, vec3(0.393, 0.769, 0.189)),
    dot(color, vec3(0.349, 0.686, 0.168)),
    dot(color, vec3(0.272, 0.534, 0.131))
  );
}

void main() {
  vec4 sampleColor = texture2D(u_image, v_texCoord);
  vec3 original = sampleColor.rgb;
  vec3 target = original;
  if (u_preset == 0) {
    target = adjustSaturation(adjustContrast(original, 1.12), 1.50) * 1.03;
  } else if (u_preset == 1) {
    target = adjustSaturation(adjustContrast(original, 0.90), 0.90) * 1.08;
  } else if (u_preset == 2) {
    target = adjustSaturation(adjustContrast(original, 1.20), 0.76) * 0.95;
  } else if (u_preset == 3) {
    target = adjustSaturation(adjustContrast(original, 1.05), 1.20) * 1.03;
    target = mix(target, sepiaColor(target), 0.28);
  } else if (u_preset == 4) {
    target = adjustSaturation(adjustContrast(original, 1.25), 0.0);
  } else {
    target = adjustSaturation(adjustContrast(original, 1.08), 0.82) * 0.96;
    target = mix(target, sepiaColor(target), 0.46);
  }
  gl_FragColor = vec4(clamp(mix(original, target, u_strength), 0.0, 1.0), sampleColor.a);
}`;

function compileShader(gl: WebGLRenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);
  if (!shader) throw new Error('Safari 無法建立濾鏡著色器');
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader) ?? '未知錯誤';
    gl.deleteShader(shader);
    throw new Error('Safari 濾鏡編譯失敗：' + message);
  }
  return shader;
}

class WebGLFilterRenderer {
  readonly canvas: HTMLCanvasElement;
  private readonly gl: WebGLRenderingContext;
  private readonly program: WebGLProgram;
  private readonly texture: WebGLTexture;
  private readonly buffer: WebGLBuffer;

  constructor(width: number, height: number, preset: FilterPreset, strength: number) {
    this.canvas = document.createElement('canvas');
    this.canvas.width = width;
    this.canvas.height = height;
    const gl = this.canvas.getContext('webgl', {
      alpha: false,
      antialias: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: true,
    });
    if (!gl) throw new Error('這台 Safari 無法啟動 WebGL 濾鏡輸出');
    this.gl = gl;

    const vertex = compileShader(gl, gl.VERTEX_SHADER, FILTER_VERTEX_SHADER);
    const fragment = compileShader(gl, gl.FRAGMENT_SHADER, FILTER_FRAGMENT_SHADER);
    const program = gl.createProgram();
    if (!program) throw new Error('Safari 無法建立濾鏡程式');
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const message = gl.getProgramInfoLog(program) ?? '未知錯誤';
      gl.deleteProgram(program);
      throw new Error('Safari 濾鏡連結失敗：' + message);
    }
    this.program = program;
    gl.useProgram(program);

    const buffer = gl.createBuffer();
    if (!buffer) throw new Error('Safari 無法建立濾鏡頂點資料');
    this.buffer = buffer;
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1, 0, 0,
       1, -1, 1, 0,
      -1,  1, 0, 1,
       1,  1, 1, 1,
    ]), gl.STATIC_DRAW);
    const position = gl.getAttribLocation(program, 'a_position');
    const texCoord = gl.getAttribLocation(program, 'a_texCoord');
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(texCoord);
    gl.vertexAttribPointer(texCoord, 2, gl.FLOAT, false, 16, 8);

    const texture = gl.createTexture();
    if (!texture) throw new Error('Safari 無法建立影片濾鏡材質');
    this.texture = texture;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);
    gl.uniform1i(gl.getUniformLocation(program, 'u_image'), 0);
    gl.uniform1i(gl.getUniformLocation(program, 'u_preset'), FILTER_INDEX[preset]);
    gl.uniform1f(gl.getUniformLocation(program, 'u_strength'), clamp(strength, 0, 1));
    gl.viewport(0, 0, width, height);
  }

  render(video: HTMLVideoElement) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.flush();
  }

  dispose() {
    this.gl.deleteTexture(this.texture);
    this.gl.deleteBuffer(this.buffer);
    this.gl.deleteProgram(this.program);
  }
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

function withTimeout<T>(operation: Promise<T>, milliseconds: number, message: string) {
  return new Promise<T>((resolve, reject) => {
    const timeout = window.setTimeout(() => reject(new Error(message)), milliseconds);
    operation.then(
      (value) => {
        window.clearTimeout(timeout);
        resolve(value);
      },
      (error) => {
        window.clearTimeout(timeout);
        reject(error);
      },
    );
  });
}

function withCancellation<T>(
  operation: Promise<T>,
  isCancelled: () => boolean,
  message = '使用者已取消輸出',
) {
  return new Promise<T>((resolve, reject) => {
    const poll = window.setInterval(() => {
      if (!isCancelled()) return;
      cleanup();
      reject(new Error(message));
    }, 100);
    const cleanup = () => window.clearInterval(poll);
    operation.then(
      (value) => {
        cleanup();
        resolve(value);
      },
      (error) => {
        cleanup();
        reject(error);
      },
    );
  });
}

function sourceOutputSize(video: HTMLVideoElement): [number, number] {
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  const scale = Math.min(1, 1280 / Math.max(sourceWidth, sourceHeight));
  return [even(sourceWidth * scale), even(sourceHeight * scale)];
}

function selectionOutputSize(box: Box): [number, number] {
  const width = Math.max(2, box[2]);
  const height = Math.max(2, box[3]);
  const scale = Math.min(1, 1280 / Math.max(width, height));
  return [even(width * scale), even(height * scale)];
}

function configureOutputCanvas(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  operation: ExportOperation,
) {
  if (operation.kind === 'crop' && operation.selectionBox) {
    [canvas.width, canvas.height] = selectionOutputSize(operation.selectionBox);
    return;
  }
  if (operation.kind === 'filter' || operation.aspect === 'source') {
    [canvas.width, canvas.height] = sourceOutputSize(video);
    return;
  }
  [canvas.width, canvas.height] = OUTPUT_SIZES[operation.aspect];
}

export function interpolateBox(path: TrackPoint[], time: number): Box {
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

export function trackedFrameCrop(
  sourceWidth: number,
  sourceHeight: number,
  box: Box,
  outputAspect: number,
  subjectScale: number,
): Box {
  const [x, y, width, height] = box;
  const scale = clamp(subjectScale, 0.25, 0.8);
  let cropHeight = Math.max(height / scale, width / (outputAspect * Math.min(0.9, scale + 0.12)));
  cropHeight = Math.min(cropHeight, sourceHeight, sourceWidth / outputAspect);
  const cropWidth = cropHeight * outputAspect;
  const centerX = x + width / 2;
  const centerY = y + height / 2;
  const cropX = clamp(centerX - cropWidth / 2, 0, Math.max(0, sourceWidth - cropWidth));
  const cropY = clamp(centerY - cropHeight / 2, 0, Math.max(0, sourceHeight - cropHeight));
  return [cropX, cropY, cropWidth, cropHeight];
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
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  const outputAspect = canvas.width / canvas.height;
  const [cropX, cropY, cropWidth, cropHeight] = trackedFrameCrop(
    sourceWidth,
    sourceHeight,
    interpolateBox(path, time),
    outputAspect,
    subjectScale,
  );

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
  if (operation.selectionBox) {
    const [rawX, rawY, rawWidth, rawHeight] = operation.selectionBox;
    const cropX = clamp(rawX, 0, Math.max(0, sourceWidth - 2));
    const cropY = clamp(rawY, 0, Math.max(0, sourceHeight - 2));
    const cropWidth = clamp(rawWidth, 2, sourceWidth - cropX);
    const cropHeight = clamp(rawHeight, 2, sourceHeight - cropY);
    context.filter = 'none';
    context.fillStyle = '#000';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(video, cropX, cropY, cropWidth, cropHeight, 0, 0, canvas.width, canvas.height);
    return;
  }
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
  renderer: WebGLFilterRenderer,
) {
  const context = canvas.getContext('2d', { alpha: false });
  if (!context || !video.videoWidth || !video.videoHeight) return;
  context.fillStyle = '#000';
  context.fillRect(0, 0, canvas.width, canvas.height);
  renderer.render(video);
  context.filter = 'none';
  context.drawImage(renderer.canvas, 0, 0, canvas.width, canvas.height);
  context.filter = 'none';
}

function drawOutputFrame(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  path: TrackPoint[],
  time: number,
  operation: ExportOperation,
  filterRenderer: WebGLFilterRenderer | null,
  backgroundRenderer: PersonBackgroundRenderer | null,
  backgroundTimeline: ModnetPreviewTimeline | null,
) {
  if (operation.kind === 'track') {
    drawTrackedFrame(video, canvas, path, time, operation.subjectScale);
    return;
  }
  if (operation.kind === 'crop') {
    drawCroppedFrame(video, canvas, operation);
    return;
  }
  if (operation.kind === 'remove-background') {
    const trackedBox = interpolateBox(path, time);
    const crop = trackedFrameCrop(
      video.videoWidth,
      video.videoHeight,
      trackedBox,
      canvas.width / canvas.height,
      operation.subjectScale,
    );
    if (backgroundTimeline) {
      backgroundTimeline.draw(
        video,
        canvas,
        trackedBox,
        operation.bodyTightness,
        crop,
        operation.blackOutline,
      );
      return;
    }
    if (!backgroundRenderer) throw new Error('人物去背模型尚未就緒');
    backgroundRenderer.render(video, canvas, trackedBox, crop, operation.bodyTightness);
    return;
  }
  if (!filterRenderer) throw new Error('濾鏡輸出器尚未就緒');
  drawFilteredFrame(video, canvas, filterRenderer);
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
    await new Promise<void>((resolve, reject) => {
      const timeout = window.setTimeout(
        () => reject(new Error('Safari 啟用原聲逾時，請重新點一次輸出')),
        6000,
      );
      this.audioContext!.resume().then(
        () => {
          window.clearTimeout(timeout);
          resolve();
        },
        (error) => {
          window.clearTimeout(timeout);
          reject(error);
        },
      );
    });
    return this.destination!.stream.getAudioTracks();
  }

  async unlockMediaForExport() {
    const audioReady = this.prepareAudio();
    const playbackReady = withTimeout(
      this.video.play(),
      6000,
      'Safari 預先啟用影片播放逾時，請重新點一次輸出',
    );
    try {
      await Promise.all([audioReady, playbackReady]);
    } finally {
      this.video.pause();
    }
  }

  async export(
    path: TrackPoint[],
    canvas: HTMLCanvasElement,
    options: ExportOptions,
  ): Promise<ExportResult> {
    if ((options.operation.kind === 'track' || options.operation.kind === 'remove-background') && path.length < 2) {
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
    const smoothedPath = options.operation.kind === 'track' || options.operation.kind === 'remove-background'
      ? smoothTrackPath(path, options.operation.smoothness)
      : path;
    const filterRenderer = options.operation.kind === 'filter'
      ? new WebGLFilterRenderer(width, height, options.operation.preset, options.operation.strength)
      : null;
    const originalTime = this.video.currentTime;
    const originalRate = this.video.playbackRate;
    options.onStage?.('audio');
    const audioTracks = await withCancellation(this.prepareAudio(), options.isCancelled);
    options.onStage?.('recorder');
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
    let backgroundRenderer: PersonBackgroundRenderer | null = null;
    try {
      if (options.operation.kind === 'remove-background' && !options.backgroundTimeline) {
        const { PersonBackgroundRenderer: Renderer } = await import('./person-background-removal');
        backgroundRenderer = await Renderer.create();
      }
      this.video.pause();
      options.onStage?.('seek');
      await withCancellation(seek(this.video, 0), options.isCancelled);
      this.video.playbackRate = 1;
      drawOutputFrame(this.video, canvas, smoothedPath, 0, options.operation, filterRenderer, backgroundRenderer, options.backgroundTimeline ?? null);
      recorder.start(1000);
      recorderStarted = true;
      options.onStage?.('playback');

      await new Promise<void>((resolve, reject) => {
        let finished = false;
        const playbackTimeout = window.setTimeout(
          () => settle(new Error('Safari 啟動影片編碼播放逾時，請重新點一次輸出')),
          10000,
        );
        const cancelPoll = window.setInterval(() => {
          if (options.isCancelled()) settle(new Error('使用者已取消輸出'));
        }, 100);
        const cleanup = () => {
          window.clearTimeout(playbackTimeout);
          window.clearInterval(cancelPoll);
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
          window.clearTimeout(playbackTimeout);
          if (options.isCancelled()) {
            this.video.pause();
            settle(new Error('使用者已取消輸出'));
            return false;
          }
          try {
            drawOutputFrame(this.video, canvas, smoothedPath, mediaTime, options.operation, filterRenderer, backgroundRenderer, options.backgroundTimeline ?? null);
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
          window.clearTimeout(playbackTimeout);
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
      filterRenderer?.dispose();
      backgroundRenderer?.close();
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
