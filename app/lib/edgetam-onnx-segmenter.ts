import * as ort from 'onnxruntime-web/webgpu';

const SIZE = 1024;
const MASK_SIZE = 256;
const MASK_VALUES = MASK_SIZE * MASK_SIZE;
const IMAGE_VALUES = 3 * SIZE * SIZE;
const NMM = 7;
const MAX_POINTERS = 16;
const SPATIAL_TOKENS = 512;
const MEM_CHANNELS = 64;
const SPATIAL_VALUES = SPATIAL_TOKENS * MEM_CHANNELS;
const MEMORY_TOKENS = NMM * SPATIAL_TOKENS + MAX_POINTERS * 4;
const MEMORY_VALUES = MEMORY_TOKENS * MEM_CHANNELS;
const START_INPUT_VALUES = IMAGE_VALUES + 768;
const TRACK_INPUT_VALUES = IMAGE_VALUES + MEMORY_VALUES * 2 + MEMORY_TOKENS;
const OUTPUT_VALUES = MASK_VALUES + 256 + SPATIAL_VALUES * 2;
const NO_OBJECT_LOGIT = -1024;
const MEAN = [0.485, 0.456, 0.406] as const;
const STD = [0.229, 0.224, 0.225] as const;

type SpatialMemory = {
  frame: number;
  values: Float32Array;
  positions: Float32Array;
};

type ObjectPointer = {
  frame: number;
  values: Float32Array;
};

type GpuNavigator = Navigator & {
  gpu?: {
    requestAdapter(): Promise<{
      features: { has(name: string): boolean };
    } | null>;
  };
};

export type EdgeTamMask = {
  width: number;
  height: number;
  alpha: Uint8ClampedArray;
  foregroundPixels: number;
  inferenceMs: number;
};

export type EdgeTamLoadProgress = (progress: number, label: string) => void;

function assetUrl(base: string, file: string) {
  return new URL(file, base.endsWith('/') ? base : base + '/').href;
}

async function readFloats(url: string, expectedLength: number) {
  const response = await fetch(url);
  if (!response.ok) throw new Error('無法載入去背模型常數：HTTP ' + response.status);
  const values = new Float32Array(await response.arrayBuffer());
  if (values.length !== expectedLength) {
    throw new Error('去背模型常數長度錯誤：' + values.length + ' / ' + expectedLength);
  }
  return new Float32Array(values);
}

async function fetchModelBytes(
  url: string,
  onProgress?: (progress: number) => void,
  isCancelled?: () => boolean,
) {
  if (isCancelled?.()) throw new Error('使用者已取消去背');
  const response = await fetch(url);
  if (!response.ok) throw new Error('無法下載去背模型：HTTP ' + response.status);
  const total = Number(response.headers.get('content-length')) || 0;
  if (!response.body) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (isCancelled?.()) throw new Error('使用者已取消去背');
    onProgress?.(1);
    return bytes;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  while (true) {
    if (isCancelled?.()) {
      await reader.cancel();
      throw new Error('使用者已取消去背');
    }
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total > 0) onProgress?.(Math.min(1, received / total));
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  onProgress?.(1);
  return bytes;
}

export async function assertEdgeTamWebGpuSupport() {
  const gpu = (navigator as GpuNavigator).gpu;
  if (!gpu) throw new Error('這台裝置沒有 WebGPU；仍可使用原本的無特效輸出');
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error('Safari 無法取得 WebGPU；仍可使用原本的無特效輸出');
  if (!adapter.features.has('shader-f16')) {
    throw new Error('這台裝置缺少 WebGPU FP16；為避免等待過久，去背已停用');
  }
}

function cleanSelectedComponent(
  logits: Float32Array,
  width: number,
  height: number,
  seed?: [number, number],
) {
  if (logits.length !== width * height) throw new Error('遮罩尺寸與像素數量不符');
  const foreground = new Uint8Array(logits.length);
  for (let index = 0; index < logits.length; index += 1) {
    if (logits[index] > 0) foreground[index] = 1;
  }

  const labelComponents = (binary: Uint8Array) => {
    const labels = new Int32Array(binary.length);
    const queue = new Int32Array(binary.length);
    let nextLabel = 0;
    let largestLabel = 0;
    let largestSize = 0;
    const sizes = [0];
    for (let start = 0; start < binary.length; start += 1) {
      if (!binary[start] || labels[start] !== 0) continue;
      nextLabel += 1;
      let head = 0;
      let tail = 1;
      let size = 0;
      labels[start] = nextLabel;
      queue[0] = start;
      while (head < tail) {
        const index = queue[head++];
        size += 1;
        const x = index % width;
        const y = Math.floor(index / width);
        for (let nextY = Math.max(0, y - 1); nextY <= Math.min(height - 1, y + 1); nextY += 1) {
          for (let nextX = Math.max(0, x - 1); nextX <= Math.min(width - 1, x + 1); nextX += 1) {
            const neighbor = nextY * width + nextX;
            if (labels[neighbor] !== 0 || !binary[neighbor]) continue;
            labels[neighbor] = nextLabel;
            queue[tail++] = neighbor;
          }
        }
      }
      sizes[nextLabel] = size;
      if (size > largestSize) {
        largestSize = size;
        largestLabel = nextLabel;
      }
    }
    return { labels, largestLabel, largestSize, sizes };
  };

  const seededLabel = (
    components: ReturnType<typeof labelComponents>,
    point?: [number, number],
  ) => {
    if (!point) return components.largestLabel;
    const seedX = Math.max(0, Math.min(width - 1, Math.round(point[0])));
    const seedY = Math.max(0, Math.min(height - 1, Math.round(point[1])));
    const direct = components.labels[seedY * width + seedX];
    if (direct) return direct;
    let closestLabel = 0;
    let closestDistance = Number.POSITIVE_INFINITY;
    for (let index = 0; index < components.labels.length; index += 1) {
      const label = components.labels[index];
      if (!label || components.sizes[label] < 4) continue;
      const x = index % width;
      const y = Math.floor(index / width);
      const distance = (x - seedX) ** 2 + (y - seedY) ** 2;
      if (distance < closestDistance) {
        closestDistance = distance;
        closestLabel = label;
      }
    }
    return closestLabel || components.largestLabel;
  };

  const original = labelComponents(foreground);
  const originalLabel = seededLabel(original, seed);
  if (originalLabel === 0) return { foregroundPixels: 0, centroid: seed ?? [width / 2, height / 2] };
  const selected = new Uint8Array(foreground.length);
  for (let index = 0; index < selected.length; index += 1) {
    if (original.labels[index] === originalLabel) selected[index] = 1;
  }
  const selectedSize = original.sizes[originalLabel];

  // Remove one-pixel bridges before selecting the component. Thin background
  // objects that merely touch an arm or leg are separated, while the person's
  // torso and limb cores remain. The original alpha edge is restored below.
  const eroded = new Uint8Array(selected.length);
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const index = y * width + x;
      if (!selected[index]) continue;
      let solid = true;
      for (let offsetY = -1; offsetY <= 1 && solid; offsetY += 1) {
        for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
          if (!selected[index + offsetY * width + offsetX]) {
            solid = false;
            break;
          }
        }
      }
      if (solid) eroded[index] = 1;
    }
  }
  const core = labelComponents(eroded);
  const coreLabel = seededLabel(core, seed);
  const coreSize = core.sizes[coreLabel] ?? 0;
  const useCore = coreLabel !== 0 && coreSize >= Math.max(8, selectedSize * 0.12);
  let keep = new Uint8Array(logits.length);
  if (useCore) {
    for (let index = 0; index < keep.length; index += 1) {
      if (core.labels[index] === coreLabel) keep[index] = 1;
    }
    for (let pass = 0; pass < 2; pass += 1) {
      const expanded = new Uint8Array(keep);
      for (let index = 0; index < keep.length; index += 1) {
        if (!keep[index]) continue;
        const x = index % width;
        const y = Math.floor(index / width);
        for (let nextY = Math.max(0, y - 1); nextY <= Math.min(height - 1, y + 1); nextY += 1) {
          for (let nextX = Math.max(0, x - 1); nextX <= Math.min(width - 1, x + 1); nextX += 1) {
            expanded[nextY * width + nextX] = 1;
          }
        }
      }
      keep = expanded;
    }
    for (let index = 0; index < keep.length; index += 1) keep[index] &= selected[index];
  } else {
    keep.set(selected);
  }

  let keptForeground = 0;
  let centerX = 0;
  let centerY = 0;
  for (let index = 0; index < keep.length; index += 1) {
    if (!keep[index]) continue;
    keptForeground += 1;
    centerX += index % width;
    centerY += Math.floor(index / width);
  }
  // Do not expand the edge into nearby background pixels. The model's positive
  // logits already retain a fractional alpha edge; extra dilation becomes a
  // visible coloured halo after a small subject is enlarged for 9:16 output.
  for (let index = 0; index < logits.length; index += 1) {
    if (!keep[index]) logits[index] = NO_OBJECT_LOGIT;
  }
  return {
    foregroundPixels: keptForeground,
    centroid: keptForeground > 0
      ? [centerX / keptForeground, centerY / keptForeground] as [number, number]
      : seed ?? [width / 2, height / 2] as [number, number],
  };
}

export function retainLargestComponent(
  logits: Float32Array,
  width = MASK_SIZE,
  height = MASK_SIZE,
) {
  return cleanSelectedComponent(logits, width, height).foregroundPixels;
}

function alphaFromLogits(logits: Float32Array) {
  const alpha = new Uint8ClampedArray(logits.length);
  for (let index = 0; index < logits.length; index += 1) {
    const value = logits[index];
    alpha[index] = value <= -0.5 ? 0 : value >= 0.5 ? 255 : Math.round((value + 0.5) * 255);
  }
  return alpha;
}

export class EdgeTamOnnxSegmenter {
  private readonly startSession: ort.InferenceSession;
  private readonly trackSession: ort.InferenceSession;
  private readonly prompt: Float32Array;
  private readonly temporalPositions: Float32Array;
  private readonly inputCanvas = document.createElement('canvas');
  private readonly inputContext: CanvasRenderingContext2D;
  private readonly startInput = new Float32Array(START_INPUT_VALUES);
  private readonly trackInput = new Float32Array(TRACK_INPUT_VALUES);
  private readonly spatialBank: SpatialMemory[] = [];
  private readonly pointerBank: ObjectPointer[] = [];
  private frameIndex = -1;
  private previousArea = 0;
  private previousCentroid: [number, number] | null = null;
  private selfTested = false;

  private constructor(
    startSession: ort.InferenceSession,
    trackSession: ort.InferenceSession,
    prompt: Float32Array,
    temporalPositions: Float32Array,
  ) {
    this.startSession = startSession;
    this.trackSession = trackSession;
    this.prompt = prompt;
    this.temporalPositions = temporalPositions;
    this.inputCanvas.width = SIZE;
    this.inputCanvas.height = SIZE;
    const context = this.inputCanvas.getContext('2d', { alpha: false, willReadFrequently: true });
    if (!context) throw new Error('Safari 無法建立去背影像畫布');
    this.inputContext = context;
  }

  static async create(
    modelBaseUrl: string,
    wasmBaseUrl: string,
    onProgress?: EdgeTamLoadProgress,
    isCancelled?: () => boolean,
  ) {
    await assertEdgeTamWebGpuSupport();
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    ort.env.wasm.wasmPaths = wasmBaseUrl;
    ort.env.webgpu.powerPreference = 'high-performance';

    onProgress?.(0.02, '載入主角提示');
    const [prompt, temporalPositions] = await Promise.all([
      readFloats(assetUrl(modelBaseUrl, 'box_prompt.bin'), 1024),
      readFloats(assetUrl(modelBaseUrl, 'mtpe.bin'), NMM * MEM_CHANNELS),
    ]);

    // ORT WebGPU session compilation must be sequential on Safari.
    let lastStartPercent = -1;
    let startBytes: Uint8Array | null = await fetchModelBytes(assetUrl(modelBaseUrl, 'start.onnx'), (progress) => {
      const percent = Math.round(progress * 100);
      if (percent === lastStartPercent) return;
      lastStartPercent = percent;
      onProgress?.(0.05 + progress * 0.27, '下載首幀去背模型 · ' + percent + '%');
    }, isCancelled);
    onProgress?.(0.34, '編譯首幀去背');
    const startSession = await ort.InferenceSession.create(startBytes, {
      executionProviders: ['webgpu'],
      graphOptimizationLevel: 'all',
      executionMode: 'sequential',
    });
    startBytes = null;
    try {
      let lastTrackPercent = -1;
      let trackBytes: Uint8Array | null = await fetchModelBytes(assetUrl(modelBaseUrl, 'track.onnx'), (progress) => {
        const percent = Math.round(progress * 100);
        if (percent === lastTrackPercent) return;
        lastTrackPercent = percent;
        onProgress?.(0.48 + progress * 0.35, '下載連續去背模型 · ' + percent + '%');
      }, isCancelled);
      onProgress?.(0.86, '編譯連續去背');
      const trackSession = await ort.InferenceSession.create(trackBytes, {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
        executionMode: 'sequential',
      });
      trackBytes = null;
      onProgress?.(1, '去背模型就緒');
      return new EdgeTamOnnxSegmenter(startSession, trackSession, prompt, temporalPositions);
    } catch (error) {
      await startSession.release();
      throw error;
    }
  }

  reset() {
    this.frameIndex = -1;
    this.previousArea = 0;
    this.previousCentroid = null;
    this.spatialBank.length = 0;
    this.pointerBank.length = 0;
  }

  private preprocess(image: CanvasImageSource, destination: Float32Array) {
    this.inputContext.drawImage(image, 0, 0, SIZE, SIZE);
    const pixels = this.inputContext.getImageData(0, 0, SIZE, SIZE).data;
    const plane = SIZE * SIZE;
    for (let index = 0; index < plane; index += 1) {
      const pixel = index * 4;
      destination[index] = (pixels[pixel] / 255 - MEAN[0]) / STD[0];
      destination[plane + index] = (pixels[pixel + 1] / 255 - MEAN[1]) / STD[1];
      destination[plane * 2 + index] = (pixels[pixel + 2] / 255 - MEAN[2]) / STD[2];
    }
  }

  private boxSparse(normalizedBox: [number, number, number, number]) {
    const sparse = new Float32Array(768);
    const gaussian = this.prompt.subarray(0, 256);
    const embeddings = [this.prompt.subarray(256, 512), this.prompt.subarray(512, 768)] as const;
    const left = Math.max(0, Math.min(1, normalizedBox[0]));
    const top = Math.max(0, Math.min(1, normalizedBox[1]));
    const right = Math.max(left, Math.min(1, normalizedBox[0] + normalizedBox[2]));
    const bottom = Math.max(top, Math.min(1, normalizedBox[1] + normalizedBox[3]));
    const corners = [[left, top], [right, bottom]] as const;
    corners.forEach(([normalizedX, normalizedY], corner) => {
      const ccx = 2 * (normalizedX + 0.5 / SIZE) - 1;
      const ccy = 2 * (normalizedY + 0.5 / SIZE) - 1;
      const embedding = embeddings[corner];
      const offset = corner * 256;
      for (let index = 0; index < 128; index += 1) {
        const projection = Math.PI * 2 * (
          ccx * gaussian[index] + ccy * gaussian[128 + index]
        );
        sparse[offset + index] = Math.sin(projection) + embedding[index];
        sparse[offset + 128 + index] = Math.cos(projection) + embedding[128 + index];
      }
    });
    sparse.set(this.prompt.subarray(768, 1024), 512);
    return sparse;
  }

  private sinePosition(offset: number) {
    const result = new Float32Array(MEM_CHANNELS);
    const position = offset / 15;
    for (let index = 0; index < 32; index += 1) {
      const dimension = Math.pow(10000, (2 * Math.floor(index / 2)) / 32);
      const value = position / dimension;
      result[index] = Math.sin(value);
      result[32 + index] = Math.cos(value);
    }
    return result;
  }

  private assembleMemory(frame: number) {
    if (this.spatialBank.length === 0) throw new Error('去背模型尚未指定主角');
    const memoryOffset = IMAGE_VALUES;
    const positionOffset = memoryOffset + MEMORY_VALUES;
    const keyMaskOffset = positionOffset + MEMORY_VALUES;
    this.trackInput.fill(0, memoryOffset, keyMaskOffset);
    this.trackInput.fill(-1e9, keyMaskOffset);

    const conditioningFrame = this.spatialBank[0].frame;
    const spatial: Array<{ item: SpatialMemory; temporalIndex: number }> = [
      { item: this.spatialBank[0], temporalIndex: NMM - 1 },
    ];
    for (let offset = NMM - 1; offset >= 1; offset -= 1) {
      const wanted = frame - offset;
      if (wanted === conditioningFrame) continue;
      const match = this.spatialBank.find((item) => item.frame === wanted);
      if (match) spatial.push({ item: match, temporalIndex: offset - 1 });
    }

    spatial.forEach(({ item, temporalIndex }, slot) => {
      const base = slot * SPATIAL_VALUES;
      this.trackInput.set(item.values, memoryOffset + base);
      for (let token = 0; token < SPATIAL_TOKENS; token += 1) {
        const tokenBase = base + token * MEM_CHANNELS;
        for (let channel = 0; channel < MEM_CHANNELS; channel += 1) {
          this.trackInput[positionOffset + tokenBase + channel] =
            item.positions[token * MEM_CHANNELS + channel]
            + this.temporalPositions[temporalIndex * MEM_CHANNELS + channel];
        }
      }
    });
    this.trackInput.fill(0, keyMaskOffset, keyMaskOffset + spatial.length * SPATIAL_TOKENS);

    const recentPointers = [...this.pointerBank]
      .sort((left, right) => right.frame - left.frame)
      .slice(0, MAX_POINTERS);
    let pointerToken = 0;
    const pointerBase = NMM * SPATIAL_VALUES;
    for (const pointer of recentPointers) {
      const position = this.sinePosition(frame - pointer.frame);
      for (let token = 0; token < 4; token += 1) {
        const destination = pointerBase + pointerToken * MEM_CHANNELS;
        this.trackInput.set(
          pointer.values.subarray(token * MEM_CHANNELS, (token + 1) * MEM_CHANNELS),
          memoryOffset + destination,
        );
        this.trackInput.set(position, positionOffset + destination);
        this.trackInput[keyMaskOffset + NMM * SPATIAL_TOKENS + pointerToken] = 0;
        pointerToken += 1;
      }
    }
  }

  private async run(session: ort.InferenceSession, values: Float32Array) {
    const tensor = new ort.Tensor('float32', values, [1, values.length]);
    const started = performance.now();
    try {
      const outputs = await session.run({ input: tensor });
      const output = outputs.output;
      if (!output) throw new Error('去背模型沒有輸出');
      try {
        const data = output.data;
        if (!(data instanceof Float32Array) || data.length !== OUTPUT_VALUES) {
          throw new Error('去背模型輸出格式錯誤');
        }
        const copy = new Float32Array(data);
        for (let index = 0; index < copy.length; index += 1) {
          if (!Number.isFinite(copy[index])) throw new Error('去背模型產生非數值結果');
        }
        return { values: copy, inferenceMs: performance.now() - started };
      } finally {
        output.dispose();
      }
    } finally {
      tensor.dispose();
    }
  }

  private storeOutput(
    frame: number,
    output: Float32Array,
    inferenceMs: number,
    seed?: [number, number],
  ): EdgeTamMask {
    const logits = new Float32Array(output.subarray(0, MASK_VALUES));
    const pointer = new Float32Array(output.subarray(MASK_VALUES, MASK_VALUES + 256));
    const memoryStart = MASK_VALUES + 256;
    const values = new Float32Array(output.subarray(memoryStart, memoryStart + SPATIAL_VALUES));
    const positions = new Float32Array(output.subarray(memoryStart + SPATIAL_VALUES));

    const cleaned = cleanSelectedComponent(logits, MASK_SIZE, MASK_SIZE, seed ?? this.previousCentroid ?? undefined);
    const foregroundPixels = cleaned.foregroundPixels;
    if (foregroundPixels < 32 || foregroundPixels > MASK_VALUES * 0.6) {
      throw new Error('去背遮罩範圍異常，已停止以免輸出錯誤畫面');
    }
    if (this.previousArea > 0) {
      const ratio = foregroundPixels / this.previousArea;
      if (ratio < 0.25 || ratio > 4) {
        throw new Error('去背遮罩突然失去主角，已停止以免畫面閃爍');
      }
    }

    this.spatialBank.push({ frame, values, positions });
    this.pointerBank.push({ frame, values: pointer });
    if (this.spatialBank.length > NMM) {
      const conditioning = this.spatialBank[0];
      const recent = this.spatialBank.slice(-(NMM - 1)).filter((item) => item.frame !== conditioning.frame);
      this.spatialBank.splice(0, this.spatialBank.length, conditioning, ...recent);
    }
    if (this.pointerBank.length > MAX_POINTERS) {
      this.pointerBank.splice(0, this.pointerBank.length - MAX_POINTERS);
    }
    this.previousArea = foregroundPixels;
    this.previousCentroid = cleaned.centroid;
    return {
      width: MASK_SIZE,
      height: MASK_SIZE,
      alpha: alphaFromLogits(logits),
      foregroundPixels,
      inferenceMs,
    };
  }

  async start(image: CanvasImageSource, normalizedBox: [number, number, number, number]) {
    this.reset();
    this.preprocess(image, this.startInput);
    this.startInput.set(this.boxSparse(normalizedBox), IMAGE_VALUES);

    // The failed LiteRT build changed results when the same frame was run twice.
    // Verify determinism on the real first frame before accepting any mask.
    const first = await this.run(this.startSession, this.startInput);
    let accepted = first;
    if (!this.selfTested) {
      const repeated = await this.run(this.startSession, this.startInput);
      let maxDelta = 0;
      for (let index = 0; index < MASK_VALUES; index += 1) {
        maxDelta = Math.max(maxDelta, Math.abs(first.values[index] - repeated.values[index]));
      }
      if (maxDelta > 0.02) {
        throw new Error('WebGPU 重複推論不穩定，已停用去背以避免花畫面');
      }
      accepted = repeated;
      this.selfTested = true;
    }
    this.frameIndex = 0;
    const seed: [number, number] = [
      (normalizedBox[0] + normalizedBox[2] / 2) * MASK_SIZE,
      (normalizedBox[1] + normalizedBox[3] / 2) * MASK_SIZE,
    ];
    return this.storeOutput(this.frameIndex, accepted.values, accepted.inferenceMs, seed);
  }

  async track(image: CanvasImageSource) {
    if (this.frameIndex < 0) throw new Error('去背模型尚未指定主角');
    this.frameIndex += 1;
    this.preprocess(image, this.trackInput);
    this.assembleMemory(this.frameIndex);
    const output = await this.run(this.trackSession, this.trackInput);
    return this.storeOutput(this.frameIndex, output.values, output.inferenceMs);
  }

  async close() {
    this.reset();
    await this.startSession.release();
    await this.trackSession.release();
  }
}
