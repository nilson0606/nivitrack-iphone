import * as ort from 'onnxruntime-web/webgpu';

const WIDTH = 288;
const HEIGHT = 512;
const FEATURE_WIDTH = 18;
const FEATURE_HEIGHT = 32;
const FEATURE_PIXELS = FEATURE_WIDTH * FEATURE_HEIGHT;
const KEY_DIM = 64;
const VALUE_DIM = 256;
const MAX_TEMP_FRAMES = 4;
const MAX_MEMORY_FRAMES = 1 + MAX_TEMP_FRAMES;
const MAX_TOKENS = MAX_MEMORY_FRAMES * FEATURE_PIXELS;
const MEMORY_EVERY = 5;

const STAGE_NAMES = [
  'encoder',
  'uncert',
  'memory',
  'readout',
  'decoder',
  'maskencoder',
  'objsummary',
] as const;

type StageName = (typeof STAGE_NAMES)[number];
type TensorMap = Record<string, ort.Tensor>;
type MemoryFrame = {
  key: Uint16Array;
  shrinkage: Uint16Array;
  value: Uint16Array;
};

export type MatAnyoneLoadProgress = {
  loaded: number;
  total: number;
  stage: StageName;
  backend: 'webgpu' | 'wasm';
};

export type MatteResult = {
  alpha: Float32Array;
  inferenceMs: number;
};

const scalarFloat = new Float32Array(1);
const scalarBits = new Uint32Array(scalarFloat.buffer);

function floatToHalf(value: number) {
  scalarFloat[0] = value;
  const bits = scalarBits[0];
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  let mantissa = bits & 0x7fffff;
  if (exponent === 0xff) return sign | (mantissa ? 0x7e00 : 0x7c00);
  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >>> (1 - halfExponent);
    return sign | ((mantissa + 0x1000) >>> 13);
  }
  return sign | (halfExponent << 10) | ((mantissa + 0x1000) >>> 13);
}

function halfToFloat(value: number) {
  const sign = (value & 0x8000) << 16;
  let exponent = (value >>> 10) & 0x1f;
  let mantissa = value & 0x3ff;
  if (exponent === 0) {
    if (mantissa === 0) {
      scalarBits[0] = sign;
      return scalarFloat[0];
    }
    exponent = 1;
    while ((mantissa & 0x400) === 0) {
      mantissa <<= 1;
      exponent -= 1;
    }
    mantissa &= 0x3ff;
    exponent += 112;
  } else if (exponent === 0x1f) {
    scalarBits[0] = sign | 0x7f800000 | (mantissa << 13);
    return scalarFloat[0];
  } else {
    exponent += 112;
  }
  scalarBits[0] = sign | (exponent << 23) | (mantissa << 13);
  return scalarFloat[0];
}

function floatsToHalf(values: Float32Array | number[]) {
  const output = new Uint16Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    output[index] = floatToHalf(values[index]);
  }
  return output;
}

function rawHalf(tensor: ort.Tensor) {
  const data = tensor.data as ArrayBufferView;
  return new Uint16Array(data.buffer, data.byteOffset, data.byteLength / 2);
}

function halfToFloats(tensor: ort.Tensor) {
  const data = rawHalf(tensor);
  const output = new Float32Array(data.length);
  for (let index = 0; index < data.length; index += 1) {
    output[index] = halfToFloat(data[index]);
  }
  return output;
}

function halfTensor(data: Uint16Array, dims: readonly number[]) {
  return new ort.Tensor('float16', data, dims);
}

function tensorView(tensor: ort.Tensor, dims: readonly number[]) {
  return halfTensor(rawHalf(tensor), dims);
}

function disposeMap(values: TensorMap, keep: Set<ort.Tensor> = new Set()) {
  for (const value of Object.values(values)) {
    if (!keep.has(value)) value.dispose();
  }
}

class BrowserMemoryBank {
  private permanent: MemoryFrame | null = null;
  private temporary: MemoryFrame[] = [];
  private objectMemory = new Float32Array(16 * (VALUE_DIM + 1));
  private cached: {
    key: ort.Tensor;
    shrinkage: ort.Tensor;
    value: ort.Tensor;
    valid: ort.Tensor;
  } | null = null;

  get engaged() {
    return this.permanent !== null || this.temporary.length > 0;
  }

  clear() {
    this.permanent = null;
    this.temporary = [];
    this.objectMemory.fill(0);
    this.disposeCache();
  }

  add(key: ort.Tensor, shrinkage: ort.Tensor, value: ort.Tensor, summary: ort.Tensor) {
    const frame: MemoryFrame = {
      key: new Uint16Array(rawHalf(key)),
      shrinkage: new Uint16Array(rawHalf(shrinkage)),
      value: new Uint16Array(rawHalf(value)),
    };
    if (!this.permanent) this.permanent = frame;
    else {
      this.temporary.push(frame);
      if (this.temporary.length > MAX_TEMP_FRAMES) this.temporary.shift();
    }
    const summaryValues = halfToFloats(summary);
    for (let index = 0; index < this.objectMemory.length; index += 1) {
      this.objectMemory[index] += summaryValues[index] ?? 0;
    }
    this.disposeCache();
  }

  objectTensor() {
    return halfTensor(floatsToHalf(this.objectMemory), [1, 1, 1, 16, VALUE_DIM + 1]);
  }

  tensors() {
    if (this.cached) return this.cached;
    const frames = [this.permanent, ...this.temporary].filter(
      (frame): frame is MemoryFrame => frame !== null,
    );
    const key = new Uint16Array(KEY_DIM * MAX_TOKENS);
    const shrinkage = new Uint16Array(MAX_TOKENS);
    const value = new Uint16Array(VALUE_DIM * MAX_TOKENS);
    for (let frameIndex = 0; frameIndex < frames.length; frameIndex += 1) {
      const offset = frameIndex * FEATURE_PIXELS;
      const frame = frames[frameIndex];
      shrinkage.set(frame.shrinkage, offset);
      for (let channel = 0; channel < KEY_DIM; channel += 1) {
        key.set(
          frame.key.subarray(
            channel * FEATURE_PIXELS,
            (channel + 1) * FEATURE_PIXELS,
          ),
          channel * MAX_TOKENS + offset,
        );
      }
      for (let channel = 0; channel < VALUE_DIM; channel += 1) {
        value.set(
          frame.value.subarray(
            channel * FEATURE_PIXELS,
            (channel + 1) * FEATURE_PIXELS,
          ),
          channel * MAX_TOKENS + offset,
        );
      }
    }
    const valid = new Uint16Array(MAX_TOKENS);
    valid.fill(floatToHalf(1), 0, frames.length * FEATURE_PIXELS);
    this.cached = {
      key: halfTensor(key, [1, KEY_DIM, MAX_TOKENS]),
      shrinkage: halfTensor(shrinkage, [1, 1, MAX_TOKENS]),
      value: halfTensor(value, [1, VALUE_DIM, MAX_TOKENS]),
      valid: halfTensor(valid, [1, MAX_TOKENS, 1]),
    };
    return this.cached;
  }

  private disposeCache() {
    if (!this.cached) return;
    this.cached.key.dispose();
    this.cached.shrinkage.dispose();
    this.cached.value.dispose();
    this.cached.valid.dispose();
    this.cached = null;
  }
}

export class MatAnyone2Engine {
  readonly width = WIDTH;
  readonly height = HEIGHT;
  readonly backend: 'webgpu' | 'wasm';

  private readonly sessions: Record<StageName, ort.InferenceSession>;
  private readonly canvas = document.createElement('canvas');
  private readonly context: CanvasRenderingContext2D;
  private readonly memory = new BrowserMemoryBank();
  private currentTimeIndex = -1;
  private lastMemoryTime = 0;
  private lastMask: ort.Tensor | null = null;
  private lastPixelFeatures: ort.Tensor | null = null;
  private lastMaskValue: ort.Tensor | null = null;
  private sensory = halfTensor(
    new Uint16Array(VALUE_DIM * FEATURE_PIXELS),
    [1, 1, VALUE_DIM, FEATURE_HEIGHT, FEATURE_WIDTH],
  );

  private constructor(
    sessions: Record<StageName, ort.InferenceSession>,
    backend: 'webgpu' | 'wasm',
  ) {
    this.sessions = sessions;
    this.backend = backend;
    this.canvas.width = WIDTH;
    this.canvas.height = HEIGHT;
    const context = this.canvas.getContext('2d', {
      alpha: false,
      willReadFrequently: true,
    });
    if (!context) throw new Error('Safari 無法建立 MatAnyone 影格畫布');
    this.context = context;
  }

  static async create(
    modelBaseUrl: string,
    wasmBaseUrl: string,
    onProgress?: (progress: MatAnyoneLoadProgress) => void,
  ) {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    ort.env.wasm.wasmPaths = wasmBaseUrl;
    const preferred: Array<'webgpu' | 'wasm'> = 'gpu' in navigator
      ? ['webgpu', 'wasm']
      : ['wasm'];
    let lastError: unknown;
    for (const backend of preferred) {
      const loaded = {} as Partial<Record<StageName, ort.InferenceSession>>;
      try {
        for (let index = 0; index < STAGE_NAMES.length; index += 1) {
          const stage = STAGE_NAMES[index];
          loaded[stage] = await ort.InferenceSession.create(
            new URL(`${stage}.onnx`, modelBaseUrl).href,
            {
              executionProviders: [backend],
              graphOptimizationLevel: 'all',
              executionMode: 'sequential',
            },
          );
          onProgress?.({
            loaded: index + 1,
            total: STAGE_NAMES.length,
            stage,
            backend,
          });
        }
        return new MatAnyone2Engine(
          loaded as Record<StageName, ort.InferenceSession>,
          backend,
        );
      } catch (error) {
        lastError = error;
        await Promise.all(
          Object.values(loaded).map((session) => session?.release()),
        );
      }
    }
    throw lastError instanceof Error
      ? lastError
      : new Error('MatAnyone 2 模型無法在這台 Safari 載入');
  }

  reset() {
    this.currentTimeIndex = -1;
    this.lastMemoryTime = 0;
    this.lastMask?.dispose();
    this.lastPixelFeatures?.dispose();
    this.lastMaskValue?.dispose();
    this.sensory.dispose();
    this.lastMask = null;
    this.lastPixelFeatures = null;
    this.lastMaskValue = null;
    this.sensory = halfTensor(
      new Uint16Array(VALUE_DIM * FEATURE_PIXELS),
      [1, 1, VALUE_DIM, FEATURE_HEIGHT, FEATURE_WIDTH],
    );
    this.memory.clear();
  }

  async seed(
    source: CanvasImageSource,
    seedMask: Float32Array,
    warmup = 10,
    onProgress?: (progress: number) => void,
  ): Promise<MatteResult> {
    if (seedMask.length !== WIDTH * HEIGHT) {
      throw new Error(`首幀遮罩尺寸錯誤：${seedMask.length}`);
    }
    this.reset();
    const image = this.imageTensor(source);
    const started = performance.now();
    await this.stepTensor(image, seedMask, false);
    let alpha = await this.stepTensor(image, null, true);
    onProgress?.(1 / warmup);
    for (let index = 1; index < warmup; index += 1) {
      alpha = await this.stepTensor(image, null, true);
      onProgress?.((index + 1) / warmup);
    }
    image.dispose();
    return { alpha, inferenceMs: performance.now() - started };
  }

  async step(source: CanvasImageSource): Promise<MatteResult> {
    const image = this.imageTensor(source);
    const started = performance.now();
    const alpha = await this.stepTensor(image, null, false);
    image.dispose();
    return { alpha, inferenceMs: performance.now() - started };
  }

  async close() {
    this.currentTimeIndex = -1;
    this.lastMask?.dispose();
    this.lastPixelFeatures?.dispose();
    this.lastMaskValue?.dispose();
    this.sensory.dispose();
    this.lastMask = null;
    this.lastPixelFeatures = null;
    this.lastMaskValue = null;
    this.memory.clear();
    await Promise.all(Object.values(this.sessions).map((session) => session.release()));
  }

  private imageTensor(source: CanvasImageSource) {
    this.context.drawImage(source, 0, 0, WIDTH, HEIGHT);
    const rgba = this.context.getImageData(0, 0, WIDTH, HEIGHT).data;
    const plane = WIDTH * HEIGHT;
    const data = new Uint16Array(plane * 3);
    for (let index = 0; index < plane; index += 1) {
      const pixel = index * 4;
      data[index] = floatToHalf(rgba[pixel] / 255);
      data[plane + index] = floatToHalf(rgba[pixel + 1] / 255);
      data[plane * 2 + index] = floatToHalf(rgba[pixel + 2] / 255);
    }
    return halfTensor(data, [1, 3, HEIGHT, WIDTH]);
  }

  private async run(stage: StageName, feeds: TensorMap) {
    const session = this.sessions[stage];
    const accepted: TensorMap = {};
    for (const name of session.inputNames) accepted[name] = feeds[name];
    return session.run(accepted);
  }

  private async stepTensor(
    image: ort.Tensor,
    seedMask: Float32Array | null,
    firstFramePrediction: boolean,
  ) {
    this.currentTimeIndex += 1;
    let memoryFrame = this.currentTimeIndex - this.lastMemoryTime >= MEMORY_EVERY || seedMask !== null;
    let needSegment = seedMask === null;
    let updateSensory = this.currentTimeIndex - this.lastMemoryTime >= 1
      && this.currentTimeIndex - this.lastMemoryTime <= MEMORY_EVERY;
    if (firstFramePrediction) {
      this.currentTimeIndex = 0;
      this.lastMemoryTime = 0;
      memoryFrame = true;
      needSegment = true;
      updateSensory = true;
    }

    const encoded = await this.run('encoder', { image });
    let alpha: Float32Array;
    if (needSegment) {
      alpha = await this.segment(encoded, updateSensory);
    } else {
      alpha = seedMask!;
    }

    const previousMask = this.lastMask;
    const previousPixels = this.lastPixelFeatures;
    this.lastMask = halfTensor(floatsToHalf(alpha), [1, 1, HEIGHT, WIDTH]);
    this.lastPixelFeatures = encoded.pix_feat;

    if (firstFramePrediction) this.memory.clear();
    const maskEncoded = await this.run('maskencoder', {
      image,
      pix_feat: encoded.pix_feat,
      sensory: this.sensory,
      masks: this.lastMask,
    });
    const previousMaskValue = this.lastMaskValue;
    this.lastMaskValue = maskEncoded.mask_value;
    if (memoryFrame) {
      const summarized = await this.run('objsummary', {
        masks: this.lastMask,
        mask_value: maskEncoded.mask_value,
      });
      this.memory.add(
        encoded.key,
        encoded.shrinkage,
        maskEncoded.mask_value,
        summarized.obj_summaries,
      );
      summarized.obj_summaries.dispose();
      this.lastMemoryTime = this.currentTimeIndex;
      const previousSensory = this.sensory;
      this.sensory = maskEncoded.new_sensory;
      previousSensory.dispose();
    } else {
      maskEncoded.new_sensory.dispose();
    }

    previousMask?.dispose();
    previousPixels?.dispose();
    previousMaskValue?.dispose();
    const keep = new Set<ort.Tensor>([this.lastPixelFeatures, this.lastMaskValue]);
    disposeMap(encoded, keep);
    return alpha;
  }

  private async segment(encoded: TensorMap, updateSensory: boolean) {
    if (!this.memory.engaged || !this.lastMask || !this.lastMaskValue || !this.lastPixelFeatures) {
      throw new Error('MatAnyone 2 尚未建立首幀主角記憶');
    }
    let visual: ort.Tensor;
    if (this.currentTimeIndex === 0) {
      visual = tensorView(
        this.lastMaskValue,
        [1, 1, VALUE_DIM, FEATURE_HEIGHT, FEATURE_WIDTH],
      );
    } else {
      const memory = this.memory.tensors();
      const queryKey = tensorView(encoded.key, [1, KEY_DIM, FEATURE_PIXELS]);
      const querySelection = tensorView(
        encoded.selection,
        [1, KEY_DIM, FEATURE_PIXELS],
      );
      let matched: TensorMap;
      try {
        matched = await this.run('memory', {
          memory_key: memory.key,
          memory_shrinkage: memory.shrinkage,
          memory_value: memory.value,
          valid_mask: memory.valid,
          query_key: queryKey,
          query_selection: querySelection,
        });
      } finally {
        queryKey.dispose();
        querySelection.dispose();
      }
      const readout = halfToFloats(matched.visual_readout);
      const previous = halfToFloats(this.lastMaskValue);
      const difference = new Float32Array(readout.length);
      for (let index = 0; index < difference.length; index += 1) {
        difference[index] = readout[index] - previous[index];
      }
      const memoryDifference = halfTensor(
        floatsToHalf(difference),
        [1, VALUE_DIM, FEATURE_HEIGHT, FEATURE_WIDTH],
      );
      let uncertainty: TensorMap;
      try {
        uncertainty = await this.run('uncert', {
          last_pix_feat: this.lastPixelFeatures,
          cur_pix_feat: encoded.pix_feat,
          last_mask: this.lastMask,
          mem_val_diff: memoryDifference,
        });
      } finally {
        memoryDifference.dispose();
      }
      const probability = halfToFloats(uncertainty.prob);
      const blended = new Float32Array(readout.length);
      for (let channel = 0; channel < VALUE_DIM; channel += 1) {
        const offset = channel * FEATURE_PIXELS;
        for (let pixel = 0; pixel < FEATURE_PIXELS; pixel += 1) {
          const index = offset + pixel;
          blended[index] = previous[index]
            + probability[pixel] * (readout[index] - previous[index]);
        }
      }
      visual = halfTensor(
        floatsToHalf(blended),
        [1, 1, VALUE_DIM, FEATURE_HEIGHT, FEATURE_WIDTH],
      );
      matched.visual_readout.dispose();
      uncertainty.prob.dispose();
    }

    const objectMemory = this.memory.objectTensor();
    const readout = await this.run('readout', {
      pix_feat: encoded.pix_feat,
      pixel: visual,
      sensory: this.sensory,
      last_mask: this.lastMask,
      obj_memory: objectMemory,
    });
    const decoded = await this.run('decoder', {
      f16: encoded.f16,
      f8: encoded.f8,
      f4: encoded.f4,
      f2: encoded.f2,
      f1: encoded.f1,
      memory_readout: readout.mem_readout,
      sensory: this.sensory,
    });
    if (updateSensory) {
      const previous = this.sensory;
      this.sensory = decoded.new_sensory;
      previous.dispose();
    } else {
      decoded.new_sensory.dispose();
    }
    const alpha = halfToFloats(decoded.logits);
    for (let index = 0; index < alpha.length; index += 1) {
      alpha[index] = Math.max(0, Math.min(1, alpha[index]));
    }
    visual.dispose();
    objectMemory.dispose();
    readout.mem_readout.dispose();
    decoded.logits.dispose();
    return alpha;
  }
}

export const MATANYONE2_SIZE = { width: WIDTH, height: HEIGHT } as const;
