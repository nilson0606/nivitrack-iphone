import {
  CompiledModel,
  Tensor,
  isWebGPUSupported,
  loadAndCompile,
  loadLiteRt,
  type Accelerator,
} from '@litertjs/core';

const SIZE = 1024;
const MASK_SIZE = 256;
const IE = 256 * 64 * 64;
const H0 = 32 * 256 * 256;
const H1 = 64 * 128 * 128;
const NMM = 7;
const MAX_POINTERS = 16;
const MEM_CHANNELS = 64;
const SPATIAL_TOKENS = 512;
const MEMORY_TOKENS = NMM * SPATIAL_TOKENS + MAX_POINTERS * 4;
const MEMORY_VALUES = MEMORY_TOKENS * MEM_CHANNELS;
const MASK_VALUES = MASK_SIZE * MASK_SIZE;
const DECODE_OUTPUT_VALUES = MASK_VALUES * 3 + 3 + 256 * 3 + 1;
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

export type EdgeTamMask = {
  width: number;
  height: number;
  alpha: Uint8ClampedArray;
};

type EdgeTamModels = {
  encode: CompiledModel;
  memcond: CompiledModel;
  decodeBox: CompiledModel;
  decode: CompiledModel;
  memorize: CompiledModel;
};

export type EdgeTamLoadProgress = (step: number, total: number, label: string) => void;

function assetUrl(base: string, file: string) {
  return new URL(file, base.endsWith('/') ? base : base + '/').href;
}

export function retainLargestComponent(
  logits: Float32Array,
  width = MASK_SIZE,
  height = MASK_SIZE,
) {
  if (logits.length !== width * height) throw new Error('遮罩尺寸與像素數量不符');
  const labels = new Int32Array(logits.length);
  const queue = new Int32Array(logits.length);
  const sizes = [0];
  let nextLabel = 0;
  let largestLabel = 0;
  let largestSize = 0;

  for (let start = 0; start < logits.length; start += 1) {
    if (logits[start] <= 0 || labels[start] !== 0) continue;
    nextLabel += 1;
    let head = 0;
    let tail = 0;
    let size = 0;
    labels[start] = nextLabel;
    queue[tail] = start;
    tail += 1;

    while (head < tail) {
      const index = queue[head];
      head += 1;
      size += 1;
      const x = index % width;
      const y = Math.floor(index / width);
      const left = Math.max(0, x - 1);
      const right = Math.min(width - 1, x + 1);
      const top = Math.max(0, y - 1);
      const bottom = Math.min(height - 1, y + 1);
      for (let nextY = top; nextY <= bottom; nextY += 1) {
        for (let nextX = left; nextX <= right; nextX += 1) {
          const neighbor = nextY * width + nextX;
          if (labels[neighbor] !== 0 || logits[neighbor] <= 0) continue;
          labels[neighbor] = nextLabel;
          queue[tail] = neighbor;
          tail += 1;
        }
      }
    }

    sizes[nextLabel] = size;
    if (size > largestSize) {
      largestSize = size;
      largestLabel = nextLabel;
    }
  }

  if (largestLabel === 0) return 0;
  for (let index = 0; index < logits.length; index += 1) {
    if (logits[index] > 0 && labels[index] !== largestLabel) logits[index] = NO_OBJECT_LOGIT;
  }
  return sizes[largestLabel];
}

async function readFloats(url: string, expectedLength: number) {
  const response = await fetch(url);
  if (!response.ok) throw new Error('無法載入 EdgeTAM 記憶常數：' + response.status);
  const bytes = await response.arrayBuffer();
  const values = new Float32Array(bytes);
  if (values.length !== expectedLength) {
    throw new Error('EdgeTAM 記憶常數長度錯誤：' + values.length + ' / ' + expectedLength);
  }
  return new Float32Array(values);
}

async function compileModels(
  baseUrl: string,
  accelerator: Accelerator,
  onLoadProgress?: EdgeTamLoadProgress,
): Promise<EdgeTamModels> {
  const loaded: CompiledModel[] = [];
  const stages = [
    ['encode.tflite', '影像編碼'],
    ['memcond.tflite', '時序記憶'],
    ['decode_box.tflite', '主角方框'],
    ['decode.tflite', '逐幀遮罩'],
    ['memorize.tflite', '記憶更新'],
  ] as const;
  const compile = async (file: string, label: string, step: number) => {
    onLoadProgress?.(step, stages.length, label);
    const model = await loadAndCompile(assetUrl(baseUrl, file), { accelerator });
    if (
      accelerator === 'webgpu'
      && (model.options.accelerator !== 'webgpu' || !model.isFullyAccelerated)
    ) {
      model.delete();
      throw new Error('EdgeTAM 無法完整使用 WebGPU，已停止以避免退回極慢的 CPU 模式');
    }
    loaded.push(model);
    return model;
  };
  try {
    const encode = await compile(...stages[0], 1);
    const memcond = await compile(...stages[1], 2);
    const decodeBox = await compile(...stages[2], 3);
    const decode = await compile(...stages[3], 4);
    const memorize = await compile(...stages[4], 5);
    return { encode, memcond, decodeBox, decode, memorize };
  } catch (error) {
    for (const model of loaded) model.delete();
    throw error;
  }
}

export class EdgeTamVideoSegmenter {
  private readonly models: EdgeTamModels;
  private readonly noMemory: Float32Array;
  private readonly temporalPositions: Float32Array;
  private readonly noObjectPointer: Float32Array;
  private readonly trackSparse: Float32Array;
  private readonly gaussian: Float32Array;
  private readonly boxCorner0Embedding: Float32Array;
  private readonly boxCorner1Embedding: Float32Array;
  private readonly notAPointEmbedding: Float32Array;
  private readonly inputCanvas = document.createElement('canvas');
  private readonly inputContext: CanvasRenderingContext2D;
  private readonly inputValues = new Float32Array(3 * SIZE * SIZE);
  private readonly spatialBank: SpatialMemory[] = [];
  private readonly pointerBank: ObjectPointer[] = [];
  private frameIndex = -1;

  readonly accelerator: Accelerator;

  private constructor(
    models: EdgeTamModels,
    accelerator: Accelerator,
    noMemory: Float32Array,
    temporalPositions: Float32Array,
    noObjectPointer: Float32Array,
    trackSparse: Float32Array,
    prompt: Float32Array,
  ) {
    this.models = models;
    this.accelerator = accelerator;
    this.noMemory = noMemory;
    this.temporalPositions = temporalPositions;
    this.noObjectPointer = noObjectPointer;
    this.trackSparse = trackSparse;
    this.gaussian = prompt.slice(0, 256);
    this.boxCorner0Embedding = prompt.slice(256, 512);
    this.boxCorner1Embedding = prompt.slice(512, 768);
    this.notAPointEmbedding = prompt.slice(768, 1024);
    this.inputCanvas.width = SIZE;
    this.inputCanvas.height = SIZE;
    const context = this.inputCanvas.getContext('2d', {
      alpha: false,
      willReadFrequently: true,
    });
    if (!context) throw new Error('Safari 無法建立 EdgeTAM 影像 Canvas');
    this.inputContext = context;
  }

  static async create(
    wasmBaseUrl: string,
    modelBaseUrl: string,
    onLoadProgress?: EdgeTamLoadProgress,
  ) {
    if (!isWebGPUSupported()) {
      throw new Error('Stage 1 需要 Safari 26 WebGPU；請更新 iOS 並用 Safari 重新開啟');
    }
    await loadLiteRt(wasmBaseUrl, { threads: false, jspi: false });
    const accelerator: Accelerator = 'webgpu';
    const models = await compileModels(modelBaseUrl, accelerator, onLoadProgress);

    const [noMemory, temporalPositions, noObjectPointer, trackSparse, prompt] = await Promise.all([
      readFloats(assetUrl(modelBaseUrl, 'no_memory.bin'), 256),
      readFloats(assetUrl(modelBaseUrl, 'mtpe.bin'), 7 * 64),
      readFloats(assetUrl(modelBaseUrl, 'no_objptr.bin'), 256),
      readFloats(assetUrl(modelBaseUrl, 'track_sparse.bin'), 512),
      readFloats(assetUrl(modelBaseUrl, 'box_prompt.bin'), 1024),
    ]);

    return new EdgeTamVideoSegmenter(
      models,
      accelerator,
      noMemory,
      temporalPositions,
      noObjectPointer,
      trackSparse,
      prompt,
    );
  }

  reset() {
    this.frameIndex = -1;
    this.spatialBank.length = 0;
    this.pointerBank.length = 0;
  }

  private async runModel(model: CompiledModel, input: Float32Array, shape: number[]) {
    const tensor = new Tensor(input, shape);
    let outputs: Tensor[] = [];
    try {
      outputs = await model.run(tensor);
      if (outputs.length !== 1) throw new Error('EdgeTAM 模型輸出數量不符');
      const data = await outputs[0].data();
      if (!(data instanceof Float32Array)) throw new Error('EdgeTAM 模型輸出格式不符');
      return new Float32Array(data);
    } finally {
      tensor.delete();
      for (const output of outputs) {
        if (!output.deleted) output.delete();
      }
    }
  }

  private async encode(image: CanvasImageSource) {
    this.inputContext.drawImage(image, 0, 0, SIZE, SIZE);
    const pixels = this.inputContext.getImageData(0, 0, SIZE, SIZE).data;
    const plane = SIZE * SIZE;
    for (let index = 0; index < plane; index += 1) {
      const pixel = index * 4;
      this.inputValues[index] = (pixels[pixel] / 255 - MEAN[0]) / STD[0];
      this.inputValues[plane + index] = (pixels[pixel + 1] / 255 - MEAN[1]) / STD[1];
      this.inputValues[plane * 2 + index] = (pixels[pixel + 2] / 255 - MEAN[2]) / STD[2];
    }
    return this.runModel(this.models.encode, this.inputValues, [1, 3, SIZE, SIZE]);
  }

  private boxSparse(normalizedBox: [number, number, number, number]) {
    const sparse = new Float32Array(768);
    const left = Math.max(0, Math.min(1, normalizedBox[0]));
    const top = Math.max(0, Math.min(1, normalizedBox[1]));
    const right = Math.max(left, Math.min(1, normalizedBox[0] + normalizedBox[2]));
    const bottom = Math.max(top, Math.min(1, normalizedBox[1] + normalizedBox[3]));
    const corners = [[left, top], [right, bottom]] as const;
    const embeddings = [this.boxCorner0Embedding, this.boxCorner1Embedding] as const;
    corners.forEach(([normalizedX, normalizedY], corner) => {
      const modelX = normalizedX * SIZE;
      const modelY = normalizedY * SIZE;
      const ccx = 2 * ((modelX + 0.5) / SIZE) - 1;
      const ccy = 2 * ((modelY + 0.5) / SIZE) - 1;
      const embedding = embeddings[corner];
      const offset = corner * 256;
      for (let index = 0; index < 128; index += 1) {
        const projection = Math.PI * 2 * (
          ccx * this.gaussian[index] + ccy * this.gaussian[128 + index]
        );
        sparse[offset + index] = Math.sin(projection) + embedding[index];
        sparse[offset + 128 + index] = Math.cos(projection) + embedding[128 + index];
      }
    });
    sparse.set(this.notAPointEmbedding, 512);
    return sparse;
  }

  private sinePosition(offset: number) {
    const result = new Float32Array(64);
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
    if (this.spatialBank.length === 0) throw new Error('EdgeTAM 尚未指定主角');
    const memory = new Float32Array(MEMORY_VALUES);
    const positions = new Float32Array(MEMORY_VALUES);
    const keyMask = new Float32Array(MEMORY_TOKENS);
    keyMask.fill(-1e9);

    const conditioningFrame = this.spatialBank[0].frame;
    const spatial: Array<{ item: SpatialMemory; temporalIndex: number }> = [
      { item: this.spatialBank[0], temporalIndex: 6 },
    ];
    for (let offset = NMM - 1; offset >= 1; offset -= 1) {
      const wanted = frame - offset;
      if (wanted === conditioningFrame) continue;
      const match = this.spatialBank.find((item) => item.frame === wanted);
      if (match) spatial.push({ item: match, temporalIndex: offset - 1 });
    }

    spatial.forEach(({ item, temporalIndex }, slot) => {
      const base = slot * SPATIAL_TOKENS * MEM_CHANNELS;
      memory.set(item.values, base);
      for (let token = 0; token < SPATIAL_TOKENS; token += 1) {
        const tokenBase = base + token * MEM_CHANNELS;
        for (let channel = 0; channel < MEM_CHANNELS; channel += 1) {
          positions[tokenBase + channel] = item.positions[token * MEM_CHANNELS + channel]
            + this.temporalPositions[temporalIndex * MEM_CHANNELS + channel];
        }
      }
    });
    keyMask.fill(0, 0, spatial.length * SPATIAL_TOKENS);

    const recentPointers = [...this.pointerBank]
      .sort((left, right) => right.frame - left.frame)
      .slice(0, MAX_POINTERS);
    let pointerToken = 0;
    const pointerBase = NMM * SPATIAL_TOKENS * MEM_CHANNELS;
    for (const pointer of recentPointers) {
      const position = this.sinePosition(frame - pointer.frame);
      for (let token = 0; token < 4; token += 1) {
        const destination = pointerBase + pointerToken * MEM_CHANNELS;
        memory.set(pointer.values.subarray(token * MEM_CHANNELS, (token + 1) * MEM_CHANNELS), destination);
        positions.set(position, destination);
        keyMask[NMM * SPATIAL_TOKENS + pointerToken] = 0;
        pointerToken += 1;
      }
    }

    return { memory, positions, keyMask };
  }

  private maskForMemory(logits: Float32Array) {
    const result = new Float32Array(SIZE * SIZE);
    const scale = MASK_SIZE / SIZE;
    for (let outputY = 0; outputY < SIZE; outputY += 1) {
      const sourceY = Math.max(0, Math.min(255, (outputY + 0.5) * scale - 0.5));
      const y0 = Math.floor(sourceY);
      const y1 = Math.min(255, y0 + 1);
      const fy = sourceY - y0;
      for (let outputX = 0; outputX < SIZE; outputX += 1) {
        const sourceX = Math.max(0, Math.min(255, (outputX + 0.5) * scale - 0.5));
        const x0 = Math.floor(sourceX);
        const x1 = Math.min(255, x0 + 1);
        const fx = sourceX - x0;
        const top = logits[y0 * MASK_SIZE + x0] * (1 - fx)
          + logits[y0 * MASK_SIZE + x1] * fx;
        const bottom = logits[y1 * MASK_SIZE + x0] * (1 - fx)
          + logits[y1 * MASK_SIZE + x1] * fx;
        result[outputY * SIZE + outputX] = top * (1 - fy) + bottom * fy > 0 ? 10 : -10;
      }
    }
    return result;
  }

  private alphaFromLogits(logits: Float32Array) {
    const alpha = new Uint8ClampedArray(MASK_VALUES);
    for (let index = 0; index < MASK_VALUES; index += 1) {
      const value = logits[index];
      alpha[index] = value <= -0.5 ? 0 : value >= 0.5 ? 255 : Math.round((value + 0.5) * 255);
    }
    return alpha;
  }

  private storeMemory(frame: number, values: Float32Array, positions: Float32Array, pointer: Float32Array) {
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
  }

  private async decodeAndStore(
    frame: number,
    pixFeat: Float32Array,
    hi0: Float32Array,
    hi1: Float32Array,
    pixRaw: Float32Array,
    sparse: Float32Array,
  ) {
    if (sparse.length !== 512 && sparse.length !== 768) {
      throw new Error('EdgeTAM 提示 token 數量錯誤：' + sparse.length);
    }
    const decodeInput = new Float32Array(IE + H0 + H1 + sparse.length);
    decodeInput.set(pixFeat, 0);
    decodeInput.set(hi0, IE);
    decodeInput.set(hi1, IE + H0);
    decodeInput.set(sparse, IE + H0 + H1);
    const decoder = sparse.length === 768 ? this.models.decodeBox : this.models.decode;
    const decoded = await this.runModel(decoder, decodeInput, [1, decodeInput.length]);
    if (decoded.length !== DECODE_OUTPUT_VALUES) {
      throw new Error('EdgeTAM 遮罩輸出長度錯誤：' + decoded.length);
    }

    let best = 0;
    const iouOffset = MASK_VALUES * 3;
    for (let candidate = 1; candidate < 3; candidate += 1) {
      if (decoded[iouOffset + candidate] > decoded[iouOffset + best]) best = candidate;
    }
    const appearing = decoded[DECODE_OUTPUT_VALUES - 1] > 0;
    const logits = new Float32Array(MASK_VALUES);
    if (appearing) logits.set(decoded.subarray(best * MASK_VALUES, (best + 1) * MASK_VALUES));
    else logits.fill(NO_OBJECT_LOGIT);
    if (appearing) retainLargestComponent(logits);

    const pointer = new Float32Array(256);
    if (appearing) pointer.set(decoded.subarray(iouOffset + 3 + best * 256, iouOffset + 3 + (best + 1) * 256));
    else pointer.set(this.noObjectPointer);

    const memoryMask = this.maskForMemory(logits);
    const memoryInput = new Float32Array(IE * 2);
    memoryInput.set(pixRaw, 0);
    memoryInput.set(memoryMask, IE);
    const memoryOutput = await this.runModel(this.models.memorize, memoryInput, [1, memoryInput.length]);
    if (memoryOutput.length !== SPATIAL_TOKENS * MEM_CHANNELS * 2) {
      throw new Error('EdgeTAM 記憶輸出長度錯誤：' + memoryOutput.length);
    }
    this.storeMemory(
      frame,
      memoryOutput.slice(0, SPATIAL_TOKENS * MEM_CHANNELS),
      memoryOutput.slice(SPATIAL_TOKENS * MEM_CHANNELS),
      pointer,
    );
    return this.alphaFromLogits(logits);
  }

  async start(
    image: CanvasImageSource,
    normalizedBox: [number, number, number, number],
  ): Promise<EdgeTamMask> {
    this.reset();
    this.frameIndex = 0;
    const encoded = await this.encode(image);
    if (encoded.length !== IE + H0 + H1) throw new Error('EdgeTAM 編碼輸出長度錯誤');
    const pixRaw = encoded.subarray(0, IE);
    const hi0 = encoded.subarray(IE, IE + H0);
    const hi1 = encoded.subarray(IE + H0);
    const pixFeat = new Float32Array(IE);
    for (let channel = 0; channel < 256; channel += 1) {
      const offset = channel * 4096;
      const noMemory = this.noMemory[channel];
      for (let index = 0; index < 4096; index += 1) {
        pixFeat[offset + index] = pixRaw[offset + index] + noMemory;
      }
    }
    const alpha = await this.decodeAndStore(
      this.frameIndex,
      pixFeat,
      hi0,
      hi1,
      pixRaw,
      this.boxSparse(normalizedBox),
    );
    return { width: MASK_SIZE, height: MASK_SIZE, alpha };
  }

  async track(image: CanvasImageSource): Promise<EdgeTamMask> {
    if (this.frameIndex < 0) throw new Error('EdgeTAM 尚未指定主角');
    this.frameIndex += 1;
    const encoded = await this.encode(image);
    if (encoded.length !== IE + H0 + H1) throw new Error('EdgeTAM 編碼輸出長度錯誤');
    const pixRaw = encoded.subarray(0, IE);
    const hi0 = encoded.subarray(IE, IE + H0);
    const hi1 = encoded.subarray(IE + H0);
    const assembled = this.assembleMemory(this.frameIndex);
    const memoryInput = new Float32Array(IE + MEMORY_VALUES * 2 + MEMORY_TOKENS);
    memoryInput.set(pixRaw, 0);
    memoryInput.set(assembled.memory, IE);
    memoryInput.set(assembled.positions, IE + MEMORY_VALUES);
    memoryInput.set(assembled.keyMask, IE + MEMORY_VALUES * 2);
    const pixFeat = await this.runModel(this.models.memcond, memoryInput, [1, memoryInput.length]);
    if (pixFeat.length !== IE) throw new Error('EdgeTAM 時序記憶輸出長度錯誤');
    const alpha = await this.decodeAndStore(
      this.frameIndex,
      pixFeat,
      hi0,
      hi1,
      pixRaw,
      this.trackSparse,
    );
    return { width: MASK_SIZE, height: MASK_SIZE, alpha };
  }

  close() {
    this.reset();
    this.models.encode.delete();
    this.models.memcond.delete();
    this.models.decodeBox.delete();
    this.models.decode.delete();
    this.models.memorize.delete();
  }
}
