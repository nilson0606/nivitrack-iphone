import * as ort from 'onnxruntime-web/wasm';

export type Box = [number, number, number, number];

export type TrackResult = {
  box: Box;
  score: number;
  accepted: boolean;
  inferenceMs: number;
};

const TEMPLATE_SIZE = 128;
const SEARCH_SIZE = 256;
const MAP_SIZE = 16;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

function makeCanvas(width: number, height: number) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function sourceSize(source: CanvasImageSource): [number, number] {
  if (source instanceof HTMLVideoElement) {
    return [source.videoWidth, source.videoHeight];
  }
  if (source instanceof HTMLCanvasElement || source instanceof HTMLImageElement) {
    return [source.width, source.height];
  }
  if (source instanceof ImageBitmap) {
    return [source.width, source.height];
  }
  if (source instanceof VideoFrame) {
    return [source.displayWidth || source.codedWidth, source.displayHeight || source.codedHeight];
  }
  throw new Error('不支援的影像來源');
}

function cropToTensor(source: CanvasImageSource, box: Box, factor: number, size: number) {
  const [frameWidth, frameHeight] = sourceSize(source);
  const [x, y, width, height] = box.map((value) => Math.trunc(value)) as Box;
  const cropSize = Math.ceil(Math.sqrt(width * height) * factor);
  const x1 = x + Math.trunc((width - cropSize) / 2);
  const y1 = y + Math.trunc((height - cropSize) / 2);

  const crop = makeCanvas(cropSize, cropSize);
  const cropContext = crop.getContext('2d', {
    alpha: false,
    willReadFrequently: true,
  });
  if (!cropContext) throw new Error('無法建立影格裁切畫布');
  cropContext.fillStyle = '#000';
  cropContext.fillRect(0, 0, cropSize, cropSize);
  cropContext.drawImage(source, -x1, -y1, frameWidth, frameHeight);

  const resized = makeCanvas(size, size);
  const context = resized.getContext('2d', {
    alpha: false,
    willReadFrequently: true,
  });
  if (!context) throw new Error('無法建立模型輸入畫布');
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = 'medium';
  context.drawImage(crop, 0, 0, size, size);

  const rgba = context.getImageData(0, 0, size, size).data;
  const plane = size * size;
  const tensorData = new Float32Array(plane * 3);

  for (let index = 0; index < plane; index += 1) {
    const pixel = index * 4;
    const red = rgba[pixel] / 255;
    const green = rgba[pixel + 1] / 255;
    const blue = rgba[pixel + 2] / 255;

    // OpenCV VideoCapture supplies BGR and TrackerVit does not swap channels.
    tensorData[index] = (blue - MEAN[0]) / STD[0];
    tensorData[plane + index] = (green - MEAN[1]) / STD[1];
    tensorData[plane * 2 + index] = (red - MEAN[2]) / STD[2];
  }

  return {
    cropSize,
    tensor: new ort.Tensor('float32', tensorData, [1, 3, size, size]),
  };
}

function createHanningWindow() {
  const oneDimension = new Float32Array(MAP_SIZE);
  for (let index = 0; index < MAP_SIZE; index += 1) {
    oneDimension[index] =
      0.5 * (1 - Math.cos((2 * Math.PI * (index + 1)) / (MAP_SIZE + 1)));
  }

  const window = new Float32Array(MAP_SIZE * MAP_SIZE);
  for (let y = 0; y < MAP_SIZE; y += 1) {
    for (let x = 0; x < MAP_SIZE; x += 1) {
      window[y * MAP_SIZE + x] = oneDimension[y] * oneDimension[x];
    }
  }
  return window;
}

export class VitTracker {
  private readonly session: ort.InferenceSession;
  private readonly hanning = createHanningWindow();
  private template: ort.Tensor | null = null;
  private lastBox: Box | null = null;
  private scoreThreshold = 0.2;

  private constructor(session: ort.InferenceSession) {
    this.session = session;
  }

  static async create(modelUrl: string) {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    ort.env.wasm.wasmPaths = new URL('ort/', document.baseURI).href;

    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      executionMode: 'sequential',
    });
    return new VitTracker(session);
  }

  initialize(source: CanvasImageSource, box: Box) {
    const prepared = cropToTensor(source, box, 2, TEMPLATE_SIZE);
    this.template = prepared.tensor;
    this.lastBox = box.map((value) => Math.trunc(value)) as Box;
  }

  async update(source: CanvasImageSource): Promise<TrackResult> {
    if (!this.template || !this.lastBox) {
      throw new Error('ViT Tracker 尚未初始化');
    }

    const previous = [...this.lastBox] as Box;
    const prepared = cropToTensor(source, previous, 4, SEARCH_SIZE);
    const started = performance.now();
    const output = await this.session.run({
      template: this.template,
      search: prepared.tensor,
    });
    const inferenceMs = performance.now() - started;

    const confidence = output.output1.data as Float32Array;
    const sizeMap = output.output2.data as Float32Array;
    const offsetMap = output.output3.data as Float32Array;
    let bestIndex = 0;
    let bestScore = Number.NEGATIVE_INFINITY;

    for (let index = 0; index < MAP_SIZE * MAP_SIZE; index += 1) {
      const score = confidence[index] * this.hanning[index];
      if (score > bestScore) {
        bestScore = score;
        bestIndex = index;
      }
    }

    if (bestScore >= this.scoreThreshold) {
      const mapX = bestIndex % MAP_SIZE;
      const mapY = Math.floor(bestIndex / MAP_SIZE);
      const centerX = (mapX + offsetMap[bestIndex]) / MAP_SIZE;
      const centerY =
        (mapY + offsetMap[MAP_SIZE * MAP_SIZE + bestIndex]) / MAP_SIZE;
      const width = sizeMap[bestIndex];
      const height = sizeMap[MAP_SIZE * MAP_SIZE + bestIndex];
      const cropX =
        previous[0] + Math.trunc((previous[2] - prepared.cropSize) / 2);
      const cropY =
        previous[1] + Math.trunc((previous[3] - prepared.cropSize) / 2);
      const nextBox: Box = [
        Math.floor((centerX - width / 2) * prepared.cropSize + cropX),
        Math.floor((centerY - height / 2) * prepared.cropSize + cropY),
        Math.floor(width * prepared.cropSize),
        Math.floor(height * prepared.cropSize),
      ];
      this.lastBox = nextBox;
    }

    return {
      box: [...this.lastBox] as Box,
      score: bestScore,
      accepted: bestScore >= this.scoreThreshold,
      inferenceMs,
    };
  }
}
