import type {
  InteractiveSegmenterLegacy,
  InteractiveSegmenterLegacyResult,
} from '@mediapipe/tasks-vision';

type NormalizedKeypoint = { x: number; y: number };

function createGpuCanvas() {
  const canvas = document.createElement('canvas');
  canvas.width = 1;
  canvas.height = 1;
  return canvas;
}

export class InteractiveSubjectSegmenter {
  private constructor(private readonly segmenter: InteractiveSegmenterLegacy) {}

  static async create(wasmBaseUrl: string, modelUrl: string) {
    const { FilesetResolver, InteractiveSegmenterLegacy } = await import('@mediapipe/tasks-vision');
    const fileset = await FilesetResolver.forVisionTasks(wasmBaseUrl);
    const canvas = createGpuCanvas();
    let segmenter: InteractiveSegmenterLegacy;
    try {
      segmenter = await InteractiveSegmenterLegacy.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: modelUrl, delegate: 'GPU' },
        canvas,
        outputCategoryMask: false,
        outputConfidenceMasks: true,
      });
    } catch {
      segmenter = await InteractiveSegmenterLegacy.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: modelUrl, delegate: 'CPU' },
        outputCategoryMask: false,
        outputConfidenceMasks: true,
      });
    }
    return new InteractiveSubjectSegmenter(segmenter);
  }

  segment(image: TexImageSource, keypoint: NormalizedKeypoint): InteractiveSegmenterLegacyResult {
    return this.segmenter.segment(image, { keypoint });
  }

  close() {
    this.segmenter.close();
  }
}
