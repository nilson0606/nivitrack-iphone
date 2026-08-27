import type {
  ImageSegmenter,
  InteractiveSegmenterLegacy,
} from '@mediapipe/tasks-vision';
import { fuseSubjectAndPersonMasks } from './person-mask-fusion';

type NormalizedKeypoint = { x: number; y: number };
export type SubjectPrompt = {
  keypoint?: NormalizedKeypoint;
  scribble?: NormalizedKeypoint[];
};

export type SubjectMask = {
  values: Float32Array;
  width: number;
  height: number;
};

function createGpuCanvas() {
  const canvas = document.createElement('canvas');
  canvas.width = 1;
  canvas.height = 1;
  return canvas;
}

export class InteractiveSubjectSegmenter {
  private constructor(
    private readonly subjectSegmenter: InteractiveSegmenterLegacy,
    private readonly personSegmenter: ImageSegmenter,
    private readonly personMaskIndex: number,
  ) {}

  static async create(wasmBaseUrl: string, subjectModelUrl: string, personModelUrl: string) {
    const { FilesetResolver, ImageSegmenter, InteractiveSegmenterLegacy } = await import('@mediapipe/tasks-vision');
    const fileset = await FilesetResolver.forVisionTasks(wasmBaseUrl);
    let subjectSegmenter: InteractiveSegmenterLegacy;
    try {
      subjectSegmenter = await InteractiveSegmenterLegacy.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: subjectModelUrl, delegate: 'GPU' },
        canvas: createGpuCanvas(),
        outputCategoryMask: false,
        outputConfidenceMasks: true,
      });
    } catch {
      subjectSegmenter = await InteractiveSegmenterLegacy.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: subjectModelUrl, delegate: 'CPU' },
        outputCategoryMask: false,
        outputConfidenceMasks: true,
      });
    }

    let personSegmenter: ImageSegmenter;
    try {
      personSegmenter = await ImageSegmenter.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: personModelUrl, delegate: 'GPU' },
        canvas: createGpuCanvas(),
        runningMode: 'IMAGE',
        outputCategoryMask: false,
        outputConfidenceMasks: true,
      });
    } catch {
      personSegmenter = await ImageSegmenter.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: personModelUrl, delegate: 'CPU' },
        runningMode: 'IMAGE',
        outputCategoryMask: false,
        outputConfidenceMasks: true,
      });
    }
    const labels = personSegmenter.getLabels().map((label) => label.toLowerCase());
    const personIndex = labels.findIndex((label) => label.includes('person'));
    return new InteractiveSubjectSegmenter(
      subjectSegmenter,
      personSegmenter,
      personIndex >= 0 ? personIndex : Math.max(0, labels.length - 1),
    );
  }

  segment(image: TexImageSource, prompt: SubjectPrompt): SubjectMask {
    const subjectResult = this.subjectSegmenter.segment(image, prompt);
    const personResult = this.personSegmenter.segment(image);
    try {
      const subjectMasks = subjectResult.confidenceMasks ?? [];
      const personMasks = personResult.confidenceMasks ?? [];
      if (subjectMasks.length === 0) throw new Error('指定主角模型沒有回傳遮罩');
      const personMask = personMasks[Math.min(this.personMaskIndex, Math.max(0, personMasks.length - 1))];
      if (!personMask) throw new Error('人物模型沒有回傳遮罩');

      const promptPoints = prompt.scribble?.length
        ? prompt.scribble
        : prompt.keypoint
          ? [prompt.keypoint]
          : [{ x: 0.5, y: 0.5 }];
      let selectedSubject = subjectMasks[0];
      let selectedValues = selectedSubject.getAsFloat32Array();
      let bestPromptScore = Number.NEGATIVE_INFINITY;
      for (const candidate of subjectMasks) {
        const values = candidate.getAsFloat32Array();
        const score = promptPoints.reduce((sum, point) => {
          const x = Math.round(point.x * Math.max(0, candidate.width - 1));
          const y = Math.round(point.y * Math.max(0, candidate.height - 1));
          return sum + (values[y * candidate.width + x] ?? 0);
        }, 0) / Math.max(1, promptPoints.length);
        if (score > bestPromptScore) {
          bestPromptScore = score;
          selectedSubject = candidate;
          selectedValues = values;
        }
      }

      return {
        values: fuseSubjectAndPersonMasks(
          selectedValues,
          selectedSubject.width,
          selectedSubject.height,
          personMask.getAsFloat32Array(),
          personMask.width,
          personMask.height,
        ),
        width: selectedSubject.width,
        height: selectedSubject.height,
      };
    } finally {
      subjectResult.close();
      personResult.close();
    }
  }

  close() {
    this.subjectSegmenter.close();
    this.personSegmenter.close();
  }
}
