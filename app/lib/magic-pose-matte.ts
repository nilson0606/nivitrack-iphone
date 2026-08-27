import type {
  InteractiveSegmenterLegacy,
  PoseLandmarker,
} from '@mediapipe/tasks-vision';

export type MatteFrame = {
  alpha: Uint8ClampedArray;
  width: number;
  height: number;
};

function createGpuCanvas() {
  const canvas = document.createElement('canvas');
  canvas.width = 1;
  canvas.height = 1;
  return canvas;
}

function smoothGate(value: number) {
  const normalized = Math.max(0, Math.min(1, (value - 0.05) / 0.45));
  return normalized * normalized * (3 - 2 * normalized);
}

export class MagicPoseMatte {
  private previousPose: Float32Array | null = null;
  private previousPoseWidth = 0;
  private previousPoseHeight = 0;
  private nextTimestampMs = 0;

  private constructor(
    private readonly subject: InteractiveSegmenterLegacy,
    private readonly pose: PoseLandmarker,
  ) {}

  static async create(wasmBaseUrl: string, subjectModelUrl: string, poseModelUrl: string) {
    const {
      FilesetResolver,
      InteractiveSegmenterLegacy,
      PoseLandmarker,
    } = await import('@mediapipe/tasks-vision');
    const fileset = await FilesetResolver.forVisionTasks(wasmBaseUrl);

    let subject: InteractiveSegmenterLegacy;
    try {
      subject = await InteractiveSegmenterLegacy.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: subjectModelUrl, delegate: 'GPU' },
        canvas: createGpuCanvas(),
        outputCategoryMask: false,
        outputConfidenceMasks: true,
      });
    } catch {
      subject = await InteractiveSegmenterLegacy.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: subjectModelUrl, delegate: 'CPU' },
        outputCategoryMask: false,
        outputConfidenceMasks: true,
      });
    }

    let pose: PoseLandmarker;
    try {
      pose = await PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: poseModelUrl, delegate: 'GPU' },
        canvas: createGpuCanvas(),
        runningMode: 'VIDEO',
        numPoses: 1,
        minPoseDetectionConfidence: 0.35,
        minPosePresenceConfidence: 0.35,
        minTrackingConfidence: 0.35,
        outputSegmentationMasks: true,
      });
    } catch {
      pose = await PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: poseModelUrl, delegate: 'CPU' },
        runningMode: 'VIDEO',
        numPoses: 1,
        minPoseDetectionConfidence: 0.35,
        minPosePresenceConfidence: 0.35,
        minTrackingConfidence: 0.35,
        outputSegmentationMasks: true,
      });
    }
    return new MagicPoseMatte(subject, pose);
  }

  beginSequence() {
    this.previousPose = null;
    this.previousPoseWidth = 0;
    this.previousPoseHeight = 0;
  }

  segment(image: TexImageSource): MatteFrame {
    const subjectResult = this.subject.segment(image, {
      keypoint: { x: 0.5, y: 0.5 },
    });
    const poseResult = this.pose.detectForVideo(image, this.nextTimestampMs);
    this.nextTimestampMs += 1000 / 30;
    try {
      const subjectMasks = subjectResult.confidenceMasks ?? [];
      if (subjectMasks.length === 0) throw new Error('MagicTouch 沒有回傳主角遮罩');

      let subjectMask = subjectMasks[0];
      let subjectValues = subjectMask.getAsFloat32Array();
      let bestScore = Number.NEGATIVE_INFINITY;
      for (const candidate of subjectMasks) {
        const values = candidate.getAsFloat32Array();
        const x = Math.round(0.5 * Math.max(0, candidate.width - 1));
        const y = Math.round(0.5 * Math.max(0, candidate.height - 1));
        const score = values[y * candidate.width + x] ?? 0;
        if (score > bestScore) {
          bestScore = score;
          subjectMask = candidate;
          subjectValues = values;
        }
      }

      const poseMask = poseResult.segmentationMasks?.[0];
      if (poseMask) {
        this.previousPose = new Float32Array(poseMask.getAsFloat32Array());
        this.previousPoseWidth = poseMask.width;
        this.previousPoseHeight = poseMask.height;
      }

      const alpha = new Uint8ClampedArray(subjectValues.length);
      for (let y = 0; y < subjectMask.height; y += 1) {
        const poseY = this.previousPose
          ? Math.min(
              this.previousPoseHeight - 1,
              Math.floor(((y + 0.5) / subjectMask.height) * this.previousPoseHeight),
            )
          : 0;
        for (let x = 0; x < subjectMask.width; x += 1) {
          const index = y * subjectMask.width + x;
          let poseGate = 0;
          if (this.previousPose) {
            const poseX = Math.min(
              this.previousPoseWidth - 1,
              Math.floor(((x + 0.5) / subjectMask.width) * this.previousPoseWidth),
            );
            poseGate = smoothGate(
              this.previousPose[poseY * this.previousPoseWidth + poseX],
            );
          }
          alpha[index] = Math.round(
            Math.max(0, Math.min(1, subjectValues[index])) * poseGate * 255,
          );
        }
      }
      return { alpha, width: subjectMask.width, height: subjectMask.height };
    } finally {
      subjectResult.close();
      poseResult.close();
    }
  }

  close() {
    this.subject.close();
    this.pose.close();
    this.beginSequence();
  }
}
