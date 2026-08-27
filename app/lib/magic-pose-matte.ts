import type {
  InteractiveSegmenterLegacy,
  PoseLandmarker,
} from '@mediapipe/tasks-vision';

export type MatteFrame = {
  alpha: Uint8ClampedArray;
  bodySupport: Uint8ClampedArray | null;
  width: number;
  height: number;
};

type PosePoint = {
  x: number;
  y: number;
  visibility?: number;
  presence?: number;
};

type PixelPoint = { x: number; y: number };

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function posePointAt(points: PosePoint[], index: number, width: number, height: number) {
  const point = points[index];
  if (!point) return null;
  const confidence = Math.min(point.visibility ?? 1, point.presence ?? 1);
  if (
    confidence < 0.12
    || point.x < -0.12
    || point.x > 1.12
    || point.y < -0.12
    || point.y > 1.12
  ) {
    return null;
  }
  return {
    x: point.x * Math.max(1, width - 1),
    y: point.y * Math.max(1, height - 1),
  } satisfies PixelPoint;
}

function midpoint(first: PixelPoint, second: PixelPoint): PixelPoint {
  return { x: (first.x + second.x) / 2, y: (first.y + second.y) / 2 };
}

function distance(first: PixelPoint, second: PixelPoint) {
  return Math.hypot(first.x - second.x, first.y - second.y);
}

function paintCapsule(
  support: Uint8ClampedArray,
  width: number,
  height: number,
  start: PixelPoint,
  end: PixelPoint,
  radius: number,
) {
  const outerRadius = radius * 1.45;
  const left = Math.max(0, Math.floor(Math.min(start.x, end.x) - outerRadius));
  const right = Math.min(width - 1, Math.ceil(Math.max(start.x, end.x) + outerRadius));
  const top = Math.max(0, Math.floor(Math.min(start.y, end.y) - outerRadius));
  const bottom = Math.min(height - 1, Math.ceil(Math.max(start.y, end.y) + outerRadius));
  const segmentX = end.x - start.x;
  const segmentY = end.y - start.y;
  const segmentLengthSquared = segmentX * segmentX + segmentY * segmentY;

  for (let y = top; y <= bottom; y += 1) {
    for (let x = left; x <= right; x += 1) {
      const projection = segmentLengthSquared > 0
        ? clamp(
          ((x - start.x) * segmentX + (y - start.y) * segmentY) / segmentLengthSquared,
          0,
          1,
        )
        : 0;
      const closestX = start.x + segmentX * projection;
      const closestY = start.y + segmentY * projection;
      const at = Math.hypot(x - closestX, y - closestY);
      if (at > outerRadius) continue;
      const feather = 1 - clamp(
        (at - radius) / Math.max(1, outerRadius - radius),
        0,
        1,
      );
      const smooth = feather * feather * (3 - 2 * feather);
      const index = y * width + x;
      support[index] = Math.max(support[index], Math.round(smooth * 255));
    }
  }
}

export function createBodyCoreSupport(
  points: PosePoint[] | undefined,
  width: number,
  height: number,
) {
  if (!points || points.length < 29 || width <= 0 || height <= 0) return null;
  const leftShoulder = posePointAt(points, 11, width, height);
  const rightShoulder = posePointAt(points, 12, width, height);
  const leftHip = posePointAt(points, 23, width, height);
  const rightHip = posePointAt(points, 24, width, height);
  if (!leftShoulder || !rightShoulder || !leftHip || !rightHip) return null;

  const support = new Uint8ClampedArray(width * height);
  const shoulderMiddle = midpoint(leftShoulder, rightShoulder);
  const hipMiddle = midpoint(leftHip, rightHip);
  const shoulderWidth = distance(leftShoulder, rightShoulder);
  const hipWidth = distance(leftHip, rightHip);
  const torsoLength = distance(shoulderMiddle, hipMiddle);
  const minimumEdge = Math.min(width, height);
  const limbRadius = clamp(
    Math.max(shoulderWidth * 0.12, torsoLength * 0.09, minimumEdge * 0.012),
    minimumEdge * 0.012,
    minimumEdge * 0.052,
  );
  const torsoRadius = clamp(
    Math.max(shoulderWidth, hipWidth) * 0.52,
    minimumEdge * 0.055,
    minimumEdge * 0.19,
  );

  paintCapsule(support, width, height, shoulderMiddle, hipMiddle, torsoRadius);
  paintCapsule(support, width, height, leftShoulder, rightShoulder, limbRadius * 1.7);
  paintCapsule(support, width, height, leftHip, rightHip, limbRadius * 1.8);

  const segments: Array<[number, number, number]> = [
    [11, 13, 1.18], [13, 15, 1.05], [15, 17, 1.18], [15, 19, 1.18], [15, 21, 1.12],
    [12, 14, 1.18], [14, 16, 1.05], [16, 18, 1.18], [16, 20, 1.18], [16, 22, 1.12],
    [23, 25, 1.32], [25, 27, 1.18], [27, 29, 1.15], [27, 31, 1.2],
    [24, 26, 1.32], [26, 28, 1.18], [28, 30, 1.15], [28, 32, 1.2],
  ];
  for (const [from, to, radiusScale] of segments) {
    const start = posePointAt(points, from, width, height);
    const end = posePointAt(points, to, width, height);
    if (start && end) {
      paintCapsule(support, width, height, start, end, limbRadius * radiusScale);
    }
  }

  const nose = posePointAt(points, 0, width, height);
  const leftEar = posePointAt(points, 7, width, height);
  const rightEar = posePointAt(points, 8, width, height);
  const headCenter = leftEar && rightEar ? midpoint(leftEar, rightEar) : nose;
  if (headCenter) {
    const headRadius = clamp(
      Math.max(shoulderWidth * 0.3, torsoLength * 0.16),
      minimumEdge * 0.025,
      minimumEdge * 0.085,
    );
    paintCapsule(support, width, height, headCenter, headCenter, headRadius);
    paintCapsule(support, width, height, headCenter, shoulderMiddle, limbRadius * 1.5);
  }
  return support;
}

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
  private previousBodySupport: Uint8ClampedArray | null = null;
  private previousBodySupportWidth = 0;
  private previousBodySupportHeight = 0;
  private missingBodySupportFrames = 0;
  private nextTimestampMs = 0;
  private readonly subject: InteractiveSegmenterLegacy;
  private readonly pose: PoseLandmarker;

  private constructor(subject: InteractiveSegmenterLegacy, pose: PoseLandmarker) {
    this.subject = subject;
    this.pose = pose;
  }

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
    this.previousBodySupport = null;
    this.previousBodySupportWidth = 0;
    this.previousBodySupportHeight = 0;
    this.missingBodySupportFrames = 0;
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
      let bodySupport = createBodyCoreSupport(
        poseResult.landmarks?.[0],
        subjectMask.width,
        subjectMask.height,
      );
      if (bodySupport) {
        this.previousBodySupport = bodySupport;
        this.previousBodySupportWidth = subjectMask.width;
        this.previousBodySupportHeight = subjectMask.height;
        this.missingBodySupportFrames = 0;
      } else if (
        this.previousBodySupport
        && this.previousBodySupportWidth === subjectMask.width
        && this.previousBodySupportHeight === subjectMask.height
        && this.missingBodySupportFrames < 2
      ) {
        bodySupport = new Uint8ClampedArray(this.previousBodySupport);
        this.missingBodySupportFrames += 1;
      } else {
        this.missingBodySupportFrames += 1;
      }
      return {
        alpha,
        bodySupport,
        width: subjectMask.width,
        height: subjectMask.height,
      };
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
