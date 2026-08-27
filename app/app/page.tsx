'use client';

import {
  ChangeEvent,
  PointerEvent as ReactPointerEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import type { ObjectDetection } from '@tensorflow-models/coco-ssd';
import { Box, TrackResult, VitTracker } from '../lib/vit-tracker';
import {
  AspectPreset,
  configureOutputCanvas,
  getRecorderSupport,
  RealtimeVideoExporter,
  RecorderSupport,
  smoothTrackPath,
  TrackPoint,
} from '../lib/video-export';
import {
  BackgroundEffect,
  CloneLayout,
  DEFAULT_PERSON_EFFECTS,
  MaskCorrectionMode,
  OutlineEffect,
  PersonEffectOptions,
  PersonEffectRenderer,
  SubjectEffect,
} from '../lib/person-effects';

type Capability = {
  label: string;
  detail: string;
  available: boolean;
};

type VideoInfo = {
  name: string;
  size: string;
  duration: string;
  resolution: string;
};

type Phase = 'choose' | 'select' | 'tracking' | 'complete' | 'path-ready' | 'effect-testing' | 'exporting';

type TrackingStats = {
  frames: number;
  elapsedMs: number;
  averageInferenceMs: number;
  averageScore: number;
  acceptedFrames: number;
};

type Candidate = {
  box: Box;
  label: string;
  score: number;
};

type ExportInfo = {
  name: string;
  size: string;
  mimeType: string;
  resolution: string;
};

type EffectTestWindow = { start: number; end: number };

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1024 / 1024).toFixed(1) + ' MB';
}

function formatDuration(seconds: number) {
  if (!Number.isFinite(seconds)) return '—';
  const minutes = Math.floor(seconds / 60);
  const rest = Math.floor(seconds % 60);
  return minutes + ':' + rest.toString().padStart(2, '0');
}

function normalizeBox(start: [number, number], end: [number, number]): Box {
  return [
    Math.min(start[0], end[0]),
    Math.min(start[1], end[1]),
    Math.abs(end[0] - start[0]),
    Math.abs(end[1] - start[1]),
  ];
}

function clampTime(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragStartRef = useRef<[number, number] | null>(null);
  const cancelRef = useRef(false);
  const trackerRef = useRef<VitTracker | null>(null);
  const detectorRef = useRef<ObjectDetection | null>(null);
  const selectionRef = useRef<{ time: number; box: Box } | null>(null);
  const renderCanvasRef = useRef<HTMLCanvasElement>(null);
  const effectPreviewCanvasRef = useRef<HTMLCanvasElement>(null);
  const exporterRef = useRef<RealtimeVideoExporter | null>(null);
  const personEffectRendererRef = useRef<PersonEffectRenderer | null>(null);
  const correctionDrawingRef = useRef(false);

  const [videoUrl, setVideoUrl] = useState('');
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null>(null);
  const [capabilities, setCapabilities] = useState<Capability[]>([]);
  const [phase, setPhase] = useState<Phase>('choose');
  const [box, setBox] = useState<Box | null>(null);
  const [notice, setNotice] = useState('等待選擇影片');
  const [progress, setProgress] = useState(0);
  const [currentScore, setCurrentScore] = useState<number | null>(null);
  const [stats, setStats] = useState<TrackingStats | null>(null);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [detecting, setDetecting] = useState(false);
  const [trackPath, setTrackPath] = useState<TrackPoint[]>([]);
  const [aspect, setAspect] = useState<AspectPreset>('9:16');
  const [subjectScale, setSubjectScale] = useState(0.55);
  const [smoothness, setSmoothness] = useState(0.72);
  const [recorderSupport, setRecorderSupport] = useState<RecorderSupport>({
    h264: null,
    hevc: null,
  });
  const [exportUrl, setExportUrl] = useState('');
  const [exportBlob, setExportBlob] = useState<Blob | null>(null);
  const [exportInfo, setExportInfo] = useState<ExportInfo | null>(null);
  const [personEffects, setPersonEffects] = useState<PersonEffectOptions>(DEFAULT_PERSON_EFFECTS);
  const [showEffectPreview, setShowEffectPreview] = useState(false);
  const [effectTestPassed, setEffectTestPassed] = useState(false);
  const [editingMask, setEditingMask] = useState(false);
  const [correctionMode, setCorrectionMode] = useState<MaskCorrectionMode>('remove');
  const [correctionBrush, setCorrectionBrush] = useState(28);
  const [correctionCount, setCorrectionCount] = useState(0);
  const [correctionTime, setCorrectionTime] = useState(0);
  const [effectTestWindow, setEffectTestWindow] = useState<EffectTestWindow | null>(null);

  useEffect(() => {
    const support = getRecorderSupport();
    const capabilityFrame = requestAnimationFrame(() => {
      setRecorderSupport(support);
      setCapabilities([
        { label: '本機 AI', detail: 'WebAssembly', available: typeof WebAssembly !== 'undefined' },
        { label: '人物去背', detail: 'MagicTouch × Pose', available: typeof WebAssembly !== 'undefined' },
        { label: '背景運算', detail: 'Web Worker', available: typeof Worker !== 'undefined' },
        { label: '逐幀影像', detail: 'WebCodecs', available: typeof VideoFrame !== 'undefined' },
        { label: 'GPU 加速', detail: 'WebGPU', available: 'gpu' in navigator },
        { label: '相容分享', detail: 'H.264 / AAC MP4', available: Boolean(support.h264) },
        { label: 'HEVC 母片', detail: 'HEVC / AAC', available: Boolean(support.hevc) },
        { label: '離線安裝', detail: 'Service Worker', available: 'serviceWorker' in navigator },
      ]);
    });

    if ('serviceWorker' in navigator) {
      const workerUrl = new URL('sw.js', document.baseURI);
      navigator.serviceWorker
        .register(workerUrl.href, { scope: new URL('./', document.baseURI).pathname })
        .catch(() => {
          setNotice('離線快取尚未啟用；其餘本機功能仍可測試');
        });
    }
    return () => cancelAnimationFrame(capabilityFrame);
  }, []);

  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  useEffect(() => {
    return () => {
      if (exportUrl) URL.revokeObjectURL(exportUrl);
    };
  }, [exportUrl]);

  useEffect(() => {
    return () => {
      void exporterRef.current?.dispose();
      personEffectRendererRef.current?.close();
    };
  }, []);

  const readyCount = useMemo(
    () => capabilities.filter((item) => item.available).length,
    [capabilities],
  );

  function clearRenderedOutput() {
    personEffectRendererRef.current?.clearPrepared();
    correctionDrawingRef.current = false;
    setEditingMask(false);
    setEffectTestWindow(null);
    setShowEffectPreview(false);
    setEffectTestPassed(false);
    setExportUrl('');
    setExportBlob(null);
    setExportInfo(null);
  }

  function updatePersonEffects(patch: Partial<PersonEffectOptions>) {
    setPersonEffects((current) => ({ ...current, ...patch }));
    clearRenderedOutput();
  }

  async function getPersonEffectRenderer() {
    if (!personEffectRendererRef.current) {
      const wasmBase = new URL('mediapipe/', document.baseURI).href;
      const subjectModelUrl = new URL('models/magic_touch.tflite', document.baseURI).href;
      const poseModelUrl = new URL('models/pose_landmarker_lite.task', document.baseURI).href;
      personEffectRendererRef.current = await PersonEffectRenderer.create(
        wasmBase,
        subjectModelUrl,
        poseModelUrl,
      );
    }
    return personEffectRendererRef.current;
  }

  function openVideoPicker() {
    const input = inputRef.current;
    if (!input) return;
    input.value = '';
    input.click();
  }

  function chooseVideo(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setSourceFile(file);
    setVideoUrl(URL.createObjectURL(file));
    setVideoInfo(null);
    setBox(null);
    setStats(null);
    setCandidates([]);
    selectionRef.current = null;
    setTrackPath([]);
    setExportUrl('');
    setExportBlob(null);
    setExportInfo(null);
    setShowEffectPreview(false);
    setEffectTestPassed(false);
    personEffectRendererRef.current?.reset();
    setCorrectionCount(0);
    setProgress(0);
    setPhase('choose');
    setNotice('正在直接讀取原始影片…');
  }

  function readMetadata() {
    const video = videoRef.current;
    if (!video || !sourceFile) return;
    setVideoInfo({
      name: sourceFile.name,
      size: formatBytes(sourceFile.size),
      duration: formatDuration(video.duration),
      resolution: video.videoWidth + ' × ' + video.videoHeight,
    });
    setNotice('影片已在本機載入，沒有上傳或預先轉檔');
  }

  function drawFrame(
    targetBox: Box | null,
    score?: number,
    visibleCandidates: Candidate[] = candidates,
  ) {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth) return;
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    const context = canvas.getContext('2d');
    if (!context) return;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const candidateLine = Math.max(3, canvas.width / 500);
    context.lineWidth = candidateLine;
    context.font = '700 ' + Math.max(16, canvas.width / 65) + 'px -apple-system';
    for (const candidate of visibleCandidates) {
      const [candidateX, candidateY, candidateWidth, candidateHeight] = candidate.box;
      context.strokeStyle = '#35d292';
      context.strokeRect(candidateX, candidateY, candidateWidth, candidateHeight);
      context.fillStyle = 'rgba(16, 32, 24, 0.82)';
      const text = candidate.label + ' ' + Math.round(candidate.score * 100) + '%';
      const width = context.measureText(text).width + candidateLine * 5;
      context.fillRect(candidateX, candidateY, width, candidateLine * 9);
      context.fillStyle = '#d9f06f';
      context.fillText(text, candidateX + candidateLine * 2, candidateY + candidateLine * 6.5);
    }
    if (!targetBox) return;

    const [x, y, width, height] = targetBox;
    const lineWidth = Math.max(4, canvas.width / 320);
    context.fillStyle = 'rgba(217, 240, 111, 0.12)';
    context.fillRect(x, y, width, height);
    context.strokeStyle = '#d9f06f';
    context.lineWidth = lineWidth;
    context.strokeRect(x, y, width, height);
    context.fillStyle = '#102018';
    context.font = '700 ' + Math.max(18, canvas.width / 55) + 'px -apple-system';
    const label = score === undefined ? '主角' : 'ViT ' + score.toFixed(3);
    const labelWidth = context.measureText(label).width + lineWidth * 5;
    context.fillRect(x, Math.max(0, y - lineWidth * 9), labelWidth, lineWidth * 9);
    context.fillStyle = '#d9f06f';
    context.fillText(label, x + lineWidth * 2, Math.max(lineWidth * 6.5, y - lineWidth * 2));
  }

  function enterSelection() {
    const video = videoRef.current;
    if (!video) return;
    video.pause();
    setPhase('select');
    setBox(null);
    setStats(null);
    setCandidates([]);
    selectionRef.current = null;
    setTrackPath([]);
    setExportUrl('');
    setExportBlob(null);
    setExportInfo(null);
    setShowEffectPreview(false);
    setEffectTestPassed(false);
    personEffectRendererRef.current?.reset();
    setCorrectionCount(0);
    setNotice('用手指框住要追蹤的人物或寵物');
    requestAnimationFrame(() => drawFrame(null));
  }

  function pointerPosition(event: ReactPointerEvent<HTMLCanvasElement>): [number, number] {
    const canvas = canvasRef.current;
    if (!canvas) return [0, 0];
    const bounds = canvas.getBoundingClientRect();
    return [
      Math.max(0, Math.min(canvas.width, ((event.clientX - bounds.left) / bounds.width) * canvas.width)),
      Math.max(0, Math.min(canvas.height, ((event.clientY - bounds.top) / bounds.height) * canvas.height)),
    ];
  }

  function startBox(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (phase !== 'select') return;
    const point = pointerPosition(event);
    const hit = candidates
      .filter((item) => {
        const [x, y, width, height] = item.box;
        return point[0] >= x && point[0] <= x + width && point[1] >= y && point[1] <= y + height;
      })
      .sort((a, b) => a.box[2] * a.box[3] - b.box[2] * b.box[3])[0];
    if (hit) {
      setBox(hit.box);
      selectionRef.current = {
        time: videoRef.current?.currentTime ?? 0,
        box: [...hit.box] as Box,
      };
      drawFrame(hit.box);
      setNotice('已選擇 AI 辨識框；也可直接拖曳重新框選');
      return;
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    dragStartRef.current = point;
    setBox(null);
    drawFrame(null);
  }

  function moveBox(event: ReactPointerEvent<HTMLCanvasElement>) {
    const start = dragStartRef.current;
    if (!start || phase !== 'select') return;
    const next = normalizeBox(start, pointerPosition(event));
    setBox(next);
    drawFrame(next);
  }

  function finishBox(event: ReactPointerEvent<HTMLCanvasElement>) {
    const start = dragStartRef.current;
    if (!start || phase !== 'select') return;
    dragStartRef.current = null;
    const next = normalizeBox(start, pointerPosition(event));
    if (next[2] < 12 || next[3] < 12) {
      setBox(null);
      drawFrame(null);
      setNotice('框選範圍太小，請重新框住完整主角');
      return;
    }
    setBox(next);
    selectionRef.current = {
      time: videoRef.current?.currentTime ?? 0,
      box: [...next] as Box,
    };
    drawFrame(next);
    setNotice('主角已指定；可開始 3 秒 ViT 追蹤測試');
  }

  async function detectSubjects() {
    const video = videoRef.current;
    if (!video) return;
    setDetecting(true);
    setNotice('正在本機載入 SSDLite 並掃描目前影格…');
    try {
      if (!detectorRef.current) {
        await import('@tensorflow/tfjs');
        const cocoSsd = await import('@tensorflow-models/coco-ssd');
        detectorRef.current = await cocoSsd.load({
          base: 'lite_mobilenet_v2',
          modelUrl: new URL('models/ssdlite_mobilenet_v2/model.json', document.baseURI).href,
        });
      }
      const predictions = await detectorRef.current.detect(video, 30, 0.25);
      const nextCandidates: Candidate[] = predictions
        .filter((item) => item.class === 'person' || item.class === 'dog' || item.class === 'cat')
        .map((item) => ({
          box: item.bbox as Box,
          label: item.class === 'person' ? '人物' : item.class === 'dog' ? '狗' : '貓',
          score: item.score,
        }));
      setCandidates(nextCandidates);
      drawFrame(box, undefined, nextCandidates);
      setNotice(
        nextCandidates.length
          ? '找到 ' + nextCandidates.length + ' 個候選框；點選其中一個或手動畫框'
          : '沒有找到人物或寵物，請直接用手指框選',
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setNotice('初始辨識失敗：' + message + '；仍可手動畫框');
    } finally {
      setDetecting(false);
    }
  }

  async function seekTo(seconds: number) {
    const video = videoRef.current;
    if (!video) throw new Error('影片尚未載入');
    if (Math.abs(video.currentTime - seconds) < 0.001) {
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      return;
    }
    await new Promise<void>((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        cleanup();
        reject(new Error('影片逐幀 seek 逾時'));
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
        reject(new Error('影片 seek 失敗'));
      };
      video.addEventListener('seeked', done);
      video.addEventListener('error', failed);
      video.currentTime = seconds;
    });
  }

  async function runTracking() {
    const video = videoRef.current;
    if (!video || !box) return;
    cancelRef.current = false;
    setPhase('tracking');
    setStats(null);
    setProgress(0);
    setCurrentScore(null);

    const startTime = video.currentTime;
    const endTime = Math.min(video.duration, startTime + 3);
    const interval = 1 / 10;
    const results: TrackResult[] = [];
    const started = performance.now();

    try {
      setNotice('正在載入本機 ViT 模型…');
      if (!trackerRef.current) {
        const modelUrl = new URL('models/vittrack.onnx', document.baseURI).href;
        trackerRef.current = await VitTracker.create(modelUrl);
      }
      trackerRef.current.initialize(video, box);

      let frameIndex = 0;
      for (let at = startTime + interval; at <= endTime + 0.001; at += interval) {
        if (cancelRef.current) throw new Error('使用者已取消追蹤');
        const frameTime = Math.min(at, endTime);
        await seekTo(frameTime);
        const result = await trackerRef.current.update(video);
        results.push(result);
        frameIndex += 1;
        setBox(result.box);
        setCurrentScore(result.score);
        setProgress((frameTime - startTime) / Math.max(0.001, endTime - startTime));
        setNotice('ViT 追蹤中 · 第 ' + frameIndex + ' 幀');
        drawFrame(result.box, result.score);
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
        if (frameTime >= endTime) break;
      }

      const elapsedMs = performance.now() - started;
      const inferenceTotal = results.reduce((sum, item) => sum + item.inferenceMs, 0);
      const scoreTotal = results.reduce((sum, item) => sum + item.score, 0);
      setStats({
        frames: results.length,
        elapsedMs,
        averageInferenceMs: inferenceTotal / Math.max(1, results.length),
        averageScore: scoreTotal / Math.max(1, results.length),
        acceptedFrames: results.filter((item) => item.accepted).length,
      });
      setProgress(1);
      setPhase('complete');
      setNotice('3 秒 ViT 路徑測試完成；尚未進行影片輸出');
    } catch (error) {
      setPhase('select');
      const message = error instanceof Error ? error.message : String(error);
      setNotice(message);
      drawFrame(box);
    }
  }

  async function runFullTracking() {
    const video = videoRef.current;
    const selection = selectionRef.current;
    if (!video || !selection || !Number.isFinite(video.duration)) {
      setNotice('請先在影片畫面框選主角');
      return;
    }

    cancelRef.current = false;
    setPhase('tracking');
    setStats(null);
    setTrackPath([]);
    setExportUrl('');
    setExportBlob(null);
    setExportInfo(null);
    setShowEffectPreview(false);
    setEffectTestPassed(false);
    personEffectRendererRef.current?.reset();
    setCorrectionCount(0);
    setProgress(0);
    setCurrentScore(null);

    const interval = 1 / 10;
    const forwardCount = Math.ceil((video.duration - selection.time) / interval);
    const backwardCount = Math.ceil(selection.time / interval);
    const forwardTimes = Array.from({ length: forwardCount }, (_, index) =>
      Math.min(video.duration, selection.time + (index + 1) * interval),
    );
    const backwardTimes = Array.from({ length: backwardCount }, (_, index) =>
      Math.max(0, selection.time - (index + 1) * interval),
    );
    const totalFrames = forwardTimes.length + backwardTimes.length;
    const points: TrackPoint[] = [{
      time: selection.time,
      box: [...selection.box] as Box,
      score: 1,
      accepted: true,
    }];
    const measurements: TrackResult[] = [];
    const started = performance.now();
    let processed = 0;

    try {
      setNotice('正在載入本機 ViT 模型…');
      if (!trackerRef.current) {
        const modelUrl = new URL('models/vittrack.onnx', document.baseURI).href;
        trackerRef.current = await VitTracker.create(modelUrl);
      }

      const trackDirection = async (times: number[], label: string) => {
        await seekTo(selection.time);
        trackerRef.current!.initialize(video, selection.box);
        for (const frameTime of times) {
          if (cancelRef.current) throw new Error('使用者已取消追蹤');
          await seekTo(frameTime);
          const result = await trackerRef.current!.update(video);
          measurements.push(result);
          points.push({
            time: frameTime,
            box: [...result.box] as Box,
            score: result.score,
            accepted: result.accepted,
          });
          processed += 1;
          setBox(result.box);
          setCurrentScore(result.score);
          setProgress(processed / Math.max(1, totalFrames));
          setNotice(label + ' · ' + processed + ' / ' + totalFrames + ' 幀');
          drawFrame(result.box, result.score);
          await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
        }
      };

      await trackDirection(forwardTimes, '向後完整追蹤');
      await trackDirection(backwardTimes, '補齊選角前片段');

      points.sort((left, right) => left.time - right.time);
      const elapsedMs = performance.now() - started;
      const inferenceTotal = measurements.reduce((sum, item) => sum + item.inferenceMs, 0);
      const scoreTotal = measurements.reduce((sum, item) => sum + item.score, 0);
      setTrackPath(points);
      setStats({
        frames: measurements.length,
        elapsedMs,
        averageInferenceMs: inferenceTotal / Math.max(1, measurements.length),
        averageScore: scoreTotal / Math.max(1, measurements.length),
        acceptedFrames: measurements.filter((item) => item.accepted).length,
      });
      setProgress(1);
      setPhase('path-ready');
      await seekTo(selection.time);
      setBox(selection.box);
      drawFrame(selection.box);
      setNotice('完整 ViT 路徑已建立；可調整構圖並輸出影片');
    } catch (error) {
      setPhase('select');
      const message = error instanceof Error ? error.message : String(error);
      setNotice(message);
      await seekTo(selection.time).catch(() => undefined);
      setBox(selection.box);
      drawFrame(selection.box);
    }
  }

  function renderCorrectionPreview(at = videoRef.current?.currentTime ?? correctionTime) {
    const video = videoRef.current;
    const canvas = effectPreviewCanvasRef.current;
    const renderer = personEffectRendererRef.current;
    if (!video || !canvas || !renderer || trackPath.length < 2) return;
    configureOutputCanvas(canvas, aspect);
    renderer.render(
      video,
      canvas,
      smoothTrackPath(trackPath, smoothness),
      at,
      subjectScale,
      personEffects,
    );
    setShowEffectPreview(true);
  }

  function beginMaskCorrection() {
    const video = videoRef.current;
    const window = effectTestWindow;
    if (!video || !window || !personEffectRendererRef.current) return;
    video.pause();
    const at = clampTime(video.currentTime, window.start, window.end);
    setCorrectionTime(at);
    setEditingMask(true);
    void seekTo(at).then(() => renderCorrectionPreview(at)).catch((error) => {
      setNotice(error instanceof Error ? error.message : String(error));
    });
    setNotice('修正主角：紅色移除誤入背景，綠色補回主角；閃爍由系統自動穩定');
  }

  function closeMaskCorrection() {
    correctionDrawingRef.current = false;
    personEffectRendererRef.current?.endCorrectionStroke();
    setEditingMask(false);
    setNotice(correctionCount > 0 ? '修正已保存，正式輸出時會自動套用' : '未加入人工修正');
  }

  function seekCorrectionPreview(at: number) {
    const video = videoRef.current;
    const window = effectTestWindow;
    if (!video || !window) return;
    const next = clampTime(at, window.start, window.end);
    setCorrectionTime(next);
    video.pause();
    if (Math.abs(video.currentTime - next) < 0.001) {
      renderCorrectionPreview(next);
    } else {
      video.currentTime = next;
    }
  }

  function handleVideoSeeked() {
    if (!editingMask) return;
    const at = videoRef.current?.currentTime ?? correctionTime;
    setCorrectionTime(at);
    renderCorrectionPreview(at);
  }

  function paintMaskCorrection(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (!editingMask || !correctionDrawingRef.current) return;
    const canvas = effectPreviewCanvasRef.current;
    const video = videoRef.current;
    const renderer = personEffectRendererRef.current;
    if (!canvas || !video || !renderer) return;
    const bounds = canvas.getBoundingClientRect();
    if (bounds.width <= 0 || bounds.height <= 0) return;
    const x = clampTime((event.clientX - bounds.left) / bounds.width, 0, 1);
    const y = clampTime((event.clientY - bounds.top) / bounds.height, 0, 1);
    const brushPixels = correctionBrush * canvas.width / bounds.width;
    const count = renderer.paintCorrection(
      video.currentTime,
      x,
      y,
      brushPixels,
      correctionMode,
      video,
      canvas,
      smoothTrackPath(trackPath, smoothness),
      subjectScale,
      personEffects,
    );
    setCorrectionCount(count);
    setExportUrl('');
    setExportBlob(null);
    setExportInfo(null);
    renderCorrectionPreview(video.currentTime);
  }

  function startMaskCorrection(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (!editingMask) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    correctionDrawingRef.current = true;
    personEffectRendererRef.current?.beginCorrectionStroke();
    paintMaskCorrection(event);
  }

  function finishMaskCorrection(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (!editingMask) return;
    correctionDrawingRef.current = false;
    personEffectRendererRef.current?.endCorrectionStroke();
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function undoMaskCorrection() {
    const count = personEffectRendererRef.current?.undoCorrection() ?? 0;
    setCorrectionCount(count);
    renderCorrectionPreview();
    setNotice(count > 0 ? '已復原上一筆修正' : '所有人工修正已清除');
  }

  function clearMaskCorrections() {
    personEffectRendererRef.current?.clearCorrections();
    setCorrectionCount(0);
    renderCorrectionPreview();
    setNotice('人工修正已全部清除；保留自動時間穩定');
  }

  async function runEffectTest() {
    const video = videoRef.current;
    const previewCanvas = effectPreviewCanvasRef.current;
    const selection = selectionRef.current;
    if (!video || !previewCanvas || !selection || trackPath.length < 2) {
      setNotice('請先完成整支影片的 ViT 追蹤');
      return;
    }
    if (!personEffects.enabled) {
      setNotice('請先開啟 Stage 1 人物特效');
      return;
    }

    clearRenderedOutput();
    cancelRef.current = false;
    setPhase('effect-testing');
    setProgress(0);
    setCurrentScore(null);
    setNotice('正在載入本機指定主角去背模型與時間穩定…');
    const testStart = Math.min(video.currentTime, Math.max(0, video.duration - 3));
    const testEnd = Math.min(video.duration, testStart + 3);
    const testPreviewTime = testStart + (testEnd - testStart) * 0.5;
    const interval = 1 / 10;
    const totalFrames = Math.max(1, Math.ceil((testEnd - testStart) / interval));

    try {
      const renderer = await getPersonEffectRenderer();
      configureOutputCanvas(previewCanvas, aspect);
      const smoothedPath = smoothTrackPath(trackPath, smoothness);
      video.pause();

      await renderer.prepare(video, smoothedPath, {
        startTime: testStart,
        endTime: testEnd,
        preserveFraming: false,
        retainSourceForCorrections: true,
        onProgress: (next) => {
          setProgress(next * 0.9);
          setNotice('先逐格鎖定主角並去背 · ' + Math.round(next * 100) + '%');
        },
        isCancelled: () => cancelRef.current,
      });
      renderer.resetPlayback();

      for (let frame = 0; frame <= totalFrames; frame += 1) {
        if (cancelRef.current) throw new Error('使用者已取消特效測試');
        const at = Math.min(testEnd, testStart + frame * interval);
        await seekTo(at);
        renderer.render(video, previewCanvas, smoothedPath, at, subjectScale, personEffects);
        setShowEffectPreview(true);
        const next = frame / Math.max(1, totalFrames);
        setProgress(0.9 + next * 0.1);
        setNotice('套用特效測試 · ' + Math.round(next * 100) + '%');
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      }

      setProgress(1);
      setEffectTestPassed(true);
      setEffectTestWindow({ start: testStart, end: testEnd });
      setCorrectionTime(testPreviewTime);
      setPhase('path-ready');
      await seekTo(testPreviewTime);
      renderer.resetPlayback();
      renderer.render(video, previewCanvas, smoothedPath, testPreviewTime, subjectScale, personEffects);
      setShowEffectPreview(true);
      setNotice('3 秒自動去背完成；若有多餘物品再按「修正主角」');
    } catch (error) {
      setShowEffectPreview(false);
      setEffectTestPassed(false);
      setPhase('path-ready');
      setNotice(error instanceof Error ? error.message : String(error));
    } finally {
      video.pause();
      await seekTo(testPreviewTime).catch(() => undefined);
    }
  }
  async function exportVideo(codec: 'h264' | 'hevc') {
    const video = videoRef.current;
    const renderCanvas = renderCanvasRef.current;
    if (!video || !renderCanvas || trackPath.length < 2) {
      setNotice('請先完成整支影片的 ViT 追蹤');
      return;
    }
    if (personEffects.enabled && !effectTestPassed) {
      setNotice('請先完成 3 秒特效測試，再輸出完整影片');
      return;
    }
    cancelRef.current = false;
    setShowEffectPreview(false);
    setPhase('exporting');
    setProgress(0);
    setNotice(codec === 'hevc' ? '正在準備 HEVC 母片輸出…' : '正在準備 H.264 相容影片輸出…');

    try {
      if (!exporterRef.current) exporterRef.current = new RealtimeVideoExporter(video);
      setNotice('正在啟動 Safari 音訊編碼…');
      await exporterRef.current.primeAudio();
      const effectRenderer = personEffects.enabled ? await getPersonEffectRenderer() : undefined;
      if (effectRenderer) {
        await effectRenderer.prepare(video, smoothTrackPath(trackPath, smoothness), {
          startTime: 0,
          endTime: video.duration,
          preserveFraming: false,
          onProgress: (next) => {
            setProgress(next * 0.8);
            setNotice('先完成整支影片的主角去背 · ' + Math.round(next * 100) + '%');
          },
          isCancelled: () => cancelRef.current,
        });
        effectRenderer.resetPlayback();
        setNotice('主角去背完成，正在套用特效與編碼…');
      }
      const result = await exporterRef.current.export(trackPath, renderCanvas, {
        aspect,
        subjectScale,
        smoothness,
        codec,
        effects: personEffects,
        effectRenderer,
        onProgress: (next) => {
          setProgress(personEffects.enabled ? 0.8 + next * 0.2 : next);
          setNotice('本機編碼中 · ' + Math.round(next * 100) + '%');
        },
        isCancelled: () => cancelRef.current,
      });
      const baseName = (sourceFile?.name ?? 'NiviTrack').replace(/\.[^.]+$/, '');
      const name = baseName + '-NiviTrack' + (personEffects.enabled ? '-FX' : '') + '-' + aspect.replace(':', 'x') + '.mp4';
      setExportBlob(result.blob);
      setExportUrl(URL.createObjectURL(result.blob));
      setExportInfo({
        name,
        size: formatBytes(result.blob.size),
        mimeType: result.mimeType,
        resolution: result.width + ' × ' + result.height,
      });
      setPhase('path-ready');
      setNotice('輸出完成；影片仍在這台 iPhone，可分享或儲存到檔案');
    } catch (error) {
      setPhase('path-ready');
      const message = error instanceof Error ? error.message : String(error);
      setNotice(message);
    }
  }

  async function shareExport() {
    if (!exportBlob || !exportInfo) return;
    const file = new File([exportBlob], exportInfo.name, { type: exportInfo.mimeType });
    try {
      if (navigator.canShare?.({ files: [file] })) {
        await navigator.share({ files: [file], title: 'NiviTrack 輸出影片' });
        setNotice('已開啟 iPhone 分享選單');
        return;
      }
      const link = document.createElement('a');
      link.href = exportUrl;
      link.download = exportInfo.name;
      link.click();
      setNotice('已交給 Safari 下載；可從下載項目儲存到檔案');
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return;
      setNotice('分享失敗：' + (error instanceof Error ? error.message : String(error)));
    }
  }

  function cancelTracking() {
    cancelRef.current = true;
    setNotice(phase === 'exporting' ? '正在取消輸出…' : phase === 'effect-testing' ? '正在取消特效測試…' : '正在取消追蹤…');
  }

  const step = phase === 'choose' ? 1 : phase === 'select' ? 2 : 3;

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="NiviTrack 首頁">
          <span className="brand-mark">N</span><span>NiviTrack</span>
        </a>
        <span className="local-pill"><i aria-hidden="true" />iPhone 本機處理</span>
      </header>

      <section className="hero">
        <div className="eyebrow">IPHONE WEB APP · 技術原型</div>
        <h1>讓主角一直留在<span>畫面正中央。</span></h1>
        <p>選擇 iPhone 原始 MOV／HEVC，直接在手機裡辨識、追蹤與輸出。影片不會離開這台裝置。</p>
        <div className="steps" aria-label="處理步驟">
          <div className={'step ' + (step >= 1 ? 'active' : '')}><b>01</b><span>選擇影片</span></div>
          <div className={'step ' + (step >= 2 ? 'active' : '')}><b>02</b><span>指定主角</span></div>
          <div className={'step ' + (step >= 3 ? 'active' : '')}><b>03</b><span>追蹤與輸出</span></div>
        </div>
      </section>

      <section className="workspace">
        <div className="video-panel">
          {!videoUrl ? (
            <button className="picker" type="button" onClick={openVideoPicker}>
              <span className="picker-icon" aria-hidden="true">+</span>
              <strong>選擇一支影片</strong>
              <small>支援「照片」與「檔案」中的 MOV、HEVC、MP4</small>
              <em>選擇影片</em>
            </button>
          ) : (
            <>
              <div className="video-stage">
                <video
                  ref={videoRef}
                  className={phase === 'choose' || phase === 'exporting' || phase === 'effect-testing' || showEffectPreview ? '' : 'is-hidden'}
                  src={videoUrl}
                  controls
                  playsInline
                  preload="metadata"
                  onLoadedMetadata={readMetadata}
                  onSeeked={handleVideoSeeked}
                  onError={() => setNotice('Safari 無法解碼這支影片，請保留檔案供實機記錄')}
                />
                <canvas
                  ref={canvasRef}
                  className={phase === 'choose' || phase === 'exporting' || phase === 'effect-testing' || showEffectPreview ? 'tracking-canvas is-hidden' : 'tracking-canvas'}
                  onPointerDown={startBox}
                  onPointerMove={moveBox}
                  onPointerUp={finishBox}
                  onPointerCancel={finishBox}
                />
                <canvas
                  ref={effectPreviewCanvasRef}
                  className={(showEffectPreview ? 'effect-preview-canvas' : 'effect-preview-canvas is-hidden') + (editingMask ? ' editing-mask' : '')}
                  aria-label="3 秒人物特效預覽"
                  onPointerDown={startMaskCorrection}
                  onPointerMove={paintMaskCorrection}
                  onPointerUp={finishMaskCorrection}
                  onPointerCancel={finishMaskCorrection}
                />
                <span className="source-badge">
                  {phase === 'choose' ? '影片本機解碼' : phase === 'select' ? '手指框選主角' : phase === 'effect-testing' ? '3 秒人物特效測試' : editingMask ? (correctionMode === 'remove' ? '紅色筆刷 · 移除背景' : '綠色筆刷 · 補回主角') : showEffectPreview ? 'Stage 1 特效預覽' : phase === 'exporting' ? 'Safari 本機編碼' : phase === 'path-ready' ? '構圖與輸出' : 'ViT 本機推論'}
                </span>
                {(phase === 'tracking' || phase === 'effect-testing' || phase === 'exporting') && (
                  <div className="progress-overlay">
                    <strong>{Math.round(progress * 100)}%</strong>
                    <span>{phase === 'effect-testing' ? '人物去背＋延遲分身' : 'score ' + (currentScore === null ? '—' : currentScore.toFixed(3))}</span>
                  </div>
                )}
              </div>
              <div className="video-actions">
                {phase !== 'tracking' && phase !== 'effect-testing' && phase !== 'exporting' && (
                  <button type="button" onClick={openVideoPicker}>重新選擇影片</button>
                )}
                {phase === 'choose' && (
                  <button className="primary" type="button" disabled={!videoInfo} onClick={enterSelection}>
                    進入主角選取
                  </button>
                )}
                {phase === 'select' && (
                  <>
                    <button type="button" onClick={() => { setPhase('choose'); setBox(null); }}>返回播放</button>
                    <button type="button" disabled={detecting} onClick={detectSubjects}>
                      {detecting ? 'AI 掃描中…' : 'AI 尋找人物／寵物'}
                    </button>
                    <button type="button" disabled={!box} onClick={runTracking}>
                      測試 3 秒 ViT
                    </button>
                    <button className="primary" type="button" disabled={!box} onClick={runFullTracking}>
                      追蹤完整影片
                    </button>
                  </>
                )}
                {(phase === 'tracking' || phase === 'effect-testing' || phase === 'exporting') && (
                  <button className="danger" type="button" onClick={cancelTracking}>{phase === 'exporting' ? '取消輸出' : phase === 'effect-testing' ? '取消特效測試' : '取消追蹤'}</button>
                )}
                {phase === 'complete' && (
                  <>
                    <button type="button" onClick={enterSelection}>重新框選</button>
                    <button className="primary" type="button" onClick={runFullTracking}>追蹤完整影片</button>
                  </>
                )}
                {phase === 'path-ready' && (
                  <button type="button" onClick={enterSelection}>重新選角與追蹤</button>
                )}
              </div>
              {phase === 'path-ready' && effectTestPassed && showEffectPreview && effectTestWindow && (
                <section className={'mask-correction-panel ' + (editingMask ? 'is-editing' : '')}>
                  <div className="mask-correction-heading">
                    <div>
                      <span>光流防閃＋移動背景排除已套用</span>
                      <strong>智慧去背完成</strong>
                    </div>
                    {!editingMask ? (
                      <button type="button" onClick={beginMaskCorrection}>修正主角（最後手段）</button>
                    ) : (
                      <b>{correctionCount} 筆修正</b>
                    )}
                  </div>
                  {!editingMask ? (
                    <p>正常情況直接輸出，不需要手修；只有仍看見明顯誤入物或主角缺角時才開啟修正。</p>
                  ) : (
                    <>
                      <label className="correction-timeline">
                        <span><b>選擇錯誤畫面</b><em>{correctionTime.toFixed(1)} 秒</em></span>
                        <input
                          type="range"
                          min={effectTestWindow.start}
                          max={effectTestWindow.end}
                          step="0.033333"
                          value={correctionTime}
                          onChange={(event) => seekCorrectionPreview(Number(event.target.value))}
                        />
                      </label>
                      <div className="correction-tools" aria-label="主角修正工具">
                        <button
                          className={correctionMode === 'remove' ? 'selected remove' : ''}
                          type="button"
                          onClick={() => setCorrectionMode('remove')}
                        >移除背景</button>
                        <button
                          className={correctionMode === 'keep' ? 'selected keep' : ''}
                          type="button"
                          onClick={() => setCorrectionMode('keep')}
                        >補回主角</button>
                      </div>
                      <label className="correction-brush">
                        <span><b>筆刷大小</b><em>{correctionBrush}px</em></span>
                        <input
                          type="range"
                          min="12"
                          max="72"
                          step="4"
                          value={correctionBrush}
                          onChange={(event) => setCorrectionBrush(Number(event.target.value))}
                        />
                      </label>
                      <p>直接在上方畫面塗抹。移除用於風扇、路人及其他背景；補回用於主角被吃掉的部分。</p>
                      <div className="correction-actions">
                        <button type="button" disabled={correctionCount === 0} onClick={undoMaskCorrection}>復原上一筆</button>
                        <button type="button" disabled={correctionCount === 0} onClick={clearMaskCorrections}>全部清除</button>
                        <button className="primary" type="button" onClick={closeMaskCorrection}>完成修正</button>
                      </div>
                    </>
                  )}
                </section>
              )}
              {(phase === 'path-ready' || phase === 'effect-testing' || phase === 'exporting') && trackPath.length > 1 && (
                <section className="export-panel">
                  <div className="export-heading">
                    <div>
                      <span>完整路徑已就緒</span>
                      <strong>選擇輸出構圖</strong>
                    </div>
                    <b>{trackPath.length} 點</b>
                  </div>

                  <div className="aspect-options" aria-label="輸出比例">
                    {(['9:16', '1:1', '16:9'] as AspectPreset[]).map((preset) => (
                      <button
                        className={aspect === preset ? 'selected' : ''}
                        type="button"
                        key={preset}
                        disabled={phase === 'exporting'}
                        onClick={() => { setAspect(preset); clearRenderedOutput(); }}
                      >
                        {preset}
                      </button>
                    ))}
                  </div>

                  <label className="range-control">
                    <span><b>主角大小</b><em>{Math.round(subjectScale * 100)}%</em></span>
                    <input
                      type="range"
                      min="25"
                      max="80"
                      value={Math.round(subjectScale * 100)}
                      disabled={phase === 'exporting'}
                      onChange={(event) => { setSubjectScale(Number(event.target.value) / 100); clearRenderedOutput(); }}
                    />
                  </label>

                  <label className="range-control">
                    <span><b>置中柔順度</b><em>{Math.round(smoothness * 100)}%</em></span>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={Math.round(smoothness * 100)}
                      disabled={phase === 'exporting'}
                      onChange={(event) => { setSmoothness(Number(event.target.value) / 100); clearRenderedOutput(); }}
                    />
                  </label>

                  <div className="effect-lab">
                    <div className="effect-heading">
                      <div>
                        <span>STAGE 1</span>
                        <strong>人物去背與延遲分身</strong>
                      </div>
                      <button
                        className={personEffects.enabled ? 'selected' : ''}
                        type="button"
                        disabled={phase !== 'path-ready'}
                        onClick={() => updatePersonEffects({ enabled: !personEffects.enabled })}
                      >
                        {personEffects.enabled ? '已開啟' : '開啟特效'}
                      </button>
                    </div>
                    <p className="effect-note">ViT 追蹤維持原樣；框內使用 MagicTouch＋Pose 去背，骨架保護人體核心，再以光流延續前後遮罩與移動中的背景記憶。所有影像仍在這台 iPhone 本機處理。</p>

                    {personEffects.enabled && (
                      <>
                        <div className="effect-group">
                          <span>背景</span>
                          <div className="effect-options three">
                            {([['original', '原始'], ['black', '純黑'], ['blur', '模糊']] as Array<[BackgroundEffect, string]>).map(([value, label]) => (
                              <button
                                className={personEffects.background === value ? 'selected' : ''}
                                type="button"
                                key={value}
                                disabled={phase !== 'path-ready'}
                                onClick={() => updatePersonEffects({ background: value })}
                              >{label}</button>
                            ))}
                          </div>
                        </div>

                        <div className="effect-group">
                          <span>主角</span>
                          <div className="effect-options three">
                            {([['original', '原色'], ['blue', '藍色剪影'], ['black', '黑色剪影']] as Array<[SubjectEffect, string]>).map(([value, label]) => (
                              <button
                                className={personEffects.subject === value ? 'selected' : ''}
                                type="button"
                                key={value}
                                disabled={phase !== 'path-ready'}
                                onClick={() => updatePersonEffects({ subject: value })}
                              >{label}</button>
                            ))}
                          </div>
                        </div>

                        <div className="effect-group">
                          <span>邊框</span>
                          <div className="effect-options three">
                            {([['white', '白邊'], ['neon', '霓虹光'], ['none', '無邊框']] as Array<[OutlineEffect, string]>).map(([value, label]) => (
                              <button
                                className={personEffects.outline === value ? 'selected' : ''}
                                type="button"
                                key={value}
                                disabled={phase !== 'path-ready'}
                                onClick={() => updatePersonEffects({ outline: value })}
                              >{label}</button>
                            ))}
                          </div>
                        </div>

                        <div className="effect-group">
                          <span>分身排列</span>
                          <div className="effect-options two">
                            {([['trail', '前後殘影'], ['lineup', '左右並排']] as Array<[CloneLayout, string]>).map(([value, label]) => (
                              <button
                                className={personEffects.cloneLayout === value ? 'selected' : ''}
                                type="button"
                                key={value}
                                disabled={phase !== 'path-ready'}
                                onClick={() => updatePersonEffects({ cloneLayout: value })}
                              >{label}</button>
                            ))}
                          </div>
                        </div>

                        <div className="effect-group">
                          <span>分身數量</span>
                          <div className="effect-options five">
                            {[0, 1, 2, 3, 4].map((count) => (
                              <button
                                className={personEffects.cloneCount === count ? 'selected' : ''}
                                type="button"
                                key={count}
                                disabled={phase !== 'path-ready'}
                                onClick={() => updatePersonEffects({ cloneCount: count })}
                              >{count}</button>
                            ))}
                          </div>
                        </div>

                        <label className="range-control">
                          <span><b>分身延遲</b><em>{personEffects.cloneDelay.toFixed(1)} 秒</em></span>
                          <input
                            type="range"
                            min="0"
                            max="100"
                            step="5"
                            value={Math.round(personEffects.cloneDelay * 100)}
                            disabled={phase !== 'path-ready' || personEffects.cloneCount === 0}
                            onChange={(event) => updatePersonEffects({ cloneDelay: Number(event.target.value) / 100 })}
                          />
                        </label>

                        <label className="range-control">
                          <span><b>分身透明度</b><em>{Math.round(personEffects.cloneOpacity * 100)}%</em></span>
                          <input
                            type="range"
                            min="10"
                            max="100"
                            step="5"
                            value={Math.round(personEffects.cloneOpacity * 100)}
                            disabled={phase !== 'path-ready' || personEffects.cloneCount === 0}
                            onChange={(event) => updatePersonEffects({ cloneOpacity: Number(event.target.value) / 100 })}
                          />
                        </label>

                        <div className="color-control">
                          <label>
                            <span>分身顏色</span>
                            <input
                              type="color"
                              value={personEffects.cloneColor}
                              disabled={phase !== 'path-ready' || personEffects.cloneCount === 0}
                              onChange={(event) => updatePersonEffects({ cloneColor: event.target.value })}
                            />
                          </label>
                          <div className="color-swatches" aria-label="快速選擇分身顏色">
                            {['#165dff', '#ff2d55', '#35d292', '#d9f06f', '#9b5cff'].map((color) => (
                              <button
                                className={personEffects.cloneColor === color ? 'selected' : ''}
                                style={{ backgroundColor: color }}
                                type="button"
                                key={color}
                                disabled={phase !== 'path-ready' || personEffects.cloneCount === 0}
                                aria-label={'選擇 ' + color}
                                onClick={() => updatePersonEffects({ cloneColor: color })}
                              />
                            ))}
                          </div>
                        </div>

                        <button
                          className={'effect-test-button ' + (effectTestPassed ? 'passed' : '')}
                          type="button"
                          disabled={phase !== 'path-ready'}
                          onClick={() => void runEffectTest()}
                        >
                          {effectTestPassed ? '✓ 3 秒特效測試完成' : '測試 3 秒特效'}
                        </button>
                      </>
                    )}
                  </div>
                  <div className="export-buttons">
                    <button
                      className="primary"
                      type="button"
                      disabled={phase !== 'path-ready' || !recorderSupport.h264 || (personEffects.enabled && !effectTestPassed)}
                      onClick={() => void exportVideo('h264')}
                    >
                      輸出相容 MP4
                    </button>
                    <button
                      type="button"
                      disabled={phase !== 'path-ready' || !recorderSupport.hevc || (personEffects.enabled && !effectTestPassed)}
                      onClick={() => void exportVideo('hevc')}
                    >
                      輸出 HEVC 母片（MP4）
                    </button>
                  </div>

                  {!recorderSupport.hevc && (
                    <p className="export-note">
                      此 Safari 未提供 HEVC 網頁編碼；可先用 H.264 MP4。MOV／HEVC 硬性需求需改用原生 AVFoundation。
                    </p>
                  )}

                  {exportUrl && exportInfo && (
                    <div className="export-result">
                      <video src={exportUrl} controls playsInline preload="metadata" />
                      <div>
                        <strong>{exportInfo.name}</strong>
                        <span>{exportInfo.resolution} · {exportInfo.size}</span>
                      </div>
                      <button className="primary" type="button" onClick={() => void shareExport()}>
                        分享／儲存到 iPhone
                      </button>
                    </div>
                  )}
                </section>
              )}
              <canvas ref={renderCanvasRef} className="export-canvas" aria-hidden="true" />
            </>
          )}
          <input ref={inputRef} className="sr-only" type="file" accept="video/*,.mov,.mp4" onChange={chooseVideo} />
        </div>

        <aside className="side-panel">
          <div className="status-card">
            <div className="card-heading"><span>裝置能力</span><b>{readyCount}/{capabilities.length || 8}</b></div>
            <div className="capability-list">
              {capabilities.length === 0 ? <p className="checking">正在檢查 Safari…</p> : capabilities.map((item) => (
                <div className="capability" key={item.label}>
                  <span className={item.available ? 'ok' : 'missing'}>{item.available ? '✓' : '—'}</span>
                  <div><strong>{item.label}</strong><small>{item.detail}</small></div>
                </div>
              ))}
            </div>
          </div>

          <div className="info-card">
            <div className="card-heading"><span>{stats ? '追蹤量測' : '來源影片'}</span><b>{stats ? 'MEASURED' : videoInfo ? 'READY' : '—'}</b></div>
            {stats ? (
              <dl>
                <div><dt>處理幀</dt><dd>{stats.frames}</dd></div>
                <div><dt>總耗時</dt><dd>{(stats.elapsedMs / 1000).toFixed(2)} 秒</dd></div>
                <div><dt>推論</dt><dd>{stats.averageInferenceMs.toFixed(1)} ms／幀</dd></div>
                <div><dt>平均分數</dt><dd>{stats.averageScore.toFixed(4)}</dd></div>
                <div><dt>接受幀</dt><dd>{stats.acceptedFrames} / {stats.frames}</dd></div>
              </dl>
            ) : videoInfo ? (
              <dl>
                <div><dt>檔名</dt><dd>{videoInfo.name}</dd></div>
                <div><dt>解析度</dt><dd>{videoInfo.resolution}</dd></div>
                <div><dt>長度</dt><dd>{videoInfo.duration}</dd></div>
                <div><dt>容量</dt><dd>{videoInfo.size}</dd></div>
              </dl>
            ) : (
              <p>選片後會顯示實際讀取結果，不會先將 MOV 轉成 MP4。</p>
            )}
          </div>
        </aside>
      </section>

      <footer>
        <p><span aria-hidden="true">●</span>{notice}</p>
        <span>所有運算留在此裝置</span>
      </footer>
    </main>
  );
}
