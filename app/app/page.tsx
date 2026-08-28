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
  ExportOperation,
  FilterPreset,
  getFilterCss,
  getRecorderSupport,
  RealtimeVideoExporter,
  RecorderSupport,
  TrackPoint,
} from '../lib/video-export';
import type { ModnetPreviewTimeline } from '../lib/modnet-background-preview';

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
  aspectRatio: number;
  width: number;
  height: number;
};

type Phase = 'choose' | 'tool-ready' | 'crop-select' | 'select' | 'tracking' | 'masking' | 'previewing' | 'complete' | 'path-ready' | 'exporting';

type ToolId =
  | `filter-${FilterPreset}`
  | 'crop-9-16'
  | 'crop-square'
  | 'crop-16-9'
  | 'crop-free'
  | 'track'
  | 'remove-background';

type ToolChoice = {
  id: ToolId;
  group: '濾鏡' | '裁切' | '鎖定' | '去背';
  name: string;
  detail: string;
  fileTag: string;
};

const TOOL_CHOICES: ToolChoice[] = [
  { id: 'filter-vivid', group: '濾鏡', name: '鮮明增色', detail: '提高飽和與層次', fileTag: 'Vivid' },
  { id: 'filter-soft', group: '濾鏡', name: '柔和人像', detail: '降低反差、提亮膚色', fileTag: 'Soft' },
  { id: 'filter-cinematic', group: '濾鏡', name: '電影冷調', detail: '沉穩低彩度質感', fileTag: 'Cinema' },
  { id: 'filter-warm', group: '濾鏡', name: '暖陽色調', detail: '溫暖明亮的色彩', fileTag: 'Warm' },
  { id: 'filter-mono', group: '濾鏡', name: '黑白高反差', detail: '俐落黑白明暗', fileTag: 'Mono' },
  { id: 'filter-vintage', group: '濾鏡', name: '復古底片', detail: '低彩暖褐底片感', fileTag: 'Vintage' },
  { id: 'crop-9-16', group: '裁切', name: '直式 9:16', detail: '短影音滿版比例', fileTag: '9x16' },
  { id: 'crop-square', group: '裁切', name: '方形 1:1', detail: '社群方形構圖', fileTag: '1x1' },
  { id: 'crop-16-9', group: '裁切', name: '橫式 16:9', detail: '標準寬螢幕比例', fileTag: '16x9' },
  { id: 'crop-free', group: '裁切', name: '自由裁切', detail: '手指框出任意範圍', fileTag: 'FreeCrop' },
  { id: 'track', group: '鎖定', name: '主角鎖定置中', detail: 'ViT 追蹤人物或寵物', fileTag: 'SubjectLock' },
  { id: 'remove-background', group: '去背', name: '單一舞者去背', detail: '只留選定舞者，背景純黑', fileTag: 'SoloBlack' },
];

function filterPresetFor(tool: ToolId | null): FilterPreset | null {
  return tool?.startsWith('filter-') ? tool.slice(7) as FilterPreset : null;
}

function cropAspectFor(tool: ToolId | null): AspectPreset | 'source' | null {
  if (tool === 'crop-9-16') return '9:16';
  if (tool === 'crop-square') return '1:1';
  if (tool === 'crop-16-9') return '16:9';
  if (tool === 'crop-free') return 'source';
  return null;
}

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

type BackgroundPreview = {
  startTime: number;
  endTime: number;
  path: TrackPoint[];
};

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

function eventClock() {
  return performance.now();
}

function normalizeBox(start: [number, number], end: [number, number]): Box {
  return [
    Math.min(start[0], end[0]),
    Math.min(start[1], end[1]),
    Math.abs(end[0] - start[0]),
    Math.abs(end[1] - start[1]),
  ];
}

function previewBoxAt(path: TrackPoint[], time: number): Box {
  if (path.length === 0) return [0, 0, 1, 1];
  if (time <= path[0].time) return [...path[0].box] as Box;
  const last = path[path.length - 1];
  if (time >= last.time) return [...last.box] as Box;
  let beforeIndex = 0;
  while (beforeIndex + 1 < path.length && path[beforeIndex + 1].time < time) beforeIndex += 1;
  const before = path[beforeIndex];
  const after = path[Math.min(path.length - 1, beforeIndex + 1)];
  const amount = (time - before.time) / Math.max(0.0001, after.time - before.time);
  return before.box.map((value, index) => value + (after.box[index] - value) * amount) as Box;
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
  const exportResultRef = useRef<HTMLDivElement>(null);
  const exporterRef = useRef<RealtimeVideoExporter | null>(null);
  const backgroundPreviewRef = useRef<BackgroundPreview | null>(null);
  const modnetPreviewTimelineRef = useRef<ModnetPreviewTimeline | null>(null);
  const previewFrameCallbackRef = useRef(0);
  const previewAnimationFrameRef = useRef(0);
  const backgroundPreviewReturnPhaseRef = useRef<'complete' | 'path-ready'>('complete');

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
  const [selectionPlaying, setSelectionPlaying] = useState(false);
  const [backgroundPreviewReady, setBackgroundPreviewReady] = useState(false);
  const [trackPath, setTrackPath] = useState<TrackPoint[]>([]);
  const [cropBox, setCropBox] = useState<Box | null>(null);
  const [selectedTool, setSelectedTool] = useState<ToolId | null>(null);
  const [filterStrength, setFilterStrength] = useState(0.72);
  const [cropCenterX, setCropCenterX] = useState(0.5);
  const [cropCenterY, setCropCenterY] = useState(0.5);
  const [cropZoom, setCropZoom] = useState(1);
  const [aspect, setAspect] = useState<AspectPreset>('9:16');
  const [subjectScale, setSubjectScale] = useState(0.55);
  const [smoothness, setSmoothness] = useState(0.72);
  const [bodyTightness, setBodyTightness] = useState(0.62);
  const [recorderSupport, setRecorderSupport] = useState<RecorderSupport>({
    h264: null,
    hevc: null,
  });
  const [exportUrl, setExportUrl] = useState('');
  const [exportBlob, setExportBlob] = useState<Blob | null>(null);
  const [exportInfo, setExportInfo] = useState<ExportInfo | null>(null);

  useEffect(() => {
    const support = getRecorderSupport();
    const capabilityFrame = requestAnimationFrame(() => {
      setRecorderSupport(support);
      setCapabilities([
        { label: '本機 AI', detail: 'WebAssembly', available: typeof WebAssembly !== 'undefined' },
        { label: '人物去背', detail: 'MODNet 低記憶體預覽', available: typeof WebAssembly !== 'undefined' && typeof HTMLCanvasElement !== 'undefined' },
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
    const activeVideoRef = videoRef;
    return () => {
      void exporterRef.current?.dispose();
      const video = activeVideoRef.current;
      if (video && previewFrameCallbackRef.current && 'cancelVideoFrameCallback' in video) {
        video.cancelVideoFrameCallback(previewFrameCallbackRef.current);
      }
      if (previewAnimationFrameRef.current) cancelAnimationFrame(previewAnimationFrameRef.current);
      void trackerRef.current?.close();
      detectorRef.current?.dispose();
      modnetPreviewTimelineRef.current?.close();
    };
  }, []);

  const readyCount = useMemo(
    () => capabilities.filter((item) => item.available).length,
    [capabilities],
  );

  function openVideoPicker() {
    const input = inputRef.current;
    if (!input) return;
    input.value = '';
    input.click();
  }

  function stopBackgroundPreviewCallbacks() {
    const video = videoRef.current;
    if (video && previewFrameCallbackRef.current && 'cancelVideoFrameCallback' in video) {
      video.cancelVideoFrameCallback(previewFrameCallbackRef.current);
    }
    if (previewAnimationFrameRef.current) cancelAnimationFrame(previewAnimationFrameRef.current);
    previewFrameCallbackRef.current = 0;
    previewAnimationFrameRef.current = 0;
  }

  function resetBackgroundPreview() {
    stopBackgroundPreviewCallbacks();
    backgroundPreviewRef.current = null;
    modnetPreviewTimelineRef.current?.close();
    modnetPreviewTimelineRef.current = null;
    setBackgroundPreviewReady(false);
  }

  async function releaseTracker() {
    const tracker = trackerRef.current;
    trackerRef.current = null;
    await tracker?.close();
  }

  function releaseDetector() {
    const detector = detectorRef.current;
    detectorRef.current = null;
    detector?.dispose();
  }

  async function releaseUpstreamModels() {
    releaseDetector();
    await releaseTracker();
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  }

  function chooseVideo(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    videoRef.current?.pause();
    setSelectionPlaying(false);
    resetBackgroundPreview();
    void releaseUpstreamModels();
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setSourceFile(file);
    setVideoUrl(URL.createObjectURL(file));
    setVideoInfo(null);
    setBox(null);
    setStats(null);
    setCandidates([]);
    selectionRef.current = null;
    setTrackPath([]);
    setCropBox(null);
    setSelectedTool(null);
    setFilterStrength(0.72);
    setCropCenterX(0.5);
    setCropCenterY(0.5);
    setCropZoom(1);
    setAspect('9:16');
    setSubjectScale(0.55);
    setSmoothness(0.72);
    setBodyTightness(0.62);
    setExportUrl('');
    setExportBlob(null);
    setExportInfo(null);
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
      aspectRatio: video.videoWidth / Math.max(1, video.videoHeight),
      width: video.videoWidth,
      height: video.videoHeight,
    });
    setNotice('影片已在本機載入；請從 12 種功能中選擇一項');
  }

  function resetExportResult() {
    setExportUrl('');
    setExportBlob(null);
    setExportInfo(null);
    setProgress(0);
  }

  function chooseTool(tool: ToolId) {
    if (!videoInfo) return;
    setSelectedTool(tool);
    setFilterStrength(0.72);
    setCropCenterX(0.5);
    setCropCenterY(0.5);
    setCropZoom(1);
    setAspect('9:16');
    setSubjectScale(0.55);
    setSmoothness(0.72);
    setCropBox(null);
    resetExportResult();
    if (tool === 'track' || tool === 'remove-background') {
      requestAnimationFrame(() => enterSelection());
      return;
    }
    if (tool === 'crop-free') {
      requestAnimationFrame(() => enterCropSelection());
      return;
    }
    setPhase('tool-ready');
    setNotice('已選擇「' + TOOL_CHOICES.find((item) => item.id === tool)?.name + '」；可先播放預覽，再輸出影片');
  }

  function returnToTools() {
    videoRef.current?.pause();
    setSelectionPlaying(false);
    resetBackgroundPreview();
    setSelectedTool(null);
    setPhase('choose');
    setBox(null);
    setCandidates([]);
    setStats(null);
    setTrackPath([]);
    setCropBox(null);
    selectionRef.current = null;
    resetExportResult();
    setNotice('請從 12 種功能中選擇一項');
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

  function drawCropFrame(targetBox: Box | null) {
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
    if (!targetBox) return;
    const [x, y, width, height] = targetBox;
    const lineWidth = Math.max(4, canvas.width / 320);
    context.fillStyle = 'rgba(0, 0, 0, 0.58)';
    context.fillRect(0, 0, canvas.width, y);
    context.fillRect(0, y, x, height);
    context.fillRect(x + width, y, canvas.width - x - width, height);
    context.fillRect(0, y + height, canvas.width, canvas.height - y - height);
    context.strokeStyle = '#d9f06f';
    context.lineWidth = lineWidth;
    context.strokeRect(x, y, width, height);
    context.fillStyle = '#102018';
    context.font = '700 ' + Math.max(18, canvas.width / 55) + 'px -apple-system';
    const label = '保留範圍';
    const labelWidth = context.measureText(label).width + lineWidth * 5;
    context.fillRect(x, Math.max(0, y - lineWidth * 9), labelWidth, lineWidth * 9);
    context.fillStyle = '#d9f06f';
    context.fillText(label, x + lineWidth * 2, Math.max(lineWidth * 6.5, y - lineWidth * 2));
  }

  function enterCropSelection() {
    const video = videoRef.current;
    if (!video) return;
    video.pause();
    setPhase('crop-select');
    setCropBox(null);
    resetExportResult();
    setNotice('用手指框出成品要保留的畫面範圍，比例完全自由');
    requestAnimationFrame(() => drawCropFrame(null));
  }

  function confirmCropSelection() {
    if (!cropBox) {
      setNotice('請先用手指框出要保留的範圍');
      return;
    }
    const video = videoRef.current;
    if (video?.videoWidth && video.videoHeight) {
      const targetRatio = cropBox[2] / cropBox[3];
      const sourceRatio = video.videoWidth / video.videoHeight;
      const baseWidth = sourceRatio > targetRatio ? video.videoHeight * targetRatio : video.videoWidth;
      setCropCenterX((cropBox[0] + cropBox[2] / 2) / video.videoWidth);
      setCropCenterY((cropBox[1] + cropBox[3] / 2) / video.videoHeight);
      setCropZoom(baseWidth / cropBox[2]);
    }
    setPhase('tool-ready');
    setNotice('自由裁切框已確認；可輸出影片或重新框選');
  }

  function enterSelection() {
    const video = videoRef.current;
    if (!video) return;
    video.pause();
    setSelectionPlaying(false);
    resetBackgroundPreview();
    setPhase('select');
    setBox(null);
    setStats(null);
    setCandidates([]);
    selectionRef.current = null;
    setTrackPath([]);
    setExportUrl('');
    setExportBlob(null);
    setExportInfo(null);
    setNotice(selectedTool === 'remove-background'
      ? '用手指緊貼框住要保留的單一舞者，或使用 AI 找人物'
      : '用手指框住要追蹤的人物或寵物');
    requestAnimationFrame(() => drawFrame(null));
  }

  function toggleSelectionPlayback() {
    const video = videoRef.current;
    if (!video) return;
    if (selectionPlaying) {
      video.pause();
      return;
    }
    setBox(null);
    setCandidates([]);
    selectionRef.current = null;
    setNotice('影片播放中；請暫停在主角清楚的畫面再框選');
    void video.play().catch((error) => {
      setSelectionPlaying(false);
      setNotice('Safari 無法播放影片：' + (error instanceof Error ? error.message : String(error)));
    });
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
    if (phase === 'crop-select') {
      event.currentTarget.setPointerCapture(event.pointerId);
      dragStartRef.current = pointerPosition(event);
      setCropBox(null);
      drawCropFrame(null);
      return;
    }
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
    if (!start) return;
    const next = normalizeBox(start, pointerPosition(event));
    if (phase === 'crop-select') {
      setCropBox(next);
      drawCropFrame(next);
      return;
    }
    if (phase !== 'select') return;
    setBox(next);
    drawFrame(next);
  }

  function finishBox(event: ReactPointerEvent<HTMLCanvasElement>) {
    const start = dragStartRef.current;
    if (!start) return;
    dragStartRef.current = null;
    const next = normalizeBox(start, pointerPosition(event));
    if (phase === 'crop-select') {
      if (next[2] < 24 || next[3] < 24) {
        setCropBox(null);
        drawCropFrame(null);
        setNotice('裁切範圍太小，請重新拖曳較大的框');
        return;
      }
      setCropBox(next);
      drawCropFrame(next);
      setNotice('裁切框已畫好；確認後即可輸出');
      return;
    }
    if (phase !== 'select') return;
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
    setNotice(selectedTool === 'remove-background'
      ? '舞者已指定；可先測試 3 秒主角追蹤'
      : '主角已指定；可開始 3 秒 ViT 追蹤測試');
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
      const peopleOnly = selectedTool === 'remove-background';
      const nextCandidates: Candidate[] = predictions
        .filter((item) => item.class === 'person' || (!peopleOnly && (item.class === 'dog' || item.class === 'cat')))
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
          : peopleOnly ? '沒有找到人物，請直接用手指緊貼框住單一舞者' : '沒有找到人物或寵物，請直接用手指框選',
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

  async function buildModnetPreview(
    path: TrackPoint[],
    startTime: number,
    endTime: number,
  ) {
    const video = videoRef.current;
    if (!video) throw new Error('影片尚未載入');
    setPhase('masking');
    setProgress(0);
    setBackgroundPreviewReady(false);
    modnetPreviewTimelineRef.current?.close();
    modnetPreviewTimelineRef.current = null;
    setNotice('正在釋放追蹤模型並載入低記憶體 MODNet…');
    await releaseUpstreamModels();
    const { prepareModnetPreview } = await import('../lib/modnet-background-preview');
    const prepared = await prepareModnetPreview(video, path, {
      startTime,
      endTime,
      seekTo,
      isCancelled: () => cancelRef.current,
      onProgress: (next, frame, total) => {
        setProgress(next);
        setNotice('產生 MODNet 3 秒 Preview · ' + frame + ' / ' + total + ' 幀');
      },
    });
    modnetPreviewTimelineRef.current = prepared.timeline;
    return prepared.stats;
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
    const previewPoints: TrackPoint[] = [{
      time: startTime,
      box: [...box] as Box,
      score: 1,
      accepted: true,
    }];
    const started = eventClock();

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
        previewPoints.push({
          time: frameTime,
          box: [...result.box] as Box,
          score: result.score,
          accepted: result.accepted,
        });
        frameIndex += 1;
        setBox(result.box);
        setCurrentScore(result.score);
        setProgress((frameTime - startTime) / Math.max(0.001, endTime - startTime));
        setNotice('ViT 追蹤中 · 第 ' + frameIndex + ' 幀');
        drawFrame(result.box, result.score);
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
        if (frameTime >= endTime) break;
      }

      const elapsedMs = eventClock() - started;
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
      if (selectedTool === 'remove-background') {
        backgroundPreviewRef.current = { startTime, endTime, path: previewPoints };
        try {
          const modnetStats = await buildModnetPreview(previewPoints, startTime, endTime);
          await seekTo(startTime);
          setBox([...previewPoints[0].box] as Box);
          drawFrame(previewPoints[0].box);
          setBackgroundPreviewReady(true);
          setPhase('complete');
          setNotice('MODNet Preview 已準備 · 平均 ' + Math.round(modnetStats.averageInferenceMs) + ' ms／幀');
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          setBackgroundPreviewReady(false);
          setPhase('complete');
          setNotice('MODNet Preview 準備失敗：' + message + '；正式輸出尚未切換 v9');
        }
      } else {
        setPhase('complete');
        setNotice('3 秒 ViT 路徑測試完成；尚未進行影片輸出');
      }
    } catch (error) {
      backgroundPreviewRef.current = null;
      setBackgroundPreviewReady(false);
      setPhase('select');
      const message = error instanceof Error ? error.message : String(error);
      setNotice(message);
      drawFrame(box);
    }
  }

  function playBackgroundPreview() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const preview = backgroundPreviewRef.current;
    const timeline = modnetPreviewTimelineRef.current;
    if (!video || !canvas || !preview || !timeline || !backgroundPreviewReady) {
      setNotice('請先完成 3 秒追蹤，讓去背預覽準備完成');
      return;
    }

    stopBackgroundPreviewCallbacks();
    cancelRef.current = false;
    backgroundPreviewReturnPhaseRef.current = phase === 'path-ready' ? 'path-ready' : 'complete';
    video.pause();
    video.currentTime = preview.startTime;
    setPhase('previewing');
    setProgress(0);
    setNotice('正在播放 3 秒 MODNet 純黑背景 Preview…');

    let finished = false;
    let cleanupListeners = () => {};
    const duration = Math.max(0.001, preview.endTime - preview.startTime);

    const finish = (error?: unknown, cancelled = false) => {
      if (finished) return;
      finished = true;
      cleanupListeners();
      video.pause();
      stopBackgroundPreviewCallbacks();
      setPhase(backgroundPreviewReturnPhaseRef.current);
      if (error) {
        const message = error instanceof Error ? error.message : String(error);
        setNotice('3 秒去背預覽失敗：' + message);
      } else if (cancelled) {
        setNotice('已取消 3 秒去背預覽');
      } else {
        setProgress(1);
        setNotice('3 秒去背預覽完成；可重播或繼續追蹤完整影片');
      }
    };

    const renderFrame = (mediaTime: number) => {
      if (finished) return;
      if (cancelRef.current) {
        finish(undefined, true);
        return;
      }
      try {
        timeline.draw(
          video,
          canvas,
          previewBoxAt(preview.path, mediaTime),
          bodyTightness,
        );
        setProgress(Math.max(0, Math.min(1, (mediaTime - preview.startTime) / duration)));
        if (mediaTime >= preview.endTime - 0.01) finish();
      } catch (error) {
        finish(error);
      }
    };

    const onEnded = () => finish();
    const onError = () => finish(new Error('Safari 無法播放預覽片段'));
    const stopAtPreviewEnd = () => {
      if (video.currentTime >= preview.endTime) finish();
    };
    video.addEventListener('ended', onEnded, { once: true });
    video.addEventListener('error', onError, { once: true });
    video.addEventListener('timeupdate', stopAtPreviewEnd);
    cleanupListeners = () => {
      video.removeEventListener('ended', onEnded);
      video.removeEventListener('error', onError);
      video.removeEventListener('timeupdate', stopAtPreviewEnd);
    };

    const requestVideoFrame = (video as unknown as {
      requestVideoFrameCallback?: (callback: VideoFrameRequestCallback) => number;
    }).requestVideoFrameCallback;
    if (requestVideoFrame) {
      const nextFrame = (_now: number, metadata: VideoFrameCallbackMetadata) => {
        renderFrame(metadata.mediaTime);
        if (!finished) previewFrameCallbackRef.current = requestVideoFrame.call(video, nextFrame);
      };
      previewFrameCallbackRef.current = requestVideoFrame.call(video, nextFrame);
    } else {
      const nextFrame = () => {
        renderFrame(video.currentTime);
        if (!finished) previewAnimationFrameRef.current = requestAnimationFrame(nextFrame);
      };
      previewAnimationFrameRef.current = requestAnimationFrame(nextFrame);
    }

    video.play().catch((error) => {
      finish(error);
    });
  }

  async function prepareTrackedPathBackgroundPreview() {
    const video = videoRef.current;
    if (!video || trackPath.length < 2 || !Number.isFinite(video.duration)) {
      setNotice('請先完成整支影片的主角追蹤');
      return;
    }
    const preferredStart = selectionRef.current?.time ?? video.currentTime;
    const startTime = Math.min(Math.max(0, preferredStart), Math.max(0, video.duration - 3));
    const endTime = Math.min(video.duration, startTime + 3);
    cancelRef.current = false;
    backgroundPreviewRef.current = { startTime, endTime, path: trackPath };
    setBackgroundPreviewReady(false);
    try {
      const modnetStats = await buildModnetPreview(trackPath, startTime, endTime);
      await seekTo(startTime);
      const previewBox = previewBoxAt(trackPath, startTime);
      setBox(previewBox);
      drawFrame(previewBox);
      setBackgroundPreviewReady(true);
      setPhase('path-ready');
      setNotice('MODNet Preview 已準備 · 平均 ' + Math.round(modnetStats.averageInferenceMs) + ' ms／幀');
    } catch (error) {
      setPhase('path-ready');
      const message = error instanceof Error ? error.message : String(error);
      setNotice('MODNet Preview 準備失敗：' + message + '；正式輸出尚未切換 v9');
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
    const started = eventClock();
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
      const elapsedMs = eventClock() - started;
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
      await releaseUpstreamModels();
      setNotice(selectedTool === 'remove-background'
        ? '完整舞者路徑已建立；可調整比例、主角大小與柔順度後輸出'
        : '完整 ViT 路徑已建立；可調整構圖並輸出影片');
    } catch (error) {
      setPhase('select');
      const message = error instanceof Error ? error.message : String(error);
      setNotice(message);
      await seekTo(selection.time).catch(() => undefined);
      setBox(selection.box);
      drawFrame(selection.box);
    }
  }

  async function exportVideo(codec: 'h264' | 'hevc') {
    const video = videoRef.current;
    const renderCanvas = renderCanvasRef.current;
    const filterPreset = filterPresetFor(selectedTool);
    const cropAspect = cropAspectFor(selectedTool);
    let operation: ExportOperation | null = null;
    if (filterPreset) {
      operation = { kind: 'filter', preset: filterPreset, strength: filterStrength };
    } else if (cropAspect) {
      if (selectedTool === 'crop-free' && !cropBox) {
        setNotice('請先用手指框出自由裁切範圍');
        return;
      }
      operation = {
        kind: 'crop',
        aspect: cropAspect,
        centerX: cropCenterX,
        centerY: cropCenterY,
        zoom: cropZoom,
        selectionBox: selectedTool === 'crop-free' ? cropBox ?? undefined : undefined,
      };
    } else if (selectedTool === 'track') {
      operation = { kind: 'track', aspect, subjectScale, smoothness };
    } else if (selectedTool === 'remove-background') {
      operation = {
        kind: 'remove-background',
        aspect,
        subjectScale,
        smoothness,
        bodyTightness,
      };
    }
    if (!video || !renderCanvas || !operation) {
      setNotice('請先選擇一項後製功能');
      return;
    }
    if ((operation.kind === 'track' || operation.kind === 'remove-background') && trackPath.length < 2) {
      setNotice('請先完成整支影片的 ViT 追蹤');
      return;
    }
    cancelRef.current = false;
    setPhase('exporting');
    setProgress(0);
    setNotice(operation.kind === 'remove-background'
      ? '正在準備低記憶體 MODNet 輸出…'
      : codec === 'hevc' ? '正在準備 HEVC 母片輸出…' : '正在準備 H.264 相容影片輸出…');

    let exportTimeline: ModnetPreviewTimeline | null = null;
    try {
      if (operation.kind === 'remove-background') {
        modnetPreviewTimelineRef.current?.close();
        modnetPreviewTimelineRef.current = null;
        setBackgroundPreviewReady(false);
        await releaseUpstreamModels();
        const { prepareModnetPreview } = await import('../lib/modnet-background-preview');
        const prepared = await prepareModnetPreview(video, trackPath, {
          startTime: 0,
          endTime: video.duration,
          maxFrames: 241,
          seekTo,
          isCancelled: () => cancelRef.current,
          onProgress: (next, frame, total) => {
            setProgress(next * 0.4);
            setNotice('MODNet 輸出遮罩 · ' + frame + ' / ' + total + ' 幀');
          },
        });
        exportTimeline = prepared.timeline;
        setNotice('MODNet 已釋放；正在保留原聲並編碼影片…');
      }
      if (!exporterRef.current) exporterRef.current = new RealtimeVideoExporter(video);
      const result = await exporterRef.current.export(trackPath, renderCanvas, {
        operation,
        codec,
        backgroundTimeline: exportTimeline ?? undefined,
        onProgress: (next) => {
          const displayed = operation.kind === 'remove-background' ? 0.4 + next * 0.6 : next;
          setProgress(displayed);
          setNotice((operation.kind === 'remove-background' ? 'MODNet 去背與本機編碼中 · ' : '本機編碼中 · ') + Math.round(displayed * 100) + '%');
        },
        isCancelled: () => cancelRef.current,
      });
      const baseName = (sourceFile?.name ?? 'NiviTrack').replace(/\.[^.]+$/, '');
      const fileTag = TOOL_CHOICES.find((item) => item.id === selectedTool)?.fileTag ?? 'Edit';
      const name = baseName + '-NiviTrack-' + fileTag + '.mp4';
      setExportBlob(result.blob);
      setExportUrl(URL.createObjectURL(result.blob));
      setExportInfo({
        name,
        size: formatBytes(result.blob.size),
        mimeType: result.mimeType,
        resolution: result.width + ' × ' + result.height,
      });
      setPhase(operation.kind === 'track' || operation.kind === 'remove-background' ? 'path-ready' : 'tool-ready');
      setNotice('輸出完成；請點「分享／儲存到 iPhone」');
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          exportResultRef.current?.scrollIntoView({
            behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
            block: 'center',
          });
        });
      });
    } catch (error) {
      setPhase(selectedTool === 'track' || selectedTool === 'remove-background' ? 'path-ready' : 'tool-ready');
      const message = error instanceof Error ? error.message : String(error);
      setNotice(message);
    } finally {
      exportTimeline?.close();
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
    setNotice(phase === 'exporting'
      ? '正在取消輸出…'
      : phase === 'previewing'
        ? '正在取消預覽…'
        : phase === 'masking' ? '正在取消 MODNet Preview…' : '正在取消追蹤…');
  }

  const selectedChoice = TOOL_CHOICES.find((item) => item.id === selectedTool) ?? null;
  const selectedFilter = filterPresetFor(selectedTool);
  const selectedCropAspect = cropAspectFor(selectedTool);
  const isSimpleTool = Boolean(selectedFilter || selectedCropAspect);
  const sourceRatio = videoInfo?.aspectRatio ?? 9 / 16;
  const previewRatio = selectedTool === 'crop-free' && cropBox
    ? cropBox[2] / cropBox[3]
    : selectedCropAspect === '9:16'
    ? 9 / 16
    : selectedCropAspect === '1:1'
      ? 1
      : selectedCropAspect === '16:9'
        ? 16 / 9
        : sourceRatio;
  const cropPreviewActive = Boolean(selectedCropAspect && (phase === 'tool-ready' || phase === 'exporting'));
  const stageStyle = cropPreviewActive
    ? { aspectRatio: String(previewRatio), maxWidth: previewRatio < 1 ? `calc(62vh * ${previewRatio})` : '100%' }
    : undefined;
  const videoStyle = {
    filter: selectedFilter ? getFilterCss(selectedFilter, filterStrength) : undefined,
    objectFit: cropPreviewActive ? 'cover' as const : undefined,
    objectPosition: cropPreviewActive ? `${Math.round(cropCenterX * 100)}% ${Math.round(cropCenterY * 100)}%` : undefined,
    transform: cropPreviewActive ? `scale(${cropZoom})` : undefined,
  };
  const step = !videoInfo ? 1 : selectedTool === null ? 2 : 3;

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="NiviTrack 首頁">
          <span className="brand-mark">N</span><span>NiviTrack</span>
        </a>
        <span className="local-pill"><i aria-hidden="true" />iPhone 本機處理</span>
      </header>

      <section className="hero">
        <div className="eyebrow">IPHONE WEB APP · 12 種單一後製</div>
        <h1>選一個功能，<span>直接完成影片。</span></h1>
        <p>匯入 MOV、HEVC 或 MP4，選擇濾鏡、裁切、主角鎖定或單一舞者去背。一次處理一項；要疊加時，把輸出影片再匯入即可。</p>
        <div className="steps" aria-label="處理步驟">
          <div className={'step ' + (step >= 1 ? 'active' : '')}><b>01</b><span>選擇影片</span></div>
          <div className={'step ' + (step >= 2 ? 'active' : '')}><b>02</b><span>選擇功能</span></div>
          <div className={'step ' + (step >= 3 ? 'active' : '')}><b>03</b><span>調整與輸出</span></div>
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
              <div className={'video-stage ' + (cropPreviewActive ? 'crop-preview' : '')} style={stageStyle}>
                <video
                  ref={videoRef}
                  className={phase === 'choose' || phase === 'tool-ready' || phase === 'exporting' || phase === 'select' ? '' : 'is-hidden'}
                  style={videoStyle}
                  src={videoUrl}
                  controls
                  playsInline
                  preload="metadata"
                  onLoadedMetadata={readMetadata}
                  onPlay={() => {
                    if (phase !== 'select') return;
                    setSelectionPlaying(true);
                    setBox(null);
                    setCandidates([]);
                    selectionRef.current = null;
                    setNotice('影片播放中；請暫停在主角清楚的畫面再框選');
                  }}
                  onPause={() => {
                    if (phase !== 'select') return;
                    setSelectionPlaying(false);
                    setNotice(selectedTool === 'remove-background'
                      ? '已暫停；請框住要保留的單一舞者，或使用 AI 找人物'
                      : '已暫停；請用手指框住要追蹤的人物或寵物');
                    requestAnimationFrame(() => drawFrame(null));
                  }}
                  onSeeked={() => {
                    if (phase === 'select' && videoRef.current?.paused) requestAnimationFrame(() => drawFrame(null));
                  }}
                  onError={() => setNotice('Safari 無法解碼這支影片，請保留檔案供實機記錄')}
                />
                <canvas
                  ref={canvasRef}
                  className={phase === 'choose' || phase === 'tool-ready' || phase === 'exporting' || (phase === 'select' && selectionPlaying) ? 'tracking-canvas is-hidden' : 'tracking-canvas'}
                  onPointerDown={startBox}
                  onPointerMove={moveBox}
                  onPointerUp={finishBox}
                  onPointerCancel={finishBox}
                />
                <span className="source-badge">
                  {phase === 'choose' ? '尚未選擇功能' : phase === 'tool-ready' ? selectedChoice?.name : phase === 'crop-select' ? '手指框選保留範圍' : phase === 'select' ? (selectionPlaying ? '播放中 · 暫停後框選' : selectedTool === 'remove-background' ? '框選單一舞者' : '手指框選主角') : phase === 'masking' ? 'MODNet 逐幀產生遮罩' : phase === 'previewing' ? '3 秒 MODNet 純黑背景預覽' : phase === 'exporting' ? (selectedTool === 'remove-background' ? 'MODNet 本機去背輸出' : 'Safari 本機編碼') : phase === 'path-ready' ? (selectedTool === 'remove-background' ? '單一舞者去背與輸出' : '構圖與輸出') : 'ViT 本機推論'}
                </span>
                {(phase === 'tracking' || phase === 'masking' || phase === 'previewing' || phase === 'exporting') && (
                  <div className="progress-overlay">
                    <strong>{Math.round(progress * 100)}%</strong>
                    <span>{phase === 'exporting' ? '保留原聲' : phase === 'previewing' ? 'MODNet 預覽' : phase === 'masking' ? '單 Session · 逐幀釋放' : 'score ' + (currentScore === null ? '—' : currentScore.toFixed(3))}</span>
                  </div>
                )}
              </div>
              <div className="video-actions">
                {phase !== 'tracking' && phase !== 'masking' && phase !== 'previewing' && phase !== 'exporting' && (
                  <button type="button" onClick={openVideoPicker}>重新選擇影片</button>
                )}
                {phase === 'tool-ready' && <button type="button" onClick={returnToTools}>取消此功能</button>}
                {phase === 'tool-ready' && selectedTool === 'crop-free' && <button type="button" onClick={enterCropSelection}>重新框選裁切</button>}
                {phase === 'crop-select' && (
                  <>
                    <button type="button" onClick={returnToTools}>返回功能選單</button>
                    <button className="primary" type="button" disabled={!cropBox} onClick={confirmCropSelection}>使用此裁切框</button>
                  </>
                )}
                {phase === 'select' && (
                  <>
                    <button type="button" onClick={returnToTools}>返回功能選單</button>
                    <button type="button" onClick={toggleSelectionPlayback}>
                      {selectionPlaying ? '暫停並框選' : '播放找畫面'}
                    </button>
                    <button type="button" disabled={detecting || selectionPlaying} onClick={detectSubjects}>
                      {detecting ? 'AI 掃描中…' : selectedTool === 'remove-background' ? 'AI 尋找人物' : 'AI 尋找人物／寵物'}
                    </button>
                    <button type="button" disabled={!box || selectionPlaying} onClick={runTracking}>
                      {selectedTool === 'remove-background' ? '準備 3 秒去背預覽' : '測試 3 秒 ViT'}
                    </button>
                    <button className="primary" type="button" disabled={!box || selectionPlaying} onClick={runFullTracking}>
                      追蹤完整影片
                    </button>
                  </>
                )}
                {(phase === 'tracking' || phase === 'masking' || phase === 'previewing' || phase === 'exporting') && (
                  <button className="danger" type="button" onClick={cancelTracking}>{phase === 'exporting' ? '取消輸出' : phase === 'previewing' ? '取消預覽' : phase === 'masking' ? '取消 MODNet' : '取消追蹤'}</button>
                )}
                {phase === 'complete' && (
                  <>
                    <button type="button" onClick={returnToTools}>返回功能選單</button>
                    <button type="button" onClick={enterSelection}>重新框選</button>
                    {selectedTool === 'remove-background' && (
                      <button className="primary" type="button" disabled={!backgroundPreviewReady} onClick={playBackgroundPreview}>播放 3 秒 MODNet 預覽</button>
                    )}
                    <button className={selectedTool === 'remove-background' ? '' : 'primary'} type="button" onClick={runFullTracking}>追蹤完整影片</button>
                  </>
                )}
                {phase === 'path-ready' && (
                  <>
                    <button type="button" onClick={returnToTools}>返回功能選單</button>
                    <button type="button" onClick={enterSelection}>重新選角與追蹤</button>
                  </>
                )}
              </div>
              {phase === 'complete' && selectedTool === 'remove-background' && (
                <label className="range-control">
                  <span><b>去背收緊度</b><em>{Math.round(bodyTightness * 100)}%</em></span>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={Math.round(bodyTightness * 100)}
                    onChange={(event) => setBodyTightness(Number(event.target.value) / 100)}
                  />
                </label>
              )}
              {(phase === 'choose' || phase === 'tool-ready') && videoInfo && (
                <section className="tool-panel" aria-label="選擇一項影片後製功能">
                  <div className="tool-heading">
                    <div>
                      <span>一次選一項</span>
                      <strong>12 種影片後製</strong>
                    </div>
                    <b>{selectedChoice ? selectedChoice.name : '尚未選擇'}</b>
                  </div>
                  <div className="tool-grid">
                    {TOOL_CHOICES.map((tool, index) => (
                      <button
                        className={selectedTool === tool.id ? 'selected' : ''}
                        type="button"
                        key={tool.id}
                        onClick={() => chooseTool(tool.id)}
                      >
                        <span>{String(index + 1).padStart(2, '0')} · {tool.group}</span>
                        <strong>{tool.name}</strong>
                        <small>{tool.detail}</small>
                      </button>
                    ))}
                  </div>
                </section>
              )}
              {(phase === 'tool-ready' || (phase === 'exporting' && isSimpleTool)) && isSimpleTool && selectedChoice && (
                <section className="export-panel">
                  <div className="export-heading">
                    <div>
                      <span>{selectedChoice.group}已就緒</span>
                      <strong>{selectedChoice.name}</strong>
                    </div>
                    <b>保留原聲</b>
                  </div>

                  {selectedFilter && (
                    <label className="range-control">
                      <span><b>效果強度</b><em>{Math.round(filterStrength * 100)}%</em></span>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={Math.round(filterStrength * 100)}
                        disabled={phase === 'exporting'}
                        onChange={(event) => setFilterStrength(Number(event.target.value) / 100)}
                      />
                    </label>
                  )}

                  {selectedCropAspect && (
                    <>
                      <div className="crop-summary">
                        <span>輸出比例</span>
                        <strong>{selectedTool === 'crop-free' && cropBox ? `自由比例 ${(cropBox[2] / cropBox[3]).toFixed(2)}:1` : selectedCropAspect}</strong>
                      </div>
                      {selectedTool !== 'crop-free' && (
                        <>
                          <label className="range-control">
                            <span><b>左右位置</b><em>{Math.round(cropCenterX * 100)}%</em></span>
                            <input type="range" min="0" max="100" value={Math.round(cropCenterX * 100)} disabled={phase === 'exporting'} onChange={(event) => setCropCenterX(Number(event.target.value) / 100)} />
                          </label>
                          <label className="range-control">
                            <span><b>上下位置</b><em>{Math.round(cropCenterY * 100)}%</em></span>
                            <input type="range" min="0" max="100" value={Math.round(cropCenterY * 100)} disabled={phase === 'exporting'} onChange={(event) => setCropCenterY(Number(event.target.value) / 100)} />
                          </label>
                          <label className="range-control">
                            <span><b>畫面縮放</b><em>{Math.round(cropZoom * 100)}%</em></span>
                            <input type="range" min="100" max="250" value={Math.round(cropZoom * 100)} disabled={phase === 'exporting'} onChange={(event) => setCropZoom(Number(event.target.value) / 100)} />
                          </label>
                        </>
                      )}
                    </>
                  )}

                  <div className="export-buttons">
                    <button className="primary" type="button" disabled={phase === 'exporting' || !recorderSupport.h264} onClick={() => void exportVideo('h264')}>
                      輸出相容 MP4
                    </button>
                    <button type="button" disabled={phase === 'exporting' || !recorderSupport.hevc} onClick={() => void exportVideo('hevc')}>
                      輸出 HEVC 母片（MP4）
                    </button>
                  </div>

                  {exportUrl && exportInfo && (
                    <div className="export-result result-ready" ref={exportResultRef}>
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
              {(phase === 'path-ready' || phase === 'exporting') && trackPath.length > 1 && (
                <section className="export-panel">
                  <div className="export-heading">
                    <div>
                      <span>{selectedTool === 'remove-background' ? '單一舞者路徑已就緒' : '完整路徑已就緒'}</span>
                      <strong>{selectedTool === 'remove-background' ? '選擇去背構圖' : '選擇輸出構圖'}</strong>
                    </div>
                    <b>{selectedTool === 'remove-background' ? aspect : trackPath.length + ' 點'}</b>
                  </div>

                  {selectedTool === 'remove-background' && (
                    <>
                      <div className="crop-summary">
                        <span>輸出方式</span>
                        <strong>放大並置中選定舞者 · 其餘畫面為純黑</strong>
                      </div>
                      <div className="background-preview-actions">
                        <button
                          className="primary"
                          type="button"
                          disabled={phase === 'exporting'}
                          onClick={backgroundPreviewReady
                            ? playBackgroundPreview
                            : () => void prepareTrackedPathBackgroundPreview()}
                        >
                          {backgroundPreviewReady ? '播放 3 秒 MODNet Preview' : '準備 3 秒 MODNet Preview'}
                        </button>
                      </div>
                    </>
                  )}

                  <div className="aspect-options" aria-label="輸出比例">
                    {(['9:16', '1:1', '16:9'] as AspectPreset[]).map((preset) => (
                      <button
                        className={aspect === preset ? 'selected' : ''}
                        type="button"
                        key={preset}
                        disabled={phase === 'exporting'}
                        onClick={() => setAspect(preset)}
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
                      onChange={(event) => setSubjectScale(Number(event.target.value) / 100)}
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
                      onChange={(event) => setSmoothness(Number(event.target.value) / 100)}
                    />
                  </label>

                  {selectedTool === 'remove-background' && (
                    <label className="range-control">
                      <span><b>去背收緊度</b><em>{Math.round(bodyTightness * 100)}%</em></span>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={Math.round(bodyTightness * 100)}
                        disabled={phase === 'exporting'}
                        onChange={(event) => setBodyTightness(Number(event.target.value) / 100)}
                      />
                    </label>
                  )}

                  <div className="export-buttons">
                    <button
                      className="primary"
                      type="button"
                      disabled={phase === 'exporting' || !recorderSupport.h264}
                      onClick={() => void exportVideo('h264')}
                    >
                      輸出相容 MP4
                    </button>
                    <button
                      type="button"
                      disabled={phase === 'exporting' || !recorderSupport.hevc}
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

                  {selectedTool === 'remove-background' && (
                    <p className="export-note">
                      MODNet 已接到正式輸出：先以受控張數產生遮罩並釋放 ONNX，再保留原聲編碼；目前舊 3 秒 Preview 僅供參考，正式輸出驗收後會改成同一路徑的 3 秒試輸出。
                    </p>
                  )}

                  {exportUrl && exportInfo && (
                    <div className="export-result result-ready" ref={exportResultRef}>
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
