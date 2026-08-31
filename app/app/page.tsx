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
  OutputQuality,
  RealtimeVideoExporter,
  RecorderSupport,
  TrackPoint,
  trackedFrameCrop,
} from '../lib/video-export';
import type {
  BackgroundFillMode,
  CloneLayout,
  CloneStyle,
  PersonBackgroundEffects,
  PersonBackgroundRenderer,
} from '../lib/person-background-removal';
import {
  createMainHeadTrackingContext,
  headBoxFromTrackingContext,
  headBoxesAt,
  plausibleDetectedHeadBoxes,
  stabilizeHeadDetectionFrames,
  type FaceHeadDetector,
  type FaceMaskEffects,
  type FaceMaskStyle,
  type FaceObscuringRenderer,
  type HeadDetectionFrame,
} from '../lib/face-obscuring';
import {
  selectionCanvasPointToSource,
  selectionViewport,
  sourceBoxToSelectionCanvas,
  type SelectionFocus,
} from '../lib/selection-zoom';

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

type Phase = 'choose' | 'tool-ready' | 'crop-select' | 'select' | 'tracking' | 'previewing' | 'complete' | 'path-ready' | 'exporting';

type ToolId =
  | `filter-${FilterPreset}`
  | 'crop-9-16'
  | 'crop-square'
  | 'crop-16-9'
  | 'crop-free'
  | 'track'
  | 'remove-background'
  | 'mask-faces';

type ToolChoice = {
  id: ToolId;
  group: '濾鏡' | '裁切' | '鎖定' | '去背' | '隱私';
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
  { id: 'remove-background', group: '去背', name: '單一舞者去背', detail: '去背後加入背景、外框與分身', fileTag: 'SoloFX' },
  { id: 'mask-faces', group: '隱私', name: '旁人人臉遮罩', detail: '保留主角，遮住其他人臉', fileTag: 'FaceMask' },
];

const BACKGROUND_COLORS = [
  { name: '黑', value: '#000000' },
  { name: '白', value: '#ffffff' },
  { name: '萊姆', value: '#d9f06f' },
  { name: '藍', value: '#2563eb' },
  { name: '粉', value: '#ec4899' },
  { name: '紫', value: '#7c3aed' },
];

const BACKGROUND_PREVIEW_SIZES: Record<AspectPreset, [number, number]> = {
  '9:16': [720, 1280],
  '1:1': [720, 720],
  '16:9': [1280, 720],
};

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

type FacePreview = BackgroundPreview & {
  headFrames: HeadDetectionFrame[];
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

function sourcePreviewSize(width: number, height: number): [number, number] {
  const maxWidth = width > height ? 1280 : 720;
  const maxHeight = height > width ? 1280 : 720;
  const scale = Math.min(1, maxWidth / Math.max(2, width), maxHeight / Math.max(2, height));
  const even = (value: number) => {
    const rounded = Math.max(2, Math.round(value));
    return rounded % 2 === 0 ? rounded : rounded - 1;
  };
  return [even(width * scale), even(height * scale)];
}

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const stickerInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragStartRef = useRef<[number, number] | null>(null);
  const cancelRef = useRef(false);
  const trackerRef = useRef<VitTracker | null>(null);
  const detectorRef = useRef<ObjectDetection | null>(null);
  const faceHeadDetectorRef = useRef<FaceHeadDetector | null>(null);
  const selectionRef = useRef<{ time: number; box: Box } | null>(null);
  const selectionZoomRef = useRef(1);
  const selectionFocusRef = useRef<SelectionFocus>(null);
  const renderCanvasRef = useRef<HTMLCanvasElement>(null);
  const exportResultRef = useRef<HTMLDivElement>(null);
  const exporterRef = useRef<RealtimeVideoExporter | null>(null);
  const backgroundPreviewRef = useRef<BackgroundPreview | null>(null);
  const backgroundPreviewRendererRef = useRef<PersonBackgroundRenderer | null>(null);
  const facePreviewRef = useRef<FacePreview | null>(null);
  const facePreviewRendererRef = useRef<FaceObscuringRenderer | null>(null);
  const faceHeadFramesRef = useRef<HeadDetectionFrame[]>([]);
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
  const [selectionZoom, setSelectionZoom] = useState(1);
  const [backgroundPreviewReady, setBackgroundPreviewReady] = useState(false);
  const [facePreviewReady, setFacePreviewReady] = useState(false);
  const [trackPath, setTrackPath] = useState<TrackPoint[]>([]);
  const [cropBox, setCropBox] = useState<Box | null>(null);
  const [selectedTool, setSelectedTool] = useState<ToolId | null>(null);
  const [filterStrength, setFilterStrength] = useState(0.72);
  const [cropCenterX, setCropCenterX] = useState(0.5);
  const [cropCenterY, setCropCenterY] = useState(0.5);
  const [cropZoom, setCropZoom] = useState(1);
  const [aspect, setAspect] = useState<AspectPreset>('9:16');
  const [outputQuality, setOutputQuality] = useState<OutputQuality>('clear');
  const [subjectScale, setSubjectScale] = useState(0.55);
  const [smoothness, setSmoothness] = useState(0.72);
  const [backgroundMode, setBackgroundMode] = useState<BackgroundFillMode>('color');
  const [backgroundColor, setBackgroundColor] = useState('#000000');
  const [backgroundBlur, setBackgroundBlur] = useState(52);
  const [outlineColor, setOutlineColor] = useState('#d9f06f');
  const [outlineWidth, setOutlineWidth] = useState(0);
  const [cloneCount, setCloneCount] = useState(0);
  const [cloneLayout, setCloneLayout] = useState<CloneLayout>('trail');
  const [cloneStyle, setCloneStyle] = useState<CloneStyle>('subject');
  const [faceMaskStyle, setFaceMaskStyle] = useState<FaceMaskStyle>('strong-blur');
  const [faceMaskStrength, setFaceMaskStrength] = useState(0.72);
  const [faceMaskScale, setFaceMaskScale] = useState(1.38);
  const [faceMaskEmoji, setFaceMaskEmoji] = useState('😎');
  const [faceMaskPrivacyFirst, setFaceMaskPrivacyFirst] = useState(true);
  const [faceStickerUrl, setFaceStickerUrl] = useState('');
  const [faceStickerName, setFaceStickerName] = useState('');
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
        { label: '指定主角去背', detail: 'MagicTouch 本機物件分割', available: typeof WebAssembly !== 'undefined' && typeof HTMLCanvasElement !== 'undefined' },
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
      if (faceStickerUrl) URL.revokeObjectURL(faceStickerUrl);
    };
  }, [faceStickerUrl]);

  useEffect(() => {
    const activeVideoRef = videoRef;
    return () => {
      void exporterRef.current?.dispose();
      const video = activeVideoRef.current;
      if (video && previewFrameCallbackRef.current && 'cancelVideoFrameCallback' in video) {
        video.cancelVideoFrameCallback(previewFrameCallbackRef.current);
      }
      if (previewAnimationFrameRef.current) cancelAnimationFrame(previewAnimationFrameRef.current);
      backgroundPreviewRendererRef.current?.close();
      facePreviewRendererRef.current?.close();
      faceHeadDetectorRef.current?.close();
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
    backgroundPreviewRendererRef.current?.close();
    backgroundPreviewRendererRef.current = null;
    setBackgroundPreviewReady(false);
  }

  function resetFacePreview() {
    stopBackgroundPreviewCallbacks();
    facePreviewRef.current = null;
    facePreviewRendererRef.current?.close();
    facePreviewRendererRef.current = null;
    faceHeadFramesRef.current = [];
    setFacePreviewReady(false);
  }

  function releaseFaceHeadDetector() {
    faceHeadDetectorRef.current?.close();
    faceHeadDetectorRef.current = null;
  }

  function getBackgroundEffects(): PersonBackgroundEffects {
    return {
      backgroundMode,
      backgroundColor,
      backgroundBlur,
      outlineColor,
      outlineWidth,
      cloneCount,
      cloneLayout,
      cloneStyle,
    };
  }

  function getFaceMaskEffects(): FaceMaskEffects {
    return {
      style: faceMaskStyle,
      strength: faceMaskStrength,
      scale: faceMaskScale,
      emoji: faceMaskEmoji,
      stickerUrl: faceStickerUrl || undefined,
      privacyFirst: faceMaskPrivacyFirst,
    };
  }

  function chooseVideo(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    videoRef.current?.pause();
    setSelectionPlaying(false);
    selectionZoomRef.current = 1;
    selectionFocusRef.current = null;
    setSelectionZoom(1);
    resetBackgroundPreview();
    resetFacePreview();
    releaseFaceHeadDetector();
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
    setOutputQuality('clear');
    setSubjectScale(0.55);
    setSmoothness(0.72);
    setBackgroundMode('color');
    setBackgroundColor('#000000');
    setBackgroundBlur(52);
    setOutlineColor('#d9f06f');
    setOutlineWidth(0);
    setCloneCount(0);
    setCloneLayout('trail');
    setCloneStyle('subject');
    setFaceMaskStyle('strong-blur');
    setFaceMaskStrength(0.72);
    setFaceMaskScale(1.38);
    setFaceMaskEmoji('😎');
    setFaceMaskPrivacyFirst(true);
    setFaceStickerUrl('');
    setFaceStickerName('');
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
    setNotice('影片已在本機載入；請從 13 種功能中選擇一項');
  }

  function resetExportResult() {
    setExportUrl('');
    setExportBlob(null);
    setExportInfo(null);
    setProgress(0);
  }

  function selectOutputQuality(quality: OutputQuality) {
    setOutputQuality(quality);
    resetExportResult();
    setNotice(quality === 'clear'
      ? '已選清晰畫質；最高輸出 1080p，檔案較大、編碼較久'
      : '已選標準畫質；最高輸出 720p，速度較快');
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
    setBackgroundMode('color');
    setBackgroundColor('#000000');
    setBackgroundBlur(52);
    setOutlineColor('#d9f06f');
    setOutlineWidth(0);
    setCloneCount(0);
    setCloneLayout('trail');
    setCloneStyle('subject');
    setFaceMaskStyle('strong-blur');
    setFaceMaskStrength(0.72);
    setFaceMaskScale(1.38);
    setFaceMaskEmoji('😎');
    setFaceMaskPrivacyFirst(true);
    setFaceStickerUrl('');
    setFaceStickerName('');
    setCropBox(null);
    resetExportResult();
    if (tool === 'mask-faces') {
      detectorRef.current?.dispose();
      detectorRef.current = null;
    } else {
      releaseFaceHeadDetector();
    }
    if (tool === 'track' || tool === 'remove-background' || tool === 'mask-faces') {
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
    resetFacePreview();
    releaseFaceHeadDetector();
    setSelectedTool(null);
    setPhase('choose');
    setBox(null);
    setCandidates([]);
    setStats(null);
    setTrackPath([]);
    setCropBox(null);
    selectionRef.current = null;
    resetExportResult();
    setNotice('請從 13 種功能中選擇一項');
  }

  function drawFrame(
    targetBox: Box | null,
    score?: number,
    visibleCandidates: Candidate[] = candidates,
    useSelectionZoom = false,
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
    const viewport = useSelectionZoom
      ? selectionViewport(
          video.videoWidth,
          video.videoHeight,
          selectionZoomRef.current,
          selectionFocusRef.current,
        )
      : [0, 0, video.videoWidth, video.videoHeight] as Box;
    context.drawImage(
      video,
      viewport[0],
      viewport[1],
      viewport[2],
      viewport[3],
      0,
      0,
      canvas.width,
      canvas.height,
    );
    const candidateLine = Math.max(3, canvas.width / 500);
    context.lineWidth = candidateLine;
    context.font = '700 ' + Math.max(16, canvas.width / 65) + 'px -apple-system';
    for (const candidate of visibleCandidates) {
      const [candidateX, candidateY, candidateWidth, candidateHeight] = useSelectionZoom
        ? sourceBoxToSelectionCanvas(candidate.box, viewport, canvas.width, canvas.height)
        : candidate.box;
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

    const [x, y, width, height] = useSelectionZoom
      ? sourceBoxToSelectionCanvas(targetBox, viewport, canvas.width, canvas.height)
      : targetBox;
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

  function drawSelectionFrame(
    targetBox: Box | null,
    score?: number,
    visibleCandidates: Candidate[] = candidates,
  ) {
    drawFrame(targetBox, score, visibleCandidates, true);
  }

  function changeSelectionZoom(direction: -1 | 1) {
    const video = videoRef.current;
    if (!video) return;
    const levels = [1, 1.5, 2, 3];
    const currentIndex = Math.max(0, levels.indexOf(selectionZoomRef.current));
    const nextIndex = Math.max(0, Math.min(levels.length - 1, currentIndex + direction));
    const nextZoom = levels[nextIndex];
    if (nextZoom === selectionZoomRef.current) return;
    const focusBox = selectionRef.current?.box ?? box;
    if (nextZoom > 1 && focusBox) {
      selectionFocusRef.current = [
        focusBox[0] + focusBox[2] / 2,
        focusBox[1] + focusBox[3] / 2,
      ];
    } else if (nextZoom > 1 && !selectionFocusRef.current) {
      selectionFocusRef.current = [video.videoWidth / 2, video.videoHeight / 2];
    }
    if (nextZoom === 1) selectionFocusRef.current = null;
    selectionZoomRef.current = nextZoom;
    setSelectionZoom(nextZoom);
    drawSelectionFrame(box);
    setNotice(nextZoom === 1
      ? '已回到完整畫面；可重新框選主角'
      : '取框畫面已放大 ' + nextZoom + '×；可重新畫框微調主角範圍');
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
    selectionZoomRef.current = 1;
    selectionFocusRef.current = null;
    setSelectionZoom(1);
    resetBackgroundPreview();
    resetFacePreview();
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
      : selectedTool === 'mask-faces'
        ? '用手指框住主角的頭／臉，或使用 AI 尋找人臉'
        : '用手指框住要追蹤的人物或寵物');
    requestAnimationFrame(() => drawSelectionFrame(null));
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
    const canvasPoint: [number, number] = [
      Math.max(0, Math.min(canvas.width, ((event.clientX - bounds.left) / bounds.width) * canvas.width)),
      Math.max(0, Math.min(canvas.height, ((event.clientY - bounds.top) / bounds.height) * canvas.height)),
    ];
    if (phase !== 'select' || selectionZoomRef.current === 1) return canvasPoint;
    const video = videoRef.current;
    if (!video) return canvasPoint;
    const viewport = selectionViewport(
      video.videoWidth,
      video.videoHeight,
      selectionZoomRef.current,
      selectionFocusRef.current,
    );
    return selectionCanvasPointToSource(canvasPoint, viewport, canvas.width, canvas.height);
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
      drawSelectionFrame(hit.box);
      setNotice('已選擇 AI 辨識框；也可直接拖曳重新框選');
      return;
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    dragStartRef.current = point;
    setBox(null);
    drawSelectionFrame(null);
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
    drawSelectionFrame(next);
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
      drawSelectionFrame(null);
      setNotice(selectedTool === 'mask-faces'
        ? '框選範圍太小，請重新框住主角的頭／臉'
        : '框選範圍太小，請重新框住完整主角');
      return;
    }
    setBox(next);
    selectionRef.current = {
      time: videoRef.current?.currentTime ?? 0,
      box: [...next] as Box,
    };
    drawSelectionFrame(next);
    setNotice(selectedTool === 'remove-background'
      ? '舞者已指定；可先測試 3 秒主角追蹤'
      : selectedTool === 'mask-faces'
        ? '主角頭／臉已指定；這張臉不會被遮住'
        : '主角已指定；可開始 3 秒 ViT 追蹤測試');
  }

  async function ensurePersonDetector() {
    if (!detectorRef.current) {
      await import('@tensorflow/tfjs');
      const cocoSsd = await import('@tensorflow-models/coco-ssd');
      detectorRef.current = await cocoSsd.load({
        base: 'lite_mobilenet_v2',
        modelUrl: new URL('models/ssdlite_mobilenet_v2/model.json', document.baseURI).href,
      });
    }
    return detectorRef.current;
  }

  async function ensureFaceHeadDetector() {
    if (!faceHeadDetectorRef.current) {
      const { FaceHeadDetector } = await import('../lib/face-obscuring');
      faceHeadDetectorRef.current = await FaceHeadDetector.create();
    }
    return faceHeadDetectorRef.current;
  }

  async function detectHeadFrame(
    video: HTMLVideoElement,
    subjectBox: Box,
    time: number,
  ): Promise<HeadDetectionFrame> {
    const detector = await ensureFaceHeadDetector();
    const heads = (await detector.detect(video)).map((detection) => detection.box);
    return {
      time,
      heads: plausibleDetectedHeadBoxes(heads, subjectBox, video.videoWidth, video.videoHeight),
    };
  }

  async function detectSubjects() {
    const video = videoRef.current;
    if (!video) return;
    setDetecting(true);
    setNotice(selectedTool === 'mask-faces'
      ? '正在本機載入 360° 人頭模型並掃描目前影格…'
      : '正在本機載入 SSDLite 並掃描目前影格…');
    try {
      if (selectedTool === 'mask-faces') {
        const detector = await ensureFaceHeadDetector();
        const detections = await detector.detect(video);
        const nextCandidates: Candidate[] = detections.map((detection) => ({
          box: detection.box,
          label: '人頭',
          score: detection.score,
        }));
        setCandidates(nextCandidates);
        drawSelectionFrame(box, undefined, nextCandidates);
        setNotice(nextCandidates.length
          ? '找到 ' + nextCandidates.length + ' 顆人頭；請點選不需要遮住的主角頭，或手動框住主角頭部'
          : '沒有找到明確人頭；請用手指框住不需要遮住的主角頭部');
        return;
      }
      const detector = await ensurePersonDetector();
      const predictions = await detector.detect(video, 30, 0.25);
      const peopleOnly = selectedTool === 'remove-background';
      const nextCandidates: Candidate[] = predictions
        .filter((item) => item.class === 'person' || (!peopleOnly && (item.class === 'dog' || item.class === 'cat')))
        .map((item) => ({
          box: item.bbox as Box,
          label: item.class === 'person' ? '人物' : item.class === 'dog' ? '狗' : '貓',
          score: item.score,
        }));
      setCandidates(nextCandidates);
      drawSelectionFrame(box, undefined, nextCandidates);
      setNotice(
        nextCandidates.length
          ? '找到 ' + nextCandidates.length + ' 個候選框；點選其中一個或手動畫框'
          : peopleOnly
            ? '沒有找到人物，請直接用手指緊貼框住單一舞者'
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
    const previewPoints: TrackPoint[] = [{
      time: startTime,
      box: [...box] as Box,
      score: 1,
      accepted: true,
    }];
    const headFrames: HeadDetectionFrame[] = [];
    const started = eventClock();
    const mainHeadTrackingContext = selectedTool === 'mask-faces'
      ? createMainHeadTrackingContext(box, video.videoWidth, video.videoHeight)
      : null;

    try {
      setNotice('正在載入本機 ViT 模型…');
      if (!trackerRef.current) {
        const modelUrl = new URL('models/vittrack.onnx', document.baseURI).href;
        trackerRef.current = await VitTracker.create(modelUrl);
      }
      trackerRef.current.initialize(video, mainHeadTrackingContext?.trackerBox ?? box);
      if (selectedTool === 'mask-faces') {
        setNotice('正在載入本機 360° 人頭辨識…');
        await ensureFaceHeadDetector();
        headFrames.push(await detectHeadFrame(video, box, startTime));
      }

      let frameIndex = 0;
      for (let at = startTime + interval; at <= endTime + 0.001; at += interval) {
        if (cancelRef.current) throw new Error('使用者已取消追蹤');
        const frameTime = Math.min(at, endTime);
        await seekTo(frameTime);
        const result = await trackerRef.current.update(video);
        const resultBox = mainHeadTrackingContext
          ? headBoxFromTrackingContext(
            result.box,
            mainHeadTrackingContext,
            video.videoWidth,
            video.videoHeight,
          )
          : result.box;
        results.push(result);
        previewPoints.push({
          time: frameTime,
          box: [...resultBox] as Box,
          score: result.score,
          accepted: result.accepted,
        });
        frameIndex += 1;
        if (selectedTool === 'mask-faces' && frameIndex % 2 === 0) {
          headFrames.push(await detectHeadFrame(video, resultBox, frameTime));
        }
        setBox(resultBox);
        setCurrentScore(result.score);
        setProgress((frameTime - startTime) / Math.max(0.001, endTime - startTime));
        setNotice((selectedTool === 'mask-faces' ? 'ViT＋人頭追蹤中 · 第 ' : 'ViT 追蹤中 · 第 ') + frameIndex + ' 幀');
        drawFrame(resultBox, result.score);
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
        setNotice('3 秒追蹤完成；正在準備本機去背預覽…');
        try {
          if (!backgroundPreviewRendererRef.current) {
            const { PersonBackgroundRenderer } = await import('../lib/person-background-removal');
            backgroundPreviewRendererRef.current = await PersonBackgroundRenderer.create();
          }
          await seekTo(startTime);
          setBox([...previewPoints[0].box] as Box);
          drawFrame(previewPoints[0].box);
          setBackgroundPreviewReady(true);
          setPhase('complete');
          setNotice('3 秒去背預覽已準備；請點「播放 3 秒去背預覽」');
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          setBackgroundPreviewReady(false);
          setPhase('complete');
          setNotice('3 秒追蹤完成，但去背預覽模型準備失敗：' + message);
        }
      } else if (selectedTool === 'mask-faces') {
        const stabilizedHeadFrames = stabilizeHeadDetectionFrames(headFrames);
        faceHeadFramesRef.current = stabilizedHeadFrames;
        facePreviewRef.current = {
          startTime,
          endTime,
          path: previewPoints,
          headFrames: stabilizedHeadFrames,
        };
        setNotice('3 秒追蹤完成；正在準備旁人人臉遮罩預覽…');
        try {
          releaseFaceHeadDetector();
          if (!facePreviewRendererRef.current) {
            const { FaceObscuringRenderer } = await import('../lib/face-obscuring');
            facePreviewRendererRef.current = await FaceObscuringRenderer.create(faceStickerUrl || undefined);
          }
          await seekTo(startTime);
          setBox([...previewPoints[0].box] as Box);
          drawFrame(previewPoints[0].box);
          setFacePreviewReady(true);
          setPhase('complete');
          setNotice('3 秒旁人遮臉預覽已準備；請點「播放 3 秒遮臉預覽」');
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          setFacePreviewReady(false);
          setPhase('complete');
          setNotice('3 秒追蹤完成，但人臉遮罩模型準備失敗：' + message);
        }
      } else {
        setPhase('complete');
        setNotice('3 秒 ViT 路徑測試完成；尚未進行影片輸出');
      }
    } catch (error) {
      backgroundPreviewRef.current = null;
      setBackgroundPreviewReady(false);
      facePreviewRef.current = null;
      setFacePreviewReady(false);
      if (selectedTool === 'mask-faces') {
        releaseFaceHeadDetector();
      }
      setPhase('select');
      const message = error instanceof Error ? error.message : String(error);
      setNotice(message);
      drawSelectionFrame(box);
    }
  }

  function playBackgroundPreview() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const preview = backgroundPreviewRef.current;
    const renderer = backgroundPreviewRendererRef.current;
    if (!video || !canvas || !preview || !renderer || !backgroundPreviewReady) {
      setNotice('請先完成 3 秒追蹤，讓去背預覽準備完成');
      return;
    }

    stopBackgroundPreviewCallbacks();
    cancelRef.current = false;
    renderer.reset();
    backgroundPreviewReturnPhaseRef.current = phase === 'path-ready' ? 'path-ready' : 'complete';
    video.pause();
    video.currentTime = preview.startTime;
    setPhase('previewing');
    setProgress(0);
    setNotice('正在播放 3 秒去背特效預覽…');

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
        const previewBox = previewBoxAt(preview.path, mediaTime);
        const [previewWidth, previewHeight] = BACKGROUND_PREVIEW_SIZES[aspect];
        if (canvas.width !== previewWidth || canvas.height !== previewHeight) {
          canvas.width = previewWidth;
          canvas.height = previewHeight;
        }
        const previewCrop = trackedFrameCrop(
          video.videoWidth,
          video.videoHeight,
          previewBox,
          previewWidth / previewHeight,
          subjectScale,
        );
        renderer.render(video, canvas, previewBox, previewCrop, getBackgroundEffects());
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
    backgroundPreviewRef.current = { startTime, endTime, path: trackPath };
    setBackgroundPreviewReady(false);
    setNotice('正在準備完整路徑的 3 秒去背 Preview…');
    try {
      if (!backgroundPreviewRendererRef.current) {
        const { PersonBackgroundRenderer } = await import('../lib/person-background-removal');
        backgroundPreviewRendererRef.current = await PersonBackgroundRenderer.create();
      }
      backgroundPreviewRendererRef.current.reset();
      await seekTo(startTime);
      const previewBox = previewBoxAt(trackPath, startTime);
      setBox(previewBox);
      drawFrame(previewBox);
      setBackgroundPreviewReady(true);
      setNotice('3 秒去背 Preview 已準備；請點「播放 3 秒去背 Preview」');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setNotice('3 秒去背 Preview 準備失敗：' + message);
    }
  }

  function chooseFaceSticker(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    resetFacePreview();
    setFaceStickerUrl(URL.createObjectURL(file));
    setFaceStickerName(file.name);
    setFaceMaskStyle('sticker');
    setNotice('貼紙已載入；完成主角追蹤後可先看 3 秒遮臉預覽');
  }

  function playFacePreview() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const preview = facePreviewRef.current;
    const renderer = facePreviewRendererRef.current;
    if (!video || !canvas || !preview || !renderer || !facePreviewReady) {
      setNotice('請先完成 3 秒追蹤，讓旁人人臉遮罩預覽準備完成');
      return;
    }

    stopBackgroundPreviewCallbacks();
    cancelRef.current = false;
    renderer.reset();
    backgroundPreviewReturnPhaseRef.current = phase === 'path-ready' ? 'path-ready' : 'complete';
    video.pause();
    video.currentTime = preview.startTime;
    setPhase('previewing');
    setProgress(0);
    setNotice('正在播放 3 秒旁人人臉遮罩預覽…');

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
        setNotice('3 秒旁人遮臉預覽失敗：' + message);
      } else if (cancelled) {
        setNotice('已取消 3 秒旁人遮臉預覽');
      } else {
        setProgress(1);
        setNotice('3 秒旁人遮臉預覽完成；可重播或繼續追蹤完整影片');
      }
    };
    const renderFrame = (mediaTime: number) => {
      if (finished) return;
      if (cancelRef.current) {
        finish(undefined, true);
        return;
      }
      try {
        const previewBox = previewBoxAt(preview.path, mediaTime);
        const [previewWidth, previewHeight] = sourcePreviewSize(video.videoWidth, video.videoHeight);
        if (canvas.width !== previewWidth || canvas.height !== previewHeight) {
          canvas.width = previewWidth;
          canvas.height = previewHeight;
        }
        renderer.render(
          video,
          canvas,
          previewBox,
          getFaceMaskEffects(),
          headBoxesAt(preview.headFrames, mediaTime),
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
    video.play().catch((error) => finish(error));
  }

  async function prepareTrackedPathFacePreview() {
    const video = videoRef.current;
    if (!video || trackPath.length < 2 || !Number.isFinite(video.duration)) {
      setNotice('請先完成整支影片的主角追蹤');
      return;
    }
    const preferredStart = selectionRef.current?.time ?? video.currentTime;
    const startTime = Math.min(Math.max(0, preferredStart), Math.max(0, video.duration - 3));
    const endTime = Math.min(video.duration, startTime + 3);
    facePreviewRef.current = {
      startTime,
      endTime,
      path: trackPath,
      headFrames: faceHeadFramesRef.current,
    };
    setFacePreviewReady(false);
    setNotice('正在準備完整路徑的 3 秒旁人遮臉 Preview…');
    try {
      if (!facePreviewRendererRef.current) {
        const { FaceObscuringRenderer } = await import('../lib/face-obscuring');
        facePreviewRendererRef.current = await FaceObscuringRenderer.create(faceStickerUrl || undefined);
      }
      facePreviewRendererRef.current.reset();
      await seekTo(startTime);
      const previewBox = previewBoxAt(trackPath, startTime);
      setBox(previewBox);
      drawFrame(previewBox);
      setFacePreviewReady(true);
      setNotice('3 秒旁人遮臉 Preview 已準備；請點「播放 3 秒旁人遮臉 Preview」');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setNotice('3 秒旁人遮臉 Preview 準備失敗：' + message);
    }
  }

  async function runFullTracking() {
    const video = videoRef.current;
    const selection = selectionRef.current;
    if (!video || !selection || !Number.isFinite(video.duration)) {
      setNotice('請先在影片畫面框選主角');
      return;
    }

    if (selectedTool === 'mask-faces') resetFacePreview();

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
    const headFrames: HeadDetectionFrame[] = [];
    const measurements: TrackResult[] = [];
    const started = eventClock();
    let processed = 0;
    const mainHeadTrackingContext = selectedTool === 'mask-faces'
      ? createMainHeadTrackingContext(
        selection.box,
        video.videoWidth,
        video.videoHeight,
      )
      : null;

    try {
      setNotice('正在載入本機 ViT 模型…');
      if (!trackerRef.current) {
        const modelUrl = new URL('models/vittrack.onnx', document.baseURI).href;
        trackerRef.current = await VitTracker.create(modelUrl);
      }
      if (selectedTool === 'mask-faces') {
        setNotice('正在載入本機 360° 人頭辨識…');
        await ensureFaceHeadDetector();
        await seekTo(selection.time);
        headFrames.push(await detectHeadFrame(video, selection.box, selection.time));
      }

      const trackDirection = async (times: number[], label: string) => {
        await seekTo(selection.time);
        trackerRef.current!.initialize(
          video,
          mainHeadTrackingContext?.trackerBox ?? selection.box,
        );
        for (const frameTime of times) {
          if (cancelRef.current) throw new Error('使用者已取消追蹤');
          await seekTo(frameTime);
          const result = await trackerRef.current!.update(video);
          const resultBox = mainHeadTrackingContext
            ? headBoxFromTrackingContext(
              result.box,
              mainHeadTrackingContext,
              video.videoWidth,
              video.videoHeight,
            )
            : result.box;
          measurements.push(result);
          points.push({
            time: frameTime,
            box: [...resultBox] as Box,
            score: result.score,
            accepted: result.accepted,
          });
          processed += 1;
          if (selectedTool === 'mask-faces' && processed % 2 === 0) {
            headFrames.push(await detectHeadFrame(video, resultBox, frameTime));
          }
          setBox(resultBox);
          setCurrentScore(result.score);
          setProgress(processed / Math.max(1, totalFrames));
          setNotice((selectedTool === 'mask-faces' ? label + '＋人頭辨識' : label) + ' · ' + processed + ' / ' + totalFrames + ' 幀');
          drawFrame(resultBox, result.score);
          await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
        }
      };

      await trackDirection(forwardTimes, '向後完整追蹤');
      await trackDirection(backwardTimes, '補齊選角前片段');

      points.sort((left, right) => left.time - right.time);
      if (selectedTool === 'mask-faces') {
        faceHeadFramesRef.current = stabilizeHeadDetectionFrames(headFrames);
        releaseFaceHeadDetector();
      }
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
      setNotice(selectedTool === 'remove-background'
        ? '完整舞者路徑已建立；可調整比例、主角大小與柔順度後輸出'
        : selectedTool === 'mask-faces'
          ? '完整主角頭／臉路徑已建立；可調整旁人人臉遮罩後輸出原片構圖'
        : '完整 ViT 路徑已建立；可調整構圖並輸出影片');
    } catch (error) {
      if (selectedTool === 'mask-faces') {
        releaseFaceHeadDetector();
      }
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
        effects: getBackgroundEffects(),
      };
    } else if (selectedTool === 'mask-faces') {
      operation = {
        kind: 'mask-faces',
        smoothness,
        effects: getFaceMaskEffects(),
        headFrames: faceHeadFramesRef.current,
      };
    }
    if (!video || !renderCanvas || !operation) {
      setNotice('請先選擇一項後製功能');
      return;
    }
    if ((operation.kind === 'track' || operation.kind === 'remove-background' || operation.kind === 'mask-faces') && trackPath.length < 2) {
      setNotice('請先完成整支影片的 ViT 追蹤');
      return;
    }
    cancelRef.current = false;
    setPhase('exporting');
    setProgress(0);
    setNotice(operation.kind === 'remove-background'
      ? '正在載入本機人物去背模型…'
      : operation.kind === 'mask-faces'
        ? '正在載入本機人臉遮罩模型…'
      : codec === 'hevc' ? '正在準備 HEVC 母片輸出…' : '正在準備 H.264 相容影片輸出…');

    try {
      if (!exporterRef.current) exporterRef.current = new RealtimeVideoExporter(video);
      const result = await exporterRef.current.export(trackPath, renderCanvas, {
        operation,
        quality: outputQuality,
        codec,
        onProgress: (next) => {
          setProgress(next);
          setNotice((operation.kind === 'remove-background'
            ? '人物去背與本機編碼中 · '
            : operation.kind === 'mask-faces'
              ? '旁人人臉遮罩與本機編碼中 · '
              : '本機編碼中 · ') + Math.round(next * 100) + '%');
        },
        isCancelled: () => cancelRef.current,
      });
      const baseName = (sourceFile?.name ?? 'NiviTrack').replace(/\.[^.]+$/, '');
      const fileTag = TOOL_CHOICES.find((item) => item.id === selectedTool)?.fileTag ?? 'Edit';
      const name = baseName + '-NiviTrack-' + fileTag + '-' + (outputQuality === 'clear' ? 'Clear' : 'Standard') + '.mp4';
      setExportBlob(result.blob);
      setExportUrl(URL.createObjectURL(result.blob));
      setExportInfo({
        name,
        size: formatBytes(result.blob.size),
        mimeType: result.mimeType,
        resolution: result.width + ' × ' + result.height,
      });
      setPhase(operation.kind === 'track' || operation.kind === 'remove-background' || operation.kind === 'mask-faces' ? 'path-ready' : 'tool-ready');
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
      setPhase(selectedTool === 'track' || selectedTool === 'remove-background' || selectedTool === 'mask-faces' ? 'path-ready' : 'tool-ready');
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
    setNotice(phase === 'exporting' ? '正在取消輸出…' : phase === 'previewing' ? '正在取消預覽…' : '正在取消追蹤…');
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
  const nextStepText = !videoInfo
    ? '點「選擇一支影片」，從照片或檔案匯入影片。'
    : phase === 'choose'
      ? '從影片下方選擇一項後製功能。'
      : phase === 'tool-ready'
        ? '先播放確認效果；滿意後選擇畫質並輸出。'
        : phase === 'crop-select'
          ? cropBox
            ? '裁切框已畫好，點「使用此裁切框」。'
            : '在影片上拖曳，框出成品要保留的範圍。'
          : phase === 'select'
            ? selectionPlaying
              ? '看到主角清楚的畫面時，點「暫停並框選」。'
              : !box
                ? selectedTool === 'mask-faces'
                  ? '先點 AI 候選人臉，或用手指框住主角的頭／臉。'
                  : '先點 AI 候選人物，或用手指粗略框住主角。'
                : selectionZoom === 1
                  ? '主角已框好；可按「＋」放大後重新畫精準框。'
                  : '在放大的畫面重新畫精準框，再開始 3 秒測試或完整追蹤。'
            : phase === 'tracking'
              ? '正在追蹤，請保持此頁開啟並等待完成。'
              : phase === 'previewing'
                ? selectedTool === 'mask-faces'
                  ? '正在播放 3 秒旁人人臉遮罩預覽，請等待完成。'
                  : '正在產生 3 秒去背預覽，請等待完成。'
                : phase === 'complete'
                  ? selectedTool === 'remove-background'
                    ? '先播放 3 秒去背預覽；滿意後追蹤完整影片。'
                    : selectedTool === 'mask-faces'
                      ? '先播放 3 秒旁人遮臉預覽；滿意後追蹤完整影片。'
                    : '3 秒測試完成；確認後追蹤完整影片。'
                  : phase === 'path-ready'
                    ? '追蹤完成；調整構圖與效果後輸出影片。'
                    : phase === 'exporting'
                      ? '正在本機輸出並保留原聲，請保持此頁開啟。'
                      : '依影片下方按鈕繼續下一步。';
  const effectsBusy = phase === 'tracking' || phase === 'previewing' || phase === 'exporting';
  const backgroundEffectPanel = selectedTool === 'remove-background' && videoInfo ? (
    <section className="effect-panel" aria-label="去背後製特效">
      <div className="effect-heading">
        <div>
          <span>去背完成後套用</span>
          <strong>背景、外框與分身</strong>
        </div>
        <b>本機後製</b>
      </div>

      <div className="effect-block">
        <div className="effect-label"><b>背景</b><em>{backgroundMode === 'color' ? '純色' : '模糊原片'}</em></div>
        <div className="effect-options two">
          <button
            className={backgroundMode === 'color' ? 'selected' : ''}
            type="button"
            disabled={effectsBusy}
            aria-pressed={backgroundMode === 'color'}
            onClick={() => setBackgroundMode('color')}
          >
            背景顏色
          </button>
          <button
            className={backgroundMode === 'blur' ? 'selected' : ''}
            type="button"
            disabled={effectsBusy}
            aria-pressed={backgroundMode === 'blur'}
            onClick={() => setBackgroundMode('blur')}
          >
            模糊原片
          </button>
        </div>
        {backgroundMode === 'color' ? (
          <div className="effect-swatches" aria-label="背景顏色">
            {BACKGROUND_COLORS.map((color) => (
              <button
                className={backgroundColor === color.value ? 'selected' : ''}
                type="button"
                key={color.value}
                disabled={effectsBusy}
                aria-label={color.name + '色背景'}
                aria-pressed={backgroundColor === color.value}
                style={{ backgroundColor: color.value }}
                onClick={() => setBackgroundColor(color.value)}
              />
            ))}
            <label className="color-picker">
              <input
                type="color"
                value={backgroundColor}
                disabled={effectsBusy}
                aria-label="自訂背景顏色"
                onChange={(event) => setBackgroundColor(event.target.value)}
              />
              <span>自訂</span>
            </label>
          </div>
        ) : (
          <label className="range-control compact">
            <span><b>模糊程度</b><em>{backgroundBlur}px</em></span>
            <input
              type="range"
              min="12"
              max="64"
              step="2"
              value={backgroundBlur}
              disabled={effectsBusy}
              onChange={(event) => setBackgroundBlur(Number(event.target.value))}
            />
          </label>
        )}
      </div>

      <div className="effect-block">
        <div className="effect-label">
          <b>人物外框</b>
          <em>{outlineWidth === 0 ? '無框' : outlineWidth <= 16 ? '細框 · ' + outlineWidth + 'px' : '粗框 · ' + outlineWidth + 'px'}</em>
        </div>
        <div className="effect-options three" aria-label="外框粗細">
          <button
            className={outlineWidth === 0 ? 'selected' : ''}
            type="button"
            disabled={effectsBusy}
            aria-pressed={outlineWidth === 0}
            onClick={() => setOutlineWidth(0)}
          >
            無框
          </button>
          <button
            className={outlineWidth > 0 && outlineWidth <= 16 ? 'selected' : ''}
            type="button"
            disabled={effectsBusy}
            aria-pressed={outlineWidth > 0 && outlineWidth <= 16}
            onClick={() => setOutlineWidth(8)}
          >
            細框
          </button>
          <button
            className={outlineWidth > 16 ? 'selected' : ''}
            type="button"
            disabled={effectsBusy}
            aria-pressed={outlineWidth > 16}
            onClick={() => setOutlineWidth(32)}
          >
            粗框
          </button>
        </div>
        <div className="outline-controls">
          <label className="color-picker outline-picker">
            <input
              type="color"
              value={outlineColor}
              disabled={effectsBusy || outlineWidth === 0}
              aria-label="自訂外框顏色"
              onChange={(event) => setOutlineColor(event.target.value)}
            />
            <span>外框顏色</span>
          </label>
          <label className="range-control compact">
            <span><b>厚度</b><em>{outlineWidth === 0 ? '無框' : outlineWidth + 'px'}</em></span>
            <input
              type="range"
              min="0"
              max="48"
              value={outlineWidth}
              disabled={effectsBusy}
              onChange={(event) => setOutlineWidth(Number(event.target.value))}
            />
          </label>
        </div>
      </div>

      <div className="effect-block">
        <div className="effect-label">
          <b>分身</b>
          <em>{cloneCount === 0 ? '關閉 · 共 1 人' : '+' + cloneCount + ' 個分身 · 共 ' + (cloneCount + 1) + ' 人'}</em>
        </div>
        <div className="effect-options five" aria-label="分身數量">
          {[0, 1, 2, 3, 4].map((count) => (
            <button
              className={cloneCount === count ? 'selected' : ''}
              type="button"
              key={count}
              disabled={effectsBusy}
              aria-pressed={cloneCount === count}
              onClick={() => setCloneCount(count)}
            >
              {count === 0 ? '無' : '+' + count}
            </button>
          ))}
        </div>
        {cloneCount > 0 && (
          <>
            <div className="effect-subgroup">
              <span>分身樣式</span>
              <div className="effect-options two">
                <button
                  className={cloneStyle === 'subject' ? 'selected' : ''}
                  type="button"
                  disabled={effectsBusy}
                  aria-pressed={cloneStyle === 'subject'}
                  onClick={() => setCloneStyle('subject')}
                >
                  人物分身
                </button>
                <button
                  className={cloneStyle === 'outline' ? 'selected' : ''}
                  type="button"
                  disabled={effectsBusy}
                  aria-pressed={cloneStyle === 'outline'}
                  onClick={() => setCloneStyle('outline')}
                >
                  線框分身
                </button>
              </div>
            </div>
            <div className="effect-subgroup">
              <span>排列方式</span>
              <div className="effect-options two">
                <button
                  className={cloneLayout === 'trail' ? 'selected' : ''}
                  type="button"
                  disabled={effectsBusy}
                  aria-pressed={cloneLayout === 'trail'}
                  onClick={() => setCloneLayout('trail')}
                >
                  前後殘影
                </button>
                <button
                  className={cloneLayout === 'row' ? 'selected' : ''}
                  type="button"
                  disabled={effectsBusy}
                  aria-pressed={cloneLayout === 'row'}
                  onClick={() => setCloneLayout('row')}
                >
                  左右併排
                </button>
              </div>
            </div>
          </>
        )}
      </div>
      <p className="effect-note">構圖沿用主角鎖定裁切；處理順序為先去背，再加外框與分身，最後合成背景。</p>
    </section>
  ) : null;

  const faceMaskEffectPanel = selectedTool === 'mask-faces' && videoInfo ? (
    <section className="effect-panel" aria-label="旁人人臉遮罩設定">
      <div className="effect-heading">
        <div>
          <span>只遮住主角以外的人臉</span>
          <strong>旁人人臉遮罩</strong>
        </div>
        <b>本機辨識</b>
      </div>

      <div className="effect-block">
        <div className="effect-label"><b>遮罩樣式</b><em>共 6 種</em></div>
        <div className="effect-options three" aria-label="旁人人臉遮罩樣式">
          {([
            ['soft-blur', '柔和模糊'],
            ['strong-blur', '強力模糊'],
            ['pixelate', '馬賽克'],
            ['black-oval', '黑色橢圓'],
            ['emoji', 'Emoji'],
            ['sticker', '自選貼紙'],
          ] as [FaceMaskStyle, string][]).map(([style, label]) => (
            <button
              className={faceMaskStyle === style ? 'selected' : ''}
              type="button"
              key={style}
              disabled={effectsBusy}
              aria-pressed={faceMaskStyle === style}
              onClick={() => {
                setFaceMaskStyle(style);
                if (style === 'sticker' && !faceStickerUrl) stickerInputRef.current?.click();
              }}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {(faceMaskStyle === 'soft-blur' || faceMaskStyle === 'strong-blur' || faceMaskStyle === 'pixelate') && (
        <label className="range-control compact">
          <span><b>遮蔽強度</b><em>{Math.round(faceMaskStrength * 100)}%</em></span>
          <input
            type="range"
            min="20"
            max="100"
            value={Math.round(faceMaskStrength * 100)}
            disabled={effectsBusy}
            onChange={(event) => setFaceMaskStrength(Number(event.target.value) / 100)}
          />
        </label>
      )}

      <label className="range-control compact">
        <span><b>遮罩大小</b><em>{Math.round(faceMaskScale * 100)}%</em></span>
        <input
          type="range"
          min="100"
          max="180"
          value={Math.round(faceMaskScale * 100)}
          disabled={effectsBusy}
          onChange={(event) => setFaceMaskScale(Number(event.target.value) / 100)}
        />
      </label>

      {faceMaskStyle === 'emoji' && (
        <div className="effect-block">
          <div className="effect-label"><b>Emoji</b><em>{faceMaskEmoji}</em></div>
          <div className="effect-options five" aria-label="選擇人臉 Emoji">
            {['😎', '🥸', '🤡', '🐼', '⭐'].map((emoji) => (
              <button
                className={faceMaskEmoji === emoji ? 'selected' : ''}
                type="button"
                key={emoji}
                disabled={effectsBusy}
                aria-pressed={faceMaskEmoji === emoji}
                onClick={() => setFaceMaskEmoji(emoji)}
              >
                {emoji}
              </button>
            ))}
          </div>
        </div>
      )}

      {faceMaskStyle === 'sticker' && (
        <div className="effect-block">
          <div className="effect-label"><b>自選貼紙</b><em>{faceStickerName || '尚未選擇'}</em></div>
          <button
            type="button"
            disabled={effectsBusy}
            onClick={() => {
              const input = stickerInputRef.current;
              if (!input) return;
              input.value = '';
              input.click();
            }}
          >
            {faceStickerUrl ? '更換貼紙圖片' : '選擇貼紙圖片'}
          </button>
        </div>
      )}

      <div className="effect-block">
        <div className="effect-label"><b>隱私優先</b><em>{faceMaskPrivacyFirst ? '不確定就遮住' : '優先保留主角臉'}</em></div>
        <div className="effect-options two">
          <button
            className={faceMaskPrivacyFirst ? 'selected' : ''}
            type="button"
            disabled={effectsBusy}
            aria-pressed={faceMaskPrivacyFirst}
            onClick={() => setFaceMaskPrivacyFirst(true)}
          >
            不確定就遮住
          </button>
          <button
            className={!faceMaskPrivacyFirst ? 'selected' : ''}
            type="button"
            disabled={effectsBusy}
            aria-pressed={!faceMaskPrivacyFirst}
            onClick={() => setFaceMaskPrivacyFirst(false)}
          >
            優先保留主角
          </button>
        </div>
      </div>
      <p className="effect-note">YOLOX‑Nano 直接尋找整顆人頭，包含側面與背面；ViT 只追蹤主角頭部並將它排除，貼紙與模糊區再向外擴一圈。輸出保留原始畫面比例，全程留在這台 iPhone，不做身分辨識，也不上傳影片。</p>
    </section>
  ) : null;

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="NiviTrack 首頁">
          <span className="brand-mark">N</span><span>NiviTrack</span>
        </a>
        <span className="local-pill"><i aria-hidden="true" />iPhone 本機處理</span>
      </header>

      <section className="hero">
        <div className="eyebrow">IPHONE WEB APP · 13 種單一後製</div>
        <h1>選一個功能，<span>直接完成影片。</span></h1>
        <p>匯入 MOV、HEVC 或 MP4，選擇濾鏡、裁切、主角鎖定、單一舞者去背或旁人人臉遮罩。一次處理一項；要疊加時，把輸出影片再匯入即可。</p>
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
                      : selectedTool === 'mask-faces'
                        ? '已暫停；請框住主角的頭／臉，或使用 AI 尋找人臉'
                      : '已暫停；請用手指框住要追蹤的人物或寵物');
                    requestAnimationFrame(() => drawSelectionFrame(null));
                  }}
                  onSeeked={() => {
                    if (phase === 'select' && videoRef.current?.paused) requestAnimationFrame(() => drawSelectionFrame(null));
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
                  {phase === 'choose'
                    ? '尚未選擇功能'
                    : phase === 'tool-ready'
                      ? selectedChoice?.name
                      : phase === 'crop-select'
                        ? '手指框選保留範圍'
                        : phase === 'select'
                          ? selectionPlaying
                            ? '播放中 · 暫停後框選'
                            : selectedTool === 'remove-background'
                              ? '框選單一舞者'
                              : selectedTool === 'mask-faces'
                                ? '框選主角頭／臉'
                                : '手指框選主角'
                          : phase === 'previewing'
                            ? selectedTool === 'mask-faces' ? '3 秒旁人人臉遮罩預覽' : '3 秒去背特效預覽'
                            : phase === 'exporting'
                              ? selectedTool === 'remove-background'
                                ? '去背與特效合成'
                                : selectedTool === 'mask-faces'
                                  ? '旁人人臉遮罩與編碼'
                                  : 'Safari 本機編碼'
                              : phase === 'path-ready'
                                ? selectedTool === 'remove-background'
                                  ? '單一舞者去背與輸出'
                                  : selectedTool === 'mask-faces'
                                    ? '旁人人臉遮罩與輸出'
                                    : '構圖與輸出'
                                : 'ViT 本機推論'}
                </span>
                {(phase === 'tracking' || phase === 'previewing' || phase === 'exporting') && (
                  <div className="progress-overlay">
                    <strong>{Math.round(progress * 100)}%</strong>
                    <span>{phase === 'exporting'
                      ? '保留原聲'
                      : phase === 'previewing'
                        ? selectedTool === 'mask-faces' ? '旁人遮臉預覽' : '去背後製預覽'
                        : 'score ' + (currentScore === null ? '—' : currentScore.toFixed(3))}</span>
                  </div>
                )}
              </div>
              <div className="next-step-card" role="status" aria-live="polite">
                <span>下一步</span>
                <strong>{nextStepText}</strong>
              </div>
              {phase === 'select' && !selectionPlaying && (
                <div className="selection-zoom-bar" aria-label="取框畫面縮放">
                  <span>取框放大</span>
                  <button
                    type="button"
                    disabled={selectionZoom <= 1}
                    aria-label="縮小取框畫面"
                    onClick={() => changeSelectionZoom(-1)}
                  >−</button>
                  <strong>{selectionZoom}×</strong>
                  <button
                    type="button"
                    disabled={selectionZoom >= 3}
                    aria-label="放大取框畫面"
                    onClick={() => changeSelectionZoom(1)}
                  >＋</button>
                  <small>先粗略框選，再放大重新畫框</small>
                </div>
              )}
              <div className="video-actions">
                {phase !== 'tracking' && phase !== 'previewing' && phase !== 'exporting' && (
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
                      {detecting
                        ? 'AI 掃描中…'
                        : selectedTool === 'mask-faces'
                          ? 'AI 尋找人臉'
                          : selectedTool === 'remove-background'
                            ? 'AI 尋找人物'
                            : 'AI 尋找人物／寵物'}
                    </button>
                    <button type="button" disabled={!box || selectionPlaying} onClick={runTracking}>
                      {selectedTool === 'remove-background'
                        ? '準備 3 秒去背預覽'
                        : selectedTool === 'mask-faces'
                          ? '準備 3 秒遮臉預覽'
                          : '測試 3 秒 ViT'}
                    </button>
                    <button className="primary" type="button" disabled={!box || selectionPlaying} onClick={runFullTracking}>
                      追蹤完整影片
                    </button>
                  </>
                )}
                {(phase === 'tracking' || phase === 'previewing' || phase === 'exporting') && (
                  <button className="danger" type="button" onClick={cancelTracking}>{phase === 'exporting' ? '取消輸出' : phase === 'previewing' ? '取消預覽' : '取消追蹤'}</button>
                )}
                {phase === 'complete' && (
                  <>
                    <button type="button" onClick={returnToTools}>返回功能選單</button>
                    <button type="button" onClick={enterSelection}>重新框選</button>
                    {selectedTool === 'remove-background' && (
                      <button className="primary" type="button" disabled={!backgroundPreviewReady} onClick={playBackgroundPreview}>播放 3 秒去背預覽</button>
                    )}
                    {selectedTool === 'mask-faces' && (
                      <button className="primary" type="button" disabled={!facePreviewReady} onClick={playFacePreview}>播放 3 秒遮臉預覽</button>
                    )}
                    <button className={selectedTool === 'remove-background' || selectedTool === 'mask-faces' ? '' : 'primary'} type="button" onClick={runFullTracking}>追蹤完整影片</button>
                  </>
                )}
                {phase === 'path-ready' && (
                  <>
                    <button type="button" onClick={returnToTools}>返回功能選單</button>
                    <button type="button" onClick={enterSelection}>重新選角與追蹤</button>
                  </>
                )}
              </div>
              {backgroundEffectPanel}
              {faceMaskEffectPanel}
              {(phase === 'choose' || phase === 'tool-ready') && videoInfo && (
                <section className="tool-panel" aria-label="選擇一項影片後製功能">
                  <div className="tool-heading">
                    <div>
                      <span>一次選一項</span>
                      <strong>13 種影片後製</strong>
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

                  <div className="quality-control">
                    <span><b>輸出畫質</b><em>{outputQuality === 'clear' ? '預設 · 最高 1080p' : '較快 · 最高 720p'}</em></span>
                    <div className="aspect-options quality-options" aria-label="輸出畫質">
                      <button className={outputQuality === 'clear' ? 'selected' : ''} type="button" disabled={phase === 'exporting'} aria-pressed={outputQuality === 'clear'} onClick={() => selectOutputQuality('clear')}>
                        清晰
                      </button>
                      <button className={outputQuality === 'standard' ? 'selected' : ''} type="button" disabled={phase === 'exporting'} aria-pressed={outputQuality === 'standard'} onClick={() => selectOutputQuality('standard')}>
                        標準
                      </button>
                    </div>
                  </div>

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
                      <span>{selectedTool === 'remove-background'
                        ? '單一舞者路徑已就緒'
                        : selectedTool === 'mask-faces'
                          ? '主角臉排除路徑已就緒'
                          : '完整路徑已就緒'}</span>
                      <strong>{selectedTool === 'remove-background'
                        ? '選擇去背構圖'
                        : selectedTool === 'mask-faces'
                          ? '確認旁人人臉遮罩'
                          : '選擇輸出構圖'}</strong>
                    </div>
                    <b>{selectedTool === 'remove-background' ? aspect : selectedTool === 'mask-faces' ? '原片構圖' : trackPath.length + ' 點'}</b>
                  </div>

                  {selectedTool === 'remove-background' && (
                    <>
                      <div className="crop-summary">
                        <span>輸出方式</span>
                        <strong>
                          {backgroundMode === 'color' ? '背景顏色' : '模糊原片'}
                          {' · '}
                          {outlineWidth === 0 ? '無框' : outlineWidth + 'px 外框'}
                          {' · '}
                          {cloneCount === 0
                            ? '無分身'
                            : '主角 1 + ' + cloneCount + ' 個' + (cloneStyle === 'subject' ? '人物分身' : '線框分身')
                              + ' = 共 ' + (cloneCount + 1) + ' 人 · ' + (cloneLayout === 'trail' ? '前後殘影' : '左右併排')}
                        </strong>
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
                          {backgroundPreviewReady ? '播放 3 秒去背 Preview' : '準備 3 秒去背 Preview'}
                        </button>
                      </div>
                    </>
                  )}

                  {selectedTool === 'mask-faces' && (
                    <>
                      <div className="crop-summary">
                        <span>輸出方式</span>
                        <strong>
                          保留主角臉 · {faceMaskStyle === 'soft-blur'
                            ? '柔和模糊'
                            : faceMaskStyle === 'strong-blur'
                              ? '強力模糊'
                              : faceMaskStyle === 'pixelate'
                                ? '馬賽克'
                                : faceMaskStyle === 'black-oval'
                                  ? '黑色橢圓'
                                  : faceMaskStyle === 'emoji'
                                    ? faceMaskEmoji + ' Emoji'
                                    : faceStickerName || '自選貼紙'}
                        </strong>
                      </div>
                      <div className="background-preview-actions">
                        <button
                          className="primary"
                          type="button"
                          disabled={phase === 'exporting'}
                          onClick={facePreviewReady
                            ? playFacePreview
                            : () => void prepareTrackedPathFacePreview()}
                        >
                          {facePreviewReady ? '播放 3 秒旁人遮臉 Preview' : '準備 3 秒旁人遮臉 Preview'}
                        </button>
                      </div>
                    </>
                  )}

                  {selectedTool !== 'mask-faces' && (
                    <>
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
                    </>
                  )}

                  <div className="quality-control">
                    <span><b>輸出畫質</b><em>{outputQuality === 'clear' ? '預設 · 最高 1080p' : '較快 · 最高 720p'}</em></span>
                    <div className="aspect-options quality-options" aria-label="輸出畫質">
                      <button className={outputQuality === 'clear' ? 'selected' : ''} type="button" disabled={phase === 'exporting'} aria-pressed={outputQuality === 'clear'} onClick={() => selectOutputQuality('clear')}>
                        清晰
                      </button>
                      <button className={outputQuality === 'standard' ? 'selected' : ''} type="button" disabled={phase === 'exporting'} aria-pressed={outputQuality === 'standard'} onClick={() => selectOutputQuality('standard')}>
                        標準
                      </button>
                    </div>
                  </div>

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
                      先以 ViT 追蹤框與框內提示點辨識同一位主角，再套用背景、外框與分身；推論、後製、放大置中與原聲合成都只在這台 iPhone 執行。
                    </p>
                  )}

                  {selectedTool === 'mask-faces' && (
                    <p className="export-note">
                      YOLOX‑Nano 建立所有人頭路徑，ViT 持續排除主角頭部，再將貼紙或模糊區放大覆蓋其他人頭。保留原片構圖、原聲，全程只在這台 iPhone 執行。
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
          <input ref={stickerInputRef} className="sr-only" type="file" accept="image/png,image/webp,image/jpeg" onChange={chooseFaceSticker} />
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
