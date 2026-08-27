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

type Phase = 'choose' | 'select' | 'tracking' | 'complete';

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

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragStartRef = useRef<[number, number] | null>(null);
  const cancelRef = useRef(false);
  const trackerRef = useRef<VitTracker | null>(null);
  const detectorRef = useRef<ObjectDetection | null>(null);

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

  useEffect(() => {
    const mediaRecorder = typeof MediaRecorder !== 'undefined';
    const supportsMp4 =
      mediaRecorder &&
      ['video/mp4;codecs=avc1.42E01E,mp4a.40.2', 'video/mp4'].some((type) =>
        MediaRecorder.isTypeSupported(type),
      );
    const capabilityFrame = requestAnimationFrame(() => {
      setCapabilities([
        { label: '本機 AI', detail: 'WebAssembly', available: typeof WebAssembly !== 'undefined' },
        { label: '背景運算', detail: 'Web Worker', available: typeof Worker !== 'undefined' },
        { label: '逐幀影像', detail: 'WebCodecs', available: typeof VideoFrame !== 'undefined' },
        { label: 'GPU 加速', detail: 'WebGPU', available: 'gpu' in navigator },
        { label: '相容分享', detail: 'H.264 / AAC MP4', available: supportsMp4 },
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

  const readyCount = useMemo(
    () => capabilities.filter((item) => item.available).length,
    [capabilities],
  );

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

  function cancelTracking() {
    cancelRef.current = true;
    setNotice('正在取消追蹤…');
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
          <div className={'step ' + (step >= 3 ? 'active' : '')}><b>03</b><span>追蹤測試</span></div>
        </div>
      </section>

      <section className="workspace">
        <div className="video-panel">
          {!videoUrl ? (
            <button className="picker" type="button" onClick={() => inputRef.current?.click()}>
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
                  className={phase === 'choose' ? '' : 'is-hidden'}
                  src={videoUrl}
                  controls
                  playsInline
                  preload="metadata"
                  onLoadedMetadata={readMetadata}
                  onError={() => setNotice('Safari 無法解碼這支影片，請保留檔案供實機記錄')}
                />
                <canvas
                  ref={canvasRef}
                  className={phase === 'choose' ? 'tracking-canvas is-hidden' : 'tracking-canvas'}
                  onPointerDown={startBox}
                  onPointerMove={moveBox}
                  onPointerUp={finishBox}
                  onPointerCancel={finishBox}
                />
                <span className="source-badge">
                  {phase === 'choose' ? '原始檔直接解碼' : phase === 'select' ? '手指框選主角' : 'ViT 本機推論'}
                </span>
                {phase === 'tracking' && (
                  <div className="progress-overlay">
                    <strong>{Math.round(progress * 100)}%</strong>
                    <span>score {currentScore === null ? '—' : currentScore.toFixed(3)}</span>
                  </div>
                )}
              </div>
              <div className="video-actions">
                {phase === 'choose' && (
                  <>
                    <button type="button" onClick={() => inputRef.current?.click()}>更換影片</button>
                    <button className="primary" type="button" disabled={!videoInfo} onClick={enterSelection}>
                      進入主角選取
                    </button>
                  </>
                )}
                {phase === 'select' && (
                  <>
                    <button type="button" onClick={() => { setPhase('choose'); setBox(null); }}>返回播放</button>
                    <button type="button" disabled={detecting} onClick={detectSubjects}>
                      {detecting ? 'AI 掃描中…' : 'AI 尋找人物／寵物'}
                    </button>
                    <button className="primary" type="button" disabled={!box} onClick={runTracking}>
                      測試 3 秒 ViT
                    </button>
                  </>
                )}
                {phase === 'tracking' && (
                  <button className="danger" type="button" onClick={cancelTracking}>取消追蹤</button>
                )}
                {phase === 'complete' && (
                  <>
                    <button type="button" onClick={enterSelection}>重新框選</button>
                    <button className="primary" type="button" onClick={runTracking}>再次測試</button>
                  </>
                )}
              </div>
            </>
          )}
          <input ref={inputRef} className="sr-only" type="file" accept="video/*,.mov,.mp4" onChange={chooseVideo} />
        </div>

        <aside className="side-panel">
          <div className="status-card">
            <div className="card-heading"><span>裝置能力</span><b>{readyCount}/{capabilities.length || 6}</b></div>
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
