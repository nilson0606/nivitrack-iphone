'use client';

import {
  ChangeEvent,
  PointerEvent as ReactPointerEvent,
  useEffect,
  useRef,
  useState,
} from 'react';
import { InteractiveSubjectSegmenter } from '../../lib/interactive-subject-segmenter';
import type { MatAnyone2Engine } from '../../lib/matanyone2-engine';
import styles from './page.module.css';

const TEST_SECONDS = 3;
const TEST_FPS = 25;
const MATANYONE2_SIZE = { width: 288, height: 512 } as const;

type Phase = 'choose' | 'pause' | 'select' | 'seeded' | 'loading' | 'running' | 'ready';

function assetUrl(path: string) {
  return new URL(`./${path}`, document.baseURI).href;
}

function waitForMetadata(video: HTMLVideoElement) {
  if (video.readyState >= HTMLMediaElement.HAVE_METADATA) return Promise.resolve();
  return new Promise<void>((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error('Safari 讀取影片資訊逾時'));
    }, 10_000);
    const cleanup = () => {
      window.clearTimeout(timeout);
      video.removeEventListener('loadedmetadata', loaded);
      video.removeEventListener('error', failed);
    };
    const loaded = () => {
      cleanup();
      resolve();
    };
    const failed = () => {
      cleanup();
      reject(new Error('Safari 無法讀取這支影片'));
    };
    video.addEventListener('loadedmetadata', loaded, { once: true });
    video.addEventListener('error', failed, { once: true });
  });
}

function seekVideo(video: HTMLVideoElement, time: number) {
  if (Math.abs(video.currentTime - time) < 0.001) {
    return new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  }
  return new Promise<void>((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error('影片定位逾時'));
    }, 8_000);
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
      reject(new Error('影片定位失敗'));
    };
    video.addEventListener('seeked', done, { once: true });
    video.addEventListener('error', failed, { once: true });
    video.currentTime = time;
  });
}

function resizeMask(
  source: Float32Array,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
) {
  const output = new Float32Array(targetWidth * targetHeight);
  for (let y = 0; y < targetHeight; y += 1) {
    const sourceY = ((y + 0.5) * sourceHeight) / targetHeight - 0.5;
    const y0 = Math.max(0, Math.min(sourceHeight - 1, Math.floor(sourceY)));
    const y1 = Math.min(sourceHeight - 1, y0 + 1);
    const wy = Math.max(0, sourceY - y0);
    for (let x = 0; x < targetWidth; x += 1) {
      const sourceX = ((x + 0.5) * sourceWidth) / targetWidth - 0.5;
      const x0 = Math.max(0, Math.min(sourceWidth - 1, Math.floor(sourceX)));
      const x1 = Math.min(sourceWidth - 1, x0 + 1);
      const wx = Math.max(0, sourceX - x0);
      const top = source[y0 * sourceWidth + x0] * (1 - wx)
        + source[y0 * sourceWidth + x1] * wx;
      const bottom = source[y1 * sourceWidth + x0] * (1 - wx)
        + source[y1 * sourceWidth + x1] * wx;
      output[y * targetWidth + x] = Math.max(
        0,
        Math.min(1, top * (1 - wy) + bottom * wy),
      );
    }
  }
  return output;
}

function drawCutout(
  source: CanvasImageSource,
  alpha: Float32Array,
  output: HTMLCanvasElement,
  frame: HTMLCanvasElement,
) {
  const { width, height } = MATANYONE2_SIZE;
  output.width = width;
  output.height = height;
  frame.width = width;
  frame.height = height;
  const frameContext = frame.getContext('2d', { willReadFrequently: true });
  const outputContext = output.getContext('2d', { alpha: false });
  if (!frameContext || !outputContext) throw new Error('Safari 無法建立預覽畫布');
  frameContext.drawImage(source, 0, 0, width, height);
  const image = frameContext.getImageData(0, 0, width, height);
  for (let index = 0; index < alpha.length; index += 1) {
    const pixel = index * 4;
    const opacity = Math.max(0, Math.min(1, alpha[index]));
    image.data[pixel] = Math.round(image.data[pixel] * opacity);
    image.data[pixel + 1] = Math.round(image.data[pixel + 1] * opacity);
    image.data[pixel + 2] = Math.round(image.data[pixel + 2] * opacity);
    image.data[pixel + 3] = 255;
  }
  outputContext.putImageData(image, 0, 0);
}

function canvasFrame(canvas: HTMLCanvasElement) {
  return new Promise<string>((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error('Safari 無法保存測試影格'));
          return;
        }
        resolve(URL.createObjectURL(blob));
      },
      'image/webp',
      0.86,
    );
  });
}

export default function MatteTestClient() {
  const inputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const selectionCanvasRef = useRef<HTMLCanvasElement>(null);
  const outputCanvasRef = useRef<HTMLCanvasElement>(null);
  const frameCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const segmenterRef = useRef<InteractiveSubjectSegmenter | null>(null);
  const engineRef = useRef<MatAnyone2Engine | null>(null);
  const seedMaskRef = useRef<Float32Array | null>(null);
  const seedTimeRef = useRef(0);
  const cancelRef = useRef(false);
  const videoUrlRef = useRef('');
  const frameUrlsRef = useRef<string[]>([]);
  const replayAnimationRef = useRef(0);

  const [phase, setPhase] = useState<Phase>('choose');
  const [videoName, setVideoName] = useState('');
  const [videoUrl, setVideoUrl] = useState('');
  const [notice, setNotice] = useState('先選擇「鎖定後」輸出的影片');
  const [progress, setProgress] = useState(0);
  const [backend, setBackend] = useState('');
  const [busy, setBusy] = useState(false);
  const [frames, setFrames] = useState<string[]>([]);
  const [replayIndex, setReplayIndex] = useState(0);
  const [averageMs, setAverageMs] = useState<number | null>(null);

  const clearFrames = () => {
    cancelAnimationFrame(replayAnimationRef.current);
    for (const url of frameUrlsRef.current) URL.revokeObjectURL(url);
    frameUrlsRef.current = [];
    setFrames([]);
    setReplayIndex(0);
  };

  useEffect(() => {
    return () => {
      cancelRef.current = true;
      cancelAnimationFrame(replayAnimationRef.current);
      if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current);
      for (const url of frameUrlsRef.current) URL.revokeObjectURL(url);
      segmenterRef.current?.close();
      void engineRef.current?.close();
    };
  }, []);

  const replay = (sourceFrames = frameUrlsRef.current) => {
    if (sourceFrames.length === 0) return;
    cancelAnimationFrame(replayAnimationRef.current);
    const started = performance.now();
    const durationMs = (sourceFrames.length / TEST_FPS) * 1000;
    const tick = (now: number) => {
      const elapsed = now - started;
      const index = Math.min(
        sourceFrames.length - 1,
        Math.floor((elapsed / 1000) * TEST_FPS),
      );
      setReplayIndex(index);
      if (elapsed < durationMs) {
        replayAnimationRef.current = requestAnimationFrame(tick);
      }
    };
    setReplayIndex(0);
    replayAnimationRef.current = requestAnimationFrame(tick);
  };

  const chooseVideo = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    cancelRef.current = true;
    clearFrames();
    seedMaskRef.current = null;
    setAverageMs(null);
    setBackend('');
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current);
    const url = URL.createObjectURL(file);
    videoUrlRef.current = url;
    setVideoUrl(url);
    setVideoName(file.name);
    setPhase('pause');
    setProgress(0);
    setNotice('播放並停在主角輪廓清楚的畫面，再按「使用目前畫面」');
    requestAnimationFrame(async () => {
      const video = videoRef.current;
      if (!video) return;
      try {
        await waitForMetadata(video);
      } catch (error) {
        setNotice(error instanceof Error ? error.message : String(error));
      }
    });
  };

  const captureCurrentFrame = () => {
    const video = videoRef.current;
    const canvas = selectionCanvasRef.current;
    if (!video || !canvas || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      setNotice('影片畫面還沒準備好');
      return;
    }
    video.pause();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    if (!context) return;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    seedTimeRef.current = video.currentTime;
    seedMaskRef.current = null;
    clearFrames();
    setPhase('select');
    setNotice('直接點一下主角的身體；只用這一格建立主角種子');
  };

  const selectSubject = async (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (busy || phase === 'running' || phase === 'loading') return;
    const canvas = selectionCanvasRef.current;
    const output = outputCanvasRef.current;
    if (!canvas || !output) return;
    const rect = canvas.getBoundingClientRect();
    const point = {
      x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
    };
    setBusy(true);
    setNotice('正在建立第一格主角遮罩…');
    try {
      if (!segmenterRef.current) {
        segmenterRef.current = await InteractiveSubjectSegmenter.create(
          assetUrl('mediapipe/'),
          assetUrl('models/magic_touch.tflite'),
          assetUrl('models/deeplab_v3.tflite'),
        );
      }
      const mask = segmenterRef.current.segment(canvas, { keypoint: point });
      const resized = resizeMask(
        mask.values,
        mask.width,
        mask.height,
        MATANYONE2_SIZE.width,
        MATANYONE2_SIZE.height,
      );
      seedMaskRef.current = resized;
      frameCanvasRef.current ??= document.createElement('canvas');
      drawCutout(canvas, resized, output, frameCanvasRef.current);
      const visible = resized.reduce((sum, value) => sum + value, 0) / resized.length;
      setPhase('seeded');
      setNotice(
        visible < 0.01
          ? '遮罩幾乎是空的，請再點一次主角身體'
          : '先看黑底預覽；只有主角就可開始 3 秒延續測試',
      );
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  };

  const runTest = async () => {
    const video = videoRef.current;
    const output = outputCanvasRef.current;
    const seedMask = seedMaskRef.current;
    if (!video || !output || !seedMask) return;
    cancelRef.current = false;
    clearFrames();
    setBusy(true);
    setAverageMs(null);
    const produced: string[] = [];
    try {
      if (!engineRef.current) {
        setPhase('loading');
        setNotice('首次下載約 72 MB 模型；之後瀏覽器可使用快取');
        const { MatAnyone2Engine } = await import('../../lib/matanyone2-engine');
        engineRef.current = await MatAnyone2Engine.create(
          assetUrl('models/matanyone2/'),
          assetUrl('ort/'),
          (status) => {
            setBackend(status.backend === 'webgpu' ? 'WebGPU' : 'WASM');
            setProgress(status.loaded / status.total);
            setNotice(`載入影片遮罩模型 ${status.loaded} / ${status.total}`);
          },
        );
      }
      const engine = engineRef.current;
      setBackend(engine.backend === 'webgpu' ? 'WebGPU' : 'WASM');
      setPhase('running');
      setProgress(0);
      video.pause();
      await seekVideo(video, seedTimeRef.current);
      frameCanvasRef.current ??= document.createElement('canvas');
      setNotice('建立主角時間記憶…');
      const times: number[] = [];
      const seeded = await engine.seed(video, seedMask, 10, (warmup) => {
        setProgress(warmup * 0.08);
      });
      times.push(seeded.inferenceMs);
      drawCutout(video, seeded.alpha, output, frameCanvasRef.current);
      produced.push(await canvasFrame(output));

      const available = Math.min(
        TEST_SECONDS,
        Math.max(0, video.duration - seedTimeRef.current),
      );
      const frameCount = Math.max(2, Math.floor(available * TEST_FPS) + 1);
      for (let index = 1; index < frameCount; index += 1) {
        if (cancelRef.current) throw new Error('測試已取消');
        const time = Math.min(
          video.duration,
          seedTimeRef.current + index / TEST_FPS,
        );
        await seekVideo(video, time);
        const result = await engine.step(video);
        times.push(result.inferenceMs);
        drawCutout(video, result.alpha, output, frameCanvasRef.current);
        produced.push(await canvasFrame(output));
        setProgress(0.08 + (index / Math.max(1, frameCount - 1)) * 0.92);
        setNotice(`延續遮罩 ${index + 1} / ${frameCount} 格`);
      }
      frameUrlsRef.current = produced;
      setFrames(produced);
      const perFrame = times.slice(1);
      setAverageMs(
        perFrame.length
          ? perFrame.reduce((sum, value) => sum + value, 0) / perFrame.length
          : times[0],
      );
      setProgress(1);
      setPhase('ready');
      setNotice('3 秒影片遮罩延續完成；正在重播結果');
      replay(produced);
    } catch (error) {
      for (const url of produced) URL.revokeObjectURL(url);
      setPhase(seedMaskRef.current ? 'seeded' : 'select');
      setNotice(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className={styles.shell}>
      <header className={styles.topbar}>
        <a href="./" className={styles.brand}>NiviTrack</a>
        <span className={styles.safety}>獨立測試 · V8.1 未改</span>
      </header>

      <section className={styles.hero}>
        <p>MATANYONE2 PROTOTYPE</p>
        <h1>影片遮罩延續測試</h1>
        <span>第一格指定主角，之後沿時間追蹤同一個遮罩。這頁只測去背，不鎖定、不加特效。</span>
      </section>

      <section className={styles.steps} aria-label="測試步驟">
        {[
          ['1', '選鎖定影片'],
          ['2', '停格點主角'],
          ['3', '測試 3 秒'],
          ['4', '重播檢查'],
        ].map(([number, label], index) => {
          const order: Phase[] = ['choose', 'pause', 'seeded', 'ready'];
          const activeIndex = phase === 'select' ? 1
            : phase === 'loading' || phase === 'running' ? 2
              : order.indexOf(phase);
          return (
            <div className={index <= activeIndex ? styles.stepActive : styles.step} key={number}>
              <b>{number}</b><span>{label}</span>
            </div>
          );
        })}
      </section>

      <section className={styles.workspace}>
        <div className={styles.stage}>
          {phase === 'choose' ? (
            <button className={styles.picker} onClick={() => inputRef.current?.click()}>
              <strong>選擇鎖定後影片</strong>
              <span>MOV 或 MP4，影片只在本機處理</span>
            </button>
          ) : null}

          {videoUrl ? (
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              playsInline
              preload="metadata"
              className={phase === 'pause' ? styles.video : styles.hidden}
            />
          ) : null}

          <canvas
            ref={selectionCanvasRef}
            className={phase === 'select' ? styles.selection : styles.hidden}
            onPointerUp={selectSubject}
          />

          <div className={phase === 'seeded' || phase === 'loading' || phase === 'running'
            ? styles.outputWrap
            : styles.hidden}
          >
            <canvas ref={outputCanvasRef} className={styles.output} />
            {phase === 'running' ? <span className={styles.live}>逐格處理中</span> : null}
          </div>

          {phase === 'ready' && frames.length ? (
            <div className={styles.outputWrap}>
              {/* Blob URLs contain only locally generated black-background test frames. */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img className={styles.output} src={frames[replayIndex]} alt="三秒去背測試結果" />
              <span className={styles.live}>測試結果</span>
            </div>
          ) : null}
        </div>

        <aside className={styles.controls}>
          <div className={styles.status}>
            <small>目前狀態</small>
            <strong>{notice}</strong>
            <div className={styles.bar}><i style={{ width: `${Math.round(progress * 100)}%` }} /></div>
            <dl>
              <div><dt>影片</dt><dd>{videoName || '尚未選擇'}</dd></div>
              <div>
                <dt>後端</dt>
                <dd>
                  {backend || (
                    typeof navigator !== 'undefined' && 'gpu' in navigator
                      ? '等待本機 WASM'
                      : '等待測試'
                  )}
                </dd>
              </div>
              <div><dt>平均每格</dt><dd>{averageMs === null ? '—' : `${averageMs.toFixed(0)} ms`}</dd></div>
            </dl>
          </div>

          <input
            ref={inputRef}
            className={styles.hidden}
            type="file"
            accept="video/*,.mov,.mp4"
            onChange={chooseVideo}
          />

          {phase === 'pause' ? (
            <button className={styles.primary} onClick={captureCurrentFrame}>
              使用目前畫面
            </button>
          ) : null}
          {phase === 'select' ? (
            <p className={styles.hint}>請點主角身體中央。若遮罩不對，可以直接再點一次。</p>
          ) : null}
          {phase === 'seeded' ? (
            <button className={styles.primary} disabled={busy} onClick={runTest}>
              開始 3 秒遮罩延續
            </button>
          ) : null}
          {phase === 'loading' || phase === 'running' ? (
            <button
              className={styles.secondary}
              onClick={() => {
                cancelRef.current = true;
                setNotice('完成目前這一格後取消');
              }}
            >
              取消
            </button>
          ) : null}
          {phase === 'ready' ? (
            <>
              <button className={styles.primary} onClick={() => replay()}>重播 3 秒結果</button>
              <button
                className={styles.secondary}
                onClick={() => {
                  clearFrames();
                  setPhase('pause');
                  setNotice('可換一個停格位置重新測試');
                }}
              >
                重新選停格
              </button>
            </>
          ) : null}
          {phase !== 'choose' && phase !== 'loading' && phase !== 'running' ? (
            <button className={styles.textButton} onClick={() => inputRef.current?.click()}>
              換一支影片
            </button>
          ) : null}

          <p className={styles.boundary}>
            這個原型不會上傳影片，也不會修改正式 V8.1。模型首次載入約 72 MB，限非商業使用。
          </p>
        </aside>
      </section>
    </main>
  );
}
