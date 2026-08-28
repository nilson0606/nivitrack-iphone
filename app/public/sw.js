const VERSION = 'nivitrack-stable-recovery-v1';
const APP_CACHE = VERSION + '-app';
const MODEL_CACHE = VERSION + '-models';

const appAsset = (path) => new URL(path, self.registration.scope).href;
const APP_ENTRY = appAsset('./?app=' + VERSION);
const APP_SHELL = [
  APP_ENTRY,
  appAsset('manifest.webmanifest'),
  appAsset('og.png'),
  appAsset('manual.html'),
];
const MODEL_ASSETS = [
  appAsset('models/vittrack.onnx'),
  appAsset('models/magic_touch.tflite'),
  appAsset('models/pose_landmarker_full.task'),
  appAsset('mediapipe/vision_wasm_internal.js'),
  appAsset('mediapipe/vision_wasm_internal.wasm'),
  appAsset('models/ssdlite_mobilenet_v2/model.json'),
  appAsset('models/ssdlite_mobilenet_v2/group1-shard1of5'),
  appAsset('models/ssdlite_mobilenet_v2/group1-shard2of5'),
  appAsset('models/ssdlite_mobilenet_v2/group1-shard3of5'),
  appAsset('models/ssdlite_mobilenet_v2/group1-shard4of5'),
  appAsset('models/ssdlite_mobilenet_v2/group1-shard5of5'),
  appAsset('ort/ort-wasm-simd-threaded.mjs'),
  appAsset('ort/ort-wasm-simd-threaded.wasm'),
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(APP_CACHE)
      .then((cache) => cache.addAll(APP_SHELL))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key.startsWith('nivitrack-') && key !== APP_CACHE && key !== MODEL_CACHE)
            .map((key) => caches.delete(key)),
        ),
      )
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .then((response) => {
          if (response && response.ok && response.type === 'basic') {
            const copy = response.clone();
            caches.open(APP_CACHE).then((cache) => cache.put(request, copy));
          }
          return response;
        })
        .catch(async () => {
          const cached = await caches.match(request);
          return cached ?? (await caches.match(APP_ENTRY)) ?? Response.error();
        }),
    );
    return;
  }

  event.respondWith(
    caches.match(request).then((cached) => {
      if (cached) return cached;
      return fetch(request).then((response) => {
        if (!response || !response.ok || response.type !== 'basic') return response;
        const copy = response.clone();
        caches.open(APP_CACHE).then((cache) => cache.put(request, copy));
        return response;
      });
    }),
  );
});
