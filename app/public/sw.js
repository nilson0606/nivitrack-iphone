const VERSION = 'nivitrack-prototype-v1';
const APP_CACHE = VERSION + '-app';
const MODEL_CACHE = VERSION + '-models';

const appAsset = (path) => new URL(path, self.registration.scope).href;
const APP_SHELL = [
  appAsset('./'),
  appAsset('manifest.webmanifest'),
  appAsset('og.png'),
];
const MODEL_ASSETS = [
  appAsset('models/vittrack.onnx'),
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
    Promise.all([
      caches.open(APP_CACHE).then((cache) => cache.addAll(APP_SHELL)),
      caches.open(MODEL_CACHE).then((cache) => cache.addAll(MODEL_ASSETS)),
    ]).then(() => self.skipWaiting()),
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
