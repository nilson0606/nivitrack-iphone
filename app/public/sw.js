const VERSION = 'nivitrack-13-tools-v32-crop-redetect-guidance';
const APP_CACHE = VERSION + '-app';
// 第 13 功能只改裁切與座標流程；沿用 V30 模型快取，避免 iPhone 重新下載模型。
const MODEL_CACHE = 'nivitrack-13-tools-v30-body-identity-privacy-effects-models';

const appAsset = (path) => new URL(path, self.registration.scope).href;
const APP_ENTRY = appAsset('./?app=' + VERSION);
const APP_SHELL = [
  APP_ENTRY,
  appAsset('manifest.webmanifest'),
  appAsset('og.png'),
];
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(APP_CACHE).then((cache) => cache.addAll(APP_SHELL)).then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            // Keep older app caches while already-open pages still reference their
            // hashed CSS/JS files. Removing them during activation can leave Safari
            // displaying only the unstyled server-rendered HTML after a reload.
            .filter((key) => key.startsWith('nivitrack-') && key.endsWith('-models') && key !== MODEL_CACHE)
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
      // GitHub Pages replaces hashed assets on every deployment. Always bypass the
      // HTTP cache for HTML so a stale document never points at deleted asset names.
      fetch(request, { cache: 'no-store' })
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
        const assetCache = url.pathname.includes('/models/') || url.pathname.includes('/ort/') || url.pathname.includes('/mediapipe/')
          ? MODEL_CACHE
          : APP_CACHE;
        caches.open(assetCache).then((cache) => cache.put(request, copy));
        return response;
      });
    }),
  );
});
