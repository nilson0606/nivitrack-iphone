# NiviTrack iPhone PWA 技術可行性決策

更新日期：2026-08-27

## 決策摘要

目前繼續採用純 Web App／PWA 原型，不轉向電腦後端，也暫時不轉向原生 Swift。

已確認可在 Web 路線繼續實作：

- 從 iPhone「照片」或「檔案」選取影片。
- 以 HTMLVideoElement 直接載入 MOV／HEVC，不先完整轉成 H.264。
- 以 Canvas 取得目前影格，執行 SSDLite 初始辨識及手指 bbox 框選。
- 使用 ONNX Runtime Web 的 WASM execution provider 執行現有 ViT 模型。
- 將程式、WASM 與模型快取後離線使用。
- 以 GitHub Pages 免費公開網站提供程式；影片不會上傳。

尚未證明、不能宣稱完成：

- 真實 iPhone 上對 MOV／HEVC 的長時間逐幀 seek 是否精確穩定。
- 1080p 長影片的速度、記憶體、溫度及鎖屏／切背景行為。
- MOV／HEVC＋AAC 母片輸出。
- MP4／H.264＋AAC 的完整音訊同步與長片封裝。
- JavaScript ViT 路徑和 OpenCV TrackerVit 的逐幀數值一致性。

## 官方支援判斷

### AI 推論

- Safari 26 已在 iOS、iPadOS 與 macOS 正式加入 WebGPU。
- ONNX Runtime Web 目前的官方瀏覽器矩陣仍將 Safari iOS 的 WebGPU execution provider 列為不支援；WASM 與 WebGL 列為支援。
- 因此目前可靠基線是單執行緒 WASM。WebGPU 只在實機做能力探測，不作為完成條件。
- WebGL 已進入維護模式，不作為新的主要方案。

### 影片解碼

- Safari 16.4 起已提供視訊 WebCodecs。
- Safari 17.4 擴充 WebCodecs 的 HEVC 支援。
- WebKit 文件確認 Apple 裝置可硬體解碼 H.264、HEVC、VP9，部分裝置也可處理 AV1。
- 這代表直接讀取 iPhone MOV／HEVC 有技術基礎，但 Photos 選片、QuickTime MOV 容器與逐幀 seek 仍需真機實測。

### 影片輸出

- WebKit 官方確認 MediaRecorder 可產生 MP4／H.264＋AAC。
- 這支持「相容分享」原型，但仍需驗證 Canvas 影格、來源音訊、時長與 FPS 同步。
- 官方資料目前不足以證明 Safari Web App 能可靠產生 MOV／HEVC＋AAC。母片輸出仍是純 Web 最大阻塞點。
- 若真機證明 HEVC 編碼、MOV 封裝或長片記憶體無法達標，正式替代方案是 Swift＋AVFoundation＋Core ML，不改用電腦或雲端影片後端。

### 安裝與離線

- iPhone 可將網站加入主畫面並以 Web App 模式開啟。
- Service Worker 可快取 App shell、ONNX Runtime WASM 與模型。
- 每支 iPhone 從同一 GitHub Pages 網址各自安裝；不搬移影片或本機工作資料。
- GitHub Pages 只託管靜態程式，沒有影片上傳端點。

## 2026-08-27 真實 iPhone 里程碑

使用者已在真實 iPhone Safari 完成：

- 從 GitHub Pages 開啟 PWA，修正 base path 後所有 JavaScript 資產可正常載入。
- 從 iPhone 選取並播放 MOV。
- 在影片畫面用手指框選主角。
- 完成 3 秒、10 FPS 的 ViT 路徑測試。
- 將 NiviTrack 加入主畫面並以獨立 Web App 開啟。

尚未記錄 iPhone 型號、iOS 版本、來源 codec／解析度與量測數值，因此不能把這次結果外推為所有 iPhone 均已驗收。

## 目前完整輸出實作

- 以選角時間點為錨點，向影片結尾追蹤，再反向補齊選角前片段。
- 追蹤路徑以 10 FPS 取樣；輸出時依影片播放時間插值並做雙向平滑。
- 支援 9:16、1:1、16:9，主角大小與置中柔順度可調。
- 以同一條 ViT 路徑重播來源影片，不重新跑追蹤。
- 以 Canvas 產生 720×1280、720×720 或 1280×720 畫面。
- 來源音訊經 Web Audio 接入同一 MediaStream。
- Safari MediaRecorder 以功能探測選擇 H.264／AAC MP4；若實機回報 HEVC MIME 可用，亦提供 HEVC／AAC MP4。
- 完成 Blob 可透過 iOS Share Sheet 分享或儲存到「檔案」。

## 目前仍未完成的驗收

- 完整影片追蹤、平滑構圖、H.264／AAC MP4 的真機端到端輸出。
- 輸出影片是否有聲、時長與來源一致、FPS 流暢且音訊同步。
- HEVC 網頁編碼的實機支援與穩定度。
- QuickTime MOV 容器輸出。現行純 Web 實作不把 HEVC MP4 假稱為 MOV。
- 長片記憶體、溫度、鎖屏／切背景及取消後的資源釋放。
- 10 FPS 路徑對快速運動、交叉與遮擋的品質；必要時再提高追蹤取樣率。

若 MOV／HEVC 母片是硬性驗收條件，而 Safari 實機不能可靠提供 HEVC 編碼與 MOV 封裝，正式替代方案仍是 Swift＋AVFoundation＋Core ML，不使用電腦或雲端後端。

## 下一個實機測試

至少記錄：

- iPhone 型號、iOS 與 Safari 版本、測試日期。
- MOV／HEVC 是否可直接選取、播放、seek 與畫入 Canvas。
- SSDLite 載入時間與候選框結果。
- ViT 平均推論毫秒／幀、平均 score、失敗幀與是否換人。
- 2～5 秒片段是否能完成，不發生記憶體終止。
- VideoEncoder、MediaRecorder H.264/AAC、HEVC encoder 的 isConfigSupported 實測結果。

## 一手資料

- WebKit — Safari 26 WebGPU：https://webkit.org/blog/16993/news-from-wwdc25-web-technology-coming-this-fall-in-safari-26-beta/
- ONNX Runtime Web 支援矩陣：https://onnxruntime.ai/docs/get-started/with-javascript/web.html
- WebKit — Safari 17.4 HEVC WebCodecs：https://webkit.org/blog/15063/webkit-features-in-safari-17-4/
- WebKit — MediaRecorder MP4／H.264／AAC：https://webkit.org/blog/11353/mediarecorder-api/
- Apple — 將網站加入 iPhone 主畫面：https://support.apple.com/guide/iphone/iph42ab2f3a7/ios
- OpenCV TrackerVit 官方來源：https://github.com/opencv/opencv/blob/4.x/modules/video/src/tracking/tracker_vit.cpp
- GitHub Pages 官方限制：https://docs.github.com/pages/getting-started-with-github-pages/github-pages-limits
