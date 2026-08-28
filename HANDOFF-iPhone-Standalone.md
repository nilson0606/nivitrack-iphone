# NiviTrack iPhone 獨立執行版 — Handoff

## 1. 使用者真正要的結果

將目前的 NiviTrack 改造成可在 **iPhone 上直接執行** 的版本。

「直接執行」的定義：

- 選片、主角指定、ViT 追蹤、置中／縮放與影片輸出都在 iPhone 完成。
- 實際使用時只依賴 iPhone Safari／主畫面 Web App，不需要任何外部處理服務。
- 不把影片上傳到雲端運算。
- 可以從 iPhone「照片」或「檔案」選擇影片。
- 必須直接讀取並處理 iPhone 原始 MOV／HEVC；不得要求使用者先轉成 H.264。
- 完成影片可儲存到「檔案」、分享或加入照片；所有輸出都在 iPhone 內產生。
- 預設本機母片輸出為 iPhone 原生取向的 MOV／HEVC＋AAC。
- 只有使用者選擇「相容分享」時，才另外輸出 MP4／H.264＋AAC。
- 產品型態是可安裝到主畫面的 Web App／PWA；任何能力限制都必須用真實 iPhone 實測說明，不可暗中改用雲端後端。

使用者接下來會在新 Codex 對話繼續此工作。

## 2. iPhone 版唯一來源

本文件只描述目前 iPhone Web App；不包含或維護其他平台版本。

- `app/app/page.tsx`：手機操作流程與狀態。
- `app/lib/vit-tracker.ts` 與 `app/public/models/vittrack.onnx`：必要的主角追蹤。
- `app/lib/video-export.ts`：置中、縮放、音訊保留與影片輸出。
- `app/lib/background-removal.ts` 與 `app/lib/edgetam-onnx-segmenter.ts`：追蹤完成後可選擇的本機去背。
- `app/public/manual.html`：手機使用手冊。

目前產品流程：選片、停格、框選主角、完整 ViT 追蹤，再由使用者選擇保留原始背景或去背黑底。ViT 不可跳過，去背可以跳過。

## 3. 已驗證的重要行為

曾以一支 HEVC、1920×1080、30 FPS、約 15.33 秒的本機影片驗證；影片只供分析，不得形成特定物品、人物位置或檔名規則。

既有 ViT 追蹤測試結果：

- 460 幀零失敗。
- 平均 score 約 0.7155，最後 score 約 0.6322。
- 最後三秒仍是原先選定的女性主角。
- 輸出為 H.264 1080×1920、30 FPS、約 15.33 秒，含 AAC 音訊。

目前以 `app/lib/vit-tracker.ts` 為唯一追蹤實作；不可替換成會自動換主角的追蹤器。

目前低記憶體去背候選版測試結果：

- 從使用者指定的停格影格向影片前後雙向延續遮罩。
- 第一版在真實 iPhone 的 3 秒測試會讓 Safari 重新載入頁面，原因判定為模型初始化資源峰值，已從公開版撤下。
- 第二版改成首幀 Session 釋放後才建立連續 Session，模型網址直接交給 ONNX Runtime，不在 JavaScript 組合雙份模型資料。
- 完整 ViT 後立即釋放 ViT 每幀張量、Session 與 SSDLite 模型，只保留數字形式的 `trackPath`。
- 230 幀全部完成，平均推論約 72 ms／幀。
- 遮罩前景面積為 431～911，最大相鄰變化約 19%，未出現整片背景突然被選中的情況。
- 預覽後重用主角起始狀態的下一幀遮罩面積差為 0。
- 正式 UI 本機流程無例外、無下載失敗；可讀取的 JavaScript heap 約由 4.6 MB 上升至最高 11.6 MB。此數字不包含瀏覽器內部與 GPU 記憶體。
- 以上是本機 WebGPU 驗證數據；在正式發布前仍只允許一次真實 iPhone Safari 純載入驗收。

## 4. 絕對不能做的事

- 不得依影片時間、人物位置或衣服顏色寫死規則。
- 不得只針對單一測試影片或某一種背景物品修補。
- 不得自動改追畫面中最大的人。
- 不得把 SSD 漏檢誤認為 ViT 追蹤失敗。
- 不得在開始追蹤前把整支 MOV 預先轉成 H.264；追蹤應直接取得 iPhone 原生解碼後的影格。
- 不得把 H.264 當成唯一或預設本機輸出；H.264 是相容分享用格式。
- 不得修改或刪除目前 iPhone 專案以外的任何內容。
- 不得未經使用者同意發布網站、建立雲端影片儲存或上傳影片。
- 不得宣稱「iPhone 可用」卻仍要求外部裝置協助處理。

## 5. iPhone 專案邊界

所有新建與修改都必須留在本 iPhone 專案儲存庫。其他專案不屬於交付範圍，也不可編輯或刪除。

## 6. 第一階段：先做有證據的技術可行性研究

這些能力在 iOS Safari 上會隨版本改變，必須查閱最新的一手官方資料並實測，不要憑印象：

1. iPhone Safari／加入主畫面的 Web App 對 WebGPU、WebAssembly SIMD、Web Workers、OffscreenCanvas 的支援。
2. ONNX Runtime Web 在 iOS Safari 可用的 execution provider：WebGPU、WebGL 或 WASM。
3. Safari 對 WebCodecs、VideoFrame、VideoEncoder、AudioEncoder、MediaRecorder 與 MP4 輸出的實際支援。
4. 從 Photos 選取 MOV／HEVC 後，HTMLVideoElement 是否可可靠解碼與逐幀 seek。
5. PWA Service Worker、模型離線快取、單檔與總快取容量限制。
6. 長影片在 iPhone 的記憶體、溫度、背景／鎖屏暫停與檔案輸出限制。

優先使用 Apple WebKit、Apple Developer、ONNX Runtime、TensorFlow.js 等官方文件。技術支援矩陣要記錄 iOS 版本、iPhone 機型與測試日期。

研究完成後先寫一份簡短決策：

- 純 Web／PWA 是否可完成「影片選取 → ViT → MP4」。
- 哪些步驟能離線執行。
- 預估可接受的最高輸入解析度、輸出解析度與影片長度。
- 若有阻塞，阻塞是 API 不存在、模型不相容、速度不足、記憶體不足，還是音訊／MP4 封裝問題。

## 7. 第二階段：最小可行原型

不要一開始就移植全部 UI。先完成一個可量測的 iPhone 原型：

1. 從 Photos／Files 選一支短影片。
2. 直接載入 MOV／HEVC 並顯示影片畫面與時間軸，不做輸入預轉檔。
3. 執行現有 SSDLite／COCO-SSD 初始辨識。
4. 保留手指拖曳 bbox 的後備選取方式。
5. 對 2～5 秒片段執行 ViT 追蹤。
6. 在畫面上畫出每幀 bbox 與 score。
7. 顯示 FPS、每幀推論時間、記憶體錯誤及取消按鈕。
8. 以固定影格序列檢查相同輸入是否得到穩定、可重現的追蹤路徑。

只有這個原型在真實 iPhone 上證明不會換人，才進入完整輸出。

## 8. ViT 移植的核心工作

目前 ONNX 檔不能只用 ONNX Runtime Web 呼叫一次就算完成。必須重現 OpenCV TrackerVit 的整套邏輯：

- 初始 template crop。
- 每幀 search region crop。
- resize、padding、色彩與 normalization。
- ONNX 輸入／輸出 tensor 名稱、shape 與資料排列。
- 模型輸出轉 bbox。
- bbox clip、尺度更新、template／state 是否更新。
- score 的解析與失敗處理。

建立 JavaScript 固定 frame-by-frame 測試，檢查 tensor、bbox、score 與每幀結果；允許極小浮點差異，但不能只靠肉眼說相似。

## 9. 完整 iPhone 版必須保留的產品行為

- 任意人物或寵物都能由使用者指定。
- iPhone 拍攝的 MOV／HEVC 可直接進入辨識與追蹤，不先轉 H.264。
- 有 SSD 框可點；沒框可用手指拖曳。
- 不自動換主角。
- 支援 9:16、1:1、16:9。
- 追蹤柔順度與主角大小可調。
- 預覽與輸出使用同一條追蹤路徑。
- 輸出時間、FPS 與音訊長度要和來源一致。
- 提供兩個清楚分開的輸出動作：
  - 「儲存到 iPhone」：MOV／HEVC＋AAC，作為本機高效率母片。
  - 「相容分享」：MP4／H.264＋AAC，供跨平台或社群分享。
- 兩種輸出共用同一條 ViT 追蹤路徑，不得重新跑追蹤。
- 若同時產生兩種格式，應直接從原始影格與追蹤路徑分別編碼，避免把已壓縮的 MOV 再轉成 H.264 造成額外畫質損失。
- 提供真實進度、取消與錯誤訊息，不可長時間假停在 0%。
- 處理完成後可透過 iOS Share Sheet、Files 或 Photos 保存。
- 模型與程式可在安裝後離線使用；影片不得上傳。

## 10. 影片解碼與輸出策略

請用實測決定，不要先假定 FFmpeg WASM 一定可行。

輸入與輸出必須分開看待：

- **輸入：** iPhone 本身原生支援 MOV／HEVC，應直接解碼原始檔並取得影格給 SSD 與 ViT，不先轉 H.264。
- **本機輸出：** 追蹤後畫面已重新裁切與置中，因此需要重新編碼；預設使用 MOV／HEVC＋AAC 留在 iPhone，兼顧畫質與容量。
- **分享輸出：** 只有使用者按「相容分享」時，才使用同一條追蹤路徑另行產生 MP4／H.264＋AAC。
- 若系統 Share Sheet 因分享目的地而自動做相容性轉換，屬於 iOS 的輸出分享行為，不是 NiviTrack 的輸入轉檔。

優先評估：

1. Safari 原生直接解碼 MOV／HEVC + Canvas／VideoFrame 逐幀讀取。
2. WebCodecs／MediaRecorder 原生硬體編碼（若目前 iOS 支援足夠）。
3. MP4 muxer 只負責封裝，避免整支影片同時放在記憶體。
4. 音訊盡量直接 remux；無法 remux 才重編碼。
5. 使用 chunk／stream 或分段處理，避免 1080p 長影片造成記憶體崩潰。

若 Safari 無法可靠輸出指定格式，必須清楚顯示限制與可用的本機相容格式，不可改走遠端後端。

## 11. PWA 與隱私

- PWA 若需安裝與 Service Worker，正式使用通常需要 HTTPS。
- 已決定使用每月費用為 0 的 GitHub Pages：採 GitHub Free 公開 repository、GitHub 提供的免費 `*.github.io` 網址，不購買自訂網域。
- GitHub Pages 只提供 HTML、CSS、JavaScript、WASM、ONNX 模型、Web App Manifest 與 Service Worker；不得提供影片處理 API。
- 每台 iPhone 從同一網址各自加入主畫面並快取程式與模型；程式可移到不同 iPhone 使用，但不處理跨手機的影片或多媒體轉移。
- 影片選取、辨識、ViT 追蹤、編碼與輸出仍必須完全留在當下使用的 iPhone。
- 未經使用者同意不得部署。
- 若日後部署，頁面要明確顯示「影片不會上傳」且程式中不得存在影片上傳端點。
- 快取要有版本管理，避免更新模型後舊快取造成不一致。

## 12. 驗收標準

至少測試：

- 一支 iPhone MOV／HEVC。
- 確認該 MOV／HEVC 從選取到開始追蹤之間沒有完整預轉檔步驟。
- 一支 H.264 MP4。
- 同一支多人影片選兩位不同主角。
- 一位沒有 SSD 辨識框、只能手動框選的後排人物。
- 人物交叉、短暫遮擋與最後兩秒。
- 直式 9:16 與橫式 16:9 輸出。
- 輸出可在 iPhone Photos／Files 正常播放，影像流暢且有聲音。
- MOV／HEVC 本機母片與 MP4／H.264 分享版的時間、FPS、音訊同步及主角構圖一致。

驗收時必須記錄：

- iPhone 型號與 iOS 版本。
- 原始影片 codec、解析度、FPS、時長。
- 追蹤耗時、平均推論 FPS、最高記憶體問題。
- 輸出 codec、解析度、FPS、時長與音訊。
- 是否從頭到尾保持同一主角。

## 13. 目前版本與下一個行動

- `stable-no-effects-v1` 是使用者已確認的無特效穩定基準，不得破壞。
- 公開版已恢復無特效穩定流程並保留 Safari 快取修正。
- 低記憶體去背候選工作位於 `codex/background-removal-low-memory-v2`，尚未取代公開版。
- 正式流程固定為：選片 → 停格 → 框主角 → 完整 ViT 追蹤 → 選擇原始背景或去背黑底。
- ViT 追蹤不可跳過；去背可以跳過。選原始背景時必須走原本穩定輸出路徑。
- 去背只在使用者選擇後才載入 EdgeTAM；先做 3 秒預覽，通過後才允許完整去背輸出。
- 未經使用者同意不得部署。若一次真實 iPhone 純載入驗收仍讓 Safari 重新載入，立即停止 EdgeTAM 路線，不再反覆調整後要求重測。

## 14. 使用者偏好

- 使用繁體中文溝通。
- 直接指出真正問題，不要用漂亮話掩蓋限制。
- 修改要通用，不能只救單一影片。
- 優先追蹤準確、影片流暢與畫質，再考慮速度。
- 使用者會更換不同主角測試。
