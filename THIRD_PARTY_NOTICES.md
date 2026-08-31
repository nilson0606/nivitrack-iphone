# Third-party notices

NiviTrack 的第 12 項「單一舞者去背」使用下列開源元件：

- `@mediapipe/tasks-vision` 1.0.1 — Copyright The MediaPipe Authors，Apache License 2.0。
- MediaPipe MagicTouch Interactive Segmenter (`magic_touch.tflite`) — Google MediaPipe 所提供的點提示物件分割模型；隨本專案靜態發布，僅在使用者裝置本機推論。
- MediaPipe Selfie Segmenter (`selfie_segmenter.tflite`) — Google MediaPipe 所提供的人像分割模型；隨本專案靜態發布，僅在使用者裝置本機推論。

MediaPipe 專案與授權：<https://github.com/google-ai-edge/mediapipe>

模型來源：

- MagicTouch：<https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite>
- Selfie Segmenter：<https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite>

NiviTrack 的自動選角使用 `@tensorflow-models/coco-ssd` / SSDLite MobileNet V2 尋找人物候選位置。

NiviTrack 的第 13 項「旁人人臉遮罩」另外使用：

- PINTO Model Zoo YOLOX‑Nano Body／Head／Hand (`yolox_n_body_head_hand_256x320.onnx`) — Copyright 2023 Katsuya Hyodo，Apache License 2.0。第 13 項只讀取 `head` 類別，在裝置本機尋找整顆人頭；模型輸入為 256×320。

模型來源：

- YOLOX Body／Head／Hand：<https://github.com/PINTO0309/PINTO_model_zoo/tree/main/426_YOLOX-Body-Head-Hand>
- YOLOX 模型授權：<https://github.com/PINTO0309/PINTO_model_zoo/blob/main/426_YOLOX-Body-Head-Hand/LICENSE>

本專案未將 MediaPipe、模型、使用者影片或人臉資料傳送到付費 API。首次下載靜態資產後，影片處理在瀏覽器內完成。
