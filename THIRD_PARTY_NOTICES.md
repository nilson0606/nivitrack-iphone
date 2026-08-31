# Third-party notices

NiviTrack 的第 12 項「單一舞者去背」使用下列開源元件：

- `@mediapipe/tasks-vision` 1.0.1 — Copyright The MediaPipe Authors，Apache License 2.0。
- MediaPipe MagicTouch Interactive Segmenter (`magic_touch.tflite`) — Google MediaPipe 所提供的點提示物件分割模型；隨本專案靜態發布，僅在使用者裝置本機推論。
- MediaPipe Selfie Segmenter (`selfie_segmenter.tflite`) — Google MediaPipe 所提供的人像分割模型；隨本專案靜態發布，僅在使用者裝置本機推論。

MediaPipe 專案與授權：<https://github.com/google-ai-edge/mediapipe>

模型來源：

- MagicTouch：<https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite>
- Selfie Segmenter：<https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite>

NiviTrack 的第 13 項「旁人人臉遮罩」另外使用：

- MediaPipe BlazeFace Short Range Face Detector (`blaze_face_short_range.tflite`) — Google MediaPipe 所提供的人臉位置偵測模型；隨本專案靜態發布，只偵測畫面中的人臉位置，不建立人臉身分資料庫。

模型來源：<https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite>

本專案未將 MediaPipe、模型、使用者影片或人臉資料傳送到付費 API。首次下載靜態資產後，影片處理在瀏覽器內完成。
