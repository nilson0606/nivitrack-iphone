# Third-party notices

NiviTrack 的第 12 項「單一舞者去背」使用下列開源元件：

- `@mediapipe/tasks-vision` 1.0.1 — Copyright The MediaPipe Authors，Apache License 2.0。
- MediaPipe Pose Landmarker Lite (`pose_landmarker_lite.task`) — Google MediaPipe 所提供的姿勢、人物實例遮罩模型；隨本專案靜態發布，僅在使用者裝置本機推論。
- MediaPipe Selfie Segmenter (`selfie_segmenter.tflite`) — Google MediaPipe 所提供的人像分割模型；隨本專案靜態發布，僅在使用者裝置本機推論。

MediaPipe 專案與授權：<https://github.com/google-ai-edge/mediapipe>

Pose 模型來源：<https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task>

Selfie Segmenter 模型來源：<https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite>

本專案未將 MediaPipe、模型或使用者影片傳送到付費 API。首次下載靜態資產後，影片處理在瀏覽器內完成。
