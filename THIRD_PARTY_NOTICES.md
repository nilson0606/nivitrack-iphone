# Third-party notices

NiviTrack Stage 1 includes the following third-party components for on-device person segmentation.

## MediaPipe Tasks Vision

- Package: `@mediapipe/tasks-vision` 1.0.1
- Project: <https://github.com/google-ai-edge/mediapipe>
- License: Apache License 2.0
- Local assets:
  - `app/public/mediapipe/vision_wasm_internal.js`
  - `app/public/mediapipe/vision_wasm_internal.wasm`

## MediaPipe Selfie Segmenter model

- Source: <https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite>
- Documentation: <https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter>
- Local asset: `app/public/models/magic_touch.tflite`
- SHA-256: `191AC9529AE506EE0BEEFA6B2C945A172DAB9D07D1E802A290A4E4038226658B`

MediaPipe processes the selected video frames on the user's device. NiviTrack does not send those frames to Google.
