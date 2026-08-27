# Third-party notices

NiviTrack Stage 1 includes the following third-party components for on-device person segmentation.

## MediaPipe Tasks Vision

- Package: `@mediapipe/tasks-vision` 1.0.1
- Project: <https://github.com/google-ai-edge/mediapipe>
- License: Apache License 2.0
- Local assets:
  - `app/public/mediapipe/vision_wasm_internal.js`
  - `app/public/mediapipe/vision_wasm_internal.wasm`

## MediaPipe MagicTouch Interactive Segmenter model

- Source: <https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite>
- Documentation: <https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter>
- Local asset: `app/public/models/magic_touch.tflite`
- SHA-256: `E24338A717C1B7AD8D159666677EF400BABB7F33B8AD60C4D96DB4ECF694CD25`

## MediaPipe DeepLab-V3 model

- Source: <https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/latest/deeplab_v3.tflite>
- Documentation: <https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter>
- Local asset: `app/public/models/deeplab_v3.tflite`
- SHA-256: `FF36E24D40547FE9E645E2F4E8745D1876D6E38B332D39A82F0BF0F5D1D561B3`

MediaPipe processes the selected video frames on the user's device. NiviTrack does not send those frames to Google.
