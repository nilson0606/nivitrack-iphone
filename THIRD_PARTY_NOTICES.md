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

## MatAnyone2 model weights

- Project: <https://github.com/pq-yang/MatAnyone2>
- Source revision: `0079197acd6d16a741f71558809c06c586c579e0`
- License: NTU S-Lab License 1.0, non-commercial use only
- Conversion: the original MatAnyone2 weights were split into seven fixed-size FP16 ONNX stages for local browser inference.
- Local assets: `app/public/models/matanyone2/*.onnx`
- License copy: `app/public/models/matanyone2/MODEL-LICENSE.txt`

The MatAnyone2 test processes video frames locally. The model files are downloaded by the browser only when the separate test page is opened.

## MatAnyone2Kit implementation reference

- Project: <https://github.com/flowtyone/MatAnyone2Kit>
- Reference revision: `d85a029870b5149af6aa122cbc13c90db11e5b35`
- License: GNU GPL 3.0 for the Swift package source; bundled model weights remain under the NTU S-Lab License 1.0.

MatAnyone2Kit's six-stage Core ML decomposition was used as an implementation reference for the browser-stage decomposition. Its Swift and Core ML files are not bundled in NiviTrack.
