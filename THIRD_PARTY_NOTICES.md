# Third-party notices

NiviTrack Stage 1 includes the following third-party components for on-device person segmentation.

## EdgeTAM Video

- Original project: <https://github.com/facebookresearch/EdgeTAM>
- Browser model source: <https://huggingface.co/yonigozlan/EdgeTAM-hf>
- LiteRT conversion reference: <https://github.com/john-rocky/LiteRT-Models/tree/main/edgetam-video>
- Original model license: Apache License 2.0
- Purpose: video object segmentation with rolling spatial memory and object-pointer memory for the user-selected protagonist.
- Local fp16 graph assets:
  - `app/public/models/edgetam-video/encode.tflite` — SHA-256 `2E38353F85AD2A24D4FA3F3B9DDDEECA7A3574C414A3491C379A897AD0D58810`
  - `app/public/models/edgetam-video/memcond.tflite` — SHA-256 `280CF7269DCEF2B0BD2B6278F714F960C929C2540E95056C290E8CDF01958C52`
  - `app/public/models/edgetam-video/decode_box.tflite` — SHA-256 `3C01AC9CA03FC8129B0C91C7D97E0757AC0912F6515B385B03C4A31EACE76363`
  - `app/public/models/edgetam-video/decode.tflite` — SHA-256 `0F17571EE724FAB5DF208DF3872FF71249EF7D153AF87C795B835049AA2AB54A`
  - `app/public/models/edgetam-video/memorize.tflite` — SHA-256 `FE2C9D0EB2CFA7C2CAB2BBF65D90AE4AEE3332C9C54D46469667EE4E23C097E6`
- The `.bin` files in the same directory are model constants exported from the same EdgeTAM weights. `box_prompt.bin` SHA-256: `A97634DB16B23282C15A375794B717EEB7351C9D156AE63657F78BE7145640A7`.

## LiteRT.js

- Package: `@litertjs/core` 2.5.3
- Project: <https://github.com/google-ai-edge/LiteRT/tree/main/litert/js>
- License: Apache License 2.0
- Purpose: execute the EdgeTAM `.tflite` graphs locally in Safari using WebGPU, with WebAssembly/XNNPACK fallback.

## MediaPipe Tasks Vision

- Package: `@mediapipe/tasks-vision` 1.0.1
- Project: <https://github.com/google-ai-edge/mediapipe>
- License: Apache License 2.0
- Local assets:
  - `app/public/mediapipe/vision_wasm_internal.js`
  - `app/public/mediapipe/vision_wasm_internal.wasm`
- Status: retained as a legacy asset; the EdgeTAM candidate pipeline does not use it for Stage 1 masks.

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

## MediaPipe Pose Landmarker Lite model

- Source: <https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task>
- Documentation: <https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker>
- Local asset: `app/public/models/pose_landmarker_lite.task`
- SHA-256: `59929E1D1EE95287735DDD833B19CF4AC46D29BC7AFDDBBF6753C459690D574A`
- Status: retained for reproducibility of the earlier V8.1 pipeline; the EdgeTAM candidate does not combine it with the final mask.

MediaPipe processes the selected video frames on the user's device. NiviTrack does not send those frames to Google.
