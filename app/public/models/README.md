# Bundled MediaPipe pose model

`pose_landmarker_lite.task` is the official MediaPipe Pose Landmarker Lite
float16 model bundle used as a single-person body prior for background removal.

- Source: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
- Model card: https://developers.google.com/static/ml-kit/images/vision/pose-detection/pose_model_card.pdf
- License: Apache License 2.0 (see the model card)
- SHA-256: `59929E1D1EE95287735DDD833B19CF4AC46D29BC7AFDDBBF6753C459690D574A`

The model is served from this WebApp and inference runs locally in the browser.

# Bundled MODNet preview model

`modnet_quantized.onnx` is the quantized MODNet portrait-matting model used to
precompute bounded background-removal timelines for preview and v10 export. The
WebApp runs it locally with ONNX Runtime Web at 384 x 384, one frame at a time,
downsamples stored timeline masks to 256 x 256, and releases the session before
playback or media encoding.

- Source revision: https://huggingface.co/Xenova/modnet/tree/fa2fa546052fba4c08921230a26cc69a333fca12
- Upstream project: https://github.com/ZHKKKe/MODNet
- License: Apache License 2.0
- Size: `6,632,188` bytes
- SHA-256: `92E49898C3E05A6D7A944FC67A8CB87C4AAD754FFB6EBD949528C7D1105FEE3A`
