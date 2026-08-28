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

Export generation also checks the tracked dancer in upper, middle, and lower
body bands. A band that suddenly collapses receives one short, time-based
50 ms release from the most recent reliable mask. Persistent changes are then
accepted immediately to avoid a delayed silhouette during fast movement.

Before playback or export, masks are solidified once to keep the current
subject opaque, then inset by an adjustable 0-6 pixel black safety edge. The
50 ms retained mask is also clipped to the current subject neighborhood so it
cannot reveal a distant patch of the current background. Styling mutates each
stored mask in place and uses one reusable distance buffer, preserving the
bounded timeline memory design.

- Source revision: https://huggingface.co/Xenova/modnet/tree/fa2fa546052fba4c08921230a26cc69a333fca12
- Upstream project: https://github.com/ZHKKKe/MODNet
- License: Apache License 2.0
- Size: `6,632,188` bytes
- SHA-256: `92E49898C3E05A6D7A944FC67A8CB87C4AAD754FFB6EBD949528C7D1105FEE3A`

# Bundled MediaPipe MagicTouch experiment

`interactive_segmentation.task` is Google's official int8 MagicTouch v2 task
bundle used only by the isolated 3-second selected-object experiment. It
contains separate encoder and decoder TFLite models. The tracking box produces
positive torso strokes and negative surrounding points automatically; the user
does not draw a brush stroke.

The experiment runs at 5 mask frames per second in a dedicated Web Worker,
closes every returned `MPMask`, terminates the worker before playback, and is
not connected to formal export.

- Source: https://storage.googleapis.com/mediapipe-models/interactive_segmenter_v2/magic_touch/int8/1/interactive_segmentation.task
- Official Web reference: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/vision/README.md
- MediaPipe Tasks license: Apache License 2.0
- Size: `30,525,312` bytes
- SHA-256: `38431BC66B883404E8397F74C3579404315B9B52B04A46C6346FE906A7309B03`
