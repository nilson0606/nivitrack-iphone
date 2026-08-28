# Third-party notices

NiviTrack includes or derives from the following open-source software and model work:

- **EdgeTAM** — Meta Platforms, Inc.; Apache License 2.0. The ONNX files in
  `app/public/models/edgetam-onnx/` are locally converted FP16 graphs derived
  from the EdgeTAM video object-segmentation pipeline.
  Source: <https://github.com/facebookresearch/EdgeTAM>
- **LiteRT-Models EdgeTAM Video conversion reference** — used to preserve the
  EdgeTAM/SAM2 rolling-memory graph contract during ONNX export.
  Source: <https://github.com/john-rocky/LiteRT-Models>
- **ONNX Runtime Web** — Microsoft Corporation; MIT License.
  Source: <https://github.com/microsoft/onnxruntime>

Model integrity:

- `start.onnx`: `13e7978ab8d552f0d5fcce8df524c9f0747d3687f1b86ef35dad73f5f4c91db2`
- `track.onnx`: `914355805fd288faeafde60ca2602538ba7a9da4abdc41baf16226ec49d5eca6`
