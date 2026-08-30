import type { Box } from './vit-tracker';

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

export type SelectionFocus = [number, number] | null;

export function selectionViewport(
  sourceWidth: number,
  sourceHeight: number,
  zoom: number,
  focus: SelectionFocus,
): Box {
  const safeZoom = clamp(zoom, 1, 3);
  const width = sourceWidth / safeZoom;
  const height = sourceHeight / safeZoom;
  const centerX = focus?.[0] ?? sourceWidth / 2;
  const centerY = focus?.[1] ?? sourceHeight / 2;
  const x = clamp(centerX - width / 2, 0, Math.max(0, sourceWidth - width));
  const y = clamp(centerY - height / 2, 0, Math.max(0, sourceHeight - height));
  return [x, y, width, height];
}

export function sourceBoxToSelectionCanvas(
  box: Box,
  viewport: Box,
  canvasWidth: number,
  canvasHeight: number,
): Box {
  const scaleX = canvasWidth / Math.max(1, viewport[2]);
  const scaleY = canvasHeight / Math.max(1, viewport[3]);
  return [
    (box[0] - viewport[0]) * scaleX,
    (box[1] - viewport[1]) * scaleY,
    box[2] * scaleX,
    box[3] * scaleY,
  ];
}

export function selectionCanvasPointToSource(
  point: [number, number],
  viewport: Box,
  canvasWidth: number,
  canvasHeight: number,
): [number, number] {
  return [
    clamp(viewport[0] + point[0] * viewport[2] / Math.max(1, canvasWidth), viewport[0], viewport[0] + viewport[2]),
    clamp(viewport[1] + point[1] * viewport[3] / Math.max(1, canvasHeight), viewport[1], viewport[1] + viewport[3]),
  ];
}
