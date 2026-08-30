import {
  selectionCanvasPointToSource,
  selectionViewport,
  sourceBoxToSelectionCanvas,
} from '../lib/selection-zoom.ts';

const viewport = selectionViewport(1920, 1080, 2, [1400, 500]);
const sourceBox = [1200, 300, 240, 480];
const canvasBox = sourceBoxToSelectionCanvas(sourceBox, viewport, 1920, 1080);
const sourceCenter = [sourceBox[0] + sourceBox[2] / 2, sourceBox[1] + sourceBox[3] / 2];
const canvasCenter = [canvasBox[0] + canvasBox[2] / 2, canvasBox[1] + canvasBox[3] / 2];
const roundTripCenter = selectionCanvasPointToSource(canvasCenter, viewport, 1920, 1080);
const edgeViewport = selectionViewport(1920, 1080, 3, [40, 40]);

const result = {
  zoomTwoUsesHalfFrame: viewport[2] === 960 && viewport[3] === 540,
  boxDoublesOnCanvas: Math.abs(canvasBox[2] - sourceBox[2] * 2) < 0.001
    && Math.abs(canvasBox[3] - sourceBox[3] * 2) < 0.001,
  pointerRoundTrip: Math.abs(roundTripCenter[0] - sourceCenter[0]) < 0.001
    && Math.abs(roundTripCenter[1] - sourceCenter[1]) < 0.001,
  edgeFocusStaysInside: edgeViewport[0] === 0
    && edgeViewport[1] === 0
    && edgeViewport[0] + edgeViewport[2] <= 1920
    && edgeViewport[1] + edgeViewport[3] <= 1080,
};
const pass = Object.values(result).every(Boolean);
console.log(JSON.stringify({ ...result, pass }, null, 2));
if (!pass) process.exitCode = 1;
