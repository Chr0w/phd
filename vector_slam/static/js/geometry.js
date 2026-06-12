/** World ↔ screen coordinate transforms with pan/zoom. */

const MAP_SIZE_M = 50;
const METERS_PER_PIXEL = 0.05;
const CANVAS_SIZE = MAP_SIZE_M / METERS_PER_PIXEL; // 1000
const CENTER = CANVAS_SIZE / 2; // 500
const BASE_PIXELS_PER_METER = 1 / METERS_PER_PIXEL;

const View = {
  panX: 0,
  panY: 0,
  zoom: 1,
  minZoom: 0.25,
  maxZoom: 8,
};

function getPixelsPerMeter() {
  return BASE_PIXELS_PER_METER * View.zoom;
}

function worldToScreen(wx, wy) {
  const ppm = getPixelsPerMeter();
  return {
    x: CENTER + (wx - View.panX) * ppm,
    y: CENTER - (wy - View.panY) * ppm,
  };
}

function screenToWorld(sx, sy) {
  const ppm = getPixelsPerMeter();
  return {
    x: View.panX + (sx - CENTER) / ppm,
    y: View.panY - (sy - CENTER) / ppm,
  };
}

function clampWorld(wx, wy) {
  const half = MAP_SIZE_M / 2;
  return {
    x: Math.max(-half, Math.min(half, wx)),
    y: Math.max(-half, Math.min(half, wy)),
  };
}

function normalizeAngle(rad) {
  let a = rad;
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}

function acuteAngleDeg(vx1, vy1, vx2, vy2) {
  const len1 = Math.hypot(vx1, vy1);
  const len2 = Math.hypot(vx2, vy2);
  if (len1 < 1e-12 || len2 < 1e-12) return 0;
  const cosSigned = (vx1 * vx2 + vy1 * vy2) / (len1 * len2);
  const angle = (Math.acos(Math.max(-1, Math.min(1, cosSigned))) * 180) / Math.PI;
  return Math.min(angle, 180 - angle);
}

function zoomAt(screenX, screenY, factor) {
  const worldBefore = screenToWorld(screenX, screenY);
  View.zoom = Math.max(View.minZoom, Math.min(View.maxZoom, View.zoom * factor));
  const worldAfter = screenToWorld(screenX, screenY);
  View.panX += worldBefore.x - worldAfter.x;
  View.panY += worldBefore.y - worldAfter.y;
}

function resetView() {
  View.panX = 0;
  View.panY = 0;
  View.zoom = 1;
}

function panToWorld(wx, wy) {
  View.panX = wx;
  View.panY = wy;
}

function dist(ax, ay, bx, by) {
  const dx = ax - bx;
  const dy = ay - by;
  return Math.sqrt(dx * dx + dy * dy);
}

function formatPoint(p) {
  return `(${p[0].toFixed(2)}, ${p[1].toFixed(2)})`;
}
