/** Canvas rendering and pointer interaction. */

const CanvasView = (function () {
  const HIT_RADIUS = 8;
  const SENSOR_DOT_RADIUS = 4;
  const HEADING_LENGTH_M = 0.5;
  const ARC_RADIUS_M = 0.35;
  const STATIC_COLOR = "#000000";
  const SCAN_COLOR = "#e74c3c";
  const SCAN_EXCLUDED_COLOR = "#b87878";
  const ALIGNED_SCAN_COLOR = "#27ae60";
  const MIDPOINT_RADIUS = 3;
  const TO_EP_COLOR = "#bbbbbb";
  const ARC_COLOR = "#888888";

  let canvas, ctx;
  let gameState = null;
  let mode = "default"; // "default" | "place-line"
  let placeAnchor = null;
  let dragState = null; // { lineId, endpoint: "p1"|"p2" }
  let hoverEndpoint = null;

  function init(canvasEl) {
    canvas = canvasEl;
    ctx = canvas.getContext("2d");
  }

  function setMode(m) {
    mode = m;
    placeAnchor = null;
    updateCursor();
  }

  function setState(state) {
    gameState = state;
    render();
  }

  function getState() {
    return gameState;
  }

  function updateCursor() {
    if (!canvas) return;
    canvas.classList.remove("dragging", "can-drag");
    if (mode === "place-line") {
      canvas.style.cursor = "crosshair";
    } else if (dragState) {
      canvas.classList.add("dragging");
    } else if (hoverEndpoint) {
      canvas.classList.add("can-drag");
    } else {
      canvas.style.cursor = "crosshair";
    }
  }

  function drawLine(p1, p2, color, width) {
    const s1 = worldToScreen(p1[0], p1[1]);
    const s2 = worldToScreen(p2[0], p2[1]);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(s1.x, s1.y);
    ctx.lineTo(s2.x, s2.y);
    ctx.stroke();
  }

  function drawEndpoint(p, color, radius) {
    const s = worldToScreen(p[0], p[1]);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(s.x, s.y, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  function worldVecScreenAngle(vx, vy) {
    return Math.atan2(-vy, vx);
  }

  function drawRelativeAngleArc(mid, scanP1, scanP2, ep) {
    const vxLine = scanP2[0] - scanP1[0];
    const vyLine = scanP2[1] - scanP1[1];
    const vxEp = ep.x - mid[0];
    const vyEp = ep.y - mid[1];

    const lenLine = Math.hypot(vxLine, vyLine);
    const lenEp = Math.hypot(vxEp, vyEp);
    if (lenLine < 1e-9 || lenEp < 1e-9) return;

    const acuteRad =
      (acuteAngleDeg(vxLine, vyLine, vxEp, vyEp) * Math.PI) / 180;
    const aLine = worldVecScreenAngle(vxLine, vyLine);
    const aEp = worldVecScreenAngle(vxEp, vyEp);

    // Pick the ray along the undirected scan line that bounds the acute wedge.
    let start = aLine;
    let diff = normalizeAngle(aEp - start);
    if (Math.abs(diff) > Math.PI / 2) {
      start = aLine + Math.PI;
      diff = normalizeAngle(aEp - start);
    }
    const end = start + (diff >= 0 ? acuteRad : -acuteRad);

    const center = worldToScreen(mid[0], mid[1]);
    const radius = ARC_RADIUS_M * getPixelsPerMeter();

    ctx.strokeStyle = ARC_COLOR;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(center.x, center.y, radius, start, end, diff < 0);
    ctx.stroke();
  }

  function drawVectorToEp(mid, ep) {
    drawLine(mid, [ep.x, ep.y], TO_EP_COLOR, 1.5);
  }

  function drawLineMidpoint(p1, p2, color) {
    const mid = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2];
    drawEndpoint(mid, color, MIDPOINT_RADIUS);
  }

  function drawScanLinePair(pair) {
    const scan = pair.scan_line;
    const color = pair.excluded ? SCAN_EXCLUDED_COLOR : SCAN_COLOR;
    drawLine(scan.p1, scan.p2, color, 2);
    drawLineMidpoint(scan.p1, scan.p2, color);

    if (gameState.estimated_pose && pair.relative_angle_deg !== null) {
      drawVectorToEp(pair.midpoint, gameState.estimated_pose);
      drawRelativeAngleArc(
        pair.midpoint,
        scan.p1,
        scan.p2,
        gameState.estimated_pose
      );
    }
  }

  function drawSensor(pose, alpha) {
    const { x, y, theta } = pose;
    const center = worldToScreen(x, y);
    const thetaRad = (theta * Math.PI) / 180;
    const endX = x + HEADING_LENGTH_M * Math.cos(thetaRad);
    const endY = y + HEADING_LENGTH_M * Math.sin(thetaRad);
    const end = worldToScreen(endX, endY);

    ctx.globalAlpha = alpha;
    ctx.fillStyle = "#000";
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 2;

    ctx.beginPath();
    ctx.arc(center.x, center.y, SENSOR_DOT_RADIUS, 0, Math.PI * 2);
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(center.x, center.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function render() {
    if (!ctx) return;

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    if (!gameState) return;

    const pairs = gameState.line_pairs || [];
    for (const pair of pairs) {
      drawScanLinePair(pair);
    }

    for (const line of gameState.static_lines) {
      let p1 = line.p1;
      let p2 = line.p2;
      if (dragState && dragState.lineId === line.id) {
        if (dragState.endpoint === "p1" && dragState.preview) p1 = dragState.preview;
        if (dragState.endpoint === "p2" && dragState.preview) p2 = dragState.preview;
      }
      drawLine(p1, p2, STATIC_COLOR, 2);
      drawLineMidpoint(p1, p2, STATIC_COLOR);

      const isHover =
        hoverEndpoint && hoverEndpoint.lineId === line.id;
      const handleRadius = isHover || (dragState && dragState.lineId === line.id) ? 6 : 4;
      if (mode !== "place-line") {
        drawEndpoint(p1, isHover && hoverEndpoint.endpoint === "p1" ? "#3498db" : "#333", handleRadius);
        drawEndpoint(p2, isHover && hoverEndpoint.endpoint === "p2" ? "#3498db" : "#333", handleRadius);
      }
    }

    for (const line of gameState.aligned_scan_lines || []) {
      drawLine(line.p1, line.p2, ALIGNED_SCAN_COLOR, 2);
      drawLineMidpoint(line.p1, line.p2, ALIGNED_SCAN_COLOR);
    }

    if (gameState.estimated_pose) {
      drawSensor(gameState.estimated_pose, 0.4);
    }

    drawSensor(gameState.true_pose, 1.0);

    if (placeAnchor) {
      drawEndpoint(placeAnchor, "#3498db", 5);
    }
  }

  function findEndpointHit(sx, sy) {
    if (!gameState || mode === "place-line") return null;
    for (const line of gameState.static_lines) {
      const s1 = worldToScreen(line.p1[0], line.p1[1]);
      const s2 = worldToScreen(line.p2[0], line.p2[1]);
      if (dist(sx, sy, s1.x, s1.y) <= HIT_RADIUS) {
        return { lineId: line.id, endpoint: "p1" };
      }
      if (dist(sx, sy, s2.x, s2.y) <= HIT_RADIUS) {
        return { lineId: line.id, endpoint: "p2" };
      }
    }
    return null;
  }

  function getCanvasCoords(evt) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (evt.clientX - rect.left) * scaleX,
      y: (evt.clientY - rect.top) * scaleY,
    };
  }

  function onMouseDown(evt) {
    const { x, y } = getCanvasCoords(evt);

    if (mode === "place-line") {
      const world = clampWorld(screenToWorld(x, y).x, screenToWorld(x, y).y);
      const pt = [world.x, world.y];
      if (!placeAnchor) {
        placeAnchor = pt;
        render();
        return { action: "place-anchor" };
      }
      const p1 = placeAnchor;
      const p2 = pt;
      placeAnchor = null;
      return { action: "create-line", p1, p2 };
    }

    const hit = findEndpointHit(x, y);
    if (hit) {
      dragState = { ...hit, preview: null };
      updateCursor();
      return { action: "drag-start", ...hit };
    }
    return { action: "none" };
  }

  function onMouseMove(evt) {
    const { x, y } = getCanvasCoords(evt);

    if (dragState) {
      const world = clampWorld(screenToWorld(x, y).x, screenToWorld(x, y).y);
      dragState.preview = [world.x, world.y];
      render();
      return { action: "drag-move" };
    }

    const hit = findEndpointHit(x, y);
    hoverEndpoint = hit;
    updateCursor();
    if (hit) render();
    return { action: "hover" };
  }

  function onMouseUp(evt) {
    if (!dragState) return { action: "none" };

    const { lineId, endpoint, preview } = dragState;
    dragState = null;
    updateCursor();
    render();

    if (preview) {
      return { action: "drag-end", lineId, endpoint, point: preview };
    }
    return { action: "none" };
  }

  function onMouseLeave() {
    hoverEndpoint = null;
    if (!dragState) updateCursor();
  }

  function onWheel(evt) {
    evt.preventDefault();
    const { x, y } = getCanvasCoords(evt);
    const factor = evt.deltaY < 0 ? 1.12 : 1 / 1.12;
    zoomAt(x, y, factor);
    render();
    return View.zoom;
  }

  function zoomBy(factor) {
    zoomAt(CENTER, CENTER, factor);
    render();
    return View.zoom;
  }

  return {
    init,
    setMode,
    setState,
    getState,
    render,
    onMouseDown,
    onMouseMove,
    onMouseUp,
    onMouseLeave,
    onWheel,
    zoomBy,
  };
})();
