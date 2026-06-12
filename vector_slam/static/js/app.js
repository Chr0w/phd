/** UI wiring, API calls, and side panel updates. */

(function () {
  const canvas = document.getElementById("map-canvas");
  const btnMakeLine = document.getElementById("btn-make-line");
  const btnMove = document.getElementById("btn-move");
  const btnReset = document.getElementById("btn-reset");
  const inputX = document.getElementById("input-x");
  const inputY = document.getElementById("input-y");
  const inputTheta = document.getElementById("input-theta");
  const modeIndicator = document.getElementById("mode-indicator");
  const panelContent = document.getElementById("panel-content");
  const btnZoomIn = document.getElementById("btn-zoom-in");
  const btnZoomOut = document.getElementById("btn-zoom-out");
  const zoomLabel = document.getElementById("zoom-label");

  let placeLineMode = false;
  let gmmCanvas = null;

  CanvasView.init(canvas);

  async function api(method, path, body) {
    const opts = { method, headers: { "Content-Type": "application/json" } };
    if (body !== undefined) opts.body = JSON.stringify(body);
    const res = await fetch(path, opts);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || res.statusText);
    }
    return res.json();
  }

  function applyState(state) {
    CanvasView.setState(state);
    updatePanel(state);
    if (state.estimated_pose) {
      inputX.value = state.estimated_pose.x.toFixed(2);
      inputY.value = state.estimated_pose.y.toFixed(2);
      inputTheta.value = state.estimated_pose.theta.toFixed(1);
    }
  }

  function lineWithMid(line) {
    const mid = line.midpoint
      ? formatPoint(line.midpoint)
      : formatPoint([
          (line.p1[0] + line.p2[0]) / 2,
          (line.p1[1] + line.p2[1]) / 2,
        ]);
    return `${formatPoint(line.p1)} → ${formatPoint(line.p2)} · mid ${mid}`;
  }

  function updatePanel(state) {
    const est = state.estimated_pose;
    const threshold = state.excluded_angle_threshold_deg ?? 80;
    let html = "";

    html += "<h3>True sensor</h3>";
    html += `<p>${formatPose(state.true_pose)}</p>`;

    html += "<h3>Estimated sensor</h3>";
    html += `<p>${est ? formatPose(est) : "none"}</p>`;

    const pairs = state.line_pairs || [];
    const excluded = state.counts.excluded || 0;
    html += `<h3>Line pairs (${pairs.length})</h3>`;
    if (pairs.length === 0) {
      html += "<p>none</p>";
    } else {
      if (excluded > 0) {
        html += `<p class="excluded-summary">${excluded} excluded (&lt;${threshold}°)</p>`;
      }
      html += '<div class="line-pairs">';
      pairs.forEach((pair) => {
        const cls = pair.excluded ? "line-pair excluded" : "line-pair";
        html += `<div class="${cls}">`;
        html += `<div class="pair-header">Pair ${pair.index} (LS_${pair.index})</div>`;
        if (pair.static_line) {
          html += `<div>LS: ${lineWithMid(pair.static_line)}</div>`;
        }
        if (pair.scan_line) {
          html += `<div>scan: ${lineWithMid(pair.scan_line)}</div>`;
        }
        if (pair.relative_angle_deg !== null) {
          const tag = pair.excluded ? " (excluded)" : "";
          html += `<div class="relative-angle">relative angle: ${pair.relative_angle_deg.toFixed(1)}°${tag}</div>`;
        } else {
          html += `<div class="relative-angle muted">relative angle: —</div>`;
        }
        html += "</div>";
      });
      html += "</div>";
    }

    const diffs = state.pair_angle_diffs || [];
    html += `<h3>Pair angle differences (${diffs.length})</h3>`;
    if (diffs.length === 0) {
      html += "<p>none</p>";
    } else {
      html += '<ul class="angle-diff-list">';
      diffs.forEach((d) => {
        html += `<li>Pair ${d.index}: ${d.angle_deg.toFixed(4)}°</li>`;
      });
      html += "</ul>";
      html += '<div class="gmm-plot-wrap">';
      html += '<canvas id="gmm-plot" width="280" height="140"></canvas>';
      html += "</div>";
    }

    panelContent.innerHTML = html;

    if (diffs.length > 0) {
      gmmCanvas = document.getElementById("gmm-plot");
      drawAngleGmmPlot(gmmCanvas, state.angle_gmm);
    } else {
      gmmCanvas = null;
    }
  }

  function formatPose(pose) {
    return `(${pose.x.toFixed(2)}, ${pose.y.toFixed(2)}, ${pose.theta.toFixed(1)}°)`;
  }

  function updateZoomLabel() {
    zoomLabel.textContent = `${Math.round(View.zoom * 100)}%`;
  }

  function setPlaceLineMode(active) {
    placeLineMode = active;
    btnMakeLine.classList.toggle("active", active);
    CanvasView.setMode(active ? "place-line" : "default");
    modeIndicator.textContent = active
      ? "Click two points to draw a static line · scroll to zoom"
      : "Drag endpoints to move static lines · scroll to zoom";
  }

  async function loadState() {
    const state = await api("GET", "/api/state");
    applyState(state);
  }

  btnMakeLine.addEventListener("click", () => {
    setPlaceLineMode(!placeLineMode);
  });

  btnMove.addEventListener("click", async () => {
    try {
      const state = await api("POST", "/api/move_sensor", {
        x: parseFloat(inputX.value) || 0,
        y: parseFloat(inputY.value) || 0,
        theta: parseFloat(inputTheta.value) || 0,
      });
      applyState(state);
    } catch (e) {
      alert("Move failed: " + e.message);
    }
  });

  btnReset.addEventListener("click", async () => {
    try {
      const state = await api("POST", "/api/reset");
      applyState(state);
      setPlaceLineMode(false);
      inputX.value = "0";
      inputY.value = "0";
      inputTheta.value = "0";
    } catch (e) {
      alert("Reset failed: " + e.message);
    }
  });

  canvas.addEventListener("mousedown", async (evt) => {
    const result = CanvasView.onMouseDown(evt);
    if (result.action === "create-line") {
      try {
        const state = await api("POST", "/api/static_line", {
          p1: result.p1,
          p2: result.p2,
        });
        applyState(state);
        setPlaceLineMode(false);
      } catch (e) {
        alert("Create line failed: " + e.message);
      }
    }
  });

  canvas.addEventListener("mousemove", (evt) => {
    CanvasView.onMouseMove(evt);
  });

  canvas.addEventListener("mouseup", async (evt) => {
    const result = CanvasView.onMouseUp(evt);
    if (result.action === "drag-end") {
      const payload = {};
      payload[result.endpoint] = result.point;
      try {
        const state = await api("PATCH", `/api/static_line/${result.lineId}`, payload);
        applyState(state);
      } catch (e) {
        alert("Update line failed: " + e.message);
        loadState();
      }
    }
  });

  canvas.addEventListener("mouseleave", () => {
    CanvasView.onMouseLeave();
  });

  canvas.addEventListener(
    "wheel",
    (evt) => {
      CanvasView.onWheel(evt);
      updateZoomLabel();
    },
    { passive: false }
  );

  btnZoomIn.addEventListener("click", () => {
    CanvasView.zoomBy(1.25);
    updateZoomLabel();
  });

  btnZoomOut.addEventListener("click", () => {
    CanvasView.zoomBy(1 / 1.25);
    updateZoomLabel();
  });

  updateZoomLabel();

  loadState().catch((e) => {
    panelContent.innerHTML = `<p>Error loading state: ${e.message}</p>`;
  });
})();
