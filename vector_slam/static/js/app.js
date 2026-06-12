/** UI wiring, API calls, and side panel updates. */

(function () {
  const canvas = document.getElementById("map-canvas");
  const btnMakeLine = document.getElementById("btn-make-line");
  const btnRandomPose = document.getElementById("btn-random-pose");
  const btnBatchRun = document.getElementById("btn-batch-run");
  const inputBatchN = document.getElementById("input-batch-n");
  const btnReset = document.getElementById("btn-reset");
  const sliderX = document.getElementById("slider-x");
  const sliderY = document.getElementById("slider-y");
  const sliderTheta = document.getElementById("slider-theta");
  const sliderMinAngle = document.getElementById("slider-min-angle");
  const sliderNoise = document.getElementById("slider-noise");
  const labelX = document.getElementById("label-x");
  const labelY = document.getElementById("label-y");
  const labelTheta = document.getElementById("label-theta");
  const labelMinAngle = document.getElementById("label-min-angle");
  const labelNoise = document.getElementById("label-noise");
  const modeIndicator = document.getElementById("mode-indicator");
  const panelContent = document.getElementById("panel-content");
  const batchPlotSection = document.getElementById("batch-plot-section");
  const batchPlotCanvas = document.getElementById("batch-plot");
  const batchStatsEl = document.getElementById("batch-stats");
  const btnZoomIn = document.getElementById("btn-zoom-in");
  const btnZoomOut = document.getElementById("btn-zoom-out");
  const zoomLabel = document.getElementById("zoom-label");

  let placeLineMode = false;
  let gmmCanvas = null;
  let syncingSliders = false;
  let syncingSettings = false;
  let moveInFlight = false;
  let moveQueued = false;
  let settingsInFlight = false;
  let settingsQueued = false;
  let lastBatchResults = null;

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

  function getPoseFromSliders() {
    return {
      x: parseFloat(sliderX.value) || 0,
      y: parseFloat(sliderY.value) || 0,
      theta: parseFloat(sliderTheta.value) || 0,
    };
  }

  function getSettingsFromSliders() {
    return {
      min_intersection_angle_deg: parseFloat(sliderMinAngle.value) || 0,
      scan_angle_noise_std_deg: parseFloat(sliderNoise.value) || 0,
    };
  }

  function updateSliderLabels() {
    labelX.textContent = parseFloat(sliderX.value).toFixed(2);
    labelY.textContent = parseFloat(sliderY.value).toFixed(2);
    labelTheta.textContent = `${parseFloat(sliderTheta.value).toFixed(1)}°`;
  }

  function updateSettingsLabels() {
    labelMinAngle.textContent = `${parseFloat(sliderMinAngle.value).toFixed(1)}°`;
    labelNoise.textContent = `${parseFloat(sliderNoise.value).toFixed(2)}°`;
  }

  function syncSlidersFromPose(pose) {
    syncingSliders = true;
    if (pose) {
      sliderX.value = Math.max(-10, Math.min(10, pose.x));
      sliderY.value = Math.max(-10, Math.min(10, pose.y));
      sliderTheta.value = Math.max(-135, Math.min(135, pose.theta));
    } else {
      sliderX.value = "0";
      sliderY.value = "0";
      sliderTheta.value = "0";
    }
    updateSliderLabels();
    syncingSliders = false;
  }

  function syncSettingsFromState(state) {
    syncingSettings = true;
    sliderMinAngle.value = String(state.min_intersection_angle_deg ?? 5);
    sliderNoise.value = String(state.scan_angle_noise_std_deg ?? 0.1);
    updateSettingsLabels();
    syncingSettings = false;
  }

  function applyState(state) {
    CanvasView.setState(state);
    syncSlidersFromPose(state.estimated_pose);
    syncSettingsFromState(state);
    if (state.batch_results) {
      lastBatchResults = state.batch_results;
    }
    updatePanel(state);
    updateBatchPlot();
  }

  function formatStat(value) {
    if (value == null) return "—";
    return value.toFixed(4);
  }

  function renderBatchStats(batch) {
    const pose = batch.pose_statistics;
    const yaw = batch.yaw_statistics;
    if (!pose && !yaw) return "";

    let html = '<table class="batch-stats-table">';
    html += "<thead><tr><th></th><th>mean</th><th>median</th><th>mode</th><th>RMS</th></tr></thead>";
    html += "<tbody>";
    if (pose) {
      html += "<tr>";
      html += "<td>pose (m)</td>";
      html += `<td>${formatStat(pose.mean)}</td>`;
      html += `<td>${formatStat(pose.median)}</td>`;
      html += `<td>${formatStat(pose.mode)}</td>`;
      html += `<td>${formatStat(pose.rms)}</td>`;
      html += "</tr>";
    }
    if (yaw) {
      html += "<tr>";
      html += "<td>yaw (deg)</td>";
      html += `<td>${formatStat(yaw.mean)}</td>`;
      html += `<td>${formatStat(yaw.median)}</td>`;
      html += `<td>${formatStat(yaw.mode)}</td>`;
      html += `<td>${formatStat(yaw.rms)}</td>`;
      html += "</tr>";
    }
    html += "</tbody></table>";
    return html;
  }

  function updateBatchPlot() {
    if (!batchPlotSection || !batchPlotCanvas) return;
    if (!lastBatchResults) {
      batchPlotSection.classList.add("hidden");
      if (batchStatsEl) batchStatsEl.innerHTML = "";
      return;
    }
    batchPlotSection.classList.remove("hidden");
    if (batchStatsEl) {
      batchStatsEl.innerHTML = renderBatchStats(lastBatchResults);
    }
    drawBatchErrorPlot(batchPlotCanvas, lastBatchResults);
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
    const minAngle = state.min_intersection_angle_deg ?? 5;
    let html = "";

    html += "<h3>True sensor</h3>";
    html += `<p>${formatPose(state.true_pose)}</p>`;

    html += "<h3>Estimated sensor</h3>";
    html += `<p>${est ? formatPose(est) : "none"}</p>`;

    const intersections = state.probability_intersections || [];
    const includedCount = state.counts?.intersections_included ?? 0;
    html += `<h3>Weighted intersection position</h3>`;
    if (state.weighted_position) {
      const wp = state.weighted_position;
      html += `<p class="weighted-position">(${wp.x.toFixed(4)}, ${wp.y.toFixed(4)})</p>`;
      html += `<p class="intersection-summary">${includedCount} of ${intersections.length} points included (≥${minAngle.toFixed(1)}°)</p>`;
      html += `<button type="button" id="btn-goto-weighted" class="panel-btn">Center on weighted position</button>`;
    } else {
      html += "<p class=\"muted\">none</p>";
    }

    html += `<h3>Intersection points (${intersections.length})</h3>`;
    if (intersections.length === 0) {
      html += "<p>none</p>";
    } else {
      html += '<table class="intersection-table">';
      html += "<thead><tr><th>#</th><th>position</th><th>angle</th><th>weight</th></tr></thead>";
      html += "<tbody>";
      intersections.forEach((item) => {
        const cls = item.excluded ? "excluded" : "";
        const tag = item.excluded ? " ✗" : "";
        html += `<tr class="${cls}">`;
        html += `<td>${item.nr}</td>`;
        html += `<td>${formatPoint(item.point)}</td>`;
        html += `<td>${item.angle_deg.toFixed(1)}°${tag}</td>`;
        html += `<td>${item.weight.toFixed(3)}</td>`;
        html += "</tr>";
      });
      html += "</tbody></table>";
    }

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
      if (state.alpha_peak_deg != null) {
        html += `<p class="alpha-peak">α_peak: ${state.alpha_peak_deg.toFixed(4)}° · green: +α_peak</p>`;
      }
      const corrections = state.correction_vectors || [];
      if (corrections.length > 0) {
        html += `<h3>Correction vectors (${corrections.length})</h3>`;
        html += '<ul class="correction-list">';
        corrections.forEach((c) => {
          if (c.excluded) return;
          html += `<li>Pair ${c.index}: distance ${c.correction_distance.toFixed(4)} m</li>`;
        });
        html += "</ul>";
      }
      html += '<div class="gmm-plot-wrap">';
      html += '<canvas id="gmm-plot" width="280" height="140"></canvas>';
      html += "</div>";
    }

    panelContent.innerHTML = html;

    const gotoWeighted = document.getElementById("btn-goto-weighted");
    if (gotoWeighted && state.weighted_position) {
      gotoWeighted.addEventListener("click", () => {
        const wp = state.weighted_position;
        panToWorld(wp.x, wp.y);
        CanvasView.render();
      });
    }

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

  async function moveSensorFromSliders() {
    if (moveInFlight) {
      moveQueued = true;
      return;
    }
    moveInFlight = true;
    try {
      const pose = getPoseFromSliders();
      const state = await api("POST", "/api/move_sensor", pose);
      applyState(state);
    } catch (e) {
      alert("Move failed: " + e.message);
    } finally {
      moveInFlight = false;
      if (moveQueued) {
        moveQueued = false;
        moveSensorFromSliders();
      }
    }
  }

  async function updateSettingsFromSliders() {
    if (settingsInFlight) {
      settingsQueued = true;
      return;
    }
    settingsInFlight = true;
    try {
      const settings = getSettingsFromSliders();
      const state = await api("POST", "/api/move_sensor", settings);
      applyState(state);
    } catch (e) {
      alert("Settings update failed: " + e.message);
    } finally {
      settingsInFlight = false;
      if (settingsQueued) {
        settingsQueued = false;
        updateSettingsFromSliders();
      }
    }
  }

  function onSliderInput() {
    updateSliderLabels();
    if (!syncingSliders) {
      moveSensorFromSliders();
    }
  }

  function onSettingsInput() {
    updateSettingsLabels();
    if (!syncingSettings) {
      updateSettingsFromSliders();
    }
  }

  async function loadState() {
    const state = await api("GET", "/api/state");
    applyState(state);
  }

  btnMakeLine.addEventListener("click", () => {
    setPlaceLineMode(!placeLineMode);
  });

  btnRandomPose.addEventListener("click", async () => {
    try {
      const state = await api("POST", "/api/random_pose");
      applyState(state);
      setPlaceLineMode(false);
    } catch (e) {
      alert("Random pose failed: " + e.message);
    }
  });

  btnBatchRun.addEventListener("click", async () => {
    const n = Math.max(1, Math.min(500, parseInt(inputBatchN.value, 10) || 50));
    inputBatchN.value = String(n);
    btnBatchRun.disabled = true;
    modeIndicator.textContent = `Running batch (${n})…`;
    try {
      const state = await api("POST", "/api/batch_simulate", { n });
      applyState(state);
      setPlaceLineMode(false);
      modeIndicator.textContent = `Batch done (${n} trials)`;
    } catch (e) {
      alert("Batch simulation failed: " + e.message);
      modeIndicator.textContent = "Drag endpoints to move static lines · scroll to zoom";
    } finally {
      btnBatchRun.disabled = false;
    }
  });

  btnReset.addEventListener("click", async () => {
    try {
      const state = await api("POST", "/api/reset");
      lastBatchResults = null;
      applyState(state);
      setPlaceLineMode(false);
    } catch (e) {
      alert("Reset failed: " + e.message);
    }
  });

  [sliderX, sliderY, sliderTheta].forEach((slider) => {
    slider.addEventListener("input", onSliderInput);
  });

  [sliderMinAngle, sliderNoise].forEach((slider) => {
    slider.addEventListener("input", onSettingsInput);
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
  updateSliderLabels();
  updateSettingsLabels();

  loadState().catch((e) => {
    panelContent.innerHTML = `<p>Error loading state: ${e.message}</p>`;
  });
})();
