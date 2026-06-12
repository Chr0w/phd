/** Batch simulation error plots for the overview panel. */

function drawBatchErrorPlot(canvas, batchResults) {
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const pad = { left: 40, right: 10, top: 18, bottom: 20 };
  const gap = 14;
  const panelH = (h - pad.top - pad.bottom - gap) / 2;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fafafa";
  ctx.fillRect(0, 0, w, h);

  if (!batchResults || !batchResults.trials || batchResults.trials.length === 0) {
    ctx.fillStyle = "#999";
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillText("No batch results", pad.left, h / 2);
    return;
  }

  const trials = batchResults.trials;
  const poseErrors = trials.map((t) =>
    t.pose_error_m == null ? null : t.pose_error_m
  );
  const yawErrors = trials.map((t) =>
    t.yaw_error_deg == null ? null : t.yaw_error_deg
  );
  const singleLine = trials.map((t) => t.line_count === 1);

  drawBatchPanel(
    ctx,
    pad.left,
    pad.top,
    w - pad.left - pad.right,
    panelH,
    trials.length,
    poseErrors,
    singleLine,
    "Pose error (m)",
    "#c0392b"
  );

  drawBatchPanel(
    ctx,
    pad.left,
    pad.top + panelH + gap,
    w - pad.left - pad.right,
    panelH,
    trials.length,
    yawErrors,
    singleLine,
    "Yaw error (deg)",
    "#2980b9"
  );

  ctx.fillStyle = "#444";
  ctx.font = "11px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(`Batch: ${trials.length} trials`, w / 2, h - 4);
  ctx.textAlign = "left";

  ctx.fillStyle = "#e67e22";
  ctx.font = "9px system-ui, sans-serif";
  ctx.fillText("□ 1 line", pad.left, pad.top - 4);
}

function drawBatchPanel(
  ctx,
  x,
  y,
  width,
  height,
  count,
  values,
  singleLine,
  label,
  color
) {
  const valid = values.filter((v) => v != null && Number.isFinite(v));
  const yMax = valid.length ? Math.max(...valid, 1e-6) : 1;
  const yMin = 0;
  const plotH = height - 16;

  ctx.fillStyle = "#555";
  ctx.font = "10px system-ui, sans-serif";
  ctx.fillText(label, x, y + 10);

  const baseY = y + 16;
  ctx.strokeStyle = "#ddd";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, baseY);
  ctx.lineTo(x, baseY + plotH);
  ctx.lineTo(x + width, baseY + plotH);
  ctx.stroke();

  function trialX(i) {
    return count <= 1 ? x + width / 2 : x + (i / (count - 1)) * width;
  }

  function trialY(v) {
    return baseY + plotH - ((v - yMin) / (yMax - yMin || 1)) * plotH;
  }

  if (count <= 1) {
    const v = values[0];
    if (v == null) return;
    drawTrialMarker(ctx, trialX(0), trialY(v), color, singleLine[0]);
    return;
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < count; i++) {
    const v = values[i];
    if (v == null) {
      started = false;
      continue;
    }
    const px = trialX(i);
    const py = trialY(v);
    if (!started) {
      ctx.moveTo(px, py);
      started = true;
    } else {
      ctx.lineTo(px, py);
    }
  }
  ctx.stroke();

  for (let i = 0; i < count; i++) {
    const v = values[i];
    if (v == null) continue;
    drawTrialMarker(ctx, trialX(i), trialY(v), color, singleLine[i]);
  }

  ctx.fillStyle = "#888";
  ctx.font = "9px system-ui, sans-serif";
  ctx.fillText("0", x - 4, baseY + plotH + 1);
  ctx.textAlign = "right";
  ctx.fillText(yMax.toFixed(3), x - 4, baseY + 10);
  ctx.textAlign = "left";
}

function drawTrialMarker(ctx, px, py, color, isSingleLine) {
  if (isSingleLine) {
    const s = 4;
    ctx.strokeStyle = "#e67e22";
    ctx.lineWidth = 2;
    ctx.strokeRect(px - s, py - s, s * 2, s * 2);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(px, py, 2, 0, Math.PI * 2);
    ctx.fill();
    return;
  }

  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(px, py, 2, 0, Math.PI * 2);
  ctx.fill();
}
