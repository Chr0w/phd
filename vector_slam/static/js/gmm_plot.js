/** Draw GMM PDF and sample markers in the side-panel overview chart. */

function drawAngleGmmPlot(canvas, gmm) {
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const pad = { left: 36, right: 10, top: 10, bottom: 24 };

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fafafa";
  ctx.fillRect(0, 0, w, h);

  if (!gmm || !gmm.x || gmm.x.length === 0) {
    ctx.fillStyle = "#999";
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillText("No angle data", pad.left, h / 2);
    return;
  }

  const xs = gmm.x;
  const ys = gmm.y;
  const samples = gmm.samples || [];
  const xMin = Math.min(...xs, ...(samples.length ? samples : xs));
  const xMax = Math.max(...xs, ...(samples.length ? samples : xs));
  const yMax = Math.max(...ys, 1e-6);
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  const toX = (v) => pad.left + ((v - xMin) / (xMax - xMin || 1)) * plotW;
  const toY = (v) => pad.top + plotH - (v / yMax) * plotH;

  ctx.strokeStyle = "#ddd";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();

  ctx.fillStyle = "#666";
  ctx.font = "10px system-ui, sans-serif";
  ctx.fillText(xMin.toFixed(2) + "°", pad.left, h - 6);
  ctx.textAlign = "right";
  ctx.fillText(xMax.toFixed(2) + "°", pad.left + plotW, h - 6);
  ctx.textAlign = "left";
  ctx.fillText("PDF", 4, pad.top + 10);

  ctx.strokeStyle = "#3498db";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < xs.length; i++) {
    const px = toX(xs[i]);
    const py = toY(ys[i]);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();

  if (gmm.peak) {
    const px = toX(gmm.peak.x);
    const py = toY(gmm.peak.y);
    ctx.fillStyle = "#2980b9";
    ctx.beginPath();
    ctx.arc(px, py, 3.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#2980b9";
    ctx.font = "10px system-ui, sans-serif";
    ctx.textAlign = "left";
    const peakLabel = `peak: ${gmm.peak.x.toFixed(2)}° · ${gmm.peak.y.toFixed(3)}`;
    const labelX = Math.min(px + 5, pad.left + plotW - 80);
    const labelY = Math.max(py - 6, pad.top + 12);
    ctx.fillText(peakLabel, labelX, labelY);
  }

  for (const s of samples) {
    const px = toX(s);
    ctx.strokeStyle = "#e74c3c";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(px, pad.top + plotH);
    ctx.lineTo(px, pad.top + plotH - 6);
    ctx.stroke();
  }

  if (gmm.components && gmm.components.length > 0) {
    const legendY = pad.top + 4;
    ctx.font = "9px monospace";
    gmm.components.forEach((c, i) => {
      ctx.fillStyle = "#555";
      ctx.fillText(
        `C${i + 1}: μ=${c.mean.toFixed(2)}° σ=${c.std.toFixed(3)}° w=${c.weight.toFixed(2)}`,
        pad.left + 2,
        legendY + i * 11
      );
    });
  }
}
