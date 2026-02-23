const state = {
  rows: [],
};

const palette = ["#0f766e", "#b45309", "#4f46e5", "#c026d3", "#0f172a", "#dc2626"];

const csvFile = document.getElementById("csvFile");
const metricSelect = document.getElementById("metricSelect");
const chart = document.getElementById("chart");
const legend = document.getElementById("legend");
const tableBody = document.querySelector("#resultsTable tbody");

csvFile.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }
  const text = await file.text();
  state.rows = parseCsv(text);
  render();
});

metricSelect.addEventListener("change", () => render());

function parseCsv(text) {
  const lines = text.split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) {
    return [];
  }
  const headers = splitCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const cells = splitCsvLine(line);
    const row = {};
    headers.forEach((h, i) => {
      row[h] = cells[i] ?? "";
    });
    row.noise_sigma = Number(row.noise_sigma);
    row.rmse = Number(row.rmse);
    row.nrmse = Number(row.nrmse);
    return row;
  });
}

function splitCsvLine(line) {
  const out = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"' && line[i + 1] === '"') {
      cur += '"';
      i += 1;
      continue;
    }
    if (ch === '"') {
      inQuotes = !inQuotes;
      continue;
    }
    if (ch === "," && !inQuotes) {
      out.push(cur);
      cur = "";
      continue;
    }
    cur += ch;
  }
  out.push(cur);
  return out;
}

function render() {
  renderTable();
  renderChart();
}

function renderTable() {
  tableBody.innerHTML = "";
  if (state.rows.length === 0) {
    tableBody.innerHTML = '<tr><td colspan="4">Upload a CSV file to view results.</td></tr>';
    return;
  }
  const sorted = [...state.rows].sort((a, b) => a.noise_sigma - b.noise_sigma || a.algorithm.localeCompare(b.algorithm));
  for (const row of sorted) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(row.algorithm)}</td>
      <td>${row.noise_sigma.toFixed(4)}</td>
      <td>${formatNum(row.rmse)}</td>
      <td>${formatNum(row.nrmse)}</td>
    `;
    tableBody.appendChild(tr);
  }
}

function renderChart() {
  const metric = metricSelect.value;
  chart.innerHTML = "";
  legend.innerHTML = "";

  if (state.rows.length === 0) {
    drawText(40, 190, "Upload noise_sweep_results.csv to render chart.");
    return;
  }

  const grouped = groupByAlgorithm(state.rows);
  const allX = state.rows.map((r) => r.noise_sigma).filter(Number.isFinite);
  const allY = state.rows.map((r) => r[metric]).filter(Number.isFinite);
  if (allX.length === 0 || allY.length === 0) {
    drawText(40, 190, "CSV format looks invalid.");
    return;
  }

  const xMin = Math.min(...allX);
  const xMax = Math.max(...allX);
  const yMin = Math.min(...allY);
  const yMax = Math.max(...allY);

  const w = 900;
  const h = 380;
  const m = { left: 62, right: 24, top: 20, bottom: 48 };
  const cw = w - m.left - m.right;
  const ch = h - m.top - m.bottom;

  const xScale = (v) => m.left + ((v - xMin) / safeDen(xMax - xMin)) * cw;
  const yScale = (v) => m.top + (1 - (v - yMin) / safeDen(yMax - yMin)) * ch;

  drawLine(m.left, m.top, m.left, h - m.bottom, "#c6bea9", 1);
  drawLine(m.left, h - m.bottom, w - m.right, h - m.bottom, "#c6bea9", 1);

  for (let i = 0; i <= 5; i += 1) {
    const t = i / 5;
    const yVal = yMin + t * (yMax - yMin);
    const y = yScale(yVal);
    drawLine(m.left, y, w - m.right, y, "#ece6d8", 1);
    drawText(8, y + 4, yVal.toFixed(3), 12, "#6b7280");
  }

  for (let i = 0; i <= 5; i += 1) {
    const t = i / 5;
    const xVal = xMin + t * (xMax - xMin);
    const x = xScale(xVal);
    drawLine(x, m.top, x, h - m.bottom, "#f0ebdf", 1);
    drawText(x - 15, h - 16, xVal.toFixed(3), 12, "#6b7280");
  }

  drawText(8, 18, metric.toUpperCase(), 13, "#334155", "700");
  drawText(w / 2 - 80, h - 4, "Noise level (sigma)", 12, "#334155");

  Object.entries(grouped)
    .sort(([a], [b]) => a.localeCompare(b))
    .forEach(([algorithm, rows], idx) => {
      const color = palette[idx % palette.length];
      const sorted = rows.sort((a, b) => a.noise_sigma - b.noise_sigma);
      const points = sorted.map((r) => `${xScale(r.noise_sigma)},${yScale(r[metric])}`).join(" ");
      const poly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      poly.setAttribute("points", points);
      poly.setAttribute("fill", "none");
      poly.setAttribute("stroke", color);
      poly.setAttribute("stroke-width", "2.6");
      chart.appendChild(poly);

      sorted.forEach((r) => {
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", xScale(r.noise_sigma));
        circle.setAttribute("cy", yScale(r[metric]));
        circle.setAttribute("r", "3.6");
        circle.setAttribute("fill", color);
        chart.appendChild(circle);
      });

      const item = document.createElement("span");
      item.className = "legend__item";
      item.innerHTML = `<span class="legend__dot" style="background:${color}"></span>${escapeHtml(algorithm)}`;
      legend.appendChild(item);
    });
}

function groupByAlgorithm(rows) {
  const out = {};
  for (const row of rows) {
    if (!out[row.algorithm]) {
      out[row.algorithm] = [];
    }
    out[row.algorithm].push(row);
  }
  return out;
}

function drawLine(x1, y1, x2, y2, color, width) {
  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", x1);
  line.setAttribute("y1", y1);
  line.setAttribute("x2", x2);
  line.setAttribute("y2", y2);
  line.setAttribute("stroke", color);
  line.setAttribute("stroke-width", width);
  chart.appendChild(line);
}

function drawText(x, y, text, size = 14, color = "#374151", weight = "400") {
  const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
  t.setAttribute("x", x);
  t.setAttribute("y", y);
  t.setAttribute("font-size", String(size));
  t.setAttribute("fill", color);
  t.setAttribute("font-family", "IBM Plex Mono, monospace");
  t.setAttribute("font-weight", weight);
  t.textContent = text;
  chart.appendChild(t);
}

function formatNum(v) {
  if (!Number.isFinite(v)) {
    return "-";
  }
  return v.toFixed(6);
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function safeDen(v) {
  return v === 0 ? 1 : v;
}

render();

