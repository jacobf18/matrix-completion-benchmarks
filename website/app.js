const state = {
  rows: [],
  headers: [],
  numericColumns: [],
  jsonEvalEntries: [],
};

const palette = ["#0f766e", "#b45309", "#4f46e5", "#c026d3", "#0f172a", "#dc2626"];

const csvFile = document.getElementById("csvFile");
const xAxisSelect = document.getElementById("xAxisSelect");
const metricSelect = document.getElementById("metricSelect");
const chart = document.getElementById("chart");
const legend = document.getElementById("legend");
const tableBody = document.querySelector("#resultsTable tbody");

const jsonFiles = document.getElementById("jsonFiles");
const jsonMetricSelect = document.getElementById("jsonMetricSelect");
const jsonChart = document.getElementById("jsonChart");
const jsonTableBody = document.querySelector("#jsonResultsTable tbody");
const loadDemoCsvBtn = document.getElementById("loadDemoCsvBtn");
const loadDemoJsonBtn = document.getElementById("loadDemoJsonBtn");
const demoStatus = document.getElementById("demoStatus");

csvFile.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }
  const text = await file.text();
  const parsed = parseCsv(text);
  state.rows = parsed.rows;
  state.headers = parsed.headers;
  state.numericColumns = parsed.numericColumns;
  populateSelectors();
  render();
});

metricSelect.addEventListener("change", () => render());
xAxisSelect.addEventListener("change", () => render());

jsonFiles.addEventListener("change", async (event) => {
  const files = Array.from(event.target.files ?? []);
  if (files.length === 0) {
    return;
  }
  const entries = [];
  for (const file of files) {
    try {
      const text = await file.text();
      const payload = JSON.parse(text);
      entries.push(normalizeEvalJson(payload, file.name));
    } catch {
      // Ignore malformed file and continue.
    }
  }
  state.jsonEvalEntries = entries;
  populateJsonMetricSelector();
  renderJsonExplorer();
});

jsonMetricSelect.addEventListener("change", () => renderJsonExplorer());
loadDemoCsvBtn.addEventListener("click", () => loadDemoCsv());
loadDemoJsonBtn.addEventListener("click", () => loadDemoJson());

function parseCsv(text) {
  const lines = text.split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) {
    return { rows: [], headers: [], numericColumns: [] };
  }
  const headers = splitCsvLine(lines[0]);
  const rows = lines.slice(1).map((line) => {
    const cells = splitCsvLine(line);
    const row = {};
    headers.forEach((h, i) => {
      row[h] = cells[i] ?? "";
    });
    return row;
  });

  const numericColumns = headers.filter((h) => {
    let seen = 0;
    for (const row of rows) {
      const raw = String(row[h] ?? "").trim();
      if (raw === "") {
        continue;
      }
      const val = Number(raw);
      if (!Number.isFinite(val)) {
        return false;
      }
      seen += 1;
    }
    return seen > 0;
  });

  for (const row of rows) {
    for (const col of numericColumns) {
      row[col] = Number(row[col]);
    }
  }

  return { rows, headers, numericColumns };
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

function populateSelectors() {
  xAxisSelect.innerHTML = "";
  metricSelect.innerHTML = "";
  if (state.numericColumns.length === 0) {
    return;
  }

  const preferredX = ["noise_sigma", "missing_fraction"];
  const preferredMetric = ["nrmse", "nrmse_missing", "rmse_missing", "rmse", "mae"];

  for (const col of state.numericColumns) {
    const optX = document.createElement("option");
    optX.value = col;
    optX.textContent = col;
    xAxisSelect.appendChild(optX);

    const optM = document.createElement("option");
    optM.value = col;
    optM.textContent = col;
    metricSelect.appendChild(optM);
  }

  const xDefault = preferredX.find((c) => state.numericColumns.includes(c)) ?? state.numericColumns[0];
  const mDefault = preferredMetric.find((c) => state.numericColumns.includes(c))
    ?? state.numericColumns.find((c) => c !== xDefault)
    ?? state.numericColumns[0];
  xAxisSelect.value = xDefault;
  metricSelect.value = mDefault;
}

function renderTable() {
  tableBody.innerHTML = "";
  if (state.rows.length === 0) {
    tableBody.innerHTML = '<tr><td colspan="4">Upload a CSV file to view results.</td></tr>';
    return;
  }
  const xAxis = xAxisSelect.value || "noise_sigma";
  const metric = metricSelect.value || "nrmse";
  const sorted = [...state.rows].sort((a, b) => {
    const ax = Number(a[xAxis]);
    const bx = Number(b[xAxis]);
    const aSeries = seriesKey(a);
    const bSeries = seriesKey(b);
    return (Number.isFinite(ax) && Number.isFinite(bx) ? ax - bx : 0) || aSeries.localeCompare(bSeries);
  });

  for (const row of sorted) {
    const id = row.preset_id ?? row.dataset_id ?? "-";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(seriesKey(row))}</td>
      <td>${escapeHtml(String(id))}</td>
      <td>${formatNum(Number(row[xAxis]))}</td>
      <td>${formatNum(Number(row[metric]))}</td>
    `;
    tableBody.appendChild(tr);
  }
}

function renderChart() {
  const metric = metricSelect.value;
  const xAxis = xAxisSelect.value;
  chart.innerHTML = "";
  legend.innerHTML = "";

  if (state.rows.length === 0) {
    drawText(40, 190, "Upload a benchmark CSV file to render chart.");
    return;
  }
  if (!metric || !xAxis) {
    drawText(40, 190, "Select x-axis and metric columns.");
    return;
  }

  const grouped = groupByAlgorithm(state.rows);
  const allX = state.rows.map((r) => Number(r[xAxis])).filter(Number.isFinite);
  const allY = state.rows.map((r) => Number(r[metric])).filter(Number.isFinite);
  if (allX.length === 0 || allY.length === 0) {
    drawText(40, 190, "Selected columns are not plottable.");
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
  drawText(w / 2 - 80, h - 4, xAxis, 12, "#334155");

  Object.entries(grouped)
    .sort(([a], [b]) => a.localeCompare(b))
    .forEach(([algorithm, rows], idx) => {
      const color = palette[idx % palette.length];
      const sorted = rows
        .filter((r) => Number.isFinite(Number(r[xAxis])) && Number.isFinite(Number(r[metric])))
        .sort((a, b) => Number(a[xAxis]) - Number(b[xAxis]));
      if (sorted.length === 0) {
        return;
      }
      const points = sorted.map((r) => `${xScale(Number(r[xAxis]))},${yScale(Number(r[metric]))}`).join(" ");
      const poly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      poly.setAttribute("points", points);
      poly.setAttribute("fill", "none");
      poly.setAttribute("stroke", color);
      poly.setAttribute("stroke-width", "2.6");
      chart.appendChild(poly);

      sorted.forEach((r) => {
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", xScale(Number(r[xAxis])));
        circle.setAttribute("cy", yScale(Number(r[metric])));
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

function normalizeEvalJson(payload, fileName) {
  const method = methodLabel(payload, fileName);
  const task = String(payload.task ?? "unknown");
  const targetCol = payload.target_col ?? "-";
  const metrics = {};
  if (payload.downstream_metrics && typeof payload.downstream_metrics === "object") {
    Object.assign(metrics, payload.downstream_metrics);
  }
  if (payload.multiple_imputation_metrics && typeof payload.multiple_imputation_metrics === "object") {
    Object.assign(metrics, payload.multiple_imputation_metrics);
  }
  if (payload.imputation_metrics && typeof payload.imputation_metrics === "object") {
    Object.assign(metrics, payload.imputation_metrics);
  }
  return { fileName, method, task, targetCol, metrics };
}

function methodLabel(payload, fileName) {
  const paths = Array.isArray(payload.prediction_paths) ? payload.prediction_paths : [];
  if (paths.length > 0) {
    const p = String(paths[0]);
    const parts = p.split("/").filter(Boolean);
    if (parts.length >= 2) {
      return parts[parts.length - 2];
    }
  }
  return fileName.replace(/\.json$/i, "");
}

function getJsonMetricKeys() {
  const keySet = new Set();
  for (const entry of state.jsonEvalEntries) {
    for (const [k, v] of Object.entries(entry.metrics)) {
      if (typeof v === "number" && Number.isFinite(v)) {
        keySet.add(k);
      }
    }
  }
  return Array.from(keySet).sort();
}

function populateJsonMetricSelector() {
  jsonMetricSelect.innerHTML = "";
  const keys = getJsonMetricKeys();
  for (const key of keys) {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = key;
    jsonMetricSelect.appendChild(opt);
  }
  const preferred = [
    "downstream_accuracy_xgboost",
    "downstream_accuracy_random_forest",
    "downstream_accuracy_linear",
    "downstream_accuracy_xgboost_mi_pooled",
    "downstream_accuracy_random_forest_mi_pooled",
    "downstream_accuracy_linear_mi_pooled",
    "nrmse",
    "nrmse_mi_mean",
  ];
  jsonMetricSelect.value = preferred.find((k) => keys.includes(k)) ?? keys[0] ?? "";
}

function renderJsonExplorer() {
  renderJsonTable();
  renderJsonChart();
}

function renderJsonTable() {
  jsonTableBody.innerHTML = "";
  if (state.jsonEvalEntries.length === 0) {
    jsonTableBody.innerHTML = '<tr><td colspan="5">Upload tabular evaluation JSON files to view results.</td></tr>';
    return;
  }
  const metricKey = jsonMetricSelect.value;
  for (const entry of state.jsonEvalEntries) {
    const value = Number(entry.metrics[metricKey]);
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(entry.method)}</td>
      <td>${escapeHtml(entry.task)}</td>
      <td>${escapeHtml(String(entry.targetCol))}</td>
      <td>${escapeHtml(metricKey || "-")}</td>
      <td>${formatNum(value)}</td>
    `;
    jsonTableBody.appendChild(tr);
  }
}

function renderJsonChart() {
  jsonChart.innerHTML = "";
  const metricKey = jsonMetricSelect.value;
  if (state.jsonEvalEntries.length === 0) {
    drawJsonText(30, 170, "Upload evaluation JSON files to render chart.");
    return;
  }
  if (!metricKey) {
    drawJsonText(30, 170, "Select a downstream metric.");
    return;
  }
  const data = state.jsonEvalEntries
    .map((entry) => ({ label: entry.method, value: Number(entry.metrics[metricKey]) }))
    .filter((d) => Number.isFinite(d.value));
  if (data.length === 0) {
    drawJsonText(30, 170, `No numeric values for metric: ${metricKey}`);
    return;
  }

  const w = 900;
  const h = 340;
  const m = { left: 60, right: 25, top: 30, bottom: 70 };
  const cw = w - m.left - m.right;
  const ch = h - m.top - m.bottom;
  const maxVal = Math.max(...data.map((d) => d.value));
  const barW = Math.max(18, Math.min(80, cw / Math.max(data.length * 1.6, 1)));
  const gap = barW * 0.6;

  drawJsonLine(m.left, m.top, m.left, h - m.bottom, "#c6bea9", 1);
  drawJsonLine(m.left, h - m.bottom, w - m.right, h - m.bottom, "#c6bea9", 1);
  drawJsonText(10, 20, metricKey, 13, "#334155", "700");

  data.forEach((d, idx) => {
    const x = m.left + idx * (barW + gap) + gap * 0.5;
    const y = m.top + (1 - d.value / safeDen(maxVal)) * ch;
    const bar = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    bar.setAttribute("x", x);
    bar.setAttribute("y", y);
    bar.setAttribute("width", barW);
    bar.setAttribute("height", Math.max(1, h - m.bottom - y));
    bar.setAttribute("fill", palette[idx % palette.length]);
    bar.setAttribute("rx", "4");
    jsonChart.appendChild(bar);

    drawJsonText(x, h - m.bottom + 18, d.label.slice(0, 14), 11, "#334155");
    drawJsonText(x, y - 6, formatNum(d.value), 11, "#334155");
  });
}

function groupByAlgorithm(rows) {
  const out = {};
  for (const row of rows) {
    const key = seriesKey(row);
    if (!out[key]) {
      out[key] = [];
    }
    out[key].push(row);
  }
  return out;
}

function seriesKey(row) {
  return String(row.algorithm ?? row.method ?? row.model ?? "series");
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

function drawJsonLine(x1, y1, x2, y2, color, width) {
  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", x1);
  line.setAttribute("y1", y1);
  line.setAttribute("x2", x2);
  line.setAttribute("y2", y2);
  line.setAttribute("stroke", color);
  line.setAttribute("stroke-width", width);
  jsonChart.appendChild(line);
}

function drawJsonText(x, y, text, size = 14, color = "#374151", weight = "400") {
  const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
  t.setAttribute("x", x);
  t.setAttribute("y", y);
  t.setAttribute("font-size", String(size));
  t.setAttribute("fill", color);
  t.setAttribute("font-family", "IBM Plex Mono, monospace");
  t.setAttribute("font-weight", weight);
  t.textContent = text;
  jsonChart.appendChild(t);
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

async function loadDemoCsv() {
  const candidates = [
    "./data/noise_sweep_results.csv",
    "./data/hankel_results.csv",
  ];
  for (const path of candidates) {
    try {
      const res = await fetch(path, { cache: "no-store" });
      if (!res.ok) {
        continue;
      }
      const text = await res.text();
      const parsed = parseCsv(text);
      if (parsed.rows.length === 0) {
        continue;
      }
      state.rows = parsed.rows;
      state.headers = parsed.headers;
      state.numericColumns = parsed.numericColumns;
      populateSelectors();
      render();
      setDemoStatus(`Loaded demo CSV: ${path}`);
      return;
    } catch {
      // try next path
    }
  }
  setDemoStatus("No demo CSV found in website/data/. Add noise_sweep_results.csv or hankel_results.csv.");
}

async function loadDemoJson() {
  const candidates = [
    "./data/tabular_soft_impute_eval.json",
    "./data/tabular_mi_eval.json",
    "./data/tabular_eval.json",
  ];
  const entries = [];
  for (const path of candidates) {
    try {
      const res = await fetch(path, { cache: "no-store" });
      if (!res.ok) {
        continue;
      }
      const payload = await res.json();
      entries.push(normalizeEvalJson(payload, path.split("/").pop() ?? path));
    } catch {
      // ignore candidate
    }
  }
  if (entries.length === 0) {
    setDemoStatus("No demo JSON found in website/data/. Add tabular_*_eval.json files.");
    return;
  }
  state.jsonEvalEntries = entries;
  populateJsonMetricSelector();
  renderJsonExplorer();
  setDemoStatus(`Loaded ${entries.length} demo JSON file(s) from website/data/.`);
}

function setDemoStatus(message) {
  demoStatus.textContent = message;
}

render();
renderJsonExplorer();
