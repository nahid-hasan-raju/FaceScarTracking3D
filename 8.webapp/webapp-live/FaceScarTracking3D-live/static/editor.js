/*
 * Burn Region Polygon Editor
 * Works against three endpoints configured in window.EDITOR_CONFIG:
 *   imageUrl          -> GET, returns the scan image (PNG/JPEG bytes)
 *   polygonsUrl        -> GET, returns JSON: { scan_id, image_size:[w,h], regions:[...] }
 *   polygonsSaveUrl     -> POST, body = same JSON shape, saves back to disk
 *
 * Region shape:
 *   { id, label, source: "sam2"|"manual"|"manual_edit", confidence, polygon: [[x,y], ...] }
 *
 * If your existing _burn_polygons.json uses different field names, adjust the
 * translation in normalizeIncoming() / denormalizeOutgoing() below -- everything
 * else in this file works on the normalized shape.
 */

const COLORS = ["#4fb2e0", "#e0654f", "#4fd18b", "#e0c84f", "#b04fe0", "#e04f9a"];

const state = {
  image: null,
  imageSize: [0, 0],
  regions: [],       // normalized regions
  activeRegionId: null,
  drawingRegion: null, // region currently being drawn (not yet closed)
  dragging: null,       // { regionId, pointIndex }
  panning: false,
  panStart: null,
  offset: { x: 0, y: 0 },
  scale: 1,
  dirty: false,
  nextId: 1,
  history: [],
};

const canvas = document.getElementById("editor-canvas");
const ctx = canvas.getContext("2d");
const wrap = document.getElementById("canvas-wrap");
const regionList = document.getElementById("region-list");
const statusEl = document.getElementById("status");

function normalizeIncoming(data) {
  const regions = (data.regions || []).map((r, i) => ({
    id: r.id ?? i + 1,
    label: r.label || `region_${i + 1}`,
    source: r.source || "sam2",
    confidence: r.confidence ?? null,
    polygon: (r.polygon || []).map((p) => [p[0], p[1]]),
  }));
  return { imageSize: data.image_size || [0, 0], regions };
}

function denormalizeOutgoing() {
  return {
    scan_id: window.EDITOR_CONFIG.scanId,
    image_size: state.imageSize,
    regions: state.regions
      .filter((r) => r.polygon.length >= 3)
      .map((r) => ({
        id: r.id,
        label: r.label,
        source: r.source,
        confidence: r.confidence,
        polygon: r.polygon,
      })),
  };
}

function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = "status" + (cls ? " " + cls : "");
}

function markDirty() {
  state.dirty = true;
  setStatus("Unsaved changes", "dirty");
}

function pushHistory() {
  state.history.push(JSON.stringify(state.regions));
  if (state.history.length > 50) state.history.shift();
}

function undo() {
  if (!state.history.length) return;
  state.regions = JSON.parse(state.history.pop());
  render();
  markDirty();
}

// ---------- coordinate transforms ----------
function imgToScreen(x, y) {
  return [x * state.scale + state.offset.x, y * state.scale + state.offset.y];
}
function screenToImg(x, y) {
  return [(x - state.offset.x) / state.scale, (y - state.offset.y) / state.scale];
}

function fitToScreen() {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  canvas.width = w;
  canvas.height = h;
  const [iw, ih] = state.imageSize;
  if (!iw || !ih) return;
  const scale = Math.min(w / iw, h / ih) * 0.92;
  state.scale = scale;
  state.offset.x = (w - iw * scale) / 2;
  state.offset.y = (h - ih * scale) / 2;
}

// ---------- rendering ----------
function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (state.image) {
    const [x0, y0] = imgToScreen(0, 0);
    ctx.drawImage(state.image, x0, y0, state.imageSize[0] * state.scale, state.imageSize[1] * state.scale);
  }

  state.regions.forEach((r, i) => drawRegion(r, colorFor(r, i)));
  if (state.drawingRegion) {
    drawRegion(state.drawingRegion, "#ffffff", true);
    // Highlight the first point specially once closing is possible, so it's
    // obvious where to click to close the loop.
    if (state.drawingRegion.polygon.length >= 3) {
      const [fx, fy] = imgToScreen(state.drawingRegion.polygon[0][0], state.drawingRegion.polygon[0][1]);
      ctx.beginPath();
      ctx.arc(fx, fy, HIT_RADIUS + 4, 0, Math.PI * 2);
      ctx.strokeStyle = "#4fd18b";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  renderRegionList();
  updateToolbarState();
}

function colorFor(r, i) {
  return COLORS[i % COLORS.length];
}

function drawRegion(region, color, isDrawing) {
  const pts = region.polygon.map(([x, y]) => imgToScreen(x, y));
  if (pts.length === 0) return;

  ctx.beginPath();
  pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
  if (!isDrawing && pts.length > 2) ctx.closePath();
  ctx.strokeStyle = color;
  ctx.lineWidth = region.id === state.activeRegionId ? 3 : 2;
  ctx.stroke();

  if (!isDrawing && pts.length > 2) {
    ctx.fillStyle = color + "33";
    ctx.fill();
  }

  pts.forEach(([x, y]) => {
    ctx.beginPath();
    ctx.arc(x, y, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = "#0d1117";
    ctx.lineWidth = 1;
    ctx.stroke();
  });
}

// ---------- region list panel ----------
function renderRegionList() {
  regionList.innerHTML = "";
  if (!state.regions.length) {
    const p = document.createElement("div");
    p.className = "empty-note";
    p.textContent = "No regions yet. Click \"New Polygon\" then click on the image to start drawing.";
    regionList.appendChild(p);
    return;
  }
  state.regions.forEach((r, i) => {
    const div = document.createElement("div");
    div.className = "region-item" + (r.id === state.activeRegionId ? " active" : "");
    div.innerHTML = `
      <div class="region-swatch" style="background:${colorFor(r, i)}"></div>
      <div style="flex:1">
        <input type="text" value="${r.label}" data-id="${r.id}" />
        <div class="region-meta">${r.source}${r.confidence != null ? " · conf " + r.confidence : ""} · ${r.polygon.length} pts</div>
      </div>
      <button class="btn danger" data-delete="${r.id}" title="Delete region">✕</button>
    `;
    div.querySelector("input").addEventListener("click", (e) => e.stopPropagation());
    div.querySelector("input").addEventListener("change", (e) => {
      const reg = state.regions.find((rr) => rr.id === r.id);
      reg.label = e.target.value;
      markDirty();
    });
    div.querySelector("[data-delete]").addEventListener("click", (e) => {
      e.stopPropagation();
      pushHistory();
      state.regions = state.regions.filter((rr) => rr.id !== r.id);
      if (state.activeRegionId === r.id) state.activeRegionId = null;
      render();
      markDirty();
    });
    div.addEventListener("click", () => {
      state.activeRegionId = r.id;
      render();
    });
    regionList.appendChild(div);
  });
}

// ---------- interaction ----------
const HIT_RADIUS = 8;

function findVertexAt(sx, sy) {
  for (const r of state.regions) {
    for (let i = 0; i < r.polygon.length; i++) {
      const [px, py] = imgToScreen(r.polygon[i][0], r.polygon[i][1]);
      if (Math.hypot(px - sx, py - sy) <= HIT_RADIUS) return { region: r, index: i };
    }
  }
  return null;
}

canvas.addEventListener("mousedown", (e) => {
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left, sy = e.clientY - rect.top;

  if (e.button === 1 || (e.button === 0 && e.altKey)) {
    state.panning = true;
    state.panStart = { x: e.clientX, y: e.clientY, ox: state.offset.x, oy: state.offset.y };
    return;
  }

  if (state.drawingRegion) {
    const pts = state.drawingRegion.polygon;
    // Clicking near the first placed point closes the polygon (standard
    // polygon-tool convention) -- this is the most reliable way to close.
    if (pts.length >= 3) {
      const [fx, fy] = imgToScreen(pts[0][0], pts[0][1]);
      if (Math.hypot(fx - sx, fy - sy) <= HIT_RADIUS + 4) {
        finishDrawing();
        return;
      }
    }
    const [ix, iy] = screenToImg(sx, sy);
    pts.push([ix, iy]);
    render();
    return;
  }

  const hit = findVertexAt(sx, sy);
  if (hit && e.button === 0) {
    pushHistory();
    state.dragging = { regionId: hit.region.id, pointIndex: hit.index };
    state.activeRegionId = hit.region.id;
    render();
    return;
  }
  if (hit && e.button === 2) {
    e.preventDefault();
    pushHistory();
    hit.region.polygon.splice(hit.index, 1);
    render();
    markDirty();
  }
});

window.addEventListener("mousemove", (e) => {
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left, sy = e.clientY - rect.top;

  if (state.panning) {
    state.offset.x = state.panStart.ox + (e.clientX - state.panStart.x);
    state.offset.y = state.panStart.oy + (e.clientY - state.panStart.y);
    render();
    return;
  }
  if (state.dragging) {
    const [ix, iy] = screenToImg(sx, sy);
    const region = state.regions.find((r) => r.id === state.dragging.regionId);
    region.polygon[state.dragging.pointIndex] = [ix, iy];
    render();
  }
});

window.addEventListener("mouseup", () => {
  if (state.dragging) markDirty();
  state.dragging = null;
  state.panning = false;
});

canvas.addEventListener("contextmenu", (e) => e.preventDefault());

canvas.addEventListener("dblclick", () => {
  if (!state.drawingRegion) return;
  // Each click of the double-click already added a point via mousedown;
  // drop those two duplicate points before closing.
  state.drawingRegion.polygon.splice(-2, 2);
  if (state.drawingRegion.polygon.length >= 3) {
    finishDrawing();
  } else {
    render();
  }
});

canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
  const [ix, iy] = screenToImg(sx, sy);
  const factor = e.deltaY < 0 ? 1.1 : 0.9;
  state.scale *= factor;
  const [nx, ny] = imgToScreen(ix, iy);
  state.offset.x += sx - nx;
  state.offset.y += sy - ny;
  render();
}, { passive: false });

window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && state.drawingRegion) {
    state.drawingRegion = null;
    updateToolbarState();
    render();
  }
  if (e.key === "Enter" && state.drawingRegion && state.drawingRegion.polygon.length >= 3) {
    finishDrawing();
  }
  if ((e.ctrlKey || e.metaKey) && e.key === "z") {
    e.preventDefault();
    undo();
  }
  if (e.key === "Delete" && state.activeRegionId != null && !state.drawingRegion) {
    pushHistory();
    state.regions = state.regions.filter((r) => r.id !== state.activeRegionId);
    state.activeRegionId = null;
    render();
    markDirty();
  }
});

function finishDrawing() {
  pushHistory();
  state.drawingRegion.source = "manual";
  state.regions.push(state.drawingRegion);
  state.activeRegionId = state.drawingRegion.id;
  state.drawingRegion = null;
  render();
  markDirty();
}

// ---------- toolbar ----------
const btnNewPolygon = document.getElementById("btn-new-polygon");
const btnFinishPolygon = document.getElementById("btn-finish-polygon");

btnNewPolygon.addEventListener("click", () => {
  if (state.drawingRegion) return; // already drawing
  state.drawingRegion = {
    id: state.nextId++,
    label: `region_${state.regions.length + 1}`,
    source: "manual",
    confidence: null,
    polygon: [],
  };
  updateToolbarState();
  render();
});

btnFinishPolygon.addEventListener("click", () => {
  if (!state.drawingRegion) return;
  if (state.drawingRegion.polygon.length < 3) {
    setStatus("Need at least 3 points to close a polygon", "dirty");
    return;
  }
  finishDrawing();
});

function updateToolbarState() {
  const drawing = !!state.drawingRegion;
  btnNewPolygon.disabled = drawing;
  btnFinishPolygon.disabled = !drawing || state.drawingRegion.polygon.length < 3;
  btnFinishPolygon.textContent = drawing
    ? `Finish Polygon (${state.drawingRegion.polygon.length} pts)`
    : "Finish Polygon";
}

document.getElementById("btn-undo").addEventListener("click", undo);

document.getElementById("btn-reset-view").addEventListener("click", () => {
  fitToScreen();
  render();
});

document.getElementById("btn-save").addEventListener("click", async () => {
  if (state.drawingRegion) {
    if (state.drawingRegion.polygon.length >= 3) {
      finishDrawing(); // don't let an in-progress polygon get silently dropped
    } else {
      setStatus("Finish or cancel (Esc) the polygon you're drawing before saving", "dirty");
      return;
    }
  }
  const payload = denormalizeOutgoing();
  setStatus("Saving...", "");
  try {
    const res = await fetch(window.EDITOR_CONFIG.polygonsSaveUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await res.text());
    state.dirty = false;
    setStatus("Saved", "saved");
  } catch (err) {
    setStatus("Save failed: " + err.message, "dirty");
  }
});

window.addEventListener("beforeunload", (e) => {
  if (state.dirty) {
    e.preventDefault();
    e.returnValue = "";
  }
});

// ---------- boot ----------
async function boot() {
  setStatus("Loading...", "");
  const [polyRes, img] = await Promise.all([
    fetch(window.EDITOR_CONFIG.polygonsUrl).then((r) => r.json()),
    loadImage(window.EDITOR_CONFIG.imageUrl),
  ]);
  const norm = normalizeIncoming(polyRes);
  state.imageSize = norm.imageSize.length ? norm.imageSize : [img.naturalWidth, img.naturalHeight];
  state.regions = norm.regions;
  state.nextId = Math.max(1, ...state.regions.map((r) => r.id + 1), 1);
  state.image = img;
  fitToScreen();
  render();
  setStatus("Loaded", "saved");
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

window.addEventListener("resize", () => {
  fitToScreen();
  render();
});

boot();