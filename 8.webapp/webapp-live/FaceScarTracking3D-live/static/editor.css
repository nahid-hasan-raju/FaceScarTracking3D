:root {
  --bg: #1b1e24;
  --panel: #23262e;
  --panel-2: #2b2f38;
  --border: #383c46;
  --text: #e6e8ec;
  --muted: #9198a6;
  --accent: #4fb2e0;
  --accent-2: #e0654f;
  --good: #4fd18b;
  --danger: #e05a5a;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  height: 100vh;
  overflow: hidden;
}

.app {
  display: grid;
  grid-template-columns: 260px 1fr 280px;
  grid-template-rows: 48px 1fr;
  height: 100vh;
}

.topbar {
  grid-column: 1 / 4;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 16px;
  background: var(--panel);
  border-bottom: 1px solid var(--border);
}

.topbar .scan-id {
  font-weight: 600;
  letter-spacing: 0.02em;
}

.topbar .status {
  font-size: 12px;
  color: var(--muted);
}

.topbar .status.dirty { color: var(--accent-2); }
.topbar .status.saved { color: var(--good); }

.btn {
  background: var(--panel-2);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 6px 12px;
  border-radius: 6px;
  font-size: 13px;
  cursor: pointer;
}
.btn:hover { border-color: var(--accent); }
.btn.primary { background: var(--accent); color: #0d1117; border-color: var(--accent); font-weight: 600; }
.btn.danger { color: var(--danger); }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }

.sidebar-left, .sidebar-right {
  background: var(--panel);
  border-right: 1px solid var(--border);
  overflow-y: auto;
  padding: 12px;
}
.sidebar-right { border-right: none; border-left: 1px solid var(--border); }

.section-title {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin: 12px 0 6px;
}

.region-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px;
  border-radius: 6px;
  border: 1px solid transparent;
  cursor: pointer;
  margin-bottom: 4px;
}
.region-item:hover { background: var(--panel-2); }
.region-item.active { border-color: var(--accent); background: var(--panel-2); }
.region-swatch { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }
.region-item input {
  background: transparent;
  border: none;
  color: var(--text);
  font-size: 13px;
  width: 100%;
}
.region-item input:focus { outline: none; border-bottom: 1px solid var(--accent); }
.region-meta { font-size: 11px; color: var(--muted); }

.canvas-wrap {
  position: relative;
  overflow: hidden;
  background: repeating-conic-gradient(#20232a 0% 25%, #1b1e24 0% 50%) 50% / 24px 24px;
  cursor: crosshair;
}
canvas { display: block; }

.hint {
  position: absolute;
  bottom: 10px;
  left: 10px;
  font-size: 12px;
  color: var(--muted);
  background: rgba(0,0,0,0.4);
  padding: 6px 10px;
  border-radius: 6px;
  max-width: 60%;
  line-height: 1.5;
}

.field { margin-bottom: 10px; }
.field label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
.field select, .field input[type=text] {
  width: 100%;
  background: var(--panel-2);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 5px 8px;
  border-radius: 5px;
  font-size: 13px;
}

.empty-note {
  color: var(--muted);
  font-size: 12px;
  padding: 8px;
}

/* --- Other Files panel --- */
.extra-file-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 7px 8px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 12px;
  margin-bottom: 2px;
}
.extra-file-row:hover { background: var(--panel-2); }
.extra-file-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
}
.extra-file-tag {
  flex-shrink: 0;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: 2px 6px;
  border-radius: 999px;
  border: 1px solid var(--border);
  color: var(--muted);
}
.extra-file-tag.tag-mesh_3d { color: var(--accent); border-color: var(--accent); }
.extra-file-tag.tag-seg_image, .extra-file-tag.tag-image { color: var(--good); border-color: var(--good); }
.extra-file-tag.tag-data { color: var(--accent-2); border-color: var(--accent-2); }

/* --- Modal --- */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}
.modal-overlay.hidden { display: none; }
.modal-box {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  width: min(900px, 90vw);
  max-height: 85vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  font-size: 13px;
  font-weight: 600;
}
.modal-body {
  padding: 14px;
  overflow: auto;
}
.modal-image {
  max-width: 100%;
  max-height: 70vh;
  display: block;
  margin: 0 auto;
  border-radius: 6px;
}
.modal-json {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--text);
  margin: 0;
}
.viewer3d-container {
  width: 100%;
  height: 70vh;
  position: relative;
  border-radius: 6px;
  overflow: hidden;
}
.viewer3d-loading {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: var(--muted);
  font-size: 13px;
}