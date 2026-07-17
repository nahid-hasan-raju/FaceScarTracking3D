// scan_files.js
// Populates the "Other Files" sidebar section (everything besides the main
// .tif and the burn_polygons.json, which the editor itself already handles)
// and wires up a modal for previewing JSON data and .ply 3D models.

(function () {
  const listEl = document.getElementById("extra-files-list");
  const overlay = document.getElementById("modal-overlay");
  const modalTitle = document.getElementById("modal-title");
  const modalBody = document.getElementById("modal-body");
  const modalClose = document.getElementById("modal-close");
  const scanId = window.EDITOR_CONFIG.scanId;

  function closeModal() {
    overlay.classList.add("hidden");
    modalBody.innerHTML = "";
  }
  modalClose.addEventListener("click", closeModal);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) closeModal();
  });

  function openModal(title) {
    modalTitle.textContent = title;
    modalBody.innerHTML = "";
    overlay.classList.remove("hidden");
  }

  function labelFor(category) {
    return {
      mesh_3d: "3D",
      seg_image: "Image",
      image: "Image",
      data: "Data",
      other: "File",
    }[category] || "File";
  }

  function renderList(files) {
    if (!files.length) {
      listEl.innerHTML = '<div class="empty-note">No other files for this scan.</div>';
      return;
    }
    listEl.innerHTML = "";
    for (const f of files) {
      const row = document.createElement("div");
      row.className = "extra-file-row";

      const nameSpan = document.createElement("span");
      nameSpan.className = "extra-file-name";
      nameSpan.textContent = f.name;
      nameSpan.title = f.name;

      const tag = document.createElement("span");
      tag.className = "extra-file-tag tag-" + f.category;
      tag.textContent = labelFor(f.category);

      row.appendChild(nameSpan);
      row.appendChild(tag);
      row.addEventListener("click", () => handleOpen(f));
      listEl.appendChild(row);
    }
  }

  function handleOpen(f) {
    if (f.category === "mesh_3d") {
      openModal(f.name);
      const container = document.createElement("div");
      container.className = "viewer3d-container";
      modalBody.appendChild(container);
      const rawUrl = `/api/raw/${encodeURIComponent(scanId)}/${encodeURIComponent(f.id)}`;
      if (window.render3DModel) {
        window.render3DModel(container, rawUrl);
      } else {
        container.textContent = "3D viewer failed to load.";
      }
    } else if (f.category === "seg_image") {
      openModal(f.name);
      const img = document.createElement("img");
      img.className = "modal-image";
      img.src = `/api/preview_tif/${encodeURIComponent(scanId)}/${encodeURIComponent(f.id)}`;
      modalBody.appendChild(img);
    } else if (f.category === "image") {
      openModal(f.name);
      const img = document.createElement("img");
      img.className = "modal-image";
      img.src = `/api/raw/${encodeURIComponent(scanId)}/${encodeURIComponent(f.id)}`;
      modalBody.appendChild(img);
    } else if (f.category === "data") {
      openModal(f.name);
      const pre = document.createElement("pre");
      pre.className = "modal-json";
      pre.textContent = "Loading…";
      modalBody.appendChild(pre);
      fetch(`/api/preview_json/${encodeURIComponent(scanId)}/${encodeURIComponent(f.id)}`)
        .then((r) => r.json().catch(() => r.text()))
        .then((data) => {
          pre.textContent = typeof data === "string" ? data : JSON.stringify(data, null, 2);
        })
        .catch((err) => {
          pre.textContent = "Failed to load: " + err;
        });
    } else {
      // Unrecognized file type -- just offer it as a plain download/open link.
      window.open(`/api/raw/${encodeURIComponent(scanId)}/${encodeURIComponent(f.id)}`, "_blank");
    }
  }

  fetch(window.EDITOR_CONFIG.scanFilesUrl)
    .then((r) => r.json())
    .then(renderList)
    .catch(() => {
      listEl.innerHTML = '<div class="empty-note">Could not load extra files.</div>';
    });
})();
