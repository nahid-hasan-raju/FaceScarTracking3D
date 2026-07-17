// viewer3d.js
// Loads and displays a .ply point cloud / mesh with mouse-drag orbit,
// scroll zoom, and right-drag pan -- exposed as window.render3DModel()
// so the classic (non-module) scan_files.js can call into it.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

let currentCleanup = null;

async function render3DModel(container, plyUrl) {
  // Tear down any previously-open viewer before starting a new one.
  if (currentCleanup) {
    currentCleanup();
    currentCleanup = null;
  }
  container.innerHTML = "";

  const width = container.clientWidth || 600;
  const height = container.clientHeight || 480;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1b1e24);

  const camera = new THREE.PerspectiveCamera(50, width / height, 0.01, 1000);
  camera.position.set(0, 0, 2);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  scene.add(new THREE.AmbientLight(0xffffff, 0.9));
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
  dirLight.position.set(1, 1, 1);
  scene.add(dirLight);

  const loadingDiv = document.createElement("div");
  loadingDiv.className = "viewer3d-loading";
  loadingDiv.textContent = "Loading model…";
  container.appendChild(loadingDiv);

  const loader = new PLYLoader();
  let mesh = null;

  try {
    const geometry = await loader.loadAsync(plyUrl);
    geometry.computeVertexNormals();

    let material;
    if (geometry.hasAttribute("color")) {
      material = new THREE.PointsMaterial({ size: 0.01, vertexColors: true });
      mesh = geometry.index
        ? new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ vertexColors: true }))
        : new THREE.Points(geometry, material);
    } else if (geometry.index) {
      material = new THREE.MeshStandardMaterial({ color: 0x4fb2e0, flatShading: false });
      mesh = new THREE.Mesh(geometry, material);
    } else {
      material = new THREE.PointsMaterial({ size: 0.01, color: 0x4fb2e0 });
      mesh = new THREE.Points(geometry, material);
    }

    // Center + scale the model so it's always framed nicely regardless
    // of what units/scale the source .ply used.
    geometry.computeBoundingSphere();
    const sphere = geometry.boundingSphere;
    if (sphere && sphere.radius > 0) {
      mesh.position.sub(sphere.center);
      const scale = 1 / sphere.radius;
      mesh.scale.setScalar(scale);
    }

    scene.add(mesh);
    loadingDiv.remove();
  } catch (err) {
    loadingDiv.textContent = "Could not load 3D model: " + (err?.message || err);
    console.error(err);
  }

  let frameId;
  function animate() {
    frameId = requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  function handleResize() {
    const w = container.clientWidth || width;
    const h = container.clientHeight || height;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }
  window.addEventListener("resize", handleResize);

  currentCleanup = () => {
    cancelAnimationFrame(frameId);
    window.removeEventListener("resize", handleResize);
    controls.dispose();
    renderer.dispose();
    if (mesh) {
      mesh.geometry?.dispose();
      if (Array.isArray(mesh.material)) mesh.material.forEach((m) => m.dispose());
      else mesh.material?.dispose();
    }
  };

  return () => currentCleanup && currentCleanup();
}

window.render3DModel = render3DModel;
