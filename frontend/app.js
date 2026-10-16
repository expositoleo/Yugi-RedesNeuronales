// Front-end del reconocedor de cartas Yu-Gi-Oh!
// Soporta:
// 1. Escaneo y detección dinámica en tiempo real (exclusivamente cuando hay una carta real en encuadre).
// 2. Filtros de presencia de carta (luminancia, varianza espacial y bordes) para evitar falsos positivos con cámara apagada o fondos vacíos.
// 3. Auto-congelamiento (auto-lock) únicamente cuando una carta es reconocida positivamente por el modelo.
// 4. Sistema de feedback: buscador interactivo, promediado L2 de embeddings y alta de nuevas cartas.

// --- Elementos del DOM ---
const tabUpload = document.getElementById("tab-upload");
const tabCamera = document.getElementById("tab-camera");
const uploadMode = document.getElementById("upload-mode");
const cameraMode = document.getElementById("camera-mode");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const captureBtn = document.getElementById("capture-btn");
const resumeScanBtn = document.getElementById("resume-scan-btn");
const autoDetectToggle = document.getElementById("auto-detect-toggle");
const detectionIndicator = document.getElementById("detection-indicator");
const detectionIndicatorText = document.getElementById("detection-indicator-text");
const cardGuide = document.getElementById("card-guide");

// Elementos de la nueva pestaña: Mesa Completa (Multi-carta)
const tabMulti = document.getElementById("tab-multi");
const multiMode = document.getElementById("multi-mode");
const multiVideo = document.getElementById("multi-video");
const multiOverlayCanvas = document.getElementById("multi-overlay-canvas");
const multiHiddenCanvas = document.getElementById("multi-hidden-canvas");
const multiScanBtn = document.getElementById("multi-scan-btn");
const multiAutoToggle = document.getElementById("multi-auto-toggle");
const multiDetectionIndicator = document.getElementById("multi-detection-indicator");
const multiDetectionText = document.getElementById("multi-detection-text");
const multiScanLaser = document.getElementById("multi-scan-laser");
const multiUploadBtn = document.getElementById("multi-upload-btn");
const multiFileInput = document.getElementById("multi-file-input");

const previewBox = document.getElementById("preview-box");
const preview = document.getElementById("preview");
const identifyBtn = document.getElementById("identify-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const resultsFooter = document.getElementById("results-footer");
const unrecognizedBox = document.getElementById("unrecognized-box");
const unrecCalibrateBtn = document.getElementById("unrec-calibrate-btn");
const unrecAddBtn = document.getElementById("unrec-add-btn");
const openCalibrateBtn = document.getElementById("open-calibrate-btn");

// Modal elementos
const calibrateModal = document.getElementById("calibrate-modal");
const closeModalBtn = document.getElementById("close-modal-btn");
const modalThumb = document.getElementById("modal-thumb");
const modalTabSearch = document.getElementById("modal-tab-search");
const modalTabCreate = document.getElementById("modal-tab-create");
const modalPanelSearch = document.getElementById("modal-panel-search");
const modalPanelCreate = document.getElementById("modal-panel-create");
const cardSearchInput = document.getElementById("card-search-input");
const clearSearchBtn = document.getElementById("clear-search-btn");
const cardSearchResults = document.getElementById("card-search-results");
const newCardName = document.getElementById("new-card-name");
const newCardId = document.getElementById("new-card-id");
const submitNewCardBtn = document.getElementById("submit-new-card-btn");

const toast = document.getElementById("toast");

// --- Constantes y Estado ---
const CARD_ASPECT = 268 / 391; // Aspect ratio exacto de cartas Yu-Gi-Oh
let selectedBlob = null;
let cameraStream = null;
let cameraSessionId = 0;

// Control del loop de tiempo real
let isRealtimeActive = true;
let isScanLoopRunning = false;
let isDetecting = false;
let isFrozen = false;
let realtimeIntervalId = null;
let lastSamplePixels = null;
let steadyCount = 0;

// Canvas oculto para análisis rápido de luminancia, contraste y movimiento
const motionCanvas = document.createElement("canvas");
motionCanvas.width = 80;
motionCanvas.height = 116;
const motionCtx = motionCanvas.getContext("2d", { willReadFrequently: true });

// --- Inicialización y Pestañas ---
tabUpload.addEventListener("click", () => switchTab("upload"));
tabCamera.addEventListener("click", () => switchTab("camera"));
tabMulti.addEventListener("click", () => switchTab("multi"));

function switchTab(mode) {
  const isUpload = mode === "upload";
  const isCamera = mode === "camera";
  const isMulti = mode === "multi";

  tabUpload.classList.toggle("active", isUpload);
  tabCamera.classList.toggle("active", isCamera);
  tabMulti.classList.toggle("active", isMulti);

  uploadMode.hidden = !isUpload;
  cameraMode.hidden = !isCamera;
  multiMode.hidden = !isMulti;

  stopScanLoop();
  if (typeof stopMultiScanLoop === "function") stopMultiScanLoop();
  if (typeof clearMultiOverlay === "function") clearMultiOverlay();

  if (isUpload) {
    stopCamera();
    if (typeof stopMultiCamera === "function") stopMultiCamera();
    previewBox.hidden = !selectedBlob;
    resultsFooter.hidden = false;
    setStatus("Elegí una imagen o activá la cámara para comenzar.");
  } else if (isCamera) {
    if (typeof stopMultiCamera === "function") stopMultiCamera();
    previewBox.hidden = !isFrozen;
    resultsFooter.hidden = false;
    startCamera();
  } else if (isMulti) {
    stopCamera();
    previewBox.hidden = true;
    resultsFooter.hidden = true;
    unrecognizedBox.hidden = true;
    startMultiCamera();
  }
}

// --- Manejo Seguro de la Cámara ---
function isCameraStreaming() {
  if (cameraMode.hidden || !cameraStream || !video.srcObject) return false;
  const tracks = cameraStream.getVideoTracks();
  if (!tracks || tracks.length === 0 || tracks[0].readyState !== "live") return false;
  if (video.paused || video.ended || video.readyState < 2) return false;
  if (!video.videoWidth || !video.videoHeight) return false;
  return true;
}

async function startCamera() {
  const currentSession = ++cameraSessionId;
  stopScanLoop();
  stopCameraTracks();

  setDetectionBadge("idle", "Iniciando cámara...");
  setStatus("Esperando que la cámara se active...");

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "environment",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });

    // Si el usuario cambió de pestaña mientras pedía permisos, descartar stream
    if (currentSession !== cameraSessionId || cameraMode.hidden) {
      stream.getTracks().forEach((t) => t.stop());
      return;
    }

    cameraStream = stream;
    video.srcObject = cameraStream;
    await video.play();

    isFrozen = false;
    resumeScanBtn.hidden = true;
    captureBtn.hidden = false;
    cameraMode.classList.toggle("realtime-active", isRealtimeActive);

    setStatus("Alineá una carta dentro del marco dorado.");
    setDetectionBadge("scanning", "Buscando carta...");

    startScanLoop();
  } catch (err) {
    if (currentSession === cameraSessionId) {
      setStatus("No se pudo acceder a la cámara: " + err.message, "warn");
      setDetectionBadge("idle", "Cámara apagada");
    }
  }
}

function stopCameraTracks() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((t) => t.stop());
    cameraStream = null;
  }
  if (video.srcObject) {
    video.srcObject = null;
  }
}

function stopCamera() {
  cameraSessionId++;
  stopScanLoop();
  stopCameraTracks();
  setDetectionBadge("idle", "Cámara apagada");
}

function guideRectInVideo() {
  const vw = video.videoWidth || 0;
  const vh = video.videoHeight || 0;
  if (vw === 0 || vh === 0) return { x: 0, y: 0, w: 0, h: 0 };

  const guideH = vh * 0.8;
  const guideW = guideH * CARD_ASPECT;
  const x = (vw - guideW) / 2;
  const y = (vh - guideH) / 2;
  return {
    x: Math.max(0, Math.round(x)),
    y: Math.max(0, Math.round(y)),
    w: Math.min(vw, Math.round(guideW)),
    h: Math.min(vh, Math.round(guideH)),
  };
}

// --- Loop de Detección Dinámica en Real-Time ---
autoDetectToggle.addEventListener("change", (e) => {
  isRealtimeActive = e.target.checked;
  cameraMode.classList.toggle("realtime-active", isRealtimeActive);
  if (isRealtimeActive) {
    setDetectionBadge("scanning", "Escaneando en vivo...");
  } else {
    setDetectionBadge("idle", "Modo manual");
  }
});

function startScanLoop() {
  if (isScanLoopRunning) return;
  isScanLoopRunning = true;
  lastSamplePixels = null;
  steadyCount = 0;

  // Evalúa cada 350ms
  realtimeIntervalId = setInterval(processRealtimeTick, 350);
}

function stopScanLoop() {
  isScanLoopRunning = false;
  if (realtimeIntervalId) {
    clearInterval(realtimeIntervalId);
    realtimeIntervalId = null;
  }
}

/**
 * Evalúa si dentro del marco de la guía hay visualmente un objeto con las
 * características de una carta (luminancia media, varianza espacial y bordes)
 * en lugar de una cámara apagada, tapada o un fondo liso uniforme (mesa/pared).
 */
function evaluateCardPresence(pixels) {
  let sumL = 0;
  const totalPixels = pixels.length / 4;

  for (let i = 0; i < pixels.length; i += 4) {
    const l = 0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2];
    sumL += l;
  }
  const meanL = sumL / totalPixels;

  // 1. Cámara apagada, tapada o en oscuridad extrema (< 28)
  if (meanL < 28) {
    return { hasCard: false, reason: "dark" };
  }
  // 2. Sobreexposición completa (> 245)
  if (meanL > 245) {
    return { hasCard: false, reason: "bright" };
  }

  // 3. Varianza y contraste de bordes
  let sumVar = 0;
  let edgeSum = 0;
  const width = motionCanvas.width;   // 80
  const height = motionCanvas.height; // 116

  for (let y = 1; y < height - 1; y += 2) {
    for (let x = 1; x < width - 1; x += 2) {
      const idx = (y * width + x) * 4;
      const l = 0.299 * pixels[idx] + 0.587 * pixels[idx + 1] + 0.114 * pixels[idx + 2];
      const diff = l - meanL;
      sumVar += diff * diff;

      const idxRight = (y * width + (x + 1)) * 4;
      const idxDown = ((y + 1) * width + x) * 4;
      const lRight = 0.299 * pixels[idxRight] + 0.587 * pixels[idxRight + 1] + 0.114 * pixels[idxRight + 2];
      const lDown = 0.299 * pixels[idxDown] + 0.587 * pixels[idxDown + 1] + 0.114 * pixels[idxDown + 2];

      edgeSum += Math.abs(l - lRight) + Math.abs(l - lDown);
    }
  }

  const sampleCount = ((height - 2) / 2) * ((width - 2) / 2);
  const stdDev = Math.sqrt(sumVar / sampleCount);
  const avgEdge = edgeSum / sampleCount;

  // Superficie uniforme / mesa vacía / sin textura de carta
  if (stdDev < 18 || avgEdge < 8.5) {
    return { hasCard: false, reason: "flat" };
  }

  return { hasCard: true };
}

async function processRealtimeTick() {
  // Verificación estricta: si la cámara no está transmitiendo activamente, salir inmediatamente
  if (!isCameraStreaming() || isFrozen || !isRealtimeActive || isDetecting) {
    return;
  }

  const { x, y, w, h } = guideRectInVideo();
  if (w < 30 || h < 30) return;

  // 1. Extraer miniatura para análisis de presencia y movimiento
  motionCtx.drawImage(video, x, y, w, h, 0, 0, motionCanvas.width, motionCanvas.height);
  const currentPixels = motionCtx.getImageData(0, 0, motionCanvas.width, motionCanvas.height).data;

  // 2. Comprobar presencia visual de carta (evita falsos positivos con cámara apagada o mesa vacía)
  const presence = evaluateCardPresence(currentPixels);
  if (!presence.hasCard) {
    steadyCount = 0;
    lastSamplePixels = null;
    if (presence.reason === "dark") {
      setDetectionBadge("idle", "Cámara oscura o apagada");
    } else {
      setDetectionBadge("scanning", "Apuntá a una carta...");
    }
    return;
  }

  // 3. Evaluar si la carta está quieta o en movimiento
  let motionScore = 0;
  if (lastSamplePixels) {
    const step = 8;
    let diffSum = 0;
    let count = 0;
    for (let i = 0; i < currentPixels.length; i += step * 4) {
      const diffR = Math.abs(currentPixels[i] - lastSamplePixels[i]);
      const diffG = Math.abs(currentPixels[i + 1] - lastSamplePixels[i + 1]);
      const diffB = Math.abs(currentPixels[i + 2] - lastSamplePixels[i + 2]);
      diffSum += (diffR + diffG + diffB) / 3;
      count++;
    }
    motionScore = diffSum / count;
  }
  lastSamplePixels = new Uint8ClampedArray(currentPixels);

  // Si la mano o la carta aún se están moviendo
  if (motionScore > 13) {
    steadyCount = 0;
    setDetectionBadge("scanning", "Alineando carta...");
    return;
  }

  steadyCount++;
  // Esperar a que esté estable durante 2 ticks seguidos
  if (steadyCount < 2) {
    setDetectionBadge("steady", "Enfocando carta...");
    return;
  }

  // 4. Inferencia en el backend
  isDetecting = true;
  setDetectionBadge("steady", "Analizando carta...");

  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, x, y, w, h, 0, 0, w, h);

  canvas.toBlob(async (blob) => {
    if (!blob || isFrozen || !isRealtimeActive || !isCameraStreaming()) {
      isDetecting = false;
      return;
    }

    try {
      const formData = new FormData();
      formData.append("file", blob, "query_realtime.jpg");

      const resp = await fetch("/api/identify", { method: "POST", body: formData });
      if (!resp.ok) throw new Error("Error en inferencia");

      const data = await resp.json();

      // AUTO-LOCK: ÚNICAMENTE cuando la carta es reconocida positivamente por el modelo
      if (data.recognized) {
        lockDetection(blob, data, `¡Carta detectada: ${data.matches[0].name}!`);
      } else {
        // Si no se reconoce con confianza, NO congelar; seguir buscando
        setDetectionBadge("scanning", "Buscando carta...");
      }
    } catch (err) {
      console.warn("Fallo en inferencia periódica:", err);
    } finally {
      isDetecting = false;
    }
  }, "image/jpeg", 0.92);
}

// Bloquea/congela la captura tomada cuando es reconocida
function lockDetection(blob, data, badgeText) {
  isFrozen = true;
  selectedBlob = blob;

  // Actualizar imagen en preview
  preview.src = URL.createObjectURL(blob);
  previewBox.hidden = false;
  modalThumb.src = preview.src;

  // Efecto visual en el marco
  cardGuide.classList.add("detected");
  setTimeout(() => cardGuide.classList.remove("detected"), 700);

  setDetectionBadge("locked", badgeText);
  resumeScanBtn.hidden = false;

  renderResults(data);
}

// Reanudar escaneo en vivo
resumeScanBtn.addEventListener("click", () => {
  resumeScan();
});

function resumeScan() {
  isFrozen = false;
  resumeScanBtn.hidden = true;
  previewBox.hidden = true;
  unrecognizedBox.hidden = true;
  resultsFooter.hidden = true;
  resultsEl.innerHTML = "";
  setStatus("Alineá una carta dentro del marco dorado.");
  setDetectionBadge("scanning", "Buscando carta...");
  steadyCount = 0;
  lastSamplePixels = null;
}

// Bandera para identificar si la inferencia provino de un clic en "Captura manual"
let isManualCapture = false;

// Captura manual como alternativa directa
captureBtn.addEventListener("click", () => {
  if (!isCameraStreaming()) {
    setStatus("La cámara aún no está lista o no está activa.", "warn");
    return;
  }
  const { x, y, w, h } = guideRectInVideo();
  if (w <= 0 || h <= 0) return;

  canvas.width = w;
  canvas.height = h;
  canvas.getContext("2d").drawImage(video, x, y, w, h, 0, 0, w, h);

  canvas.toBlob(async (blob) => {
    isFrozen = true;
    selectedBlob = blob;
    preview.src = URL.createObjectURL(blob);
    previewBox.hidden = false;
    resumeScanBtn.hidden = false;
    setDetectionBadge("locked", "Captura manual fija");

    setStatus("Calculando embedding...");
    // Indicar que fue captura manual para abrir el modal si no se reconoce
    isManualCapture = true;
    await executeIdentify(blob);
  }, "image/jpeg", 0.92);
});

function setDetectionBadge(state, text) {
  detectionIndicator.className = "detection-badge " + state;
  detectionIndicatorText.textContent = text;
}

// --- Selección de Archivo ---
fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) processSelectedFile(fileInput.files[0]);
});

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) processSelectedFile(e.dataTransfer.files[0]);
});

async function processSelectedFile(file) {
  selectedBlob = file;
  preview.src = URL.createObjectURL(file);
  previewBox.hidden = false;
  setStatus("Imagen cargada. Procesando embedding...");
  isManualCapture = false;
  await executeIdentify(file);
}

// Ejecución manual de identificación si se pulsa el botón del preview
identifyBtn.addEventListener("click", async () => {
  if (selectedBlob) {
    isManualCapture = false;
    await executeIdentify(selectedBlob);
  }
});

async function executeIdentify(blob) {
  identifyBtn.disabled = true;
  setStatus("Buscando en la galería de embeddings...");

  try {
    const formData = new FormData();
    formData.append("file", blob, "query.jpg");

    const resp = await fetch("/api/identify", { method: "POST", body: formData });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `Error HTTP ${resp.status}`);
    }

    const data = await resp.json();
    renderResults(data);

    // EL POPUP DEBE APARECER SOLO CUANDO SE CAPTURA MANUALMENTE Y NO SE RECONOCE LA CARTA
    if (isManualCapture && !data.recognized) {
      openModal("search");
    }
  } catch (err) {
    setStatus("Error: " + err.message, "warn");
  } finally {
    identifyBtn.disabled = false;
    isManualCapture = false;
  }
}

// --- Renderizado de Resultados ---
function renderResults(data) {
  resultsEl.innerHTML = "";
  resultsFooter.hidden = false;

  if (data.recognized) {
    setStatus(`✓ Carta reconocida: ${data.matches[0].name}`, "ok");
    unrecognizedBox.hidden = true;
  } else {
    setStatus("Carta con similitud baja o desconocida. Se muestran los candidatos más cercanos.", "warn");
    unrecognizedBox.hidden = false;
  }

  data.matches.forEach((m, i) => {
    // Distancia L2 [0..2] -> similitud porcentual
    const similarity = Math.max(0, 1 - m.distance / 2) * 100;

    const div = document.createElement("div");
    div.className = "match" + (i === 0 && data.recognized ? " best" : "");
    div.innerHTML = `
      <span class="rank">#${i + 1}</span>
      <img src="${m.image_url}" alt="${escapeHtml(m.name)}" loading="lazy" />
      <div class="info">
        <div class="name">${escapeHtml(m.name)}</div>
        <div class="distance">Distancia métrica: ${m.distance.toFixed(4)}</div>
        <div class="bar"><div style="width:${similarity.toFixed(1)}%"></div></div>
      </div>
    `;
    resultsEl.appendChild(div);
  });
}

function setStatus(message, kind) {
  statusEl.textContent = message;
  statusEl.className = "status" + (kind ? " " + kind : "");
}

// --- Modal de Calibración / Feedback de Embeddings ---
unrecCalibrateBtn.addEventListener("click", () => openModal("search"));
unrecAddBtn.addEventListener("click", () => openModal("create"));
openCalibrateBtn.addEventListener("click", () => openModal("search"));
closeModalBtn.addEventListener("click", closeModal);

calibrateModal.addEventListener("click", (e) => {
  if (e.target === calibrateModal) closeModal();
});

// Cerrar modal al presionar Escape
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !calibrateModal.hidden) {
    closeModal();
  }
});

modalTabSearch.addEventListener("click", () => switchModalTab("search"));
modalTabCreate.addEventListener("click", () => switchModalTab("create"));

function openModal(initialTab = "search") {
  if (!selectedBlob) {
    showToast("Primero capturá o subí una imagen para calibrar", "warn");
    return;
  }
  modalThumb.src = URL.createObjectURL(selectedBlob);
  modalThumb.style.display = "block";
  calibrateModal.hidden = false;
  switchModalTab(initialTab);
  if (initialTab === "search") {
    loadCardSearchResults("");
    setTimeout(() => cardSearchInput.focus(), 100);
  }
}

function closeModal() {
  calibrateModal.hidden = true;
  isManualCapture = false;
}

function switchModalTab(tab) {
  const isSearch = tab === "search";
  modalTabSearch.classList.toggle("active", isSearch);
  modalTabCreate.classList.toggle("active", !isSearch);
  modalPanelSearch.hidden = !isSearch;
  modalPanelCreate.hidden = isSearch;
}

// --- Búsqueda de Cartas en el Modal ---
let searchDebounceTimeout = null;
cardSearchInput.addEventListener("input", (e) => {
  const query = e.target.value;
  clearSearchBtn.hidden = !query;

  clearTimeout(searchDebounceTimeout);
  searchDebounceTimeout = setTimeout(() => {
    loadCardSearchResults(query);
  }, 220);
});

clearSearchBtn.addEventListener("click", () => {
  cardSearchInput.value = "";
  clearSearchBtn.hidden = true;
  loadCardSearchResults("");
  cardSearchInput.focus();
});

async function loadCardSearchResults(query) {
  cardSearchResults.innerHTML = '<div class="hint">Buscando cartas...</div>';
  try {
    const resp = await fetch(`/api/cards/search?q=${encodeURIComponent(query)}&limit=25`);
    if (!resp.ok) throw new Error("Error consultando el catálogo");
    const data = await resp.json();
    renderSearchResults(data.cards);
  } catch (err) {
    cardSearchResults.innerHTML = `<div class="status warn">Error al buscar: ${err.message}</div>`;
  }
}

function renderSearchResults(cards) {
  cardSearchResults.innerHTML = "";
  if (!cards || cards.length === 0) {
    cardSearchResults.innerHTML = '<div class="hint">No se encontraron cartas que coincidan.</div>';
    return;
  }

  cards.forEach((card) => {
    const item = document.createElement("div");
    item.className = "search-item";
    item.innerHTML = `
      <img src="${card.image_url}" alt="${escapeHtml(card.name)}" loading="lazy" />
      <div class="search-item-info">
        <div class="search-item-name">${escapeHtml(card.name)}</div>
        <div class="search-item-meta">
          <span>${escapeHtml(card.type || "Carta")}</span>
          <span class="${card.in_gallery ? "badge-in-gallery" : "badge-new-catalog"}">
            ${card.in_gallery ? "En Galería" : "En Catálogo"}
          </span>
        </div>
      </div>
      <button class="search-item-btn" data-id="${card.id}" data-name="${escapeHtml(card.name)}">
        Promediar con esta
      </button>
    `;

    const btn = item.querySelector(".search-item-btn");
    btn.addEventListener("click", () => {
      averageWithCard(card.id, card.name, btn);
    });

    cardSearchResults.appendChild(item);
  });
}

// Promediar embedding con la carta seleccionada
async function averageWithCard(targetId, targetName, triggerBtn) {
  if (!selectedBlob) return;
  triggerBtn.disabled = true;
  triggerBtn.textContent = "Calculando...";

  try {
    const formData = new FormData();
    formData.append("file", selectedBlob, "captured.jpg");
    formData.append("target_id", targetId);
    formData.append("target_name", targetName);
    formData.append("alpha", "0.5");

    const resp = await fetch("/api/gallery/average", { method: "POST", body: formData });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || "Error al actualizar embedding");
    }

    const result = await resp.json();
    closeModal();
    showToast(`✓ Embedding de "${result.card_name}" promediado exitosamente (nueva dist: ${result.new_distance})`);

    // Volver a evaluar para mostrar el resultado actualizado en primer lugar
    await executeIdentify(selectedBlob);
  } catch (err) {
    showToast("Error: " + err.message, "warn");
    triggerBtn.disabled = false;
    triggerBtn.textContent = "Promediar con esta";
  }
}

// Registrar como Nueva Carta
submitNewCardBtn.addEventListener("click", async () => {
  const name = newCardName.value.trim();
  const id = newCardId.value.trim();

  if (!name) {
    showToast("Por favor escribí el nombre de la carta", "warn");
    return;
  }
  if (!selectedBlob) {
    showToast("No hay imagen capturada", "warn");
    return;
  }

  submitNewCardBtn.disabled = true;
  submitNewCardBtn.textContent = "Creando embedding...";

  try {
    const formData = new FormData();
    formData.append("file", selectedBlob, "captured.jpg");
    formData.append("name", name);
    if (id) formData.append("id", id);

    const resp = await fetch("/api/gallery/add", { method: "POST", body: formData });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || "Error al agregar carta");
    }

    const result = await resp.json();
    closeModal();
    newCardName.value = "";
    newCardId.value = "";
    showToast(`✓ Carta "${result.card_name}" agregada con nuevo embedding`);

    // Volver a evaluar para verificar que ahora es reconocida
    await executeIdentify(selectedBlob);
  } catch (err) {
    showToast("Error: " + err.message, "warn");
  } finally {
    submitNewCardBtn.disabled = false;
    submitNewCardBtn.textContent = "Guardar y generar embedding";
  }
});

// --- Toast Helper ---
let toastTimeout = null;
function showToast(message, kind = "ok") {
  toast.textContent = message;
  toast.className = "toast " + kind;
  toast.hidden = false;
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => {
    toast.hidden = true;
  }, 4000);
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/[&<>"']/g, function (m) {
    return {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[m];
  });
}

// ==========================================================================
// Control y Lógica de la Pestaña: Mesa Completa (Multi-carta)
// ==========================================================================
let multiStream = null;
let multiSessionId = 0;
let isMultiScanning = false;
let multiIntervalId = null;

function isMultiCameraStreaming() {
  if (multiMode.hidden || !multiStream || !multiVideo.srcObject) return false;
  const tracks = multiStream.getVideoTracks();
  if (!tracks || tracks.length === 0 || tracks[0].readyState !== "live") return false;
  if (multiVideo.paused || multiVideo.ended || multiVideo.readyState < 2) return false;
  if (!multiVideo.videoWidth || !multiVideo.videoHeight) return false;
  return true;
}

async function startMultiCamera() {
  const currentSession = ++multiSessionId;
  stopMultiScanLoop();
  stopMultiCameraTracks();

  setMultiBadge("idle", "Iniciando cámara...");
  setStatus("Esperando que la cámara se active...");

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "environment",
        width: { ideal: 1920, min: 1280 },
        height: { ideal: 1080, min: 720 },
      },
      audio: false,
    });

    if (currentSession !== multiSessionId || multiMode.hidden) {
      stream.getTracks().forEach((t) => t.stop());
      return;
    }

    multiStream = stream;
    multiVideo.srcObject = multiStream;
    await multiVideo.play();

    setMultiBadge("scanning", "Cámara activa (Mesa completa)");
    setStatus("Enfocá la mesa o tapete con una o más cartas.");
    syncMultiOverlaySize();

    if (multiAutoToggle.checked) {
      startMultiScanLoop();
    }
  } catch (err) {
    if (currentSession === multiSessionId) {
      setStatus("No se pudo acceder a la cámara: " + err.message, "warn");
      setMultiBadge("idle", "Cámara apagada");
    }
  }
}

function stopMultiCameraTracks() {
  if (multiStream) {
    multiStream.getTracks().forEach((t) => t.stop());
    multiStream = null;
  }
  if (multiVideo.srcObject) {
    multiVideo.srcObject = null;
  }
}

function stopMultiCamera() {
  multiSessionId++;
  stopMultiScanLoop();
  stopMultiCameraTracks();
  clearMultiOverlay();
  setMultiBadge("idle", "Cámara apagada");
}

function setMultiBadge(state, text) {
  multiDetectionIndicator.className = "detection-badge " + state;
  multiDetectionText.textContent = text;
}

function syncMultiOverlaySize() {
  if (!multiVideo.clientWidth || !multiVideo.clientHeight) return;
  multiOverlayCanvas.width = multiVideo.clientWidth;
  multiOverlayCanvas.height = multiVideo.clientHeight;
}

function clearMultiOverlay() {
  const ctx = multiOverlayCanvas.getContext("2d");
  ctx.clearRect(0, 0, multiOverlayCanvas.width, multiOverlayCanvas.height);
}

function getVideoRenderBox(videoEl) {
  const cw = videoEl.clientWidth;
  const ch = videoEl.clientHeight;
  const vw = videoEl.videoWidth || cw;
  const vh = videoEl.videoHeight || ch;

  // Con object-fit: cover:
  const scale = Math.max(cw / vw, ch / vh);
  const rw = vw * scale;
  const rh = vh * scale;
  const offsetX = (cw - rw) / 2;
  const offsetY = (ch - rh) / 2;

  return { scale, offsetX, offsetY, cw, ch, vw, vh };
}

function drawMultiOverlay(matches, detectedCards = [], customMetrics = null) {
  syncMultiOverlaySize();
  const ctx = multiOverlayCanvas.getContext("2d");
  ctx.clearRect(0, 0, multiOverlayCanvas.width, multiOverlayCanvas.height);

  const { scale, offsetX, offsetY } = customMetrics || getVideoRenderBox(multiVideo);

  // 1. Dibujar contornos detectados por vision por computadora que no coincidieron
  const matchedPointsSet = new Set(
    (matches || []).map((m) => JSON.stringify(m.points ? m.points[0] : []))
  );

  (detectedCards || []).forEach((c, idx) => {
    const pts = c.points;
    if (!pts || pts.length < 4) return;
    const key = JSON.stringify(pts[0]);
    if (matchedPointsSet.has(key)) return; // Ya es una coincidencia confirmada

    ctx.save();
    ctx.beginPath();
    ctx.setLineDash([6, 6]);
    ctx.moveTo(offsetX + pts[0][0] * scale, offsetY + pts[0][1] * scale);
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(offsetX + pts[i][0] * scale, offsetY + pts[i][1] * scale);
    }
    ctx.closePath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(245, 195, 68, 0.75)";
    ctx.fillStyle = "rgba(245, 195, 68, 0.08)";
    ctx.fill();
    ctx.stroke();

    // Etiqueta tenue
    const minX = Math.min(...pts.map((p) => p[0])) * scale + offsetX;
    const minY = Math.min(...pts.map((p) => p[1])) * scale + offsetY;
    ctx.fillStyle = "rgba(245, 195, 68, 0.9)";
    ctx.font = "bold 10px Inter, system-ui, sans-serif";
    ctx.fillText("Carta detectada (desconocida o baja certeza)", Math.max(6, minX), Math.max(14, minY - 6));
    ctx.restore();
  });

  // 2. Dibujar coincidencias reconocidas y confirmadas en verde
  (matches || []).forEach((m, idx) => {
    const pts = m.points;
    if (!pts || pts.length < 4) return;

    ctx.save();
    ctx.beginPath();
    const p0x = offsetX + pts[0][0] * scale;
    const p0y = offsetY + pts[0][1] * scale;
    ctx.moveTo(p0x, p0y);

    for (let i = 1; i < pts.length; i++) {
      const px = offsetX + pts[i][0] * scale;
      const py = offsetY + pts[i][1] * scale;
      ctx.lineTo(px, py);
    }
    ctx.closePath();

    ctx.lineWidth = 3;
    ctx.strokeStyle = "#22c55e";
    ctx.fillStyle = "rgba(34, 197, 94, 0.18)";
    ctx.shadowColor = "rgba(34, 197, 94, 0.6)";
    ctx.shadowBlur = 8;
    ctx.fill();
    ctx.stroke();
    ctx.restore();

    // Etiqueta flotante con nombre y confianza
    const minX = Math.min(...pts.map((p) => p[0])) * scale + offsetX;
    const minY = Math.min(...pts.map((p) => p[1])) * scale + offsetY;

    const labelText = `#${idx + 1} ${m.name}`;
    const subText = `${m.similarity_pct}% (${m.distance.toFixed(3)})`;

    ctx.save();
    ctx.font = "bold 12px Inter, system-ui, sans-serif";
    const titleWidth = ctx.measureText(labelText).width;
    ctx.font = "10px Inter, system-ui, sans-serif";
    const subWidth = ctx.measureText(subText).width;
    const boxW = Math.max(titleWidth, subWidth) + 18;
    const boxH = 34;

    const boxX = Math.max(6, Math.min(multiOverlayCanvas.width - boxW - 6, minX));
    const boxY = Math.max(6, minY - boxH - 4);

    ctx.fillStyle = "rgba(18, 23, 46, 0.92)";
    ctx.strokeStyle = "#22c55e";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.roundRect(boxX, boxY, boxW, boxH, 6);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 11px Inter, system-ui, sans-serif";
    ctx.fillText(labelText, boxX + 8, boxY + 14);

    ctx.fillStyle = "#86efac";
    ctx.font = "600 10px Inter, system-ui, sans-serif";
    ctx.fillText(subText, boxX + 8, boxY + 28);
    ctx.restore();
  });
}

// Procesa una imagen de mesa (desde camara en vivo o archivo subido/muestra)
async function executeMultiScanWithBlob(blob, customImageSource = null) {
  if (isMultiScanning) return;
  isMultiScanning = true;
  multiScanLaser.hidden = false;
  setMultiBadge("scanning", "Analizando mesa...");
  setStatus("Localizando cartas en la imagen...");
  multiScanBtn.disabled = true;

  try {
    const formData = new FormData();
    formData.append("file", blob, "table_frame.jpg");

    const resp = await fetch("/api/identify-multi", {
      method: "POST",
      body: formData,
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `Error HTTP ${resp.status}`);
    }

    const data = await resp.json();
    renderMultiResults(data);

    // Calcular metricas de renderizado si se paso una imagen estatica en vez de video
    let metrics = null;
    if (customImageSource) {
      const cw = multiVideo.clientWidth;
      const ch = multiVideo.clientHeight;
      const vw = customImageSource.naturalWidth || customImageSource.width || cw;
      const vh = customImageSource.naturalHeight || customImageSource.height || ch;
      const scale = Math.max(cw / vw, ch / vh);
      const rw = vw * scale;
      const rh = vh * scale;
      const offsetX = (cw - rw) / 2;
      const offsetY = (ch - rh) / 2;
      metrics = { scale, offsetX, offsetY, cw, ch, vw, vh };
    }

    drawMultiOverlay(data.matches, data.detected_cards, metrics);

    if (data.matches && data.matches.length > 0) {
      setMultiBadge("locked", `${data.matches.length} coincidencia(s)`);
      setStatus(`✓ Se encontraron ${data.matches.length} cartas reconocidas en mesa.`, "ok");
    } else if (data.total_detected > 0) {
      setMultiBadge("steady", `${data.total_detected} cartas sin coincidencia`);
      setStatus(`Se detectaron ${data.total_detected} cuadriláteros de cartas en la mesa, pero ninguna coincide con la galería.`, "warn");
    } else {
      setMultiBadge("scanning", "Sin cartas en encuadre");
      setStatus("No se detectaron cartas. Asegurate de tener buena iluminación.", "warn");
    }
  } catch (err) {
    setStatus("Error al escanear mesa: " + err.message, "warn");
    setMultiBadge("idle", "Error de escaneo");
  } finally {
    isMultiScanning = false;
    multiScanLaser.hidden = true;
    multiScanBtn.disabled = false;
  }
}

async function scanMultiCards() {
  if (!isMultiCameraStreaming()) {
    setStatus("La cámara aún no está transmitiendo video.", "warn");
    return;
  }

  const vw = multiVideo.videoWidth;
  const vh = multiVideo.videoHeight;
  multiHiddenCanvas.width = vw;
  multiHiddenCanvas.height = vh;
  const ctx = multiHiddenCanvas.getContext("2d");
  ctx.drawImage(multiVideo, 0, 0, vw, vh);

  const blob = await new Promise((resolve) =>
    multiHiddenCanvas.toBlob(resolve, "image/jpeg", 0.92)
  );

  if (!blob) {
    setStatus("No se pudo capturar el frame de la cámara.", "warn");
    return;
  }

  await executeMultiScanWithBlob(blob);
}

// Cargar imagen de prueba desde archivo o URL de muestra
async function loadStaticImageIntoMultiView(imgUrlOrBlob) {
  stopMultiScanLoop();
  stopMultiCameraTracks();

  let blob = null;
  if (imgUrlOrBlob instanceof Blob) {
    blob = imgUrlOrBlob;
  } else {
    setStatus("Descargando imagen de prueba...");
    const res = await fetch(imgUrlOrBlob);
    blob = await res.blob();
  }

  const img = new Image();
  img.src = URL.createObjectURL(blob);
  await img.decode();

  // Pausar video y renderizar la imagen fija en el overlay
  multiVideo.poster = img.src;
  syncMultiOverlaySize();

  await executeMultiScanWithBlob(blob, img);
}

function renderMultiResults(data) {
  resultsEl.innerHTML = "";
  unrecognizedBox.hidden = true;
  resultsFooter.hidden = true;

  const matches = data.matches || [];
  const totalDetected = data.total_detected || 0;
  const threshold = data.threshold || 0.75;

  // Header resumen
  const summaryBox = document.createElement("div");
  summaryBox.className = "multi-summary-box";
  summaryBox.innerHTML = `
    <div class="multi-summary-title">
      <span>&#127183;</span>
      <span>${matches.length} coincidencia(s) confirmada(s)</span>
    </div>
    <div class="multi-summary-counts">
      <span class="multi-pill success">${matches.length} en galería</span>
      <span class="multi-pill">${totalDetected} detectada(s)</span>
      <span class="multi-pill">Umbral: ${threshold}</span>
    </div>
  `;
  resultsEl.appendChild(summaryBox);

  // Lista de coincidencias únicamente
  if (matches.length > 0) {
    matches.forEach((m, idx) => {
      const cardDiv = document.createElement("div");
      cardDiv.className = "multi-match-card";
      cardDiv.innerHTML = `
        <span class="rank">#${idx + 1}</span>
        <img class="multi-card-thumb" src="${m.image_url}" alt="${escapeHtml(m.name)}" loading="lazy" />
        <div class="multi-card-info">
          <div class="multi-card-topline">
            <span class="multi-card-name" title="${escapeHtml(m.name)}">${escapeHtml(m.name)}</span>
          </div>
          <div class="distance">Distancia métrica: <strong>${m.distance.toFixed(4)}</strong></div>
          <div class="bar">
            <div style="width: ${m.similarity_pct}%"></div>
          </div>
        </div>
      `;
      resultsEl.appendChild(cardDiv);
    });
  } else {
    const emptyDiv = document.createElement("div");
    emptyDiv.className = "multi-empty-state";
    if (totalDetected > 0) {
      emptyDiv.innerHTML = `
        <div class="icon">&#9888;</div>
        <strong>Se detectaron ${totalDetected} contorno(s) de cartas en la mesa</strong>
        <p style="margin-top: 0.5rem; color: var(--muted);">
          Sin embargo, ninguna superó el umbral métrico de certeza (${threshold}) contra la galería actual.
        </p>
      `;
    } else {
      emptyDiv.innerHTML = `
        <div class="icon">&#128269;</div>
        <strong>No se detectaron cartas en el encuadre</strong>
        <p style="margin-top: 0.5rem; color: var(--muted);">
          Apuntá la cámara hacia cartas bien iluminadas sobre la mesa o usá el botón <em>"Subir foto de mesa"</em> para probar con fotos de tu equipo.
        </p>
      `;
    }
    resultsEl.appendChild(emptyDiv);
  }
}

// Loop continuo opcional para escaneo periódico
function startMultiScanLoop() {
  stopMultiScanLoop();
  multiIntervalId = setInterval(() => {
    if (!isMultiScanning && isMultiCameraStreaming() && !multiMode.hidden) {
      scanMultiCards();
    }
  }, 1600);
}

function stopMultiScanLoop() {
  if (multiIntervalId) {
    clearInterval(multiIntervalId);
    multiIntervalId = null;
  }
}

multiScanBtn.addEventListener("click", scanMultiCards);
multiAutoToggle.addEventListener("change", () => {
  if (multiAutoToggle.checked) {
    startMultiScanLoop();
  } else {
    stopMultiScanLoop();
  }
});

// Boton de subir foto de mesa
if (multiUploadBtn && multiFileInput) {
  multiUploadBtn.addEventListener("click", () => multiFileInput.click());
  multiFileInput.addEventListener("change", () => {
    if (multiFileInput.files.length > 0) {
      loadStaticImageIntoMultiView(multiFileInput.files[0]);
    }
  });
}

// Botones de imagenes de muestra con cartas para probar
document.querySelectorAll(".sample-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const imgUrl = btn.getAttribute("data-img");
    if (imgUrl) {
      loadStaticImageIntoMultiView(imgUrl);
    }
  });
});

window.addEventListener("resize", () => {
  if (!multiMode.hidden) {
    syncMultiOverlaySize();
  }
});
