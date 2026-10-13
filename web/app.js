const state = {
  points: [],
  risk: [],
  yaw: 0.35,
  pitch: 0.22,
  xt: 0,
  cutAxis: "t",
  cut: 1,
  sliceValue: 0.45,
  slab: false,
  heat: false,
  drag: null,
  marker: null,
  studio: "sculpt",
  stand: 0.5,
  smear: 0.16,
  ambient: true,
  theme: null,
};

const stage = document.getElementById("stage");
const ctx = stage.getContext("2d");
const angleOut = document.getElementById("angle-readout");
const cutOut = document.getElementById("cut-readout");
const countOut = document.getElementById("count-readout");
const reading = document.getElementById("reading");
const probeOut = document.getElementById("probe-readout");

function resize() {
  const rect = stage.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  stage.width = Math.max(1, Math.floor(rect.width * dpr));
  stage.height = Math.max(1, Math.floor(rect.height * dpr));
  draw();
}

function axisValue(point) {
  if (state.cutAxis === "x") return point[0];
  if (state.cutAxis === "y") return point[1];
  if (state.cutAxis === "z") return point[2];
  return point[3];
}

function visiblePoints() {
  const thickness = 0.07;
  const out = [];
  for (let index = 0; index < state.points.length; index += 1) {
    const point = state.points[index];
    const value = axisValue(point);
    if (state.slab ? Math.abs(value - state.cut) <= thickness : value <= state.cut + 1e-6) {
      out.push({ point, index });
    }
  }
  return out;
}

function project(point) {
  const c = Math.cos(state.xt);
  const s = Math.sin(state.xt);
  const x = point[0] - 0.5;
  const y = point[1] - 0.5;
  const z = point[2] - 0.5;
  const t = point[3] - 0.5;
  const xr = x * c - t * s;
  const tr = x * s + t * c;
  const yaw = Math.cos(state.yaw);
  const sy = Math.sin(state.yaw);
  const cp = Math.cos(state.pitch);
  const sp = Math.sin(state.pitch);
  const vx = tr;
  const vy = y;
  const vz0 = z * 0.85 + xr * 0.45;
  const vx2 = vx * yaw - vz0 * sy;
  const vz1 = vx * sy + vz0 * yaw;
  const vy2 = vy * cp - vz1 * sp;
  const vz2 = vy * sp + vz1 * cp;
  const persp = 3.1 / (3.1 + vz2);
  return { sx: vx2 * persp, sy: -vy2 * persp, depth: vz2, persp };
}

function heatOf(risk) {
  const t = Math.max(0, Math.min(1, risk));
  const r = Math.round(40 + t * 215);
  const g = Math.round(80 + (1 - t) * 80);
  const b = Math.round(255 - t * 200);
  return `rgba(${r}, ${g}, ${b}, 0.92)`;
}

function colorOf(point, index) {
  if (state.heat && state.risk.length === state.points.length) {
    return heatOf(state.risk[index] || 0);
  }
  const tissue = point[5];
  const affect = point[6];
  const scar = point[7];
  const other = point[8];
  if (scar > 0.45) return "rgba(255, 180, 92, 0.95)";
  if (tissue > 0.45) return "rgba(255, 93, 143, 0.95)";
  if (other > 0.45) return "rgba(121, 210, 255, 0.8)";
  const warm = Math.round(90 + affect * 120);
  return `rgba(${warm}, 230, 200, 0.72)`;
}

function draw() {
  if (!stage.width) return;
  const width = stage.width;
  const height = stage.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = state.theme?.bg || "#0b0e14";
  ctx.fillRect(0, 0, width, height);
  const points = visiblePoints();
  const projected = points.map((item) => ({ ...item, view: project(item.point) }));
  projected.sort((a, b) => a.view.depth - b.view.depth);
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const item of projected) {
    minX = Math.min(minX, item.view.sx);
    maxX = Math.max(maxX, item.view.sx);
    minY = Math.min(minY, item.view.sy);
    maxY = Math.max(maxY, item.view.sy);
  }
  const span = Math.max(maxX - minX, maxY - minY, 0.08);
  const scale = (Math.min(width, height) * 0.62) / span;
  const midX = (minX + maxX) / 2;
  const midY = (minY + maxY) / 2;
  const cx = width * 0.5;
  const cy = height * 0.46;
  const dpr = window.devicePixelRatio || 1;
  const radius = Math.min(Math.max(2.2 * dpr, scale * 0.01), 11 * dpr);
  for (const item of projected) {
    const x = cx + (item.view.sx - midX) * scale;
    const y = cy + (item.view.sy - midY) * scale;
    ctx.fillStyle = colorOf(item.point, item.index);
    ctx.beginPath();
    ctx.arc(x, y, radius * item.view.persp, 0, Math.PI * 2);
    ctx.fill();
    item.screenX = x;
    item.screenY = y;
  }
  if (state.marker) {
    ctx.strokeStyle = "#f4f7fb";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(state.marker.x, state.marker.y, 8, 0, Math.PI * 2);
    ctx.stroke();
  }
  state.projected = projected;
  countOut.textContent = `showing ${points.length} of ${state.points.length} points`;
}

function canvasPoint(event) {
  const rect = stage.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  return {
    x: (event.clientX - rect.left) * dpr,
    y: (event.clientY - rect.top) * dpr,
  };
}

async function probe(point, screen) {
  const query = new URLSearchParams({
    x: point[0],
    y: point[1],
    z: point[2],
    t: point[3],
  });
  const response = await fetch(`/api/probe?${query}`);
  const payload = await response.json();
  probeOut.textContent = payload.sentence;
  state.marker = screen;
  if (payload.file) {
    state.focusFile = payload.file;
    const focus = document.getElementById("coder-focus");
    if (focus) focus.textContent = `Focused file: ${payload.file}`;
  }
  draw();
}

stage.addEventListener("pointerdown", (event) => {
  state.drag = { x: event.clientX, y: event.clientY, yaw: state.yaw, pitch: state.pitch, moved: false };
});

window.addEventListener("pointermove", (event) => {
  if (!state.drag) return;
  const dx = event.clientX - state.drag.x;
  const dy = event.clientY - state.drag.y;
  if (Math.hypot(dx, dy) > 3) state.drag.moved = true;
  state.yaw = state.drag.yaw + dx * 0.008;
  state.pitch = Math.max(-1.2, Math.min(1.2, state.drag.pitch + dy * 0.008));
  draw();
});

window.addEventListener("pointerup", async (event) => {
  if (!state.drag) return;
  const moved = state.drag.moved;
  state.drag = null;
  if (moved || !state.projected) return;
  const here = canvasPoint(event);
  let best = null;
  let bestDist = 18 * (window.devicePixelRatio || 1);
  for (const item of state.projected) {
    const dist = Math.hypot(item.screenX - here.x, item.screenY - here.y);
    if (dist < bestDist) {
      best = item;
      bestDist = dist;
    }
  }
  if (best) await probe(best.point, { x: best.screenX, y: best.screenY });
});

document.getElementById("xt").addEventListener("input", (event) => {
  state.xt = (Number(event.target.value) * Math.PI) / 180;
  angleOut.textContent = `${event.target.value}°`;
  draw();
});

document.getElementById("cut").addEventListener("input", (event) => {
  state.cut = Number(event.target.value) / 1000;
  state.sliceValue = state.cut;
  cutOut.textContent = state.cut.toFixed(2);
  draw();
  scheduleSlice();
});

document.getElementById("cut-axis").addEventListener("change", (event) => {
  state.cutAxis = event.target.value;
  draw();
  scheduleSlice();
});

document.getElementById("slab").addEventListener("change", (event) => {
  state.slab = event.target.checked;
  draw();
});

document.getElementById("heat").addEventListener("change", (event) => {
  state.heat = event.target.checked;
  draw();
});

document.getElementById("stand").addEventListener("input", (event) => {
  state.stand = Number(event.target.value) / 1000;
  document.getElementById("stand-readout").textContent = state.stand.toFixed(2);
});

document.getElementById("smear-amount").addEventListener("input", (event) => {
  state.smear = Number(event.target.value) / 100;
  document.getElementById("smear-readout").textContent = state.smear.toFixed(2);
});

let sliceTimer = 0;
function scheduleSlice() {
  window.clearTimeout(sliceTimer);
  sliceTimer = window.setTimeout(loadSlice, 80);
}

async function loadSlice() {
  const query = new URLSearchParams({
    axis: state.cutAxis,
    value: String(state.sliceValue),
  });
  const response = await fetch(`/api/slice?${query}`);
  const payload = await response.json();
  const canvas = document.getElementById("slice");
  canvas.width = payload.width;
  canvas.height = payload.height;
  const sliceCtx = canvas.getContext("2d");
  const image = sliceCtx.createImageData(payload.width, payload.height);
  for (let i = 0; i < payload.width * payload.height; i += 1) {
    image.data[i * 4] = payload.rgb[i * 3];
    image.data[i * 4 + 1] = payload.rgb[i * 3 + 1];
    image.data[i * 4 + 2] = payload.rgb[i * 3 + 2];
    image.data[i * 4 + 3] = 255;
  }
  sliceCtx.putImageData(image, 0, 0);
}

function showStudio(id) {
  state.studio = id;
  state.ambient = id === "sculpt" || id === "smear";
  for (const button of document.querySelectorAll("#studios button")) {
    button.classList.toggle("on", button.dataset.studio === id);
  }
  for (const panel of document.querySelectorAll(".studio-panel")) {
    panel.classList.toggle("hidden", panel.id !== `panel-${id}`);
  }
  if (id === "scan") loadRisk();
  if (id === "coder") loadModels();
}

document.getElementById("studios").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-studio]");
  if (button) showStudio(button.dataset.studio);
});

function applyTheme(theme) {
  state.theme = theme || null;
  document.documentElement.style.setProperty("--bg", theme?.bg || "#07080b");
  document.documentElement.style.setProperty("--mint", theme?.body || "#8ef6d2");
}

function setStatus(node, message, error) {
  node.className = error ? "error" : "";
  node.textContent = message;
}

async function hold(body, statusText) {
  const status = document.getElementById("open-status");
  setStatus(status, statusText);
  const response = await fetch("/api/open", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  setStatus(status, "");
  await boot();
}

function fillCards(node, items, onPick) {
  node.innerHTML = "";
  for (const item of items) {
    const button = document.createElement("button");
    button.type = "button";
    button.innerHTML = `<b>${item.title}</b><small>${item.blurb}</small>`;
    button.addEventListener("click", () => onPick(item, button));
    node.appendChild(button);
  }
}

async function loadRisk() {
  const readingNode = document.getElementById("scan-reading");
  const list = document.getElementById("hotspots");
  readingNode.textContent = "Scanning the solid…";
  list.innerHTML = "";
  const response = await fetch("/api/risk");
  const payload = await response.json();
  state.risk = payload.points || [];
  readingNode.textContent = payload.reading;
  for (const item of payload.hotspots || []) {
    const row = document.createElement("li");
    const lines = item.lines ? `:${item.lines[0]}-${item.lines[1]}` : "";
    row.textContent = `${item.score.toFixed(2)}  ${item.file}${lines}`;
    list.appendChild(row);
  }
  if (state.heat) draw();
}

function addChat(role, text) {
  const chat = document.getElementById("chat");
  const line = document.createElement("p");
  line.className = role === "user" ? "you" : "them";
  line.textContent = text;
  chat.appendChild(line);
  chat.scrollTop = chat.scrollHeight;
}

async function ask(question) {
  const query = question.trim();
  if (!query) return;
  addChat("user", query);
  const response = await fetch("/api/resident", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, t: state.stand }),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  addChat("them", payload.text);
}

async function boot() {
  const response = await fetch("/api/block");
  const payload = await response.json();
  state.points = payload.points;
  state.shape = payload.shape;
  state.risk = [];
  state.marker = null;
  reading.textContent = payload.reading;
  probeOut.textContent = "Click the solid.";
  document.getElementById("title").textContent = payload.name;
  document.getElementById("subtitle").textContent = payload.subtitle;
  document.title = `4D Mind — ${payload.name}`;
  applyTheme(payload.theme);
  document.getElementById("chat").innerHTML = "";
  document.getElementById("coder-chat").innerHTML = "";
  state.focusFile = "";
  document.getElementById("coder-focus").textContent = payload.has_source
    ? "Click a dot to focus a file. The coder reads that file from disk."
    : "This solid has no checkout. Hold a folder before asking the coder.";
  renderTensors(payload.tensors);
  for (const [channel, label] of Object.entries(payload.labels || {})) {
    const node = document.getElementById(`label-${channel}`);
    if (node) node.textContent = label;
  }
  resize();
  await loadSlice();
  if (state.studio === "scan") await loadRisk();
}

function renderTensors(tensors) {
  const section = document.getElementById("weights-section");
  const list = document.getElementById("tensor-list");
  list.innerHTML = "";
  section.classList.toggle("hidden", !tensors);
  const time = Math.max((state.shape?.[0] || 1) - 1, 1);
  for (const item of tensors || []) {
    const row = document.createElement("li");
    const moved = item.rel_change === undefined ? "" : `  Δ ${(item.rel_change * 100).toFixed(2)}%`;
    row.textContent = `${item.name} [${item.shape.join("×")}]  μ ${item.mean.toFixed(4)}  σ ${item.std.toFixed(4)}  outliers ${(item.outliers * 100).toFixed(2)}%${moved}`;
    row.title = `range ${item.min.toFixed(4)} … ${item.max.toFixed(4)}, ${(item.near_zero * 100).toFixed(1)}% near zero, kurtosis ${item.kurtosis.toFixed(2)}, L2 ${item.l2.toFixed(3)}`;
    row.addEventListener("click", () => {
      state.cut = item.t / time;
      state.sliceValue = state.cut;
      document.getElementById("cut").value = String(Math.round(state.cut * 1000));
      cutOut.textContent = state.cut.toFixed(2);
      state.cutAxis = "t";
      document.getElementById("cut-axis").value = "t";
      draw();
      scheduleSlice();
    });
    list.appendChild(row);
  }
}

document.getElementById("open-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.target.querySelector("button");
  const path = document.getElementById("open-path").value.trim();
  button.disabled = true;
  try {
    await hold({ path }, path ? `Rasterizing ${path}…` : "Returning to Iris…");
  } catch (error) {
    setStatus(document.getElementById("open-status"), String(error.message || error), true);
  } finally {
    button.disabled = false;
  }
});

document.getElementById("ask-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = document.getElementById("ask");
  const query = input.value;
  input.value = "";
  try {
    await ask(query);
  } catch (error) {
    addChat("them", String(error.message || error));
  }
});

function addCoderLine(role, text) {
  const chat = document.getElementById("coder-chat");
  const line = document.createElement("p");
  line.className = role === "user" ? "you" : "them";
  line.textContent = text;
  chat.appendChild(line);
  chat.scrollTop = chat.scrollHeight;
}

async function loadModels() {
  const select = document.getElementById("coder-model");
  const status = document.getElementById("key-status");
  const response = await fetch("/api/models");
  const payload = await response.json();
  const previous = select.value;
  select.innerHTML = "";
  for (const model of payload.models || []) {
    const option = document.createElement("option");
    option.value = model.name;
    option.textContent = model.coder ? `${model.name}  · coder` : model.name;
    select.appendChild(option);
  }
  const preferred = previous || payload.default;
  if (preferred && [...select.options].some((option) => option.value === preferred)) {
    select.value = preferred;
  }
  status.className = "";
  status.textContent = payload.key_set
    ? "Cloud key is set for this session."
    : "Local 4D-Lens & local Ollama models work for free. Paste an Ollama Cloud key only if you want cloud models.";
}

async function askCoder(question) {
  const query = question.trim();
  if (!query) return;
  addCoderLine("user", query);
  const response = await fetch("/api/coder", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      model: document.getElementById("coder-model").value,
      file: state.focusFile || "",
    }),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  const files = (payload.files || []).join(", ");
  addCoderLine("them", files ? `${payload.text}\n\nRead: ${files}` : payload.text);
}

document.getElementById("key-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = document.getElementById("ollama-key");
  const response = await fetch("/api/ollama/key", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key: input.value }),
  });
  const payload = await response.json();
  input.value = "";
  const status = document.getElementById("key-status");
  status.className = payload.key_set ? "" : "error";
  status.textContent = payload.key_set ? "Cloud key is set for this session." : "Key cleared.";
});

document.getElementById("coder-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = document.getElementById("coder-ask");
  const query = input.value;
  input.value = "";
  try {
    await askCoder(query);
  } catch (error) {
    addCoderLine("them", String(error.message || error));
  }
});

document.getElementById("coder-hot").addEventListener("click", () => {
  askCoder("What is the most likely bug in the focused file or the hottest spot, and what is the fix?").catch((error) => {
    addCoderLine("them", String(error.message || error));
  });
});

document.getElementById("wav-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const wav = document.getElementById("wav-path").value.trim();
  const status = document.getElementById("open-status");
  try {
    const response = await fetch("/api/smear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind: "pulse", smear: state.smear, wav }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || response.statusText);
    setStatus(status, "");
    await boot();
  } catch (error) {
    setStatus(status, String(error.message || error), true);
  }
});

async function setupPlatform() {
  const response = await fetch("/api/platform");
  const platform = await response.json();
  fillCards(
    document.getElementById("sculpture-grid"),
    platform.sculptures.map((item) => ({ id: item.id, title: item.name, blurb: item.blurb })),
    async (item) => {
      try {
        await hold({ sculpture: item.id }, `Holding ${item.title}…`);
      } catch (error) {
        setStatus(document.getElementById("open-status"), String(error.message || error), true);
      }
    },
  );
  fillCards(
    document.getElementById("smear-grid"),
    platform.smear_kinds.map((kind) => ({
      id: kind,
      title: kind,
      blurb: kind === "iris-echo" ? "Iris, bled along t" : `Synthetic ${kind} clip`,
    })),
    async (item) => {
      const status = document.getElementById("open-status");
      try {
        const response = await fetch("/api/smear", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ kind: item.id, smear: state.smear }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || response.statusText);
        setStatus(status, "");
        await boot();
      } catch (error) {
        setStatus(status, String(error.message || error), true);
      }
    },
  );
  const prompts = document.getElementById("prompts");
  prompts.innerHTML = "";
  for (const question of platform.questions || []) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = question;
    button.addEventListener("click", () => ask(question));
    prompts.appendChild(button);
  }
}

function ambientTick() {
  if (state.ambient && !state.drag) {
    state.xt += 0.003;
    const degrees = Math.round(((state.xt * 180) / Math.PI + 180) % 360 - 180);
    const slider = document.getElementById("xt");
    if (document.activeElement !== slider) {
      slider.value = String(degrees);
      angleOut.textContent = `${degrees}°`;
    }
    draw();
  }
  window.requestAnimationFrame(ambientTick);
}

window.addEventListener("resize", resize);
setupPlatform();
boot();
ambientTick();
