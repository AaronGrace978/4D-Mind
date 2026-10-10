const state = {
  points: [],
  yaw: 0.35,
  pitch: 0.22,
  xt: 0,
  cutAxis: "t",
  cut: 1,
  sliceValue: 0.45,
  slab: false,
  drag: null,
  marker: null,
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
  return state.points.filter((point) => {
    const value = axisValue(point);
    if (state.slab) return Math.abs(value - state.cut) <= thickness;
    return value <= state.cut + 1e-6;
  });
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
  // Time is the long axis. The slider turns that axis toward x.
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

function colorOf(point) {
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
  ctx.fillStyle = "#0b0e14";
  ctx.fillRect(0, 0, width, height);
  const points = visiblePoints();
  const projected = points.map((point) => ({ point, view: project(point) }));
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
    ctx.fillStyle = colorOf(item.point);
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

async function boot() {
  const response = await fetch("/api/block");
  const payload = await response.json();
  state.points = payload.points;
  state.marker = null;
  reading.textContent = payload.reading;
  probeOut.textContent = "Click the solid.";
  document.getElementById("title").textContent = payload.name;
  document.getElementById("subtitle").textContent = payload.subtitle;
  document.title = `4D Block Mind — ${payload.name}`;
  for (const [channel, label] of Object.entries(payload.labels || {})) {
    const node = document.getElementById(`label-${channel}`);
    if (node) node.textContent = label;
  }
  resize();
  await loadSlice();
}

document.getElementById("open-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.target.querySelector("button");
  const status = document.getElementById("open-status");
  const path = document.getElementById("open-path").value.trim();
  button.disabled = true;
  status.className = "";
  status.textContent = path ? `Rasterizing ${path}…` : "Returning to Iris…";
  try {
    const response = await fetch("/api/open", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || response.statusText);
    status.textContent = "";
    await boot();
  } catch (error) {
    status.className = "error";
    status.textContent = String(error.message || error);
  } finally {
    button.disabled = false;
  }
});

window.addEventListener("resize", resize);
boot();
