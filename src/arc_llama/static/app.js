// arc-llama web UI: thin polling client over /admin/status + /admin/load|stop.

const $ = (sel) => document.querySelector(sel);
const fmtVram = (mb) => mb == null ? "?" : `${(mb / 1024).toFixed(1)} GB`;
const fmtPath = (p) => {
  if (!p) return "—";
  // Show just the basename + parent dir, leave full path on hover.
  const parts = p.split("/").filter(Boolean);
  return parts.slice(-2).join("/");
};

let lastStatus = null;
let inflight = false;

async function fetchStatus() {
  if (inflight) return;
  inflight = true;
  try {
    const r = await fetch("/admin/status");
    if (!r.ok) throw new Error(`status ${r.status}`);
    lastStatus = await r.json();
    render(lastStatus);
    $("#last-updated").textContent = `updated ${new Date().toLocaleTimeString()}`;
  } catch (e) {
    $("#last-updated").textContent = `error: ${e.message}`;
  } finally {
    inflight = false;
  }
}

async function postAction(path, label) {
  try {
    const r = await fetch(path, { method: "POST" });
    if (!r.ok) {
      const t = await r.text();
      alert(`${label} failed: ${r.status} ${t}`);
      return;
    }
    await fetchStatus();
  } catch (e) {
    alert(`${label} error: ${e.message}`);
  }
}

function render(s) {
  $("#server-info").textContent = `${s.server.host}:${s.server.port}`;
  $("#policy-info").textContent =
    s.server.single_resident ? "single-resident" : "multi-resident";

  // GPUs
  const gpuBody = $("#gpus tbody");
  gpuBody.innerHTML = "";
  for (const g of s.gpus) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${g.pci_slot}</td>
      <td>${g.arch}</td>
      <td>${g.name || "—"}</td>
      <td>level_zero:${g.sycl_index}</td>
      <td>${fmtVram(g.vram_mb)}</td>
      <td>${g.enabled ? "yes" : "no"}</td>
    `;
    gpuBody.appendChild(tr);
  }

  // Models
  const modelBody = $("#models tbody");
  modelBody.innerHTML = "";
  for (const m of s.models) {
    const tr = document.createElement("tr");
    tr.className = m.loaded ? "bright" : "dim";
    const kv = `${m.cache_type_k || "?"}/${m.cache_type_v || "?"}`;
    const pill = m.loaded
      ? '<span class="pill loaded">loaded</span>'
      : '<span class="pill idle">idle</span>';
    tr.innerHTML = `
      <td>${pill}</td>
      <td>${m.name}</td>
      <td>${m.gpu_pci_slot}</td>
      <td>${m.port}</td>
      <td>${m.ctx ?? "?"}</td>
      <td>${kv}</td>
      <td class="path" title="${m.path}">${fmtPath(m.path)}</td>
      <td class="actions"></td>
    `;
    const actions = tr.querySelector(".actions");
    const wrap = document.createElement("div");
    wrap.className = "row-actions";
    if (m.loaded) {
      const stop = document.createElement("button");
      stop.textContent = "Stop";
      stop.onclick = () => postAction(`/admin/stop/${encodeURIComponent(m.name)}`, "stop");
      wrap.appendChild(stop);
    } else {
      const load = document.createElement("button");
      load.textContent = "Load";
      load.onclick = () => postAction(`/admin/load/${encodeURIComponent(m.name)}`, "load");
      wrap.appendChild(load);
    }
    actions.appendChild(wrap);
    modelBody.appendChild(tr);
  }
}

$("#refresh").onclick = fetchStatus;
$("#stop-all").onclick = () => {
  if (confirm("Stop every running llama-server?")) {
    postAction("/admin/stop-all", "stop-all");
  }
};

fetchStatus();
setInterval(fetchStatus, 5000);
