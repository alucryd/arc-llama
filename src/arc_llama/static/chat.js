const $ = (sel) => document.querySelector(sel);
const chatLog = $("#chat-log");
const emptyState = $("#empty-state");
const modelSelect = $("#model-select");
const modelStatus = $("#model-status");
const statusText = $("#status-text");
const input = $("#message-input");
const sendButton = $("#send-button");
const inputWrap = $("#input-wrap");
const commandPalette = $("#command-palette");
const attachButton = $("#attach-button");
const pdfInput = $("#pdf-input");
const attachmentStrip = $("#attachment-strip");

let models = [];
let selectedModel = null;
let loadingModel = null;
let generating = false;
let statusPoller = null;
let lastUsage = null;
let streamStartTime = null;
let streamTokenCount = 0;
const conversation = [];
let attachments = [];

const agentLog          = $("#agent-log");
const agentEmptyState   = $("#agent-empty-state");
const modeToggle        = $("#mode-toggle");
const chatHint          = $("#chat-hint");
const agentControls     = $("#agent-controls");
const agentAutoConfirm  = $("#agent-auto-confirm");
const agentMaxTurns     = $("#agent-max-turns");

let agentMode = false;
let agentRunning = false;
let agentAbort = null;

const ctxMeter   = $("#ctx-meter");
const ctxBarFill = $("#ctx-bar-fill");
const ctxLabelL  = $("#ctx-label-left");
const ctxLabelTps = $("#ctx-label-tps");
const ctxLabelR  = $("#ctx-label-right");
const settingsToggle = $("#settings-toggle");
const settingsPanel  = $("#settings-panel");
const sModelName     = $("#s-model-name");
const sFields        = $("#s-fields");
const sFeedback      = $("#s-feedback");

const historyToggle  = $("#history-toggle");
const historyPanel   = $("#history-panel");
const hNew           = $("#h-new");
const hList          = $("#h-list");
const hExport        = $("#h-export");
const hImport        = $("#h-import");
const hImportInput   = $("#h-import-input");

const HISTORY_KEY    = "arc-llama-chats";
const MAX_HISTORY    = 50;

// Configure Markdown renderer with syntax highlighting and safe defaults.
if (typeof marked !== "undefined") {
  marked.use({
    gfm: true,
    breaks: false,
    headerIds: false,
    mangle: false,
  });
}
const mdRenderer = {
  code(code, language) {
    const validLang = language && hljs.getLanguage(language) ? language : "plaintext";
    const highlighted = hljs.highlight(code, { language: validLang }).value;
    const langLabel = validLang === "plaintext" ? "" : `<span class="code-lang">${escapeHtml(validLang)}</span>`;
    return `<div class="code-block-wrapper">${langLabel}<pre><code class="hljs language-${escapeHtml(validLang)}">${highlighted}</code></pre><button class="copy-code-btn" title="Copy" aria-label="Copy code"><svg viewBox="0 0 24 24" width="14" height="14"><path d="M16 1H4a2 2 0 0 0-2 2v14h2V3h12V1zm3 4H8a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2zm0 16H8V7h11v14z"/></svg></button></div>`;
  },
  blockquote(quote) {
    return `<blockquote>${quote}</blockquote>`;
  },
  html(text) {
    return escapeHtml(text);
  },
};

function attachCopyButtons(root) {
  for (const btn of root.querySelectorAll(".copy-code-btn")) {
    btn.addEventListener("click", async () => {
      const code = btn.closest(".code-block-wrapper").querySelector("code");
      const text = code ? code.textContent : "";
      try {
        await navigator.clipboard.writeText(text);
        btn.classList.add("copied");
        btn.innerHTML = `<svg viewBox="0 0 24 24" width="14" height="14"><path d="M9 16.17 4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>`;
        setTimeout(() => {
          btn.classList.remove("copied");
          btn.innerHTML = `<svg viewBox="0 0 24 24" width="14" height="14"><path d="M16 1H4a2 2 0 0 0-2 2v14h2V3h12V1zm3 4H8a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2zm0 16H8V7h11v14z"/></svg>`;
        }, 1500);
      } catch (e) {
        console.warn("Copy failed", e);
      }
    });
  }
}

let currentChatId    = null;
let chatCache        = loadChatsFromStorage();

const KV_TYPES    = ["f16","f32","q8_0","q5_1","q5_0","q4_1","q4_0"];
const KV_CLASSES  = ["default","moe_a3b","qwen3_27b_dense","gemma_swa"];

settingsToggle.addEventListener("click", () => {
  const open = settingsPanel.classList.toggle("open");
  settingsToggle.classList.toggle("open", open);
  if (open) renderSettingsPanel();
});

historyToggle.addEventListener("click", () => {
  const open = historyPanel.classList.toggle("open");
  historyToggle.classList.toggle("open", open);
  if (open) renderHistoryPanel();
});

hNew.addEventListener("click", newChat);

if (hExport) hExport.addEventListener("click", exportChats);
if (hImport) hImport.addEventListener("click", () => hImportInput?.click());
if (hImportInput) hImportInput.addEventListener("change", importChatsFromFile);

function loadChatsFromStorage() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) return parsed;
    if (parsed && Array.isArray(parsed.chats)) return parsed.chats;
  } catch (e) {
    // storage may be full / disabled
  }
  return [];
}

function loadChats() {
  return chatCache;
}

function saveChats(chats) {
  chatCache = chats;
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(chats));
  } catch (e) {
    // storage may be full / disabled
  }
}

function serverChatToLocal(data, modelHint) {
  return {
    id: data.id,
    title: data.title || "New chat",
    model: modelHint || null,
    createdAt: Math.round((data.created_at || Date.now() / 1000) * 1000),
    updatedAt: Math.round((data.updated_at || Date.now() / 1000) * 1000),
    messages: (data.messages || []).map(m => ({ role: m.role, content: m.content })),
  };
}

async function apiRequest(path, options = {}) {
  const r = await fetch(path, options);
  if (!r.ok) {
    const t = await r.text();
    throw new Error(`${r.status} ${t}`);
  }
  return r.json();
}

async function syncChatsFromServer() {
  try {
    const data = await apiRequest("/v1/chats");
    const summaries = data.data || [];
    const map = new Map(chatCache.map(c => [c.id, c]));
    for (const s of summaries) {
      const existing = map.get(s.id);
      const updatedAt = Math.round((s.updated_at || 0) * 1000);
      if (existing) {
        existing.title = s.title;
        existing.createdAt = Math.round((s.created_at || 0) * 1000);
        existing.updatedAt = updatedAt;
        existing.message_count = s.message_count;
      } else {
        map.set(s.id, {
          id: s.id,
          title: s.title,
          model: null,
          messages: [],
          createdAt: Math.round((s.created_at || 0) * 1000),
          updatedAt: updatedAt,
          message_count: s.message_count,
        });
      }
    }
    const merged = Array.from(map.values()).sort((a, b) => b.updatedAt - a.updatedAt).slice(0, MAX_HISTORY);
    saveChats(merged);
    if (historyPanel.classList.contains("open")) renderHistoryPanel();
  } catch (e) {
    console.warn("Could not sync chats from server:", e.message);
  }
}

async function ensureServerChat(titleHint) {
  if (currentChatId) return;
  const title = truncateTitle(titleHint || "New chat");
  try {
    const data = await apiRequest("/v1/chats", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
    currentChatId = data.id;
    const now = Date.now();
    const chats = loadChats();
    chats.unshift(serverChatToLocal(data, selectedModel));
    chats[0].createdAt = now;
    chats[0].updatedAt = now;
    saveChats(chats);
  } catch (e) {
    console.warn("Could not create chat on server:", e.message);
    // Local-only fallback so the UI keeps working offline.
    const id = generateId();
    currentChatId = id;
    const now = Date.now();
    const chats = loadChats();
    chats.unshift({ id, title, model: selectedModel, messages: [], createdAt: now, updatedAt: now });
    saveChats(chats);
  }
}

async function serverAppendMessages(chatId, messages, title) {
  if (!chatId) return;
  if ((!messages || messages.length === 0) && !title) return;
  const body = {};
  if (messages && messages.length > 0) body.messages = messages;
  if (title) body.title = title;
  try {
    await apiRequest(`/v1/chats/${encodeURIComponent(chatId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (e) {
    console.warn("Could not append messages to server:", e.message);
  }
}

function generateId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return Date.now().toString(36) + Math.random().toString(36).slice(2);
}

function truncateTitle(text, max = 60) {
  if (!text) return "New chat";
  const single = text.replace(/\s+/g, " ").trim();
  if (single.length <= max) return single || "New chat";
  return single.slice(0, max - 1).trimEnd() + "…";
}

function formatRelativeTime(ms) {
  const now = Date.now();
  const diff = now - ms;
  const sec = Math.floor(diff / 1000);
  if (sec < 10) return "just now";
  if (sec < 60) return `${sec}s ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.floor(hr / 24);
  if (day === 1) return "yesterday";
  if (day < 7) return `${day} days ago`;
  const d = new Date(ms);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function saveCurrentChat() {
  if (conversation.length === 0) return;
  const firstUser = conversation.find(m => m.role === "user");
  const title = truncateTitle(firstUser ? firstUser.content : "New chat");
  const chats = loadChats();
  const now = Date.now();
  if (currentChatId) {
    const idx = chats.findIndex(c => c.id === currentChatId);
    if (idx >= 0) {
      chats[idx] = { ...chats[idx], title, model: selectedModel, messages: [...conversation], updatedAt: now };
    } else {
      chats.unshift({ id: currentChatId, title, model: selectedModel, messages: [...conversation], createdAt: now, updatedAt: now });
    }
  } else {
    const id = generateId();
    currentChatId = id;
    chats.unshift({ id, title, model: selectedModel, messages: [...conversation], createdAt: now, updatedAt: now });
  }
  chats.sort((a, b) => b.updatedAt - a.updatedAt);
  while (chats.length > MAX_HISTORY) chats.pop();
  saveChats(chats);
  // Keep the server-side title in sync (best-effort).
  if (currentChatId) {
    serverAppendMessages(currentChatId, [], title).catch(() => {});
  }
}

async function newChat() {
  conversation.length = 0;
  currentChatId = null;
  chatLog.innerHTML = "";
  chatLog.appendChild(emptyState);
  emptyState.style.display = "";
  input.value = "";
  input.style.height = "auto";
  historyPanel.classList.remove("open");
  historyToggle.classList.remove("open");
  updateCtxMeter(0, models.find(m => m.id === selectedModel)?.ctx || 131072);
  ctxMeter.classList.remove("visible");
  ctxLabelTps.textContent = "";
  await ensureServerChat("New chat");
  input.focus();
}

function renderHistoryPanel() {
  const chats = loadChats();
  hList.innerHTML = "";
  if (chats.length === 0) {
    hList.innerHTML = '<div class="h-empty">No saved chats yet.</div>';
    return;
  }
  for (const c of chats) {
    const card = document.createElement("div");
    card.className = "h-card";
    card.dataset.id = c.id;
    const ICON_CLOSE = '<svg viewBox="0 0 24 24" width="14" height="14"><path d="M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>';
    card.innerHTML = `
      <div class="h-card-title">${escapeHtml(c.title)}</div>
      <div class="h-card-meta">
        <span>${escapeHtml(c.model || "unknown")}</span>
        <span>${formatRelativeTime(c.updatedAt)}</span>
      </div>
      <button class="h-delete" aria-label="Delete chat">${ICON_CLOSE}</button>
    `;
    card.addEventListener("click", (e) => {
      if (e.target.closest(".h-delete")) return;
      loadChat(c.id);
    });
    card.querySelector(".h-delete").addEventListener("click", (e) => {
      e.stopPropagation();
      deleteChat(c.id);
    });
    hList.appendChild(card);
  }
}

async function loadChat(id) {
  let chat = chatCache.find(c => c.id === id);
  if (!chat) {
    try {
      const data = await apiRequest(`/v1/chats/${encodeURIComponent(id)}`);
      chat = serverChatToLocal(data, null);
      chatCache.push(chat);
      saveChats(chatCache);
    } catch (e) {
      console.warn("Could not load chat from server:", e.message);
      return;
    }
  } else if (!chat.messages || chat.messages.length === 0) {
    try {
      const data = await apiRequest(`/v1/chats/${encodeURIComponent(id)}`);
      const updated = serverChatToLocal(data, chat.model);
      const idx = chatCache.findIndex(c => c.id === id);
      if (idx >= 0) chatCache[idx] = updated; else chatCache.push(updated);
      saveChats(chatCache);
      chat = updated;
    } catch (e) {
      console.warn("Could not load chat details from server:", e.message);
    }
  }
  if (!chat) return;
  conversation.length = 0;
  if (Array.isArray(chat.messages)) {
    conversation.push(...chat.messages);
  }
  currentChatId = chat.id;
  chatLog.innerHTML = "";
  if (conversation.length === 0) {
    chatLog.appendChild(emptyState);
    emptyState.style.display = "";
  } else {
    for (const m of conversation) {
      if (m.role === "assistant") {
        const { div, content } = createMessage("assistant", m.content || "");
        if (m.thinking) renderThinking(div, m.thinking);
        if (m.content) renderMarkdown(content, m.content);
      } else {
        createMessage(m.role, m.content || "");
      }
    }
  }
  if (chat.model && models.some(m => m.id === chat.model)) {
    selectedModel = chat.model;
    modelSelect.value = chat.model;
    loadingModel = null;
    updatePickerStatus();
  }
  historyPanel.classList.remove("open");
  historyToggle.classList.remove("open");
  const m = models.find(x => x.id === selectedModel);
  updateCtxMeter(estimateTokens(), m?.ctx || 131072);
  if (shouldAutoScroll(chatLog)) autoScroll(chatLog);
  input.focus();
}

async function deleteChat(id) {
  try {
    await apiRequest(`/v1/chats/${encodeURIComponent(id)}`, { method: "DELETE" });
  } catch (e) {
    console.warn("Could not delete chat on server:", e.message);
  }
  const chats = loadChats().filter(c => c.id !== id);
  saveChats(chats);
  if (currentChatId === id) {
    currentChatId = null;
  }
  renderHistoryPanel();
}

async function exportChats() {
  try {
    const data = await apiRequest("/v1/chats/export");
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `arc-llama-chats-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (e) {
    console.warn("Could not export chats:", e.message);
    showError("Export failed: " + e.message);
  }
}

async function importChatsFromFile() {
  const file = hImportInput.files?.[0];
  if (!file) return;
  hImportInput.value = "";
  let body;
  try {
    const text = await file.text();
    body = JSON.parse(text);
  } catch (e) {
    showError("Import failed: invalid JSON file");
    return;
  }
  const chats = body.chats;
  if (!Array.isArray(chats)) {
    showError("Import failed: missing 'chats' array");
    return;
  }
  try {
    const r = await fetch("/v1/chats/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chats, overwrite: false }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
    await syncChatsFromServer();
    showError(`Imported ${data.imported || 0}, skipped ${data.skipped || 0}, errors ${data.errors || 0}.`);
  } catch (e) {
    showError("Import failed: " + e.message);
  }
}

function renderSettingsPanel() {
  const m = models.find(m => m.id === selectedModel);
  sModelName.textContent = selectedModel || "—";
  if (!m || (m.owned_by && m.owned_by.startsWith("upstream:"))) {
    sFields.innerHTML = '<div class="s-upstream">Settings not available for upstream models.</div>';
    return;
  }
  const ctx        = m.ctx        ?? 32768;
  const ctk        = m.cache_type_k ?? "q8_0";
  const ctv        = m.cache_type_v ?? "q8_0";
  const parallel   = m.parallel   ?? 1;
  const kvClass    = m.kv_class   ?? "default";

  const kvOpts = KV_TYPES.map(v => `<option value="${v}"${v===ctk?" selected":""}>${v}</option>`).join("");
  const kvOptsV = KV_TYPES.map(v => `<option value="${v}"${v===ctv?" selected":""}>${v}</option>`).join("");
  const classOpts = KV_CLASSES.map(v => `<option value="${v}"${v===kvClass?" selected":""}>${v}</option>`).join("");

  sFields.innerHTML = `
    <div class="s-field"><label>Context (tokens)</label>
      <input id="s-ctx" type="number" min="256" max="1048576" step="1024" value="${ctx}"></div>
    <div class="s-field"><label>KV Cache K</label>
      <select id="s-ctk">${kvOpts}</select></div>
    <div class="s-field"><label>KV Cache V</label>
      <select id="s-ctv">${kvOptsV}</select></div>
    <div class="s-field"><label>Parallel slots</label>
      <input id="s-par" type="number" min="1" max="32" value="${parallel}"></div>
    <div class="s-field"><label>KV Class</label>
      <select id="s-kvc">${classOpts}</select></div>
    <button class="s-apply" id="s-apply">Apply</button>
    <div class="s-note">Takes effect on next model load.</div>
  `;
  $("#s-apply").addEventListener("click", applySettings);
}

async function applySettings() {
  const m = models.find(m => m.id === selectedModel);
  if (!m) return;
  const btn = $("#s-apply");
  btn.disabled = true;
  sFeedback.textContent = "";
  const body = {
    ctx:          parseInt($("#s-ctx").value, 10),
    cache_type_k: $("#s-ctk").value,
    cache_type_v: $("#s-ctv").value,
    parallel:     parseInt($("#s-par").value, 10),
    kv_class:     $("#s-kvc").value,
  };
  try {
    const r = await fetch(`/admin/models/${encodeURIComponent(selectedModel)}/edit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.status);
    m.ctx          = body.ctx;
    m.cache_type_k = body.cache_type_k;
    m.cache_type_v = body.cache_type_v;
    m.parallel     = body.parallel;
    m.kv_class     = body.kv_class;
    sFeedback.style.color = "var(--accent-bright)";
    sFeedback.textContent = "Saved.";
  } catch (e) {
    sFeedback.style.color = "#e8b0b0";
    sFeedback.textContent = "Error: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

function estimateTokens() {
  const chars = conversation.reduce((n, m) => n + (m.content || "").length, 0);
  return Math.round(chars / 4);
}

function updateCtxMeter(tokens, ctx) {
  if (!ctx) return;
  const pct = Math.min(100, tokens / ctx * 100);
  ctxBarFill.style.width = pct.toFixed(1) + "%";
  ctxBarFill.className = "ctx-bar-fill" + (pct >= 90 ? " critical" : pct >= 70 ? " warn" : "");
  ctxLabelL.textContent = `${tokens.toLocaleString()} / ${ctx.toLocaleString()} tokens`;
  ctxLabelR.textContent = pct.toFixed(1) + "%";
  ctxMeter.classList.add("visible");
}

async function fetchModels() {
  try {
    const r = await fetch("/v1/models");
    if (!r.ok) throw new Error(`status ${r.status}`);
    const data = await r.json();
    const local = (data.data || []).filter(m => m.object === "model" && m.owned_by !== "arc-llama-alias");
    models = local;
    renderModelPicker();
  } catch (e) {
    showError("Could not fetch models: " + e.message);
  }
}

function renderModelPicker() {
  const current = selectedModel || modelSelect.value;
  modelSelect.innerHTML = "";
  if (models.length === 0) {
    const opt = document.createElement("option");
    opt.textContent = "No models available";
    opt.disabled = true;
    opt.selected = true;
    modelSelect.appendChild(opt);
    selectedModel = null;
    updateStatus("swapping");
    return;
  }
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.id;
    modelSelect.appendChild(opt);
  }
  if (current && models.some(m => m.id === current)) {
    modelSelect.value = current;
    selectedModel = current;
  } else {
    selectedModel = models[0].id;
    modelSelect.value = selectedModel;
  }
}

async function fetchStatus() {
  try {
    const r = await fetch("/admin/status");
    if (!r.ok) return;
    const data = await r.json();
    const modelMap = new Map((data.models || []).map(m => [m.name, m]));
    models = models.map(m => {
      const s = modelMap.get(m.id);
      if (s) {
        m.loaded        = s.loaded;
        m.ctx           = s.ctx           ?? m.ctx;
        m.cache_type_k  = s.cache_type_k  ?? m.cache_type_k;
        m.cache_type_v  = s.cache_type_v  ?? m.cache_type_v;
        m.kv_class      = s.kv_class      ?? m.kv_class;
      }
      return m;
    });
    updatePickerStatus();
  } catch (e) {
    // silent — the chat endpoint will surface real errors
  }
}

function updatePickerStatus() {
  const m = models.find(m => m.id === selectedModel);
  if (!m) {
    updateStatus("swapping");
    return;
  }
  if (loadingModel === selectedModel) {
    updateStatus("loading");
  } else if (m.loaded) {
    updateStatus("ready");
  } else {
    updateStatus("swapping");
  }
}

function updateStatus(state) {
  modelStatus.className = "model-status " + state;
  statusText.textContent = state;
}

modelSelect.addEventListener("change", () => {
  selectedModel = modelSelect.value;
  loadingModel = null;
  updatePickerStatus();
  if (settingsPanel.classList.contains("open")) renderSettingsPanel();
});

function createMessage(role, text = "") {
  if (emptyState) emptyState.style.display = "none";
  const div = document.createElement("div");
  div.className = "message " + role;
  const roleLabel = document.createElement("div");
  roleLabel.className = "role";
  roleLabel.textContent = role === "user" ? "You" : role === "system" ? "System" : "Assistant";
  div.appendChild(roleLabel);
  if (role === "assistant") {
    const indicator = document.createElement("span");
    indicator.id = "streaming-indicator";
    indicator.textContent = "●";
    indicator.style.color = "var(--accent-bright)";
    indicator.style.opacity = "0";
    roleLabel.appendChild(indicator);
    const thinkingBlock = document.createElement("div");
    thinkingBlock.className = "thinking-block";
    thinkingBlock.style.display = "none";
    const thinkingToggle = document.createElement("div");
    thinkingToggle.className = "thinking-toggle";
    thinkingToggle.innerHTML = '<span class="chevron">▶</span><span>Thinking</span>';
    thinkingToggle.addEventListener("click", () => {
      thinkingToggle.classList.toggle("open");
      thinkingContent.classList.toggle("open");
    });
    const thinkingContent = document.createElement("div");
    thinkingContent.className = "thinking-content";
    thinkingBlock.appendChild(thinkingToggle);
    thinkingBlock.appendChild(thinkingContent);
    div.appendChild(thinkingBlock);
  }
  const content = document.createElement("div");
  content.className = "content";
  content.textContent = text;
  div.appendChild(content);
  chatLog.appendChild(div);
  if (shouldAutoScroll(chatLog)) autoScroll(chatLog);
  return { div, content };
}

function showError(text) {
  const { content } = createMessage("error", text);
  content.parentElement.classList.add("error-card");
  content.parentElement.querySelector(".role").textContent = "Error";
}

async function ensureModelLoaded() {
  const m = models.find(m => m.id === selectedModel);
  if (!m) throw new Error("No model selected");
  if (m.loaded || (m.owned_by && m.owned_by.startsWith("upstream:"))) return;
  loadingModel = selectedModel;
  updatePickerStatus();
  try {
    const r = await fetch(`/admin/load/${encodeURIComponent(selectedModel)}`, { method: "POST" });
    if (!r.ok) {
      const t = await r.text();
      throw new Error(`Load failed: ${r.status} ${t}`);
    }
    m.loaded = true;
  } catch (e) {
    throw e;
  } finally {
    loadingModel = null;
    updatePickerStatus();
  }
}

async function sendMessage() {
  const text = input.value.trim();
  if (generating || !selectedModel) return;
  if (!text && !hasReadyAttachments()) return;
  if (hasProcessingAttachments()) {
    showError("Please wait for attachments to finish processing.");
    return;
  }

  const attachmentText = buildAttachmentText();
  const fullText = text
    ? attachmentText ? `${text}\n\n${attachmentText}` : text
    : attachmentText;

  input.value = "";
  input.style.height = "auto";
  clearAttachments();
  conversation.push({ role: "user", content: fullText });
  createMessage("user", fullText);

  // Persist this conversation on the server (best-effort).
  await ensureServerChat(fullText);
  serverAppendMessages(currentChatId, [{ role: "user", content: fullText }]);

  generating = true;
  sendButton.disabled = true;
  inputWrap.classList.add("generating");
  const streamingDot = $("#streaming-indicator");
  if (streamingDot) streamingDot.style.opacity = "1";

  try {
    await ensureModelLoaded();
  } catch (e) {
    showError(e.message);
    finishGeneration();
    return;
  }

  const assistantMsg = createMessage("assistant");
  conversation.push({ role: "assistant", content: "", thinking: "" });
  const convoIndex = conversation.length - 1;

  let streamRaw = "";
  let lastDisplayedContent = "";
  let currentThinking = "";

  try {
    const r = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: selectedModel,
        messages: conversation.slice(0, -1),
        stream: true,
        stream_options: { include_usage: true },
      }),
    });
    if (!r.ok) {
      const t = await r.text();
      throw new Error(`${r.status} ${t}`);
    }
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        const chunkText = processSseLine(line);
        if (chunkText != null) {
          if (streamStartTime === null) streamStartTime = Date.now();
          streamTokenCount += Math.max(1, Math.round(chunkText.length / 4));
          streamRaw += chunkText;
          const parsed = parseThinking(streamRaw);
          if (!parsed.hasPartialTag) {
            const newContent = parsed.content.slice(lastDisplayedContent.length);
            lastDisplayedContent = parsed.content;
            currentThinking = parsed.thinking;
            if (newContent) {
              conversation[convoIndex].content = parsed.content;
              appendChunk(assistantMsg.content, newContent);
              if (shouldAutoScroll(chatLog)) autoScroll(chatLog);
            }
            const thinkingBlock = assistantMsg.div.querySelector(".thinking-block");
            const thinkingContent = assistantMsg.div.querySelector(".thinking-content");
            if (thinkingBlock && thinkingContent) {
              const trimmedThinking = currentThinking.trim();
              if (trimmedThinking) {
                thinkingBlock.style.display = "";
                thinkingContent.textContent = trimmedThinking;
              } else {
                thinkingBlock.style.display = "none";
                thinkingContent.textContent = "";
              }
            }
          }
        }
      }
    }
    const finalParsed = parseThinking(streamRaw);
    conversation[convoIndex].content = finalParsed.content;
    conversation[convoIndex].thinking = finalParsed.thinking;
    renderMarkdown(assistantMsg.content, finalParsed.content);
    const m = models.find(m => m.id === selectedModel);
    const completionToks = lastUsage ? lastUsage.completion_tokens : streamTokenCount;
    const totalToks      = lastUsage ? lastUsage.total_tokens      : estimateTokens();
    const elapsed        = streamStartTime ? (Date.now() - streamStartTime) / 1000 : null;
    const tps            = (elapsed && elapsed > 0 && completionToks > 0)
                           ? (completionToks / elapsed).toFixed(1)
                           : null;
    if (tps) ctxLabelTps.textContent = tps + " tok/s";
    updateCtxMeter(totalToks, m?.ctx || 131072);
    serverAppendMessages(currentChatId, [{ role: "assistant", content: conversation[convoIndex].content }]);
    saveCurrentChat();
  } catch (e) {
    assistantMsg.div.remove();
    conversation.pop();
    showError("Generation failed: " + e.message);
  } finally {
    lastUsage = null;
    streamStartTime = null;
    streamTokenCount = 0;
    finishGeneration();
  }
}

function processSseLine(line) {
  const trimmed = line.trim();
  if (!trimmed || !trimmed.startsWith("data:")) return null;
  const payload = trimmed.slice(5).trim();
  if (payload === "[DONE]") return null;
  try {
    const obj = JSON.parse(payload);
    if (obj.usage) lastUsage = obj.usage;
    const delta = obj.choices?.[0]?.delta;
    if (!delta) return null;
    let text = "";
    if (delta.reasoning_content) {
      text += "<think>" + delta.reasoning_content + "</think>";
    }
    if (delta.content != null) {
      text += delta.content;
    }
    return text || null;
  } catch (e) {
    return null;
  }
}

function parseThinking(text) {
  const tail = text.slice(-15);
  const lastLt = tail.lastIndexOf("<");
  if (lastLt !== -1) {
    const afterLt = tail.slice(lastLt);
    const possible = ["<think>", "<thinking>", "</think>", "</thinking>"];
    for (const tag of possible) {
      if (tag.startsWith(afterLt) && afterLt.length < tag.length) {
        return { thinking: "", content: text, hasPartialTag: true };
      }
    }
  }
  let thinking = "";
  let content = text;
  const thinkMatches = [...text.matchAll(/<think>([\s\S]*?)<\/think>/g)];
  for (const m of thinkMatches) thinking += (thinking ? "\n" : "") + m[1];
  content = content.replace(/<think>[\s\S]*?<\/think>/g, "");
  const thinkingMatches = [...text.matchAll(/<thinking>([\s\S]*?)<\/thinking>/g)];
  for (const m of thinkingMatches) thinking += (thinking ? "\n" : "") + m[1];
  content = content.replace(/<thinking>[\s\S]*?<\/thinking>/g, "");
  const unclosedThink = content.match(/<think>([\s\S]*)$/);
  const unclosedThinking = content.match(/<thinking>([\s\S]*)$/);
  if (unclosedThink) {
    thinking += (thinking ? "\n" : "") + unclosedThink[1];
    content = content.replace(/<think>[\s\S]*$/, "");
  } else if (unclosedThinking) {
    thinking += (thinking ? "\n" : "") + unclosedThinking[1];
    content = content.replace(/<thinking>[\s\S]*$/, "");
  }
  return { thinking: thinking.trim(), content: content.trimEnd(), hasPartialTag: false };
}

function appendChunk(container, text) {
  const span = document.createElement("span");
  span.className = "token-chunk";
  span.textContent = text;
  container.appendChild(span);
  requestAnimationFrame(() => span.classList.add("revealed"));
}

function renderMarkdown(container, text) {
  if (typeof marked === "undefined" || typeof hljs === "undefined") {
    container.textContent = text;
    return;
  }
  const raw = text
    .replace(/<think>[\s\S]*?<\/think>/g, "")
    .replace(/<thinking>[\s\S]*?<\/thinking>/g, "")
    .replace(/[\n\r]+$/, "")
    .trimEnd();
  try {
    let html = marked.parse(raw, { renderer: mdRenderer });
    html = html.replace(/<p>\s*<\/p>/g, "").replace(/<p><br\s*\/?><\/p>/g, "");
    container.innerHTML = html;
    attachCopyButtons(container);
  } catch (e) {
    console.warn("Markdown render failed, falling back to plain text", e);
    container.textContent = text;
  }
}

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function renderThinking(messageDiv, thinkingText) {
  const thinkingBlock = messageDiv.querySelector(".thinking-block");
  if (!thinkingBlock) return;
  const thinkingContent = thinkingBlock.querySelector(".thinking-content");
  const trimmed = String(thinkingText || "").trim();
  thinkingContent.textContent = trimmed;
  thinkingBlock.style.display = trimmed ? "" : "none";
}

function finishGeneration() {
  generating = false;
  sendButton.disabled = false;
  inputWrap.classList.remove("generating");
  const indicator = $("#streaming-indicator");
  if (indicator) indicator.style.opacity = "0";
  // Collapse any thinking blocks that are open and hide empty ones
  const openThinking = document.querySelectorAll(".thinking-toggle.open");
  for (const t of openThinking) {
    t.classList.remove("open");
    t.nextElementSibling?.classList.remove("open");
  }
  for (const block of document.querySelectorAll(".thinking-block")) {
    const content = block.querySelector(".thinking-content");
    if (content && !content.textContent.trim()) {
      block.style.display = "none";
    }
  }
  input.focus();
}

// ------------------------------------------------------------------
// File attachments
// ------------------------------------------------------------------

const TEXT_EXTENSIONS = new Set([".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv"]);

function isTextFile(file) {
  if (file.type.startsWith("text/")) return true;
  const name = file.name.toLowerCase();
  for (const ext of TEXT_EXTENSIONS) {
    if (name.endsWith(ext)) return true;
  }
  return false;
}

function isPdfFile(file) {
  return file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
}

function generateAttachmentId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID();
  return Date.now().toString(36) + Math.random().toString(36).slice(2);
}

function renderAttachments() {
  attachmentStrip.innerHTML = "";
  if (attachments.length === 0) return;
  for (const a of attachments) {
    const chip = document.createElement("div");
    chip.className = "attachment-chip" + (a.error ? " error" : a.processing ? " processing" : "");
    chip.dataset.id = a.id;
    const ICON_CLIP = '<svg viewBox="0 0 24 24" width="14" height="14"><path d="M16.5 6v11.5c0 2.485-2.015 4.5-4.5 4.5S7.5 19.985 7.5 17.5V5c0-1.657 1.343-3 3-3s3 1.343 3 3v12.5c0 .828-.672 1.5-1.5 1.5s-1.5-.672-1.5-1.5V6h-2v11.5c0 1.933 1.567 3.5 3.5 3.5s3.5-1.567 3.5-3.5V5c0-2.761-2.239-5-5-5S5 2.239 5 5v12.5c0 3.59 2.91 6.5 6.5 6.5s6.5-2.91 6.5-6.5V6h-2z"/></svg>';
    const ICON_CLOSE = '<svg viewBox="0 0 24 24" width="14" height="14"><path d="M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>';
    const icon = a.processing ? '<span class="spinner"></span>'
                 : a.error ? '<span>!</span>'
                 : `<span class="attachment-icon">${ICON_CLIP}</span>`;
    chip.innerHTML = `
      ${icon}
      <span class="filename" title="${escapeHtml(a.file.name)}">${escapeHtml(a.file.name)}</span>
      <button class="remove" aria-label="Remove attachment">${ICON_CLOSE}</button>
    `;
    chip.querySelector(".remove").addEventListener("click", () => removeAttachment(a.id));
    attachmentStrip.appendChild(chip);
  }
}

function addAttachment(file) {
  const id = generateAttachmentId();
  const a = { id, file, text: "", processing: true, error: "" };
  attachments.push(a);
  renderAttachments();
  processAttachment(a).finally(renderAttachments);
}

async function processAttachment(a) {
  try {
    if (isPdfFile(a.file)) {
      const form = new FormData();
      form.append("file", a.file);
      const r = await fetch("/admin/parse-pdf", { method: "POST", body: form });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      a.text = data.text || "";
    } else if (isTextFile(a.file)) {
      a.text = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => reject(new Error("Could not read file"));
        reader.readAsText(a.file);
      });
    } else {
      throw new Error("Unsupported file type");
    }
    a.error = "";
  } catch (e) {
    a.error = e.message;
    a.text = "";
  } finally {
    a.processing = false;
  }
}

function removeAttachment(id) {
  attachments = attachments.filter(a => a.id !== id);
  renderAttachments();
}

function clearAttachments() {
  attachments = [];
  renderAttachments();
}

function buildAttachmentText() {
  const parts = [];
  for (const a of attachments) {
    if (a.error || a.processing || !a.text) continue;
    parts.push(`[Attachment: ${a.file.name}]\n${a.text.trim()}`);
  }
  return parts.join("\n\n");
}

function hasReadyAttachments() {
  return attachments.some(a => !a.processing && !a.error && a.text);
}

function hasProcessingAttachments() {
  return attachments.some(a => a.processing);
}

attachButton.addEventListener("click", () => pdfInput.click());

pdfInput.addEventListener("change", () => {
  const files = Array.from(pdfInput.files || []);
  pdfInput.value = "";
  for (const file of files) addAttachment(file);
});

// Drag-and-drop on the input area
inputWrap.addEventListener("dragover", (e) => {
  e.preventDefault();
  e.stopPropagation();
  inputWrap.style.borderColor = "var(--accent-bright)";
});
inputWrap.addEventListener("dragleave", (e) => {
  e.preventDefault();
  e.stopPropagation();
  inputWrap.style.borderColor = "";
});
inputWrap.addEventListener("drop", (e) => {
  e.preventDefault();
  e.stopPropagation();
  inputWrap.style.borderColor = "";
  const files = Array.from(e.dataTransfer?.files || []);
  for (const file of files) addAttachment(file);
});

// ------------------------------------------------------------------
// Agent mode
// ------------------------------------------------------------------

function setMode(mode) {
  agentMode = mode === "agent";
  for (const btn of modeToggle.querySelectorAll("button")) {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  }
  chatLog.classList.toggle("hidden", agentMode);
  agentLog.classList.toggle("active", agentMode);
  chatHint.style.display = agentMode ? "none" : "";
  agentControls.style.display = agentMode ? "flex" : "none";
  attachButton.style.display = agentMode ? "none" : "flex";
  attachmentStrip.style.display = agentMode ? "none" : "flex";
  sendButton.title = agentMode ? "Run task" : "Send";
  input.placeholder = agentMode ? "Describe the coding task…" : "Message arc-llama…";
  ctxMeter.classList.toggle("visible", !agentMode);
  hideCommandPalette();
  input.focus();
}

for (const btn of modeToggle.querySelectorAll("button")) {
  btn.addEventListener("click", () => setMode(btn.dataset.mode));
}

const ICON_SEARCH = '<svg viewBox="0 0 24 24"><path d="M15.5 14h-.79l-.28-.27A6.471 6.471 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zM9.5 14C7.57 14 6 12.43 6 10.5S7.57 7 9.5 7 13 8.57 13 10.5 11.43 14 9.5 14z"/></svg>';
const ICON_READ = '<svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6zm4 18H6V4h7v5h5v11z"/></svg>';
const ICON_WRITE = '<svg viewBox="0 0 24 24"><path d="M19 12h-2v8H7V4h8V2H5v20h14v-6h2v6c0 1.1-.9 2-2 2H5c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10l6 6v2h-2V7l-4-4zM8 13h6v-2H8v2zm0-4h4V7H8v2zm11.7 4.5-1.4-1.4-5.3 5.3-2.3-2.3-1.4 1.4 3.7 3.7 6.7-6.7z"/></svg>';
const ICON_COMMAND = '<svg viewBox="0 0 24 24"><path d="M4 5h16v2H4V5zm0 6h10v2H4v-2zm0 6h13v2H4v-2z"/></svg>';
const ICON_LIST = '<svg viewBox="0 0 24 24"><path d="M3 5h2v2H3V5zm4 0h14v2H7V5zM3 11h2v2H3v-2zm4 0h14v2H7v-2zm-4 6h2v2H3v-2zm4 0h14v2H7v-2z"/></svg>';
const ICON_CHECK = '<svg viewBox="0 0 24 24"><path d="M9 16.17 4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>';
const ICON_X = '<svg viewBox="0 0 24 24"><path d="M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>';
const ICON_WARN = '<svg viewBox="0 0 24 24"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/></svg>';
const ICON_BOT = '<svg viewBox="0 0 24 24"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zM7 9h10v2H7V9zm7-3H7V4h7v2zm5 6H7v-2h12v2z"/></svg>';

function getToolIcon(name) {
  if (name.includes("search")) return ICON_SEARCH;
  if (name.includes("write")) return ICON_WRITE;
  if (name.includes("read")) return ICON_READ;
  if (name === "list_directory") return ICON_LIST;
  if (name === "run_command") return ICON_COMMAND;
  return ICON_BOT;
}

function getToolKind(name) {
  if (name.includes("search")) return "Search";
  if (name.includes("write")) return "Write file";
  if (name.includes("read")) return "Read file";
  if (name === "list_directory") return "List directory";
  if (name === "run_command") return "Shell";
  return "Tool";
}

function makeToggle(container, toggleClass, bodyClass) {
  const toggle = container.querySelector(toggleClass);
  const body = container.querySelector(bodyClass);
  if (!toggle || !body) return;
  toggle.classList.toggle("open");
  body.classList.toggle("open");
}

// Robot llama mascot for the agent background. Every frame is built from the
// same skeleton via center()/llamaBox() so widths can never drift between
// frames - that's what kept the old hand-typed art from jittering as it
// "animated" by swapping state.
const LLAMA_WIDTH = 24;

function center(str) {
  str = str || "";
  const pad = LLAMA_WIDTH - str.length;
  const left = Math.floor(pad / 2);
  const right = pad - left;
  return " ".repeat(Math.max(left, 0)) + str + " ".repeat(Math.max(right, 0));
}

function llamaBox(left, innerWidth, content, right) {
  content = content || "";
  const pad = innerWidth - content.length;
  const padLeft = Math.floor(pad / 2);
  const padRight = pad - padLeft;
  const inner = " ".repeat(Math.max(padLeft, 0)) + content + " ".repeat(Math.max(padRight, 0));
  return center(left + inner + right);
}

function llamaFrame({ eyes = "", mouth = "", panel1 = "", panel2 = "" }) {
  return [
    center("/\\        /\\"),
    center("/  \\      /  \\"),
    center(".------------."),
    llamaBox("|", 12, eyes, "|"),
    llamaBox("|", 12, "", "|"),
    llamaBox("|", 12, mouth, "|"),
    center("'----,  ,----'"),
    center("|  |"),
    center(".--'--'--."),
    llamaBox("| ", 8, panel1, " |"),
    llamaBox("| ", 8, panel2, " |"),
    center("'--------'"),
  ].join("\n");
}

// Each state is a list of frame descriptors cycled by the interval below -
// eyes/mouth/panel text change, the skeleton never does.
const AGENT_ASCII = {
  idle: [
    { eyes: "o    o", mouth: "____", panel1: "ARC-LLAMA", panel2: "" },
    { eyes: "-    -", mouth: "____", panel1: "ARC-LLAMA", panel2: "" },
  ],
  thinking: [
    { eyes: "o    o", mouth: "....", panel1: "THINKING", panel2: "" },
    { eyes: "o    o", mouth: "....", panel1: "THINKING", panel2: "." },
    { eyes: "o    o", mouth: "....", panel1: "THINKING", panel2: ".." },
    { eyes: "o    o", mouth: "....", panel1: "THINKING", panel2: "..." },
  ],
  read: [
    { eyes: "v    v", mouth: "____", panel1: "READING", panel2: "[=  ]" },
    { eyes: "v    v", mouth: "____", panel1: "READING", panel2: "[ = ]" },
    { eyes: "v    v", mouth: "____", panel1: "READING", panel2: "[  =]" },
    { eyes: "v    v", mouth: "____", panel1: "READING", panel2: "[ = ]" },
  ],
  write: [
    { eyes: "o    o", mouth: "____", panel1: "WRITING", panel2: "code_" },
    { eyes: "o    o", mouth: "____", panel1: "WRITING", panel2: "code " },
  ],
  command: [
    { eyes: "o    o", mouth: "____", panel1: "$ EXEC", panel2: "/" },
    { eyes: "o    o", mouth: "____", panel1: "$ EXEC", panel2: "-" },
    { eyes: "o    o", mouth: "____", panel1: "$ EXEC", panel2: "\\" },
    { eyes: "o    o", mouth: "____", panel1: "$ EXEC", panel2: "|" },
  ],
  search: [
    { eyes: "O    O", mouth: "____", panel1: "SEARCH", panel2: "[o   ]" },
    { eyes: "O    O", mouth: "____", panel1: "SEARCH", panel2: "[ o  ]" },
    { eyes: "O    O", mouth: "____", panel1: "SEARCH", panel2: "[  o ]" },
    { eyes: "O    O", mouth: "____", panel1: "SEARCH", panel2: "[   o]" },
  ],
  list: [
    { eyes: "o    o", mouth: "____", panel1: "FILES", panel2: "* - -" },
    { eyes: "o    o", mouth: "____", panel1: "FILES", panel2: "- * -" },
    { eyes: "o    o", mouth: "____", panel1: "FILES", panel2: "- - *" },
  ],
  working: [
    { eyes: "o    o", mouth: "____", panel1: "WORKING", panel2: "/" },
    { eyes: "o    o", mouth: "____", panel1: "WORKING", panel2: "-" },
    { eyes: "o    o", mouth: "____", panel1: "WORKING", panel2: "\\" },
    { eyes: "o    o", mouth: "____", panel1: "WORKING", panel2: "|" },
  ],
  success: [
    { eyes: "^    ^", mouth: "\\__/", panel1: "DONE", panel2: "" },
    { eyes: "^    ^", mouth: "\\__/", panel1: "DONE", panel2: "\\o/" },
  ],
  done: [
    { eyes: "^    ^", mouth: "\\__/", panel1: "DONE", panel2: "" },
    { eyes: "^    ^", mouth: "\\__/", panel1: "DONE", panel2: "\\o/" },
  ],
  error: [
    { eyes: "x    x", mouth: "/??\\", panel1: "ERROR", panel2: "!!!" },
    { eyes: "x    x", mouth: "/??\\", panel1: "ERROR", panel2: "" },
  ],
};

const AGENT_ASCII_FRAME_MS = 450;
const agentAsciiBg = $("#agent-ascii-bg");
let agentAsciiState = "idle";
let agentAsciiFrameIndex = 0;

function renderAgentAscii() {
  if (!agentAsciiBg) return;
  const frames = AGENT_ASCII[agentAsciiState] || AGENT_ASCII.idle;
  agentAsciiBg.textContent = llamaFrame(frames[agentAsciiFrameIndex % frames.length]);
  agentAsciiBg.className = "agent-ascii-bg state-" + agentAsciiState;
}

function setAgentAscii(state) {
  if (!agentAsciiBg) return;
  if (state === agentAsciiState) return;
  agentAsciiState = AGENT_ASCII[state] ? state : "idle";
  agentAsciiFrameIndex = 0;
  renderAgentAscii();
}

renderAgentAscii();
setInterval(() => {
  agentAsciiFrameIndex += 1;
  renderAgentAscii();
}, AGENT_ASCII_FRAME_MS);

function shouldAutoScroll(container) {
  if (!container) return true;
  const threshold = 60;
  return container.scrollHeight - container.scrollTop - container.clientHeight <= threshold;
}

function autoScroll(container) {
  if (!container) return;
  container.scrollTop = container.scrollHeight;
}

const agentRenderer = {
  thinkingEl: null,
  toolCards: new Map(),
  stepNumber: 0,

  clear() {
    const bg = $("#agent-ascii-bg");
    agentLog.innerHTML = "";
    if (bg) agentLog.appendChild(bg);
    if (agentEmptyState) agentEmptyState.style.display = "";
    this.toolCards.clear();
    this.thinkingEl = null;
    this.stepNumber = 0;
    setAgentAscii("idle");
  },

  renderPrompt(task) {
    if (agentEmptyState) agentEmptyState.style.display = "none";
    const el = document.createElement("div");
    el.className = "agent-prompt";
    el.textContent = task;
    agentLog.appendChild(el);
  },

  setThinking(active) {
    if (active) {
      if (this.thinkingEl) return;
      if (agentEmptyState) agentEmptyState.style.display = "none";
      const el = document.createElement("div");
      el.className = "agent-log-line thinking";
      el.innerHTML = `<div class="agent-thinking"><span>Thinking</span><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>`;
      agentLog.appendChild(el);
      if (shouldAutoScroll(agentLog)) autoScroll(agentLog);
      this.thinkingEl = el;
      setAgentAscii("thinking");
    } else {
      if (this.thinkingEl) {
        this.thinkingEl.remove();
        this.thinkingEl = null;
      }
    }
  },

  addEvent(event) {
    if (agentEmptyState) agentEmptyState.style.display = "none";
    switch (event.type) {
      case "status":
        this.renderStatus(event);
        break;
      case "assistant":
        this.renderAssistant(event);
        break;
      case "tool_call":
        this.renderToolCall(event);
        break;
      case "tool_result":
        this.renderToolResult(event);
        break;
      case "confirm_required":
        this.renderConfirm(event);
        break;
      case "error":
        this.renderError(event);
        break;
      case "done":
        this.renderDone(event);
        break;
      default:
        this.renderRaw(event);
    }
    if (shouldAutoScroll(agentLog)) autoScroll(agentLog);
  },

  renderStatus(event) {
    const el = document.createElement("div");
    el.className = "agent-log-line status";
    el.textContent = "# " + (event.message || "");
    agentLog.appendChild(el);
  },

  renderAssistant(event) {
    this.setThinking(false);
    const parsed = parseThinking(event.content || "");
    const line = document.createElement("div");
    line.className = "agent-log-line assistant";

    let html = "";
    if (parsed.thinking) {
      html += `
        <div class="agent-thinking-block">
          <div class="agent-thinking-toggle"><span class="chevron">▶</span> Thinking</div>
          <div class="agent-thinking-content"></div>
        </div>`;
    }
    html += `<div class="agent-assistant"></div>`;
    line.innerHTML = html;
    agentLog.appendChild(line);

    const thinkingContent = line.querySelector(".agent-thinking-content");
    if (thinkingContent && parsed.thinking) {
      thinkingContent.textContent = parsed.thinking;
    }
    const assistantContent = line.querySelector(".agent-assistant");
    if (assistantContent) {
      renderMarkdown(assistantContent, parsed.content);
    }

    const toggle = line.querySelector(".agent-thinking-toggle");
    if (toggle) {
      toggle.addEventListener("click", () => makeToggle(line, ".agent-thinking-toggle", ".agent-thinking-content"));
    }
  },

  _makeToolDetail(label, content, open = false) {
    const detail = document.createElement("div");
    detail.className = "agent-tool-detail";
    const trimmed = String(content || "").trimEnd();
    detail.innerHTML = `
      <div class="agent-tool-detail-toggle${open ? " open" : ""}"><span class="chevron">▶</span> ${escapeHtml(label)}</div>
      <div class="agent-tool-detail-body${open ? " open" : ""}"><pre>${escapeHtml(trimmed)}</pre></div>`;
    const toggle = detail.querySelector(".agent-tool-detail-toggle");
    toggle.addEventListener("click", () => makeToggle(detail, ".agent-tool-detail-toggle", ".agent-tool-detail-body"));
    return detail;
  },

  _asciiForTool(name) {
    if (name.includes("search")) return "search";
    if (name.includes("write")) return "write";
    if (name.includes("read")) return "read";
    if (name === "list_directory") return "list";
    if (name === "run_command") return "command";
    return "working";
  },

  renderToolCall(event) {
    this.setThinking(false);
    this.stepNumber += 1;
    const id = event.id || `orphan-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const name = event.name || "unknown";
    const args = event.arguments || {};
    const icon = getToolIcon(name);
    const kind = getToolKind(name);
    const target = this._toolTarget(name, args);

    setAgentAscii(this._asciiForTool(name));

    const line = document.createElement("div");
    line.className = "agent-log-line tool agent-tool pending";
    line.dataset.callId = id;
    line.innerHTML = `
      <div class="agent-tool-header">
        <span class="agent-step-number">${this.stepNumber}</span>
        <span class="agent-tool-icon">${icon}</span>
        <span class="agent-tool-title">
          <span class="agent-tool-kind">${escapeHtml(kind)}</span>
          ${target ? `<span class="agent-tool-target">${escapeHtml(target)}</span>` : ""}
        </span>
        <span class="agent-tool-status"><span class="spinner"></span> Running…</span>
      </div>
      <div class="agent-tool-details"></div>`;
    agentLog.appendChild(line);

    const details = line.querySelector(".agent-tool-details");
    details.appendChild(this._makeToolDetail("Arguments", JSON.stringify(args, null, 2), false));
    this.toolCards.set(id, { el: line, step: this.stepNumber });
  },

  _toolTarget(name, args) {
    if (args.path) return String(args.path);
    if (args.file_path) return String(args.file_path);
    if (args.directory) return String(args.directory);
    if (args.command) {
      const cmd = String(args.command);
      return cmd.length > 60 ? cmd.slice(0, 57) + "…" : cmd;
    }
    if (args.query) {
      const q = String(args.query);
      return q.length > 60 ? q.slice(0, 57) + "…" : q;
    }
    return "";
  },

  renderToolResult(event) {
    const id = event.id;
    let card = id ? this.toolCards.get(id) : null;
    if (!card) {
      const pending = [...this.toolCards.values()].find((c) => c.el.classList.contains("pending"));
      card = pending;
    }

    if (!card) {
      const line = document.createElement("div");
      line.className = "agent-log-line tool agent-tool" + (event.error ? " error" : " success");
      line.innerHTML = `
        <div class="agent-tool-row">
          <span class="agent-tool-icon">${event.error ? ICON_X : ICON_CHECK}</span>
          <span class="agent-tool-name"><code>${escapeHtml(event.name || "")}</code></span>
          <span class="agent-tool-status">${event.error ? "Failed" : "Done"}</span>
        </div>
        <div class="agent-tool-details"></div>`;
      agentLog.appendChild(line);
      const details = line.querySelector(".agent-tool-details");
      details.appendChild(this._makeToolDetail("Result", event.content || "", true));
      return;
    }

    const line = card.el;
    line.classList.remove("pending");
    line.classList.add(event.error ? "error" : "success");

    const status = line.querySelector(".agent-tool-status");
    if (status) {
      status.innerHTML = event.error
        ? `<span class="status-icon">${ICON_X}</span> Failed`
        : `<span class="status-icon">${ICON_CHECK}</span> Done`;
    }

    const iconWrap = line.querySelector(".agent-tool-icon");
    if (iconWrap) iconWrap.innerHTML = event.error ? ICON_X : getToolIcon(event.name || "");

    const details = line.querySelector(".agent-tool-details");
    const existingResult = details.querySelector(".agent-tool-result");
    if (existingResult) existingResult.remove();
    const resultDetail = this._makeToolDetail("Result", event.content || "", true);
    resultDetail.classList.add("agent-tool-result");
    details.appendChild(resultDetail);

    setAgentAscii(event.error ? "error" : "success");
    this.setThinking(true);
  },

  renderConfirm(event) {
    this.setThinking(false);
    const runId = event.run_id;
    const id = event.id;
    const tool = event.tool || "unknown";
    const args = event.arguments || {};

    let card = id ? this.toolCards.get(id) : null;

    if (!card) {
      // Build an inline confirmation line when no preceding tool_call card exists.
      const line = document.createElement("div");
      line.className = "agent-log-line tool agent-tool confirm-required";
      line.dataset.runId = runId || "";
      line.dataset.callId = id || "";
      line.innerHTML = `
        <div class="agent-confirm-row">
          <span class="agent-tool-icon">${ICON_WARN}</span>
          <span class="agent-tool-name"><code>${escapeHtml(tool)}</code> <span style="color:var(--fg-mute)">needs approval</span></span>
          <span class="agent-confirm-actions">
            <button class="agent-confirm-btn approve" data-action="approve">${ICON_CHECK} Allow</button>
            <button class="agent-confirm-btn deny" data-action="deny">${ICON_X} Deny</button>
          </span>
        </div>
        <div class="agent-tool-details"></div>`;
      agentLog.appendChild(line);
      const details = line.querySelector(".agent-tool-details");
      details.appendChild(this._makeToolDetail("Arguments", JSON.stringify(args, null, 2), false));
      card = { el: line };
    } else {
      // Replace the running status on the existing tool card with confirm buttons.
      const line = card.el;
      line.classList.add("confirm-required");
      const status = line.querySelector(".agent-tool-status");
      if (status) {
        status.innerHTML = `
          <span class="agent-confirm-actions">
            <button class="agent-confirm-btn approve" data-action="approve">${ICON_CHECK} Allow</button>
            <button class="agent-confirm-btn deny" data-action="deny">${ICON_X} Deny</button>
          </span>`;
      }
      line.dataset.runId = runId || "";
    }

    if (!runId) return;
    const line = card.el;
    const approveBtn = line.querySelector('.agent-confirm-btn[data-action="approve"]');
    const denyBtn = line.querySelector('.agent-confirm-btn[data-action="deny"]');

    const submitConfirmation = async (approved) => {
      if (approveBtn) approveBtn.disabled = true;
      if (denyBtn) denyBtn.disabled = true;
      const spinner = '<span class="spinner"></span>';
      const label = approved ? "Allowing…" : "Denying…";
      const clicked = approved ? approveBtn : denyBtn;
      if (clicked) clicked.innerHTML = spinner + " " + label;
      try {
        const r = await fetch(`/v1/agent/${encodeURIComponent(runId)}/confirm`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ approved }),
        });
        if (!r.ok) {
          const t = await r.text();
          throw new Error(`${r.status} ${t}`);
        }
      } catch (e) {
        if (approveBtn) {
          approveBtn.disabled = false;
          approveBtn.innerHTML = `${ICON_CHECK} Allow`;
        }
        if (denyBtn) {
          denyBtn.disabled = false;
          denyBtn.innerHTML = `${ICON_X} Deny`;
        }
        this.renderError({ message: "Confirmation failed: " + e.message });
      }
    };

    if (approveBtn) approveBtn.addEventListener("click", () => submitConfirmation(true));
    if (denyBtn) denyBtn.addEventListener("click", () => submitConfirmation(false));
  },

  renderError(event) {
    this.setThinking(false);
    setAgentAscii("error");
    const line = document.createElement("div");
    line.className = "agent-log-line error";
    line.innerHTML = `<div class="agent-error"><span style="color:#d8a0a0">Error:</span> ${escapeHtml(event.message || "")}</div>`;
    agentLog.appendChild(line);
  },

  renderDone(event) {
    this.setThinking(false);
    setAgentAscii("done");
    const line = document.createElement("div");
    line.className = "agent-log-line done";
    line.innerHTML = `<div class="agent-done">${ICON_CHECK} Agent finished.</div>`;
    agentLog.appendChild(line);
  },

  renderRaw(event) {
    this.setThinking(false);
    const line = document.createElement("div");
    line.className = "agent-log-line";
    line.innerHTML = `<pre>${escapeHtml(JSON.stringify(event, null, 2))}</pre>`;
    agentLog.appendChild(line);
  }
};

async function runAgentTask() {
  const task = input.value.trim();
  if (!task || agentRunning || !selectedModel) return;

  input.value = "";
  input.style.height = "auto";
  agentRenderer.clear();
  agentRenderer.renderPrompt(task);
  agentRunning = true;
  sendButton.disabled = true;
  inputWrap.classList.add("generating");

  // Ensure model is loaded
  try {
    await ensureModelLoaded();
  } catch (e) {
    agentRenderer.addEvent({ type: "error", message: e.message });
    finishAgentRun();
    return;
  }

  const autoConfirm = agentAutoConfirm.checked;
  const maxTurns = parseInt(agentMaxTurns.value, 10) || 30;

  const abortController = new AbortController();
  agentAbort = () => abortController.abort();

  agentRenderer.addEvent({ type: "status", message: `Running task with ${autoConfirm ? "auto-confirm" : "manual confirmation"}` });
  agentRenderer.setThinking(true);

  let agentStartTime = null;
  let agentTokenCount = 0;

  try {
    const r = await fetch("/v1/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: selectedModel,
        task: task,
        auto_confirm: autoConfirm,
        max_turns: maxTurns,
      }),
      signal: abortController.signal,
    });
    if (!r.ok) {
      const t = await r.text();
      throw new Error(`${r.status} ${t}`);
    }
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice(5).trim();
        if (payload === "[DONE]") continue;
        try {
          const event = JSON.parse(payload);
          agentRenderer.addEvent(event);
          if (event.type === "assistant" && event.content) {
            if (agentStartTime === null) agentStartTime = Date.now();
            agentTokenCount += Math.max(1, Math.round(event.content.length / 4));
          }
        } catch (e) {
          // ignore malformed SSE payloads
        }
      }
    }
    if (agentStartTime && agentTokenCount > 0) {
      const elapsed = (Date.now() - agentStartTime) / 1000;
      const tps = elapsed > 0 ? (agentTokenCount / elapsed).toFixed(1) : null;
      if (tps) ctxLabelTps.textContent = tps + " tok/s";
    }
  } catch (e) {
    if (e.name !== "AbortError") {
      agentRenderer.addEvent({ type: "error", message: "Agent failed: " + e.message });
    }
  } finally {
    agentAbort = null;
    finishAgentRun();
  }
}

function finishAgentRun() {
  agentRunning = false;
  sendButton.disabled = false;
  inputWrap.classList.remove("generating");
  agentRenderer.setThinking(false);
  input.focus();
}

// ------------------------------------------------------------------
// Slash commands
// ------------------------------------------------------------------

const SLASH_COMMANDS = [
  { name: "help", desc: "Show available slash commands", needsArgs: false },
  { name: "clear", desc: "Clear the current conversation", needsArgs: false },
  { name: "new", desc: "Start a new chat", needsArgs: false },
  { name: "model", desc: "Switch model, e.g. /model <id>", needsArgs: true },
  { name: "compact", desc: "Summarize context, optional: /compact <focus>", needsArgs: false },
];

let paletteSelectedIndex = -1;

function parseSlashCommand(text) {
  const trimmed = text.trim();
  if (!trimmed.startsWith("/")) return null;
  const withoutSlash = trimmed.slice(1);
  const firstSpace = withoutSlash.search(/\s/);
  const command = firstSpace === -1 ? withoutSlash : withoutSlash.slice(0, firstSpace);
  const rest = firstSpace === -1 ? "" : withoutSlash.slice(firstSpace + 1).trim();
  return { command: command.toLowerCase(), rest, raw: trimmed };
}

function getFilteredCommands(prefix) {
  const p = prefix.toLowerCase();
  return SLASH_COMMANDS.filter((c) => c.name.startsWith(p));
}

function hideCommandPalette() {
  commandPalette.classList.remove("open");
  commandPalette.innerHTML = "";
  paletteSelectedIndex = -1;
}

function renderCommandPalette(filter = "") {
  const items = filter === "" ? SLASH_COMMANDS.slice() : getFilteredCommands(filter);
  commandPalette.innerHTML = "";
  if (items.length === 0) {
    hideCommandPalette();
    return;
  }
  paletteSelectedIndex = Math.min(Math.max(paletteSelectedIndex, 0), items.length - 1);
  for (let i = 0; i < items.length; i++) {
    const cmd = items[i];
    const div = document.createElement("div");
    div.className = "command-item" + (i === paletteSelectedIndex ? " selected" : "");
    div.setAttribute("role", "option");
    div.setAttribute("aria-selected", String(i === paletteSelectedIndex));
    div.innerHTML = `
      <span class="cmd-name">/${escapeHtml(cmd.name)}</span>
      <span class="cmd-desc">${escapeHtml(cmd.desc)}</span>
      <span class="cmd-hint">${cmd.needsArgs ? "args" : "enter"}</span>
    `;
    div.addEventListener("click", () => {
      input.value = "/" + cmd.name + " ";
      input.focus();
      hideCommandPalette();
      input.dispatchEvent(new Event("input"));
    });
    div.addEventListener("mouseenter", () => {
      paletteSelectedIndex = i;
      renderCommandPalette(filter);
    });
    commandPalette.appendChild(div);
  }
  commandPalette.classList.add("open");
}

function updateCommandPalette() {
  const text = input.value;
  if (!agentMode && text.startsWith("/") && !text.includes(" ")) {
    const prefix = text.slice(1);
    renderCommandPalette(prefix);
  } else {
    hideCommandPalette();
  }
}

function executeSlashCommand(rawText) {
  const parsed = parseSlashCommand(rawText);
  if (!parsed) return false;

  const known = SLASH_COMMANDS.find((c) => c.name === parsed.command);
  if (!known) {
    showError(`Unknown command: /${escapeHtml(parsed.command)}. Type /help for available commands.`);
    return true;
  }

  switch (parsed.command) {
    case "help":
      renderHelpMessage();
      break;
    case "clear":
      clearChat();
      break;
    case "new":
      newChat();
      break;
    case "model":
      switchModel(parsed.rest);
      break;
    case "compact":
      compactConversation(parsed.rest);
      break;
  }
  return true;
}

function renderHelpMessage() {
  if (emptyState) emptyState.style.display = "none";
  const div = document.createElement("div");
  div.className = "message system command-hint";
  const roleLabel = document.createElement("div");
  roleLabel.className = "role";
  roleLabel.textContent = "Slash commands";
  div.appendChild(roleLabel);
  const content = document.createElement("div");
  content.className = "content";
  let html = "";
  for (const cmd of SLASH_COMMANDS) {
    html += `<p><code>/${cmd.name}</code> <strong>—</strong> ${escapeHtml(cmd.desc)}</p>`;
  }
  content.innerHTML = html;
  div.appendChild(content);
  chatLog.appendChild(div);
  if (shouldAutoScroll(chatLog)) autoScroll(chatLog);
}

function clearChat() {
  conversation.length = 0;
  chatLog.innerHTML = "";
  if (emptyState) emptyState.style.display = "";
  updateCtxMeter(0, models.find((m) => m.id === selectedModel)?.ctx || 131072);
  ctxLabelTps.textContent = "";
  saveCurrentChat();
}

function switchModel(modelId) {
  if (!modelId) {
    showError("Usage: /model <model-id>");
    return;
  }
  const m = models.find((x) => x.id === modelId || x.id.endsWith("/" + modelId) || (x.display_name && x.display_name.toLowerCase() === modelId.toLowerCase()));
  if (!m) {
    showError(`Model not found: ${escapeHtml(modelId)}`);
    return;
  }
  selectedModel = m.id;
  modelSelect.value = m.id;
  updatePickerStatus();
  if (settingsPanel.classList.contains("open")) renderSettingsPanel();
  ensureModelLoaded().catch((e) => showError(e.message));
}

async function compactConversation(instruction) {
  if (conversation.length === 0) {
    showError("Nothing to compact.");
    return;
  }
  if (!selectedModel) {
    showError("Select a model first.");
    return;
  }
  try {
    await ensureModelLoaded();
  } catch (e) {
    showError(e.message);
    return;
  }

  const systemPrompt = instruction
    ? `Summarize the following conversation concisely. Focus on: ${instruction}. Preserve key facts, decisions, code snippets, and user intent. Return only the summary.`
    : "Summarize the following conversation concisely. Preserve key facts, decisions, code snippets, and user intent. Return only the summary.";

  const summaryMsg = { role: "system", content: systemPrompt };
  const messages = [summaryMsg, ...conversation];

  try {
    const r = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: selectedModel, messages, stream: false }),
    });
    if (!r.ok) {
      const t = await r.text();
      throw new Error(`${r.status} ${t}`);
    }
    const data = await r.json();
    const summary = data.choices?.[0]?.message?.content?.trim();
    if (!summary) {
      throw new Error("Model returned an empty summary.");
    }

    conversation.length = 0;
    conversation.push({ role: "system", content: "Summary of prior conversation:\n\n" + summary });

    chatLog.innerHTML = "";
    if (emptyState) emptyState.style.display = "none";
    const msg = createMessage("system", "Context compacted. Summary:\n\n" + summary);
    if (shouldAutoScroll(chatLog)) autoScroll(chatLog);

    const m = models.find((x) => x.id === selectedModel);
    updateCtxMeter(estimateTokens(), m?.ctx || 131072);
    saveCurrentChat();
  } catch (e) {
    showError("Compact failed: " + e.message);
  }
}

input.addEventListener("keydown", (e) => {
  if (commandPalette.classList.contains("open")) {
    const items = commandPalette.querySelectorAll(".command-item");
    if (e.key === "ArrowDown") {
      e.preventDefault();
      paletteSelectedIndex = (paletteSelectedIndex + 1) % items.length;
      renderCommandPalette(input.value.slice(1));
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      paletteSelectedIndex = (paletteSelectedIndex - 1 + items.length) % items.length;
      renderCommandPalette(input.value.slice(1));
      return;
    }
    if (e.key === "Escape") {
      e.preventDefault();
      hideCommandPalette();
      return;
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const selected = items[paletteSelectedIndex];
      if (selected) selected.click();
      return;
    }
  }

  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    const text = input.value.trim();
    if (!agentMode && executeSlashCommand(text)) {
      input.value = "";
      input.style.height = "auto";
      hideCommandPalette();
      return;
    }
    if (agentMode) {
      runAgentTask();
    } else {
      sendMessage();
    }
  }
});

sendButton.addEventListener("click", () => {
  const text = input.value.trim();
  if (!agentMode && executeSlashCommand(text)) {
    input.value = "";
    input.style.height = "auto";
    hideCommandPalette();
    return;
  }
  if (agentMode) {
    runAgentTask();
  } else {
    sendMessage();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 96) + "px";
  updateCommandPalette();
});

(async function init() {
  await fetchModels();
  await syncChatsFromServer();
})();
statusPoller = setInterval(fetchStatus, 3000);