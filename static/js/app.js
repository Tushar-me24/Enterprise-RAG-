const chat = document.getElementById("chat");
const form = document.getElementById("chatForm");
const question = document.getElementById("question");
const trace = document.getElementById("trace");
const sourceUsed = document.getElementById("sourceUsed");
const sourceDescription = document.getElementById("sourceDescription");
const sendBtn = document.getElementById("sendBtn");
const runStatus = document.getElementById("runStatus");
const runDuration = document.getElementById("runDuration");
const runCaption = document.getElementById("runCaption");
const traceCount = document.getElementById("traceCount");

const escapeHtml = (value = "") => String(value).replace(/[&<>'"]/g, char => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
}[char]));

function formatText(value = "") {
  return escapeHtml(value).replace(/\n/g, "<br>");
}

function addMessage(role, text, source = "", citations = []) {
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;

  const validCitations = Array.isArray(citations) ? citations : [];
  const citationHtml = validCitations.length
    ? `<div class="citations"><strong>Sources</strong>${validCitations.map(item => {
        const title = escapeHtml(item.title || "Source");
        const url = typeof item.url === "string" ? item.url : "";
        // Only render normal HTTP(S) links supplied by the backend.
        if (/^https?:\/\//i.test(url)) {
          return `<div><a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${title}</a></div>`;
        }
        return `<div>${title}</div>`;
      }).join("")}</div>`
    : "";

  wrap.innerHTML = `
    ${role === "assistant" ? '<div class="message-avatar">✳</div>' : ""}
    <div class="bubble">${formatText(text)}
      ${source ? `<div class="answer-source">Evidence · ${escapeHtml(source)}</div>` : ""}
      ${citationHtml}
    </div>`;
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
}

function classifyTrace(item) {
  const value = String(item).toLowerCase();
  if (value.includes("weak") || value.includes("insufficient") || value.includes("stopped")) return "warn";
  if (value.includes("good") || value.includes("generation") || value.includes("retrieval")) return "success";
  return "";
}

function traceIcon(item, index) {
  const value = String(item).toLowerCase();
  if (value.includes("router")) return "⌁";
  if (value.includes("retrieval")) return "⌕";
  if (value.includes("grade")) return "◈";
  if (value.includes("web")) return "↗";
  if (value.includes("rewrite")) return "⟳";
  if (value.includes("generation") || value.includes("answer")) return "✳";
  if (value.includes("direct")) return "↗";
  return String(index + 1);
}

function renderTrace(items = []) {
  const safeItems = Array.isArray(items) ? items : [];
  traceCount.textContent = `${safeItems.length} EVENT${safeItems.length === 1 ? "" : "S"}`;

  if (!safeItems.length) {
    trace.innerHTML = `<div class="trace-empty"><div class="trace-empty-icon">⌘</div><b>No activity yet</b><span>Your agent's execution path will appear here.</span></div>`;
    return;
  }

  trace.innerHTML = safeItems.map((item, index) => {
    const label = String(item);
    const status = classifyTrace(label);
    return `<div class="trace-item ${status}">
      <span class="trace-marker">${traceIcon(label, index)}</span>
      <div class="trace-copy"><b>${escapeHtml(label)}</b><small>LangGraph node · ${String(index + 1).padStart(2, "0")}</small></div>
    </div>`;
  }).join("");
}

function setSource(source) {
  const normalized = String(source || "unknown");
  const labels = {
    private_kb: ["Private company KB", "Answer grounded in Nexora's indexed internal documents.", "⌑"],
    web_search: ["External web search", "Private evidence was insufficient; answer uses external information.", "↗"],
    web: ["External web search", "Answer uses external information.", "↗"],
    direct: ["Direct response", "General conversation; no knowledge-base retrieval was required.", "✳"],
    insufficient_evidence: ["Insufficient evidence", "The agent could not establish a reliable answer.", "!"],
    kb: ["Private knowledge base", "The router selected internal company knowledge.", "⌑"]
  };
  const [label, description, icon] = labels[normalized] || [normalized.replaceAll("_", " "), "Source returned by the backend.", "⌑"];
  sourceUsed.innerHTML = `<span class="source-icon">${icon}</span><span>${escapeHtml(label)}</span>`;
  sourceDescription.textContent = description;
}

function setRunning(running) {
  sendBtn.disabled = running;
  sendBtn.querySelector("span").textContent = running ? "Working" : "Ask Copilot";
  sendBtn.querySelector("b").textContent = running ? "…" : "↑";
  runStatus.textContent = running ? "RUNNING" : "IDLE";
  runStatus.className = running ? "run-status running" : "run-status";
  if (running) {
    runDuration.textContent = "…";
    runCaption.textContent = "Waiting for agent response";
    sourceUsed.innerHTML = '<span class="source-dash">—</span><span>In progress</span>';
    sourceDescription.textContent = "The agent is evaluating the best evidence source.";
  }
}

async function askAgent(rawQuestion) {
  const q = String(rawQuestion || "").trim();
  if (!q || sendBtn.disabled) return;

  addMessage("user", q);
  question.value = "";
  autoResize();
  setRunning(true);
  renderTrace(["Request submitted · awaiting LangGraph execution"]);
  const startedAt = performance.now();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q })
    });

    let data;
    try { data = await response.json(); }
    catch { throw new Error("The server returned an invalid response. Check the API logs."); }

    if (!response.ok) throw new Error(data.detail || data.message || "Request failed");
    if (!data || typeof data.answer !== "string") throw new Error("The API response is missing the answer field.");

    addMessage("assistant", data.answer, data.source_used || "", data.citations || []);
    renderTrace(data.trace || []);
    setSource(data.source_used || "unknown");
    runStatus.textContent = "COMPLETE";
    runStatus.className = "run-status complete";
    runDuration.textContent = `${((performance.now() - startedAt) / 1000).toFixed(1)}s`;
    runCaption.textContent = "Execution finished successfully";
  } catch (error) {
    addMessage("assistant", `I couldn't complete that request. ${error.message}`);
    renderTrace(["Request failed · check the API connection and server logs"]);
    runStatus.textContent = "ERROR";
    runStatus.className = "run-status warn";
    runDuration.textContent = "—";
    runCaption.textContent = "Execution did not complete";
    setSource("insufficient_evidence");
  } finally {
    setRunning(false);
    chat.scrollTop = chat.scrollHeight;
  }
}

form.addEventListener("submit", event => {
  event.preventDefault();
  askAgent(question.value);
});

document.querySelectorAll(".suggestion-card").forEach(button => {
  button.addEventListener("click", () => askAgent(button.dataset.question));
});

function autoResize() {
  question.style.height = "auto";
  question.style.height = `${Math.min(question.scrollHeight, 130)}px`;
}
question.addEventListener("input", autoResize);
question.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

// Document ingestion modal
const modal = document.getElementById("uploadModal");
const fileInput = document.getElementById("fileInput");
const fileName = document.getElementById("fileName");
const dropZone = document.getElementById("dropZone");
const uploadStatus = document.getElementById("uploadStatus");
const uploadBtn = document.getElementById("uploadBtn");

function openModal() {
  modal.classList.remove("hidden");
  uploadStatus.textContent = "";
  uploadStatus.className = "upload-status";
}
function closeModal() {
  modal.classList.add("hidden");
}
document.getElementById("openUpload").addEventListener("click", openModal);
document.getElementById("closeUpload").addEventListener("click", closeModal);
document.getElementById("modalBackdrop").addEventListener("click", closeModal);
document.addEventListener("keydown", event => {
  if (event.key === "Escape" && !modal.classList.contains("hidden")) closeModal();
});

function showSelectedFile(file) {
  if (!file) {
    fileName.textContent = "Choose a document or drag it here";
    return;
  }
  const allowed = /\.(pdf|txt|md|docx)$/i.test(file.name);
  if (!allowed) {
    fileInput.value = "";
    fileName.textContent = "Unsupported file type — choose PDF, TXT, MD or DOCX";
    uploadStatus.textContent = "Please select a supported document.";
    uploadStatus.className = "upload-status error";
    return;
  }
  fileName.textContent = file.name;
  uploadStatus.textContent = "";
  uploadStatus.className = "upload-status";
}
fileInput.addEventListener("change", () => showSelectedFile(fileInput.files[0]));

["dragenter", "dragover"].forEach(type => dropZone.addEventListener(type, event => {
  event.preventDefault();
  dropZone.classList.add("dragover");
}));
["dragleave", "drop"].forEach(type => dropZone.addEventListener(type, event => {
  event.preventDefault();
  dropZone.classList.remove("dragover");
}));
dropZone.addEventListener("drop", event => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  showSelectedFile(file);
});

uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  const key = document.getElementById("adminKey").value.trim();

  if (!file) {
    uploadStatus.textContent = "Choose a document first.";
    uploadStatus.className = "upload-status error";
    return;
  }
  if (!/\.(pdf|txt|md|docx)$/i.test(file.name)) {
    uploadStatus.textContent = "Unsupported file type.";
    uploadStatus.className = "upload-status error";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  uploadBtn.disabled = true;
  uploadBtn.querySelector("span").textContent = "Indexing document...";
  uploadStatus.textContent = "Uploading and indexing. This may take a moment.";
  uploadStatus.className = "upload-status";

  try {
    const response = await fetch("/api/ingest", {
      method: "POST",
      headers: key ? { "X-Admin-Key": key } : {},
      body: formData
    });
    let data;
    try { data = await response.json(); }
    catch { throw new Error("The server returned an invalid response."); }
    if (!response.ok) throw new Error(data.detail || data.message || "Document upload failed");

    const indexedName = data.file || file.name;
    const chunks = Number.isFinite(Number(data.chunks)) ? ` · ${data.chunks} chunks` : "";
    uploadStatus.textContent = `Successfully indexed ${indexedName}${chunks}.`;
    uploadStatus.className = "upload-status";
    fileInput.value = "";
    fileName.textContent = "Choose a document or drag it here";
  } catch (error) {
    uploadStatus.textContent = `Error: ${error.message}`;
    uploadStatus.className = "upload-status error";
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.querySelector("span").textContent = "Index document";
  }
});
