const dropzone = document.getElementById("dropzone");
const dropzoneHint = document.getElementById("dropzone-hint");
const fileInput = document.getElementById("image-input");
const preview = document.getElementById("preview");
const form = document.getElementById("upload-form");
const submitBtn = document.getElementById("submit-btn");
const resultCard = document.getElementById("result-card");
const adviceCard = document.getElementById("advice-card");
const errorBox = document.getElementById("error-box");
const adviceBtn = document.getElementById("advice-btn");
const copyAdviceBtn = document.getElementById("copy-advice-btn");
const providerSel = document.getElementById("llm-provider");
const apiKeyInput = document.getElementById("llm-api-key");
const modelInput = document.getElementById("llm-model");
const modelList = document.getElementById("llm-model-list");
const lowConfAlert = document.getElementById("low-conf-alert");
const topCandWrap = document.getElementById("top-candidates-wrap");
const topCandList = document.getElementById("top-candidates-list");

let lastResult = null;
let currentObjectURL = null;
let providersCatalog = {};
let rawAdviceMarkdown = "";

// 拉 provider/model 清单填到下拉框 + datalist。失败不阻塞核心流程，
// 用户可以手填模型 ID。
async function loadProviders() {
  try {
    const resp = await fetch("/api/llm/providers");
    if (!resp.ok) return;
    const data = await resp.json();
    providersCatalog = data.providers || {};
    providerSel.innerHTML = "";
    const auto = document.createElement("option");
    auto.value = "";
    auto.textContent = "（用服务器默认）";
    providerSel.appendChild(auto);
    for (const name of Object.keys(providersCatalog)) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      providerSel.appendChild(opt);
    }
    // sessionStorage 恢复（仅本标签页存活；关掉浏览器即清）
    const saved = JSON.parse(sessionStorage.getItem("llmSettings") || "{}");
    if (saved.provider) providerSel.value = saved.provider;
    if (saved.apiKey) apiKeyInput.value = saved.apiKey;
    if (saved.model) modelInput.value = saved.model;
    refreshModelList();
  } catch {
    // 静默：保留输入框，让用户手填
  }
}

function refreshModelList() {
  const spec = providersCatalog[providerSel.value];
  modelList.innerHTML = "";
  if (!spec) return;
  for (const m of spec.models) {
    const opt = document.createElement("option");
    opt.value = m;
    modelList.appendChild(opt);
  }
  // 用户没手填且当前值不在新清单里，重置为该 provider 的默认
  if (!modelInput.value || !spec.models.includes(modelInput.value)) {
    modelInput.value = spec.default;
  }
}

function persistLlmSettings() {
  sessionStorage.setItem(
    "llmSettings",
    JSON.stringify({
      provider: providerSel.value,
      apiKey: apiKeyInput.value,
      model: modelInput.value,
    })
  );
}

providerSel.addEventListener("change", () => {
  refreshModelList();
  persistLlmSettings();
});
apiKeyInput.addEventListener("input", persistLlmSettings);
modelInput.addEventListener("input", persistLlmSettings);

loadProviders();

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden = false;
}
function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1024 / 1024).toFixed(2) + " MB";
}

function showPreview(file) {
  if (!file || !file.type.startsWith("image/")) {
    showError("请选择图片文件");
    return;
  }
  // Revoke any previous object URL to avoid memory leaks
  if (currentObjectURL) URL.revokeObjectURL(currentObjectURL);
  currentObjectURL = URL.createObjectURL(file);

  preview.onload = () => {
    dropzone.classList.add("has-image");
  };
  preview.onerror = () => {
    showError("图片加载失败");
    dropzone.classList.remove("has-image");
  };
  preview.src = currentObjectURL;

  // Update hint text below the image (kept hidden by .has-image, but updated for next selection)
  dropzoneHint.textContent = `已选：${file.name}（${formatSize(file.size)}）`;
}

// 快速样本图片点击加载
document.querySelectorAll(".sample-chip").forEach((chip) => {
  chip.addEventListener("click", async () => {
    clearError();
    const url = chip.dataset.sample;
    const name = chip.dataset.name || "sample.jpg";
    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const blob = await resp.blob();
      const file = new File([blob], name, { type: blob.type || "image/jpeg" });

      const dt = new DataTransfer();
      dt.items.add(file);
      fileInput.files = dt.files;
      showPreview(file);

      resultCard.hidden = true;
      adviceCard.hidden = true;
    } catch (err) {
      showError("加载示例图片失败：" + err.message);
    }
  });
});

// Click on the dropzone opens the file picker
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" || ev.key === " ") {
    ev.preventDefault();
    fileInput.click();
  }
});

["dragover", "dragenter"].forEach((e) =>
  dropzone.addEventListener(e, (ev) => {
    ev.preventDefault();
    dropzone.classList.add("drag");
  })
);
["dragleave", "drop"].forEach((e) =>
  dropzone.addEventListener(e, () => dropzone.classList.remove("drag"))
);
dropzone.addEventListener("drop", (ev) => {
  ev.preventDefault();
  if (ev.dataTransfer.files[0]) {
    fileInput.files = ev.dataTransfer.files;
    showPreview(ev.dataTransfer.files[0]);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) showPreview(fileInput.files[0]);
});

function renderTopCandidates(candidates) {
  if (!topCandWrap || !topCandList) return;
  topCandList.innerHTML = "";
  if (!Array.isArray(candidates) || candidates.length === 0) {
    topCandWrap.hidden = true;
    return;
  }
  candidates.forEach((cand, idx) => {
    const pct = (cand.probability * 100).toFixed(1);
    const item = document.createElement("div");
    item.className = "top-cand-item";
    item.innerHTML = `
      <div class="top-cand-header">
        <span class="top-cand-name">${idx + 1}. ${cand.plant_class} · ${cand.disease_name} <small style="color:var(--muted)">(${cand.disease_degree})</small></span>
        <span class="top-cand-prob">${pct}%</span>
      </div>
      <div class="top-cand-bar-track">
        <div class="top-cand-bar-fill" style="width: ${Math.max(2, Math.min(100, pct))}%"></div>
      </div>
    `;
    topCandList.appendChild(item);
  });
  topCandWrap.hidden = false;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();
  if (!fileInput.files[0]) return showError("请先选择一张图片");

  submitBtn.disabled = true;
  submitBtn.textContent = "识别中…";
  resultCard.hidden = true;
  adviceCard.hidden = true;

  const fd = new FormData();
  fd.append("image", fileInput.files[0]);

  try {
    const resp = await fetch("/predict", { method: "POST", body: fd });
    if (resp.status === 413) {
      return showError("图片太大（超过 10MB），请选小一点的图");
    }
    const data = await resp.json();
    if (!data.success) return showError(data.message || "识别失败");
    lastResult = data.data;

    document.getElementById("r-plant").textContent = data.data.plant_class;
    document.getElementById("r-health").textContent = data.data.health_status;
    document.getElementById("r-disease").textContent = data.data.disease_name;
    document.getElementById("r-degree").textContent = data.data.disease_degree;
    document.getElementById("r-prob").textContent =
      (data.data.probability * 100).toFixed(2) + "%";

    // 低置信度告警展示
    if (lowConfAlert) {
      lowConfAlert.hidden = !data.data.low_confidence;
    }

    // Top-3 概率条渲染
    renderTopCandidates(data.data.top_candidates);

    resultCard.hidden = false;
  } catch (err) {
    showError("网络错误：" + err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "识别";
  }
});

// 解析 SSE 帧。SSE 一帧由若干 "field: value\n" 行组成，帧之间以空行分隔。
function parseSseFrame(raw) {
  let event = "message";
  const dataLines = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  const dataRaw = dataLines.join("\n");
  let data = "";
  try {
    data = dataRaw ? JSON.parse(dataRaw) : "";
  } catch {
    data = dataRaw;
  }
  return { event, data };
}

function renderAdvice(mdText) {
  const adviceTextEl = document.getElementById("advice-text");
  if (!adviceTextEl) return;
  if (window.marked && typeof window.marked.parse === "function") {
    adviceTextEl.innerHTML = window.marked.parse(mdText);
  } else {
    adviceTextEl.textContent = mdText;
  }
}

// 复制建议文本
if (copyAdviceBtn) {
  copyAdviceBtn.addEventListener("click", async () => {
    if (!rawAdviceMarkdown) return;
    try {
      await navigator.clipboard.writeText(rawAdviceMarkdown);
      const originalText = copyAdviceBtn.textContent;
      copyAdviceBtn.textContent = "✅ 已复制";
      setTimeout(() => {
        copyAdviceBtn.textContent = originalText;
      }, 2000);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = rawAdviceMarkdown;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      const originalText = copyAdviceBtn.textContent;
      copyAdviceBtn.textContent = "✅ 已复制";
      setTimeout(() => {
        copyAdviceBtn.textContent = originalText;
      }, 2000);
    }
  });
}

adviceBtn.addEventListener("click", async () => {
  if (!lastResult) return;
  clearError();
  adviceBtn.disabled = true;
  adviceBtn.textContent = "生成中…";
  rawAdviceMarkdown = "";
  renderAdvice("");
  adviceCard.hidden = false;

  try {
    const resp = await fetch("/get_treatment_advice", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify({
        plant_class: lastResult.plant_class,
        disease_name: lastResult.disease_name,
        disease_degree: lastResult.disease_degree,
        health_status: lastResult.health_status,
        provider: providerSel.value || undefined,
        api_key: apiKeyInput.value || undefined,
        model: modelInput.value || undefined,
      }),
    });

    // 上游已经在响应头里返回 4xx（缺 key / 缺字段），按 JSON 错误处理
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      adviceCard.hidden = true;
      return showError(data.message || `请求失败（HTTP ${resp.status}）`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let streamErr = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sep;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        if (!frame.trim()) continue;
        const { event, data } = parseSseFrame(frame);
        if (event === "chunk") {
          rawAdviceMarkdown += data;
          renderAdvice(rawAdviceMarkdown);
        } else if (event === "error") {
          streamErr = data;
        }
      }
    }

    if (streamErr) {
      adviceCard.hidden = true;
      showError(streamErr);
    }
  } catch (err) {
    adviceCard.hidden = true;
    showError("网络错误：" + err.message);
  } finally {
    adviceBtn.disabled = false;
    adviceBtn.textContent = "获取治理建议";
  }
});
