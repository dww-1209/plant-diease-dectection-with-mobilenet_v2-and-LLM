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
const providerSel = document.getElementById("llm-provider");
const apiKeyInput = document.getElementById("llm-api-key");
const modelInput = document.getElementById("llm-model");
const modelList = document.getElementById("llm-model-list");

let lastResult = null;
let currentObjectURL = null;
let providersCatalog = {};

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

// Click on the dropzone opens the file picker. Input is OUTSIDE the dropzone
// now, so there is no double-fire from <label> default behavior.
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
    resultCard.hidden = false;
  } catch (err) {
    showError("网络错误：" + err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "识别";
  }
});

// 解析 SSE 帧。SSE 一帧由若干 "field: value\n" 行组成，帧之间以空行分隔。
// 这里只关心 event 和 data 两个字段。
function parseSseFrame(raw) {
  let event = "message";
  const dataLines = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  // 后端 data 是 JSON 字符串（_sse_pack 用 json.dumps 编码）
  const dataRaw = dataLines.join("\n");
  let data = "";
  try {
    data = dataRaw ? JSON.parse(dataRaw) : "";
  } catch {
    data = dataRaw;
  }
  return { event, data };
}

adviceBtn.addEventListener("click", async () => {
  if (!lastResult) return;
  clearError();
  adviceBtn.disabled = true;
  adviceBtn.textContent = "生成中…";
  const adviceTextEl = document.getElementById("advice-text");
  adviceTextEl.textContent = "";
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
        if (event === "chunk") adviceTextEl.textContent += data;
        else if (event === "error") streamErr = data;
        // event === "done" → 自然结束
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
