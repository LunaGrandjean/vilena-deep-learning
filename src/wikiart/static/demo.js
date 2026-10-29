"use strict";
const $ = (id) => document.getElementById(id);
let imageFile = null;
let previewUrl = null;
let exampleLabel = null;
let busy = false;
let selectionVersion = 0;
let models = [];
let lastComparison = null;

function modelId() { return document.querySelector('input[name="model"]:checked')?.value || "resnet18"; }
function showError(message = "") { $("error-message").textContent = message; $("error-message").hidden = !message; }
function refreshControls() {
  $("analyse").disabled = busy || !imageFile || !models.some((model) => model.id === modelId() && model.available);
  $("compare-all").disabled = busy || !imageFile || !models.some((model) => model.available);
  document.querySelectorAll('input[name="model"]').forEach((input) => {
    input.disabled = busy || !models.some((model) => model.id === input.value && model.available);
  });
  document.querySelectorAll(".example-card").forEach((button) => { button.disabled = busy; });
  $("clear-image").disabled = busy;
  $("file-input").disabled = busy;
  $("result-heading").closest("section").setAttribute("aria-busy", String(busy));
}
function clearResult() {
  lastComparison = null;
  $("comparison").hidden = true;
  $("comparison-back").hidden = true;
  $("prediction").hidden = true;
  $("loading-state").hidden = true;
  $("empty-state").hidden = false;
  $("result-tag").textContent = "FIVE POSSIBLE STYLES";
}
function selectImage(file, truth = null) {
  if (busy) return;
  showError();
  if (!file || file.size === 0) { showError("Choose a painting image first."); return; }
  if (file.size > 10 * 1024 * 1024) { showError("This image is too large. Choose a file smaller than 10 MB."); return; }
  if (file.type && !["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    showError("Choose a JPG, PNG or WebP image."); return;
  }
  selectionVersion++;
  imageFile = file;
  exampleLabel = truth;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  $("painting-preview").src = previewUrl;
  $("painting-preview").hidden = false;
  $("upload-prompt").hidden = true;
  $("change-image").hidden = false;
  $("drop-zone").classList.add("has-image");
  $("clear-image").hidden = false;
  $("image-name").textContent = truth ? `Test-set example · ${truth}` : file.name;
  clearResult();
  refreshControls();
}
function renderPrediction(result) {
  $("comparison").hidden = true;
  $("comparison-back").hidden = !lastComparison;
  $("predicted-style").textContent = result.predicted_style;
  $("prediction-source").textContent = `Seen through ${result.model} · ${(result.elapsed_ms / 1000).toFixed(2)} s analysis time`;
  $("result-tag").textContent = result.model.toUpperCase();
  $("probability-list").replaceChildren();
  result.probabilities.forEach((item) => {
    const row = document.createElement("div"); row.className = "probability-row";
    const labels = document.createElement("div"); labels.className = "probability-label";
    const label = document.createElement("span"); label.textContent = item.label;
    const value = document.createElement("strong"); value.textContent = `${(item.probability * 100).toFixed(1)}%`;
    labels.append(label, value);
    const track = document.createElement("div"); track.className = "bar-track";
    const fill = document.createElement("div"); fill.className = "bar-fill";
    fill.style.width = `${Math.max(0, Math.min(100, item.probability * 100))}%`;
    track.append(fill); row.append(labels, track); $("probability-list").append(row);
  });
  $("example-truth").hidden = !exampleLabel;
  $("example-truth").textContent = exampleLabel ? `Dataset label: ${exampleLabel} · ${result.predicted_style === exampleLabel ? "The prediction matches this example." : "The model sees a different style in this example."}` : "";
  $("prediction").hidden = false;
  $("empty-state").hidden = true;
}
function renderComparison(result) {
  lastComparison = result;
  $("comparison-list").replaceChildren();
  result.results.forEach((prediction) => {
    const row = document.createElement("button");
    row.type = "button"; row.className = "comparison-result"; row.dataset.model = prediction.model_id;
    row.setAttribute("aria-label", `View probabilities for ${prediction.model}`);
    const name = document.createElement("strong"); name.textContent = prediction.model;
    const outcome = document.createElement("span"); outcome.className = "comparison-outcome";
    const style = document.createElement("strong"); style.textContent = prediction.predicted_style;
    const probability = document.createElement("small");
    probability.textContent = `${(prediction.probabilities[0].probability * 100).toFixed(1)}% probability`;
    outcome.append(style, probability);
    if (exampleLabel) {
      const match = document.createElement("small");
      const correct = prediction.predicted_style === exampleLabel;
      match.className = correct ? "match-label" : "different-label";
      match.textContent = correct ? "Matches dataset label" : "Different from dataset label";
      outcome.append(match);
    }
    const time = document.createElement("span"); time.className = "comparison-time";
    time.textContent = `${prediction.elapsed_ms.toFixed(1)} ms`;
    row.append(name, outcome, time);
    row.addEventListener("click", () => {
      document.querySelector(`input[value="${prediction.model_id}"]`).checked = true;
      renderPrediction(prediction); refreshControls();
    });
    $("comparison-list").append(row);
  });
  $("comparison-truth").hidden = !exampleLabel;
  $("comparison-truth").textContent = exampleLabel ? `Dataset label: ${exampleLabel}` : "";
  const messages = [];
  if (result.unavailable.length) messages.push(`Not available: ${result.unavailable.join(", ")}.`);
  if (result.errors.length) messages.push(`Could not run: ${result.errors.map((item) => item.model).join(", ")}.`);
  $("comparison-errors").textContent = messages.join(" ");
  $("comparison-errors").hidden = messages.length === 0;
  $("result-tag").textContent = `${result.results.length} MODELS`;
  $("prediction").hidden = true; $("empty-state").hidden = true;
  $("comparison").hidden = false;
}
async function analyse(compareAll = false) {
  if (!imageFile || busy) return;
  if (!compareAll) lastComparison = null;
  busy = true; showError(); refreshControls();
  $("prediction").hidden = true; $("comparison").hidden = true;
  $("empty-state").hidden = true; $("loading-state").hidden = false;
  $("loading-title").textContent = compareAll ? "Comparing the models…" : "Taking a closer look…";
  try {
    const response = await fetch(`/api/predict?model=${encodeURIComponent(compareAll ? "all" : modelId())}`, {
      method: "POST", headers: { "Content-Type": imageFile.type || "application/octet-stream" }, body: imageFile
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Something went wrong. Please try again.");
    if (compareAll) renderComparison(result); else renderPrediction(result);
  } catch (error) {
    clearResult(); showError(error.message || "Can't reach the local app. Make sure it is still running.");
  } finally {
    busy = false; $("loading-state").hidden = true; refreshControls();
  }
}
$("file-input").addEventListener("change", (event) => { if (event.target.files[0]) selectImage(event.target.files[0]); event.target.value = ""; });
$("painting-preview").addEventListener("error", () => {
  if (!busy && imageFile) {
    $("clear-image").click();
    showError("We couldn't read this image. Try another JPG, PNG or WebP.");
  }
});
$("drop-zone").addEventListener("keydown", (event) => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); if (!busy) $("file-input").click(); } });
for (const type of ["dragenter", "dragover"]) $("drop-zone").addEventListener(type, (event) => { event.preventDefault(); if (!busy) $("drop-zone").classList.add("drag-over"); });
for (const type of ["dragleave", "drop"]) $("drop-zone").addEventListener(type, (event) => { event.preventDefault(); $("drop-zone").classList.remove("drag-over"); });
$("drop-zone").addEventListener("drop", (event) => { if (event.dataTransfer.files[0]) selectImage(event.dataTransfer.files[0]); });
$("clear-image").addEventListener("click", () => {
  if (busy) return;
  selectionVersion++; imageFile = null; exampleLabel = null;
  if (previewUrl) URL.revokeObjectURL(previewUrl); previewUrl = null;
  $("painting-preview").removeAttribute("src"); $("painting-preview").hidden = true;
  $("upload-prompt").hidden = false; $("change-image").hidden = true; $("clear-image").hidden = true;
  $("drop-zone").classList.remove("has-image"); $("image-name").textContent = "Your image stays on this computer.";
  clearResult(); showError(); refreshControls();
});
$("analyse").addEventListener("click", () => analyse());
$("compare-all").addEventListener("click", () => analyse(true));
$("comparison-back").addEventListener("click", () => { if (lastComparison) renderComparison(lastComparison); });
$("model-picker").addEventListener("change", () => {
  clearResult(); showError(); refreshControls();
  if (imageFile) analyse();
});

async function initialise() {
  try {
    const response = await fetch("/api/config");
    if (!response.ok) throw new Error("Can't load the local app. Please refresh the page.");
    const config = await response.json(); models = config.models;
    for (const model of models) {
      const option = document.createElement("label"); option.className = "model-option";
      const input = document.createElement("input"); input.type = "radio"; input.name = "model"; input.value = model.id;
      input.checked = model.id === "resnet18";
      const text = document.createElement("span");
      const name = document.createElement("strong"); name.textContent = model.name;
      const description = document.createElement("small");
      description.textContent = model.available ? model.description : "Not available on this machine";
      text.append(name, description);
      const mark = document.createElement("span"); mark.className = "radio-mark";
      option.append(input, text, mark); $("model-picker").append(option);
    }
    if (!models.some((model) => model.id === modelId() && model.available)) {
      const available = models.find((model) => model.available);
      if (available) document.querySelector(`input[value="${available.id}"]`).checked = true;
      else showError("No model is available yet. Add a local checkpoint to start the demo.");
    }
    for (const item of config.examples) {
      const button = document.createElement("button"); button.type = "button"; button.className = "example-card";
      button.setAttribute("aria-label", `Try the ${item.label} example`);
      const image = document.createElement("img"); image.src = item.url; image.alt = `${item.label} painting example`; image.loading = "lazy";
      const label = document.createElement("span"); label.textContent = item.label; button.append(image, label);
      button.addEventListener("click", async () => {
        if (busy) return;
        const version = ++selectionVersion;
        try {
          const response = await fetch(item.url);
          if (!response.ok) throw new Error("This example isn't available. Try uploading your own painting.");
          const blob = await response.blob();
          if (version !== selectionVersion || busy) return;
          selectImage(new File([blob], `${item.label}.jpg`, {type: blob.type}), item.label);
          await analyse();
        } catch (error) { if (version === selectionVersion) showError(error.message); }
      });
      $("examples-grid").append(button);
    }
    $("examples-section").hidden = config.examples.length === 0;
  } catch (error) { showError(error.message); }
  refreshControls();
}
initialise();
