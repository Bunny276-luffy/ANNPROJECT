/**
 * DiaTrust-NX Healthcare Research Dashboard Application Logic
 * Implements navigation, interactive forms, live API predictions,
 * gauge rendering, explainability inspection, and benchmark comparisons.
 */

import { DiaTrustAPI } from "./api.js";

// Application State
const state = {
  activeView: "home",
  metadata: null,
  demoSamples: null,
  lastPrediction: null,
  currentXaiDataset: "early_stage",
  currentBenchTab: "early_stage"
};

// DOM Ready initialization
document.addEventListener("DOMContentLoaded", async () => {
  setupNavigation();
  setupToggleSwitches();
  setupFormHandlers();
  setupPresetHandlers();
  setupXaiControls();
  setupBenchmarkControls();

  await checkSystemStatus();
  await loadMetadataAndDemos();
});

// -------------------------------------------------------------
// Navigation Controller
// -------------------------------------------------------------
function setupNavigation() {
  const navItems = document.querySelectorAll(".nav-item");
  navItems.forEach(item => {
    item.addEventListener("click", () => {
      const targetView = item.dataset.view;
      if (targetView) navigateTo(targetView);
    });
  });

  // Delegated buttons for page jumps
  document.querySelectorAll("[data-navigate]").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      const target = btn.getAttribute("data-navigate");
      navigateTo(target);
    });
  });
}

export function navigateTo(viewId) {
  state.activeView = viewId;

  // Update navigation items active state
  document.querySelectorAll(".nav-item").forEach(item => {
    item.classList.toggle("active", item.dataset.view === viewId);
  });

  // Switch view containers
  document.querySelectorAll(".page-view").forEach(view => {
    view.classList.toggle("active", view.id === `view-${viewId}`);
  });

  window.scrollTo({ top: 0, behavior: "smooth" });

  // View specific on-load triggers
  if (viewId === "xai") {
    loadXaiView(state.currentXaiDataset);
  } else if (viewId === "benchmarks") {
    loadBenchmarkView(state.currentBenchTab);
  }
}

// -------------------------------------------------------------
// System Status & Initial Data
// -------------------------------------------------------------
async function checkSystemStatus() {
  const statusBadge = document.getElementById("system-status-indicator");
  const statusText = document.getElementById("system-status-text");

  const health = await DiaTrustAPI.checkHealth();
  if (health.status === "healthy") {
    if (statusBadge) statusBadge.style.background = "#10b981";
    if (statusText) statusText.innerText = "System Operational • 2 Models Online";
  } else {
    if (statusBadge) statusBadge.style.background = "#ef4444";
    if (statusText) statusText.innerText = "Backend Disconnected";
  }
}

async function loadMetadataAndDemos() {
  try {
    state.metadata = await DiaTrustAPI.fetchMetadata();
    state.demoSamples = await DiaTrustAPI.fetchDemoSamples();
  } catch (err) {
    console.error("Initialization warning:", err);
  }
}

// -------------------------------------------------------------
// Toggle Switch Controls (Yes/No, Gender)
// -------------------------------------------------------------
function setupToggleSwitches() {
  document.querySelectorAll(".toggle-group").forEach(group => {
    const input = group.querySelector("input[type='hidden']");
    const options = group.querySelectorAll(".toggle-option");

    options.forEach(opt => {
      opt.addEventListener("click", () => {
        options.forEach(o => {
          o.classList.remove("active");
          o.classList.remove("active-no");
        });

        const val = opt.dataset.val;
        input.value = val;

        if (val === "Yes" || val === "Male" || val === "1") {
          opt.classList.add("active");
        } else {
          opt.classList.add("active-no");
        }
      });
    });
  });
}

// -------------------------------------------------------------
// Form Prefill Demo Presets
// -------------------------------------------------------------
function setupPresetHandlers() {
  // Early stage presets
  document.querySelectorAll("[data-preset-es]").forEach(btn => {
    btn.addEventListener("click", () => {
      const type = btn.dataset.presetEs;
      if (!state.demoSamples || !state.demoSamples.early_stage) return;
      const data = state.demoSamples.early_stage[type];
      if (data) prefillEarlyStageForm(data);
    });
  });

  // CDC presets
  document.querySelectorAll("[data-preset-cdc]").forEach(btn => {
    btn.addEventListener("click", () => {
      const type = btn.dataset.presetCdc;
      if (!state.demoSamples || !state.demoSamples.cdc) return;
      const data = state.demoSamples.cdc[type];
      if (data) prefillCDCForm(data);
    });
  });
}

function prefillEarlyStageForm(data) {
  const form = document.getElementById("form-early-stage");
  if (!form) return;

  form.querySelector("#es-age").value = data.age;

  for (const [key, val] of Object.entries(data)) {
    if (key === "age") continue;
    const input = form.querySelector(`input[name='${key}']`);
    if (input) {
      input.value = val;
      const group = input.closest(".toggle-group");
      if (group) {
        group.querySelectorAll(".toggle-option").forEach(opt => {
          opt.classList.remove("active", "active-no");
          if (opt.dataset.val === val) {
            opt.classList.add(val === "Yes" || val === "Male" ? "active" : "active-no");
          }
        });
      }
    }
  }
}

function prefillCDCForm(data) {
  const form = document.getElementById("form-cdc");
  if (!form) return;

  for (const [key, val] of Object.entries(data)) {
    const input = form.querySelector(`[name='${key}']`);
    if (input) {
      input.value = val;
      const group = input.closest(".toggle-group");
      if (group) {
        group.querySelectorAll(".toggle-option").forEach(opt => {
          opt.classList.remove("active", "active-no");
          if (opt.dataset.val == val) {
            opt.classList.add(val == 1 ? "active" : "active-no");
          }
        });
      }
    }
  }
}

// -------------------------------------------------------------
// Form Submission & Live Prediction Execution
// -------------------------------------------------------------
function setupFormHandlers() {
  // 1. Early stage form
  const esForm = document.getElementById("form-early-stage");
  if (esForm) {
    esForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const submitBtn = esForm.querySelector("button[type='submit']");
      submitBtn.disabled = true;
      submitBtn.innerText = "Evaluating Symptoms...";

      try {
        const formData = new FormData(esForm);
        const payload = {
          age: parseInt(formData.get("age"), 10),
          gender: formData.get("gender"),
          polyuria: formData.get("polyuria"),
          polydipsia: formData.get("polydipsia"),
          sudden_weight_loss: formData.get("sudden_weight_loss"),
          weakness: formData.get("weakness"),
          polyphagia: formData.get("polyphagia"),
          genital_thrush: formData.get("genital_thrush"),
          visual_blurring: formData.get("visual_blurring"),
          itching: formData.get("itching"),
          irritability: formData.get("irritability"),
          delayed_healing: formData.get("delayed_healing"),
          partial_paresis: formData.get("partial_paresis"),
          muscle_stiffness: formData.get("muscle_stiffness"),
          alopecia: formData.get("alopecia"),
          obesity: formData.get("obesity")
        };

        const result = await DiaTrustAPI.predictEarlyStage(payload);
        renderPredictionResult(result);
        navigateTo("results");
      } catch (err) {
        alert("Prediction Error: " + err.message);
      } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `<span>Assess Early-Stage Risk</span> <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>`;
      }
    });
  }

  // 2. CDC form
  const cdcForm = document.getElementById("form-cdc");
  if (cdcForm) {
    cdcForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const submitBtn = cdcForm.querySelector("button[type='submit']");
      submitBtn.disabled = true;
      submitBtn.innerText = "Evaluating Lifestyle Indicators...";

      try {
        const formData = new FormData(cdcForm);
        const payload = {
          HighBP: parseInt(formData.get("HighBP"), 10),
          HighChol: parseInt(formData.get("HighChol"), 10),
          CholCheck: parseInt(formData.get("CholCheck"), 10),
          BMI: parseFloat(formData.get("BMI")),
          Smoker: parseInt(formData.get("Smoker"), 10),
          Stroke: parseInt(formData.get("Stroke"), 10),
          HeartDiseaseorAttack: parseInt(formData.get("HeartDiseaseorAttack"), 10),
          PhysActivity: parseInt(formData.get("PhysActivity"), 10),
          Fruits: parseInt(formData.get("Fruits"), 10),
          Veggies: parseInt(formData.get("Veggies"), 10),
          HvyAlcoholConsump: parseInt(formData.get("HvyAlcoholConsump"), 10),
          AnyHealthcare: parseInt(formData.get("AnyHealthcare"), 10),
          NoDocbcCost: parseInt(formData.get("NoDocbcCost"), 10),
          GenHlth: parseInt(formData.get("GenHlth"), 10),
          MentHlth: parseInt(formData.get("MentHlth"), 10),
          PhysHlth: parseInt(formData.get("PhysHlth"), 10),
          DiffWalk: parseInt(formData.get("DiffWalk"), 10),
          Sex: parseInt(formData.get("Sex"), 10),
          Age: parseInt(formData.get("Age"), 10),
          Education: parseInt(formData.get("Education"), 10),
          Income: parseInt(formData.get("Income"), 10)
        };

        const result = await DiaTrustAPI.predictCDC(payload);
        renderPredictionResult(result);
        navigateTo("results");
      } catch (err) {
        alert("Prediction Error: " + err.message);
      } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `<span>Assess CDC Population Risk</span> <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>`;
      }
    });
  }
}

// -------------------------------------------------------------
// Render Prediction Result
// -------------------------------------------------------------
function renderPredictionResult(result) {
  state.lastPrediction = result;

  const pct = Math.round(result.risk_probability * 100);
  const isHigh = result.prediction === 1;

  // 1. Gauge rendering
  const gaugeProgress = document.getElementById("res-gauge-progress");
  const gaugePercent = document.getElementById("res-gauge-percent");

  if (gaugeProgress && gaugePercent) {
    const circumference = 2 * Math.PI * 80; // r=80 in svg
    const offset = circumference - (pct / 100) * circumference;
    gaugeProgress.style.strokeDasharray = `${circumference}`;
    gaugeProgress.style.strokeDashoffset = `${offset}`;
    gaugeProgress.style.stroke = isHigh ? "var(--color-danger)" : "var(--accent-emerald)";
    gaugePercent.innerText = `${pct}%`;
    gaugePercent.style.color = isHigh ? "#f87171" : "#34d399";
  }

  // 2. Risk Badge & Labels
  const badge = document.getElementById("res-risk-badge");
  if (badge) {
    badge.className = `risk-badge ${isHigh ? 'risk-high' : 'risk-low'}`;
    badge.innerText = result.risk_label;
  }

  // 3. Metadata fields
  setText("res-dataset-name", result.dataset);
  setText("res-model-name", result.model_name);
  setText("res-threshold", `${(result.decision_threshold * 100).toFixed(1)}%`);
  setText("res-uncertainty", `${result.uncertainty_score} (${result.confidence_category})`);

  // 4. Clinical interpretation
  const adviceEl = document.getElementById("res-clinical-advice");
  if (adviceEl) {
    if (isHigh) {
      adviceEl.innerHTML = `
        <strong style="color: #f87171;">Clinical Action Advised:</strong> The patient's estimated risk (${pct}%) exceeds the validated decision threshold (${(result.decision_threshold * 100).toFixed(1)}%).
        It is strongly advised to schedule confirmatory laboratory testing, including a <strong>Fasting Plasma Glucose (FPG)</strong> or <strong>HbA1c assay</strong>, to confirm glycemic status.
      `;
    } else {
      adviceEl.innerHTML = `
        <strong style="color: #34d399;">Screening Interpretation:</strong> The patient's risk profile (${pct}%) falls below the decision threshold.
        Current symptoms or indicators do not suggest immediate acute or high chronic diabetes risk. Continue routine health maintenance.
      `;
    }
  }

  // 5. Contributing factors
  const factorList = document.getElementById("res-factor-list");
  if (factorList) {
    factorList.innerHTML = "";
    result.top_contributing_factors.forEach(f => {
      const li = document.createElement("li");
      li.className = "factor-item";
      const isPositive = f.polarity === "positive" || f.impact.includes("Risk") || f.impact.includes("Elevates");
      li.innerHTML = `
        <div class="factor-name">${f.factor}</div>
        <div class="factor-tag ${isPositive ? 'tag-risk' : 'tag-protective'}">${f.impact}</div>
      `;
      factorList.appendChild(li);
    });
  }
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.innerText = val;
}

// -------------------------------------------------------------
// Explainable AI Dashboard
// -------------------------------------------------------------
function setupXaiControls() {
  document.querySelectorAll("[data-xai-tab]").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-xai-tab]").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.currentXaiDataset = btn.dataset.xaiTab;
      loadXaiView(state.currentXaiDataset);
    });
  });
}

async function loadXaiView(dataset) {
  const isES = dataset === "early_stage";
  setText("xai-dataset-heading", isES ? "UCI Early Stage (529) — Feature Attributions" : "UCI CDC Health Indicators (891) — Feature Attributions");
  setText("xai-model-tag", isES ? "Model: Regularized Logistic Regression" : "Model: Balanced CatBoost");

  // Update SHAP image
  const shapImg = document.getElementById("xai-shap-img");
  const featImg = document.getElementById("xai-feat-img");
  if (shapImg) shapImg.src = `/plots/${isES ? 'early_stage_shap_summary.png' : 'cdc_shap_summary.png'}`;
  if (featImg) featImg.src = `/plots/${isES ? 'early_stage_feature_importance.png' : 'cdc_feature_importance.png'}`;

  // Fetch ranking table
  try {
    const list = await DiaTrustAPI.fetchFeatureImportance(dataset);
    const tbody = document.getElementById("xai-table-body");
    if (tbody) {
      tbody.innerHTML = "";
      list.slice(0, 10).forEach((row, idx) => {
        const tr = document.createElement("tr");
        const importanceVal = row.importance !== undefined ? row.importance.toFixed(4) : (row.coefficient !== undefined ? row.coefficient.toFixed(4) : "—");
        tr.innerHTML = `
          <td><strong>#${idx + 1}</strong></td>
          <td><strong>${row.feature}</strong></td>
          <td>${importanceVal}</td>
          <td>${row.clinical_relevance || "Identified as primary risk differentiator"}</td>
        `;
        tbody.appendChild(tr);
      });
    }
  } catch (err) {
    console.warn("Could not load feature table:", err);
  }
}

// -------------------------------------------------------------
// Model Performance Benchmarks
// -------------------------------------------------------------
function setupBenchmarkControls() {
  document.querySelectorAll("[data-bench-tab]").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-bench-tab]").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.currentBenchTab = btn.dataset.benchTab;
      loadBenchmarkView(state.currentBenchTab);
    });
  });
}

async function loadBenchmarkView(tab) {
  const container = document.getElementById("bench-content-container");
  if (!container) return;

  if (tab === "early_stage" || tab === "cdc") {
    try {
      const models = await DiaTrustAPI.fetchModelReports(tab);
      const isES = tab === "early_stage";
      
      let html = `
        <div class="card" style="margin-bottom: 2rem;">
          <h3 class="card-title" style="margin-bottom: 1rem;">
            ${isES ? "UCI Early Stage (529)" : "UCI CDC Health Indicators (891)"} — All Model Experiments
          </h3>
          <p class="card-desc" style="margin-bottom: 1rem; color: var(--text-secondary); font-size: 0.88rem;">
            Evaluated with zero duplicate profile leakage. Decision thresholds tuned strictly on the validation split.
          </p>
          <div class="table-responsive">
            <table class="benchmark-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Train Acc</th>
                  <th>Train ROC</th>
                  <th>Val Thresh</th>
                  <th>Val Acc</th>
                  <th>Val Bal Acc</th>
                  <th>Val Recall</th>
                  <th>Val F1</th>
                  <th>Val ROC</th>
                  <th>Val PR-AUC</th>
                </tr>
              </thead>
              <tbody>
      `;

      models.forEach(m => {
        const isSelected = (isES && m.Model.includes("Logistic")) || (!isES && m.Model.includes("CatBoost") && !m.Model.includes("Tuned"));
        html += `
          <tr class="${isSelected ? 'highlight-row' : ''}">
            <td><strong>${m.Model}</strong> ${isSelected ? '★ (Selected)' : ''}</td>
            <td>${(m['Training Accuracy'] * 100).toFixed(1)}%</td>
            <td>${m['Training ROC-AUC'].toFixed(3)}</td>
            <td>${m['Validation Threshold'].toFixed(3)}</td>
            <td>${(m['Validation Accuracy'] * 100).toFixed(1)}%</td>
            <td>${(m['Validation Balanced Accuracy'] * 100).toFixed(1)}%</td>
            <td>${(m['Validation Recall'] * 100).toFixed(1)}%</td>
            <td>${m['Validation F1'].toFixed(3)}</td>
            <td>${m['Validation ROC-AUC'].toFixed(3)}</td>
            <td>${m['Validation PR-AUC'].toFixed(3)}</td>
          </tr>
        `;
      });

      html += `
              </tbody>
            </table>
          </div>
        </div>

        <div class="plots-grid">
          <div class="plot-card">
            <div class="plot-card-header">Confusion Matrix (Untouched Test Split)</div>
            <img class="plot-img" src="/plots/${isES ? 'early_stage_confusion_matrix.png' : 'cdc_confusion_matrix.png'}" alt="Confusion Matrix">
          </div>
          <div class="plot-card">
            <div class="plot-card-header">Receiver Operating Characteristic (ROC)</div>
            <img class="plot-img" src="/plots/${isES ? 'early_stage_roc.png' : 'cdc_roc.png'}" alt="ROC Curve">
          </div>
          <div class="plot-card">
            <div class="plot-card-header">Precision-Recall Curve (PR-AUC)</div>
            <img class="plot-img" src="/plots/${isES ? 'early_stage_pr.png' : 'cdc_pr.png'}" alt="PR Curve">
          </div>
          <div class="plot-card">
            <div class="plot-card-header">Probability Calibration & Reliability Curve</div>
            <img class="plot-img" src="/plots/${isES ? 'early_stage_calibration.png' : 'cdc_calibration.png'}" alt="Calibration Plot">
          </div>
        </div>
      `;

      container.innerHTML = html;
    } catch (err) {
      container.innerHTML = `<div class="card"><p style="color: #ef4444;">Error loading benchmark data: ${err.message}</p></div>`;
    }
  } else if (tab === "ann") {
    try {
      const anns = await DiaTrustAPI.fetchANNReport();
      let html = `
        <div class="card" style="margin-bottom: 2rem;">
          <h3 class="card-title" style="margin-bottom: 1rem;">
            8 Deep Neural Network Architectures (PyTorch) Benchmark
          </h3>
          <p class="card-desc" style="margin-bottom: 1rem; color: var(--text-secondary); font-size: 0.88rem;">
            All 8 neural networks were trained with PyTorch using Focal Loss / Weighted BCE, AdamW, Early Stopping, and Learning-Rate Scheduling.
          </p>
          <div class="table-responsive">
            <table class="benchmark-table">
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>Architecture</th>
                  <th>Train Acc</th>
                  <th>Train Loss</th>
                  <th>Val Acc</th>
                  <th>Val Bal Acc</th>
                  <th>Val Recall</th>
                  <th>Val F1</th>
                  <th>Val ROC-AUC</th>
                  <th>Best Epoch</th>
                </tr>
              </thead>
              <tbody>
      `;

      anns.forEach(a => {
        const isBest = (a['Dataset'] === 'Early Stage' && a['ANN Model'].includes('Dropout')) || (a['Dataset'] === 'CDC' && a['ANN Model'].includes('Deep'));
        html += `
          <tr class="${isBest ? 'highlight-row' : ''}">
            <td>${a['Dataset']}</td>
            <td><strong>${a['ANN Model']}</strong> ${isBest ? '★ (Top ANN)' : ''}</td>
            <td>${(a['Training Accuracy'] * 100).toFixed(1)}%</td>
            <td>${a['Training Loss'].toFixed(4)}</td>
            <td>${(a['Validation Accuracy'] * 100).toFixed(1)}%</td>
            <td>${(a['Validation Balanced Accuracy'] * 100).toFixed(1)}%</td>
            <td>${(a['Validation Recall'] * 100).toFixed(1)}%</td>
            <td>${a['Validation F1'].toFixed(3)}</td>
            <td>${a['Validation ROC-AUC'].toFixed(3)}</td>
            <td>Epoch ${a['Best Epoch']}</td>
          </tr>
        `;
      });

      html += `
              </tbody>
            </table>
          </div>
        </div>

        <div class="plots-grid">
          <div class="plot-card">
            <div class="plot-card-header">Early Stage: Training vs Validation Learning Curve</div>
            <img class="plot-img" src="/plots/early_stage_ann_learning_curve.png" alt="Early Stage ANN Training Curve">
          </div>
          <div class="plot-card">
            <div class="plot-card-header">CDC: Training vs Validation Learning Curve</div>
            <img class="plot-img" src="/plots/cdc_ann_learning_curve.png" alt="CDC ANN Training Curve">
          </div>
        </div>
      `;

      container.innerHTML = html;
    } catch (err) {
      container.innerHTML = `<div class="card"><p style="color: #ef4444;">Error loading ANN benchmarks: ${err.message}</p></div>`;
    }
  }
}

