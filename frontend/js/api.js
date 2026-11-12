/**
 * DiaTrust-NX API Client Module
 * Communicates with the FastAPI backend prediction engine and report endpoints.
 */

const API_BASE = window.location.origin;

export const DiaTrustAPI = {
  /**
   * Health and status check
   */
  async checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
      return await res.json();
    } catch (err) {
      console.warn("Backend connection error:", err);
      return { status: "offline", early_stage_loaded: false, cdc_loaded: false };
    }
  },

  /**
   * Fetch model metadata
   */
  async fetchMetadata() {
    const res = await fetch(`${API_BASE}/metadata`);
    if (!res.ok) throw new Error("Failed to load model metadata");
    return await res.json();
  },

  /**
   * Fetch models info
   */
  async fetchModels() {
    const res = await fetch(`${API_BASE}/api/models`);
    if (!res.ok) throw new Error("Failed to load models list");
    return await res.json();
  },

  /**
   * Fetch all aggregated experiment reports
   */
  async fetchAllReports() {
    const res = await fetch(`${API_BASE}/api/reports`);
    if (!res.ok) throw new Error("Failed to load reports");
    return await res.json();
  },

  /**
   * Fetch comprehensive model performance comparison
   */
  async fetchModelPerformance() {
    const res = await fetch(`${API_BASE}/api/model-performance`);
    if (!res.ok) throw new Error("Failed to load model performance");
    return await res.json();
  },

  /**
   * Fetch demo patient samples for 1-click form prefill
   */
  async fetchDemoSamples() {
    const res = await fetch(`${API_BASE}/api/demo-samples`);
    if (!res.ok) throw new Error("Failed to load demo samples");
    return await res.json();
  },

  /**
   * Execute Early Stage Diabetes Prediction
   */
  async predictEarlyStage(payload) {
    const res = await fetch(`${API_BASE}/predict/early-stage`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.detail || `Prediction error: ${res.statusText}`);
    }
    return await res.json();
  },

  /**
   * Execute CDC Population Screening Prediction
   */
  async predictCDC(payload) {
    const res = await fetch(`${API_BASE}/predict/cdc`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.detail || `Prediction error: ${res.statusText}`);
    }
    return await res.json();
  },

  /**
   * Fetch ANN Architecture Comparison Report
   */
  async fetchANNReport() {
    const res = await fetch(`${API_BASE}/api/reports/ann`);
    if (!res.ok) throw new Error("Failed to load ANN report");
    return await res.json();
  },

  /**
   * Fetch Model Benchmark Experiments
   */
  async fetchModelReports(dataset) {
    const res = await fetch(`${API_BASE}/api/reports/models/${dataset}`);
    if (!res.ok) throw new Error(`Failed to load ${dataset} models report`);
    return await res.json();
  },

  /**
   * Fetch Final Dataset Comparison Summary
   */
  async fetchDatasetComparison() {
    const res = await fetch(`${API_BASE}/api/reports/dataset-comparison`);
    if (!res.ok) throw new Error("Failed to load dataset comparison");
    return await res.json();
  },

  /**
   * Fetch Feature Importance Rankings
   */
  async fetchFeatureImportance(dataset) {
    const res = await fetch(`${API_BASE}/api/reports/feature-importance/${dataset}`);
    if (!res.ok) throw new Error(`Failed to load feature importance for ${dataset}`);
    return await res.json();
  },

  /**
   * Fetch List of Generated Visualizations
   */
  async fetchPlotsList() {
    const res = await fetch(`${API_BASE}/api/plots/list`);
    if (!res.ok) throw new Error("Failed to load plots list");
    return await res.json();
  }
};

