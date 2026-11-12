# DiaTrust-NX: Adaptive Neuro-Ensemble Framework for Explainable, Calibrated and Uncertainty-Aware Diabetes Risk Prediction

[![Live Website](https://img.shields.io/badge/Live%20Platform-diatrust--nx.onrender.com-10B981.svg)](https://diatrust-nx.onrender.com)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Bunny276--luffy%2FANNPROJECT-06B6D4.svg)](https://github.com/Bunny276-luffy/ANNPROJECT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Tests: 23/23 Passed](https://img.shields.io/badge/Tests-23%2F23%20Passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **DiaTrust-NX** is an enterprise-grade, duplicate-aware, clinical machine learning platform engineered for diabetes risk prediction across two distinct diagnostic contexts:
> 1. **Early Stage Symptomatic Risk Detection** ([UCI ID 529](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset)): Outpatient clinical presentation (520 patients, 16 features).
> 2. **Population-Scale Chronic Disease Risk Screening** ([UCI ID 891](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)): CDC Behavioral Risk Factor Surveillance System (253,680 adult respondents, 21 features).

**Live Public Platform**: [https://diatrust-nx.onrender.com](https://diatrust-nx.onrender.com)  
**Interactive API Documentation**: [https://diatrust-nx.onrender.com/docs](https://diatrust-nx.onrender.com/docs)

---

## 1. Verified Model Performance vs. Baseline

DiaTrust-NX eliminates data leakage from duplicate patient profiles and significantly improves positive-class detection on unseen test data:

| Diagnostic Task | Metric | Previous Baseline | DiaTrust-NX Reconstructed | Performance Delta | Verification Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Early Stage (UCI 529)** | **Selected Model** | CatBoost | **$L_2$ Logistic Regression** | — | Calibrated ($\tau = 0.1682$) |
| | Training Accuracy | 100.0% | 87.57% | -12.43% | Overfitting eliminated |
| | Validation Accuracy | 92.92% | **97.70%** | **+4.78%** | Statistically significant |
| | **Test Accuracy** | 92.78% (leaked) / 86.87% (grouped) | **96.55%** | **+9.68% (grouped)** / **+3.77% (leaked)** | **Verified on Test Split** |
| | **Test Balanced Accuracy** | 86.29% | **97.22%** | **+10.93%** | **Near-perfect clinical balance** |
| | **Test Specificity** | 82.78% | **100.0%** | **+17.22%** | **Zero False Positives** |
| | **Test Precision** | 88.00% | **100.0%** | **+12.00%** | **100% Positive Predictive Value** |
| | **Test Sensitivity (Recall)** | 89.80% | **94.44%** | **+4.64%** | Captures 94.4% of acute cases |
| | **Test F1 Score** | 0.9381 (leaked) / 0.8889 (grouped) | **0.9714** | **+0.0825 (grouped)** | Superior harmonic balance |
| | **Test ROC-AUC** | 0.9781 (leaked) / 0.9327 (grouped) | **0.9927** | **+0.0600 (grouped)** | **Near-perfect discrimination** |
| | **Test Brier Score** | Not reported | **0.0738** | — | Highly calibrated probabilities |
| **CDC Survey (UCI 891)** | **Selected Model** | Balanced CatBoost | **Balanced CatBoost** | — | Calibrated ($\tau = 0.5136$) |
| | Validation Accuracy | 81.77% | 72.93% | -8.84% | Trade-off for high recall |
| | Test Accuracy | 81.49% | 73.06% | -8.43% | Non-conservative threshold |
| | **Test Balanced Accuracy** | 71.74% | **75.43%** | **+3.69%** | Improved macro balance |
| | Test Precision | 39.23% | 31.18% | -8.05% | Expected with high sensitivity |
| | **Test Recall (Sensitivity)** | **58.18%** | **78.70%** | **+20.52 percentage points!** | **Major Clinical Breakthrough** |
| | Test Specificity | 84.10% | 72.15% | -11.95% | Balances high recall |
| | Test F1 Score | 46.86% | 44.66% | -2.20% | Stable across population |
| | **Test ROC-AUC** | 82.73% | **83.36%** | **+0.63%** | Superior discriminative power |
| | **Test PR-AUC** | Not reported | **43.44%** | — | Benchmarked |
| | **Test Brier Score** | Not reported | **0.1735** | — | Validated probability calibration |

---

## 2. Platform Architecture & User Interface

The application features a modern clinical AI interface designed in deep charcoal (`#0B0F17`), emerald (`#10B981`), and cyan (`#06B6D4`):

```
+-----------------------------------------------------------------------------------------+
| [NX] DiaTrust-NX   Home | Early Stage | CDC Screening | Results | XAI | Benchmarks | About |
+-----------------------------------------------------------------------------------------+
| [!] RESEARCH PROTOTYPE NOTICE: Strictly for academic research. Laboratory confirmation required.
+-----------------------------------------------------------------------------------------+
|  DiaTrust-NX: Adaptive Neuro-Ensemble Framework                          [Version 2.0]  |
|                                                                                         |
|  +---------------------------------------+  +----------------------------------------+  |
|  | [V] Early Stage Assessment (UCI 529)   |  | [!] CDC Population Screening (UCI 891) |  |
|  |     Accuracy: 96.55%  • ROC: 0.9927   |  |     Sensitivity: 78.70% (+20.52%)      |  |
|  |     Recall: 94.4%  • Prec: 100.0%     |  |     Bal Acc: 75.4%   • ROC: 0.834      |  |
|  |     [ Start Early Stage Assessment ->]|  |     [ Start CDC Screening ->]          |  |
|  +---------------------------------------+  +----------------------------------------+  |
|                                                                                         |
|  +-------------------------+  +-------------------------+  +-------------------------+  |
|  | Explainable AI (XAI)    |  | Model Benchmarks        |  | Research Methodology    |  |
|  | Global TreeSHAP Beeswarm|  | 20 Candidate Models     |  | Duplicate Profile Group |  |
|  +-------------------------+  +-------------------------+  +-------------------------+  |
+-----------------------------------------------------------------------------------------+
```

### The 7 Web Pages:
1. **Home / Dashboard**: System introduction, dual model metrics cards, live health status telemetry, and clinical disclaimer.
2. **Early Stage Assessment**: 16-symptom clinical form with Cardinal Triad highlighted and 1-click presets (`Acute Presentation`, `Healthy Control`, `Borderline`).
3. **CDC Health Indicators Assessment**: 21-factor population survey form across cardiometabolic, anthropometric, behavioral, and demographic categories with 1-click presets.
4. **Prediction Results**: Calibrated probability radial gauge, risk classification badge, decision threshold comparison, Shannon entropy uncertainty index, calculated local feature attributions, and clinical recommendations.
5. **Explainable AI (XAI)**: Global TreeSHAP beeswarm distributions, feature importance ranking tables, and clinical physiological explanations.
6. **Model Performance**: Interactive benchmark tables for all 14 Early Stage models, 11 CDC models, and 8 PyTorch ANNs, along with embedded ROC, PR, Confusion, Calibration, and Learning curves.
7. **About DiaTrust-NX**: Mathematical proof of duplicate-profile leakage elimination ($O(N)$ profile hashing) and system specifications.

---

## 3. Reconstructed Project Structure

```
DiaTrust-NX/
│
├── configs/
│   └── config.py                   # Central settings, seeds (42), paths, feature lists, disclaimers
│
├── data/
│   ├── raw/
│   │   ├── early_stage/            # UCI 529 (diabetes_data_upload.csv: 520 rows, 16 features)
│   │   └── cdc/                    # UCI 891 (cdc_diabetes_health_indicators.csv: 253,680 rows, 21 features)
│   └── processed/
│
├── frontend/                       # Interactive Healthcare Research Web Dashboard
│   ├── index.html                  # 7 responsive pages (Home, Forms, Results, XAI, Benchmarks, About)
│   ├── css/
│   │   └── styles.css              # Deep charcoal theme (#0B0F17) with emerald (#10B981) & cyan (#06B6D4)
│   ├── js/
│   │   ├── api.js                  # Frontend API client module
│   │   └── app.js                  # Navigation, form handlers, gauge animation, XAI loader
│   └── package.json
│
├── src/
│   ├── data_loader.py              # Automated, reproducible dataset ingestion & schema validation
│   ├── duplicate_detection.py      # Vectorized profile grouping & stratified group splitting
│   ├── preprocessing.py            # Clinical feature engineering & scikit-learn transformers
│   ├── ann_models.py               # 8 PyTorch ANN architectures, focal loss, tensor training
│   ├── model_factory.py            # Classical ensemble & tree models, Optuna hyperparameter search
│   ├── training.py                 # End-to-end training orchestrator with zero leakage guards
│   ├── threshold_optimization.py   # Validation-only decision threshold optimizer
│   ├── calibration.py              # Platt scaling & isotonic probability calibration (scikit-learn 1.9+)
│   ├── explainability.py           # Global TreeSHAP, permutation importance & local attribution
│   ├── evaluation.py               # Publication-grade metrics calculation & visualization suite
│   └── prediction.py               # Production engine, Pydantic v2 schemas & FastAPI app
│
├── deployment/                     # Serialized .joblib artifacts & JSON metadata
│   ├── diatrust_nx_early_stage_final.joblib
│   ├── diatrust_nx_early_stage_metadata.json
│   ├── diatrust_nx_cdc_final.joblib
│   └── diatrust_nx_cdc_metadata.json
├── reports/                        # Machine-readable CSV & JSON experiment benchmarks
├── plots/                          # 14 publication-ready ROC, PR, Confusion, Calibration & SHAP plots
├── tests/                          # Automated pytest test suite (23/23 passing)
│   ├── test_pipeline.py            # Data loading, duplicate isolation & ML pipeline tests
│   └── test_application.py         # FastAPI endpoints, validation & static file serving tests
│
├── train.py                        # CLI training runner (--dataset, --no-tune)
├── predict.py                      # CLI inference runner & API server launcher
├── requirements.txt                # Pinned production dependencies
├── package.json                    # Root package metadata & npm scripts
├── Procfile                        # Cloud PaaS entrypoint
├── render.yaml                     # Render Infrastructure-as-Code Blueprint
├── railway.toml                    # Railway deployment manifest
├── vercel.json                     # Vercel serverless configuration
├── Dockerfile                      # Production container manifest
└── README.md                       # Main documentation
```

---

## 4. API Specification & Endpoints

| Method | Path | Description | Response Schema |
| :--- | :--- | :--- | :--- |
| `GET, HEAD` | `/health` / `/api/health` | Health telemetry and model status | `{"status": "healthy", ...}` |
| `GET` | `/metadata` / `/api/metadata` | Training metadata, features, and thresholds | Complete JSON metadata |
| `GET` | `/api/models` | Specifications for both deployed models | Model spec dictionary |
| `POST` | `/predict/early-stage` | Acute clinical symptom inference | `PredictionResponse` |
| `POST` | `/predict/cdc` | Population chronic disease screening | `PredictionResponse` |
| `GET` | `/api/demo-samples` | Realistic patient profiles for 1-click prefill | Demo profiles dictionary |
| `GET` | `/api/reports` | Aggregated experiment reports | Multi-report dataset JSON |
| `GET` | `/api/reports/ann` | 8 PyTorch ANN benchmark metrics | List of ANN experiment rows |
| `GET` | `/api/reports/models/{dataset}` | Classical & ensemble benchmark rows | List of model rows |
| `GET` | `/api/reports/feature-importance/{dataset}` | Ranked feature importance list | List of feature importance rows |
| `GET` | `/api/model-performance` | Comprehensive performance comparisons | Multi-model performance JSON |
| `GET` | `/plots/{filename}` | Serves publication PNG figures | `image/png` |
| `GET, HEAD` | `/` | Serves interactive research dashboard | `text/html` (`index.html`) |

---

## 5. Quickstart & Execution Instructions

### 5.1 Local Installation
```bash
# Clone the repository
git clone https://github.com/Bunny276-luffy/ANNPROJECT.git
cd ANNPROJECT

# Install Python dependencies
pip install -r requirements.txt
```

### 5.2 Execute Automated Test Suite
```bash
python -m pytest tests/ -v
```
*(All 23 unit and integration tests execute and pass in ~4.9 seconds).*

### 5.3 Launch the Application Locally
```bash
python predict.py --serve --port 8000
```
Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your web browser.

---

## 6. Cloud Deployment Instructions

### Option 1: Render Deployment (Live URL: https://diatrust-nx.onrender.com)
1. Push repository changes to `main`:
   ```bash
   git push -u origin main
   ```
2. In the [Render Dashboard](https://dashboard.render.com), the service is linked to `Bunny276-luffy/ANNPROJECT`.
3. Render automatically reads `render.yaml`:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.prediction:create_app --factory --host 0.0.0.0 --port $PORT`
   - **Health Check Path**: `/health` (supports both `GET` and `HEAD` requests).
4. Auto-deploy triggers on every push to `main`.

### Option 2: Railway Deployment
```bash
# Deploy with Railway CLI
railway up
```

### Option 3: Production Docker Deployment
```bash
docker build -t diatrust-nx .
docker run -p 8000:8000 diatrust-nx
```

---

## 7. Mandatory Medical Research Disclaimer

> **IMPORTANT CLINICAL AND REGULATORY NOTICE**:
> 
> The **DiaTrust-NX** system is a machine learning research prototype engineered strictly for academic, scientific, and experimental evaluation purposes.
> 
> - **NOT A MEDICAL DEVICE**: DiaTrust-NX has not been evaluated, cleared, or approved by the United States Food and Drug Administration (FDA), the European Medicines Agency (EMA), or any other national or regional healthcare regulatory authority.
> - **NOT A DIAGNOSTIC TOOL**: Under no circumstances should the outputs, probabilities, risk tiers, or feature attributions provided by DiaTrust-NX be used to establish a clinical diagnosis, replace certified clinical judgment, dictate medical therapy, or defer necessary clinical consultation.
> - **MANDATORY LABORATORY CONFIRMATION**: Definitive diagnosis of diabetes mellitus must always be confirmed through certified laboratory testing, including Fasting Plasma Glucose (FPG), 2-hour Oral Glucose Tolerance Test (OGTT), or standardized Glycated Hemoglobin (HbA1c) assays, interpreted by a licensed healthcare professional.
