# DiaTrust-NX: Advanced Diabetes Risk Prediction & Clinical Research System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 20/20 Passed](https://img.shields.io/badge/Tests-20%2F20%20Passing-brightgreen.svg)]()

> **DiaTrust-NX** is an enterprise-grade, duplicate-aware, clinical machine learning system engineered for diabetes risk prediction across two distinct diagnostic contexts:
> 1. **Early Stage Symptomatic Risk Detection** ([UCI ID 529](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset)): Outpatient clinical presentation (520 patients, 16 features).
> 2. **Population-Scale Chronic Disease Risk Screening** ([UCI ID 891](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)): CDC Behavioral Risk Factor Surveillance System (253,680 adult respondents, 21 features).

---

## 1. Key Performance Highlights vs. Baseline

DiaTrust-NX eliminates data leakage from duplicate patient profiles and significantly improves positive-class detection:

| Dataset | Metric | Previous Baseline | DiaTrust-NX Reconstructed | Improvement |
| :--- | :--- | :---: | :---: | :---: |
| **UCI 529 (Early Stage)** | **Test Accuracy** | 92.78% (leaked) / 86.87% (grouped) | **96.55%** | **+9.68% (grouped)** |
| | **Test Balanced Accuracy** | 86.29% | **97.22%** | **+10.93%** |
| | **Test Specificity** | 82.78% | **100.0%** | **+17.22% (Zero False Positives)** |
| | **Test Precision** | 88.00% | **100.0%** | **+12.00%** |
| | **Test Sensitivity (Recall)** | 89.80% | **94.44%** | **+4.64%** |
| | **Test F1 Score** | 0.9381 (leaked) / 0.8889 (grouped) | **0.9714** | **+0.0825 (grouped)** |
| | **Test ROC-AUC** | 0.9781 (leaked) / 0.9327 (grouped) | **0.9927** | **+0.0600 (grouped)** |
| **UCI 891 (CDC Survey)** | **Test Sensitivity (Recall)** | **58.18%** | **78.70%** | **+20.52 percentage points!** |
| | **Test Balanced Accuracy** | 71.74% | **75.43%** | **+3.69%** |
| | **Test ROC-AUC** | 82.73% | **83.36%** | **+0.63%** |
| | **Test Brier Score** | Not reported | **0.1735** | Well-calibrated probabilities |

---

## 2. Project Architecture

```
DiaTrust-NX/
│
├── configs/
│   └── config.py                   # Central configuration, random seeds, paths, feature lists
│
├── data/
│   ├── raw/
│   │   ├── early_stage/            # UCI 529 raw CSV (diabetes_data_upload.csv)
│   │   └── cdc/                    # UCI 891 raw CSV (cdc_diabetes_health_indicators.csv)
│   └── processed/
│
├── frontend/                       # Interactive Healthcare Research Web Dashboard
│   ├── index.html                  # 7 responsive dashboard views (Home, Forms, Results, XAI, Benchmarks, About)
│   ├── css/
│   │   └── styles.css              # Deep charcoal theme with emerald & cyan accents
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
├── reports/                        # Machine-readable CSV & JSON experiment benchmarks
├── plots/                          # Publication-ready ROC, PR, Confusion, Calibration & SHAP plots
├── tests/                          # Automated pytest test suite (20/20 passing)
│   ├── test_pipeline.py            # Data loading, duplicate isolation & ML pipeline tests
│   └── test_application.py         # FastAPI endpoints, validation & static file serving tests
├── docs/                           # Technical implementation, experiment benchmarks & model cards
│   ├── IMPLEMENTATION_REPORT.md    # In-depth architectural & mathematical report
│   ├── EXPERIMENT_RESULTS.md       # Comprehensive tabular benchmarks & overfitting audit
│   └── MODEL_CARD.md               # Standardized production model cards
│
├── train.py                        # CLI training runner (--dataset, --no-tune)
├── predict.py                      # CLI inference runner & API server launcher
├── requirements.txt                # Pinned production dependencies
├── package.json                    # Root package metadata & npm scripts
└── README.md                       # Main documentation
```

---

## 3. Quickstart & Application Execution

### 3.1 Environment Setup
```bash
# Clone the repository
git clone https://github.com/organization/DiaTrust-NX.git
cd DiaTrust-NX

# Install python dependencies
pip install -r requirements.txt
```

### 3.2 Automated Test Execution
Run the complete automated test suite (20 tests covering data loading, duplicate profile isolation, preprocessing, model inference, probability calibration, input validation schemas, error handling, reports, and frontend file delivery):
```bash
python -m pytest tests/ -v
```
*(All 20 tests pass with 100% compliance in ~3.8 seconds).*

### 3.3 Starting the Complete Application
Launch the unified FastAPI server (which serves both the prediction engine and the healthcare research dashboard):
```bash
python predict.py --serve --port 8000
```

Once running:
- **Local Application Dashboard**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive Swagger API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 4. Frontend Research Dashboard Features

The dashboard includes 7 integrated pages designed with a medical AI aesthetic (deep charcoal `#0B0F17`, emerald `#10B981`, and cyan `#06B6D4`):

1. **Home Dashboard**:
   - System overview with high-level performance cards for both Early Stage and CDC models.
   - Live backend health indicator (`/health`).
   - Quick navigation to assessment forms, XAI, and benchmark proofs.
   - Prominent clinical research disclaimer.
2. **Early Stage Assessment Form**:
   - 16 clinical symptom toggles with cardinal symptom triad highlighted.
   - 1-click clinical presets (`Acute Presentation`, `Healthy Control`, `Borderline Presentation`).
3. **CDC Population Screening Form**:
   - 21 survey indicators grouped into Cardiometabolic, Anthropometric, Behavioral, and Demographic sections.
   - 1-click survey presets (`Metabolic Syndrome`, `Young Active Adult`, `Pre-diabetic Alert`).
4. **Prediction Results View**:
   - Animated SVG probability gauge displaying calibrated risk percentage.
   - Risk classification badge (`High Risk / Positive` or `Low Risk / Negative`).
   - Decision threshold and Shannon entropy uncertainty index.
   - Breakdown of top contributing local factors with clinical risk/protective tags.
   - Actionable clinical guidance and mandatory disclaimer.
5. **Explainable AI (XAI) Dashboard**:
   - Live dataset switcher between Early Stage (UCI 529) and CDC (UCI 891).
   - Embedded high-resolution global SHAP beeswarm summary plots.
   - Dynamic feature ranking tables loaded directly from `/api/reports/feature-importance/{dataset}`.
6. **Model Performance Benchmarks**:
   - Interactive tables comparing all 14 Early Stage models, 11 CDC models, and 8 PyTorch ANNs.
   - Embedded Confusion Matrices, ROC curves, PR curves, and Calibration curves.
   - Baseline comparison delta breakdown.
7. **About DiaTrust-NX**:
   - In-depth documentation on the duplicate profile leakage discovery and $O(N)$ profile grouping mathematical proof.
   - Dataset metadata and system architecture.

---

## 5. CLI Inference Examples

You can also run predictions directly from the command line:
```bash
# Early Stage clinical case prediction
python predict.py --example early_stage

# CDC population screening case prediction
python predict.py --example cdc
```

---

## 6. Duplicate Profile Leakage Remediation

In survey-based datasets, identical predictor profiles are frequent:
- **UCI 529**: 520 rows, 251 unique profiles, 376 duplicate rows (72.3%).
- **UCI 891**: 253,680 rows, 227,908 unique profiles, 38,000 duplicate rows (15.0%).

When standard random splitting is applied, identical profiles leak across splits, producing misleadingly high test scores. DiaTrust-NX resolves this by:
1. Mapping every unique feature profile to a deterministic group hash via `.ngroup()`.
2. Grouping profiles so that identical feature vectors strictly reside in either Train, Validation, or Test.
3. Guaranteeing:
$$\forall x_i \in \text{Test}, \quad \nexists x_j \in \text{Train} \cup \text{Val} \quad \text{such that} \quad x_i = x_j$$

---

## 7. Medical Research Disclaimer

> **IMPORTANT CLINICAL AND REGULATORY NOTICE**:
> 
> The **DiaTrust-NX** system is a research prototype engineered strictly for academic, scientific, and experimental evaluation purposes.
> 
> - **NOT A MEDICAL DEVICE**: DiaTrust-NX has not been evaluated, cleared, or approved by the United States Food and Drug Administration (FDA), the European Medicines Agency (EMA), or any other national or regional medical regulatory agency.
> - **NOT A DIAGNOSTIC TOOL**: Under no circumstances should the outputs, probabilities, risk tiers, or feature attributions provided by DiaTrust-NX be used to establish a clinical diagnosis, replace certified clinical judgment, dictate medical therapy, or defer necessary medical consultation.
> - **MANDATORY LABORATORY CONFIRMATION**: Definitive diagnosis of diabetes mellitus must always be confirmed through certified laboratory testing, including Fasting Plasma Glucose (FPG), 2-hour Oral Glucose Tolerance Test (OGTT), or standardized Glycated Hemoglobin (HbA1c) assays, interpreted by a licensed healthcare professional.
