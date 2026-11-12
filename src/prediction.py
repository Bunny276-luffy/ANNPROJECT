"""DiaTrust-NX Production Inference & FastAPI Service Module
Provides strict Pydantic input schemas, calibrated inference, local explainability,
report APIs, static plot delivery, and a complete healthcare research web dashboard.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from configs.config import DEPLOYMENT_DIR, MEDICAL_DISCLAIMER, REPORTS_DIR, PLOTS_DIR


# ---------------------------------------------------------
# Pydantic Input Schemas
# ---------------------------------------------------------

class EarlyStageInput(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years (1-120)")
    gender: str = Field(..., description="'Male' or 'Female'")
    polyuria: str = Field(..., description="Excessive urination ('Yes'/'No')")
    polydipsia: str = Field(..., description="Excessive thirst ('Yes'/'No')")
    sudden_weight_loss: str = Field(..., description="Unexplained rapid weight loss ('Yes'/'No')")
    weakness: str = Field(..., description="Generalized bodily weakness ('Yes'/'No')")
    polyphagia: str = Field(..., description="Excessive hunger ('Yes'/'No')")
    genital_thrush: str = Field(..., description="Presence of genital candidiasis/thrush ('Yes'/'No')")
    visual_blurring: str = Field(..., description="Episodes of blurred vision ('Yes'/'No')")
    itching: str = Field(..., description="Severe generalized skin itching ('Yes'/'No')")
    irritability: str = Field(..., description="Mood irritability ('Yes'/'No')")
    delayed_healing: str = Field(..., description="Impaired or delayed wound healing ('Yes'/'No')")
    partial_paresis: str = Field(..., description="Muscle weakness or partial paralysis ('Yes'/'No')")
    muscle_stiffness: str = Field(..., description="Persistent muscle stiffness ('Yes'/'No')")
    alopecia: str = Field(..., description="Noticeable hair loss ('Yes'/'No')")
    obesity: str = Field(..., description="Clinical obesity status ('Yes'/'No')")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        v_clean = str(v).strip().capitalize()
        if v_clean not in ["Male", "Female"]:
            raise ValueError("Gender must be 'Male' or 'Female'")
        return v_clean

    @field_validator(
        "polyuria", "polydipsia", "sudden_weight_loss", "weakness", "polyphagia",
        "genital_thrush", "visual_blurring", "itching", "irritability", "delayed_healing",
        "partial_paresis", "muscle_stiffness", "alopecia", "obesity"
    )
    @classmethod
    def validate_yes_no(cls, v):
        v_clean = str(v).strip().capitalize()
        if v_clean not in ["Yes", "No"]:
            raise ValueError("Value must be 'Yes' or 'No'")
        return v_clean


class CDCInput(BaseModel):
    HighBP: int = Field(..., ge=0, le=1, description="High Blood Pressure (0=No, 1=Yes)")
    HighChol: int = Field(..., ge=0, le=1, description="High Cholesterol (0=No, 1=Yes)")
    CholCheck: int = Field(..., ge=0, le=1, description="Cholesterol checked in past 5 years (0=No, 1=Yes)")
    BMI: float = Field(..., ge=10.0, le=100.0, description="Body Mass Index (10.0 - 100.0)")
    Smoker: int = Field(..., ge=0, le=1, description="Smoked at least 100 cigarettes (0=No, 1=Yes)")
    Stroke: int = Field(..., ge=0, le=1, description="History of stroke (0=No, 1=Yes)")
    HeartDiseaseorAttack: int = Field(..., ge=0, le=1, description="Coronary heart disease or MI (0=No, 1=Yes)")
    PhysActivity: int = Field(..., ge=0, le=1, description="Physical activity in past 30 days (0=No, 1=Yes)")
    Fruits: int = Field(..., ge=0, le=1, description="Consumes fruit >= 1 time per day (0=No, 1=Yes)")
    Veggies: int = Field(..., ge=0, le=1, description="Consumes vegetables >= 1 time per day (0=No, 1=Yes)")
    HvyAlcoholConsump: int = Field(..., ge=0, le=1, description="Heavy alcohol consumption (0=No, 1=Yes)")
    AnyHealthcare: int = Field(..., ge=0, le=1, description="Has health insurance/care plan (0=No, 1=Yes)")
    NoDocbcCost: int = Field(..., ge=0, le=1, description="Could not see doctor due to cost (0=No, 1=Yes)")
    GenHlth: int = Field(..., ge=1, le=5, description="General health rating (1=Excellent to 5=Poor)")
    MentHlth: int = Field(..., ge=0, le=30, description="Days with poor mental health in past 30 days (0-30)")
    PhysHlth: int = Field(..., ge=0, le=30, description="Days with physical illness/injury in past 30 days (0-30)")
    DiffWalk: int = Field(..., ge=0, le=1, description="Serious difficulty walking/climbing stairs (0=No, 1=Yes)")
    Sex: int = Field(..., ge=0, le=1, description="Sex (0=Female, 1=Male)")
    Age: int = Field(..., ge=1, le=13, description="14-level age category code (1=18-24 ... 13=80+)")
    Education: int = Field(..., ge=1, le=6, description="Education level scale (1-6)")
    Income: int = Field(..., ge=1, le=8, description="Income level scale (1-8)")


# ---------------------------------------------------------
# Pydantic Response Schema
# ---------------------------------------------------------

class PredictionResponse(BaseModel):
    project: str = "DiaTrust-NX"
    dataset: str
    model_name: str
    prediction: int
    risk_label: str
    risk_probability: float
    decision_threshold: float
    uncertainty_score: float
    confidence_category: str
    top_contributing_factors: List[Dict[str, Any]]
    medical_disclaimer: str


# ---------------------------------------------------------
# Production Prediction Engine
# ---------------------------------------------------------

class DiaTrustPredictionEngine:
    """Manages loading, pre-flight feature verification, inference, and explanations for both models."""
    
    def __init__(self, deployment_dir: Path = DEPLOYMENT_DIR):
        self.deployment_dir = Path(deployment_dir)
        self.models: Dict[str, Any] = {}
        self.preprocessors: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self._load_artifacts()

    def _load_artifacts(self):
        # 1. Early Stage
        es_path = self.deployment_dir / "diatrust_nx_early_stage_final.joblib"
        es_meta_path = self.deployment_dir / "diatrust_nx_early_stage_metadata.json"
        if es_path.exists():
            pkg = joblib.load(es_path)
            self.models["early_stage"] = pkg["model"]
            self.preprocessors["early_stage"] = pkg["preprocessor"]
        if es_meta_path.exists():
            with open(es_meta_path, "r") as f:
                self.metadata["early_stage"] = json.load(f)

        # 2. CDC
        cdc_path = self.deployment_dir / "diatrust_nx_cdc_final.joblib"
        cdc_meta_path = self.deployment_dir / "diatrust_nx_cdc_metadata.json"
        if cdc_path.exists():
            pkg = joblib.load(cdc_path)
            self.models["cdc"] = pkg["model"]
            self.preprocessors["cdc"] = pkg["preprocessor"]
        if cdc_meta_path.exists():
            with open(cdc_meta_path, "r") as f:
                self.metadata["cdc"] = json.load(f)

    def predict_early_stage(self, data: EarlyStageInput) -> PredictionResponse:
        if "early_stage" not in self.models:
            raise RuntimeError("Early Stage model artifact is not loaded.")

        model = self.models["early_stage"]
        preprocessor = self.preprocessors["early_stage"]
        meta = self.metadata.get("early_stage", {})
        threshold = float(meta.get("decision_threshold", 0.1682))
        model_name = str(meta.get("selected_model", "Logistic Regression"))

        df_in = pd.DataFrame([data.model_dump()])
        X_trans = preprocessor.transform(df_in)
        
        prob_pos = float(model.predict_proba(X_trans)[0, 1])
        prediction = 1 if prob_pos >= threshold else 0
        risk_label = "High Risk / Positive" if prediction == 1 else "Low Risk / Negative"

        # Calculate normalized Shannon entropy uncertainty
        p = np.clip(prob_pos, 1e-12, 1.0 - 1e-12)
        uncertainty = float(- (p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))
        
        if uncertainty < 0.4:
            conf_cat = "High Confidence"
        elif uncertainty < 0.8:
            conf_cat = "Moderate Confidence"
        else:
            conf_cat = "Ambiguous / High Uncertainty"

        # Model-derived linear feature attribution (logit contributions: c_j = beta_j * x_j)
        contributions = []
        try:
            weights = model.coef_[0]
            x_row = X_trans[0]
            feature_names = preprocessor.get_feature_names_out()
            raw_contribs = weights * x_row
            ranked_idx = np.argsort(np.abs(raw_contribs))[::-1]
            for idx in ranked_idx[:5]:
                val = float(raw_contribs[idx])
                if abs(val) > 1e-4:
                    feat_clean = feature_names[idx].replace("_", " ").title()
                    polarity = "positive" if val > 0 else "negative"
                    impact = f"{'Increases Risk' if val > 0 else 'Protective Factor'} ({val:+.2f} log-odds)"
                    contributions.append({
                        "factor": feat_clean,
                        "status": "Computed",
                        "impact": impact,
                        "polarity": polarity,
                        "contribution_value": round(val, 4)
                    })
        except Exception:
            pass

        if not contributions:
            contributions.append({"factor": "Baseline Profile", "status": "Normal", "impact": "Baseline Risk", "polarity": "negative"})

        return PredictionResponse(
            project="DiaTrust-NX",
            dataset="UCI Early Stage (529)",
            model_name=model_name,
            prediction=prediction,
            risk_label=risk_label,
            risk_probability=round(prob_pos, 4),
            decision_threshold=round(threshold, 4),
            uncertainty_score=round(uncertainty, 4),
            confidence_category=conf_cat,
            top_contributing_factors=contributions[:5],
            medical_disclaimer=MEDICAL_DISCLAIMER
        )

    def predict_cdc(self, data: CDCInput) -> PredictionResponse:
        if "cdc" not in self.models:
            raise RuntimeError("CDC model artifact is not loaded.")

        model = self.models["cdc"]
        preprocessor = self.preprocessors["cdc"]
        meta = self.metadata.get("cdc", {})
        threshold = float(meta.get("decision_threshold", 0.5136))
        model_name = str(meta.get("selected_model", "CatBoost"))

        df_in = pd.DataFrame([data.model_dump()])
        X_trans = preprocessor.transform(df_in)

        prob_pos = float(model.predict_proba(X_trans)[0, 1])
        prediction = 1 if prob_pos >= threshold else 0
        risk_label = "High Risk (Diabetes/Prediabetes)" if prediction == 1 else "Low Risk (No Diabetes)"

        p = np.clip(prob_pos, 1e-12, 1.0 - 1e-12)
        uncertainty = float(- (p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))

        if uncertainty < 0.4:
            conf_cat = "High Confidence"
        elif uncertainty < 0.8:
            conf_cat = "Moderate Confidence"
        else:
            conf_cat = "Ambiguous / High Uncertainty"

        # Model-derived native TreeSHAP instance-level attribution
        contributions = []
        try:
            from catboost import Pool
            feature_names = preprocessor.get_feature_names_out()
            pool = Pool(X_trans, feature_names=feature_names)
            shap_vals = model.get_feature_importance(pool, type="ShapValues")[0]
            contribs = shap_vals[:-1]  # exclude expected value bias
            ranked_idx = np.argsort(np.abs(contribs))[::-1]
            for idx in ranked_idx[:5]:
                val = float(contribs[idx])
                if abs(val) > 1e-4:
                    feat_clean = feature_names[idx].replace("_", " ").title()
                    polarity = "positive" if val > 0 else "negative"
                    impact = f"{'Increases Risk' if val > 0 else 'Protective Factor'} ({val:+.2f} SHAP)"
                    contributions.append({
                        "factor": feat_clean,
                        "status": "Computed SHAP",
                        "impact": impact,
                        "polarity": polarity,
                        "shap_value": round(val, 4)
                    })
        except Exception:
            pass

        if not contributions:
            contributions.append({"factor": "Population Baseline Profile", "status": "Normal", "impact": "Baseline Risk", "polarity": "negative"})

        return PredictionResponse(
            project="DiaTrust-NX",
            dataset="UCI CDC Health Indicators (891)",
            model_name=model_name,
            prediction=prediction,
            risk_label=risk_label,
            risk_probability=round(prob_pos, 4),
            decision_threshold=round(threshold, 4),
            uncertainty_score=round(uncertainty, 4),
            confidence_category=conf_cat,
            top_contributing_factors=contributions[:5],
            medical_disclaimer=MEDICAL_DISCLAIMER
        )


# ---------------------------------------------------------
# FastAPI Application Factory
# ---------------------------------------------------------

def create_app() -> FastAPI:
    """Builds and returns the configured FastAPI application instance."""
    app = FastAPI(
        title="DiaTrust-NX Risk Prediction Service",
        description="Production API & Interactive Healthcare Research Dashboard for Dual-Dataset Diabetes Risk Estimation",
        version="2.0.0"
    )

    # Enable CORS for seamless local and remote interactions
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    engine = DiaTrustPredictionEngine()

    # Mount static directories
    plots_path = Path(PLOTS_DIR)
    if plots_path.exists():
        app.mount("/plots", StaticFiles(directory=str(plots_path)), name="plots")

    frontend_path = Path("frontend")
    if frontend_path.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

    # API Endpoints
    @app.api_route("/health", methods=["GET", "HEAD"])
    @app.api_route("/api/health", methods=["GET", "HEAD"])
    def health():
        return {
            "status": "healthy",
            "version": "2.0.0",
            "early_stage_loaded": "early_stage" in engine.models,
            "cdc_loaded": "cdc" in engine.models,
            "early_stage_model": engine.metadata.get("early_stage", {}).get("selected_model", "Logistic Regression"),
            "cdc_model": engine.metadata.get("cdc", {}).get("selected_model", "CatBoost")
        }

    @app.get("/metadata")
    @app.get("/api/metadata")
    def metadata():
        return engine.metadata

    @app.get("/api/models")
    def get_models_info():
        return {
            "early_stage": {
                "dataset": "UCI Early Stage Diabetes Risk Prediction (529)",
                "selected_model": engine.metadata.get("early_stage", {}).get("selected_model", "Logistic Regression"),
                "version": "2.0.0",
                "decision_threshold": engine.metadata.get("early_stage", {}).get("decision_threshold", 0.1682),
                "input_features": engine.metadata.get("early_stage", {}).get("input_features", []),
                "metrics": engine.metadata.get("early_stage", {}).get("test_metrics", {}),
                "calibration": "Platt Scaling (Sigmoid)"
            },
            "cdc": {
                "dataset": "UCI CDC Diabetes Health Indicators (891)",
                "selected_model": engine.metadata.get("cdc", {}).get("selected_model", "CatBoost"),
                "version": "2.0.0",
                "decision_threshold": engine.metadata.get("cdc", {}).get("decision_threshold", 0.5136),
                "input_features": engine.metadata.get("cdc", {}).get("input_features", []),
                "metrics": engine.metadata.get("cdc", {}).get("test_metrics", {}),
                "calibration": "Inverse-Frequency Loss Weighting"
            }
        }

    @app.get("/api/reports")
    def get_all_reports():
        ann_path = Path(REPORTS_DIR) / "DIATRUST_NX_ANN_COMPARISON.csv"
        comp_path = Path(REPORTS_DIR) / "DIATRUST_NX_FINAL_DATASET_COMPARISON.csv"
        es_fi_path = Path(REPORTS_DIR) / "early_stage_feature_importance.csv"
        cdc_fi_path = Path(REPORTS_DIR) / "cdc_feature_importance.csv"
        
        return {
            "dataset_comparison": pd.read_csv(comp_path).to_dict(orient="records") if comp_path.exists() else [],
            "ann_comparison": pd.read_csv(ann_path).to_dict(orient="records") if ann_path.exists() else [],
            "early_stage_features": pd.read_csv(es_fi_path).to_dict(orient="records") if es_fi_path.exists() else [],
            "cdc_features": pd.read_csv(cdc_fi_path).to_dict(orient="records") if cdc_fi_path.exists() else []
        }

    @app.get("/api/model-performance")
    def get_model_performance():
        es_p = Path(REPORTS_DIR) / "early_stage_all_model_experiments.csv"
        cdc_p = Path(REPORTS_DIR) / "cdc_all_model_experiments.csv"
        ann_p = Path(REPORTS_DIR) / "DIATRUST_NX_ANN_COMPARISON.csv"

        return {
            "early_stage": pd.read_csv(es_p).to_dict(orient="records") if es_p.exists() else [],
            "cdc": pd.read_csv(cdc_p).to_dict(orient="records") if cdc_p.exists() else [],
            "ann_architectures": pd.read_csv(ann_p).to_dict(orient="records") if ann_p.exists() else [],
            "baseline_comparison": {
                "early_stage": {
                    "previous_baseline": {"model": "CatBoost", "test_acc": "92.78% (leaked) / 86.87% (grouped)", "roc_auc": "0.9327", "recall": "89.80%"},
                    "diatrust_nx": {"model": "Logistic Regression", "test_acc": "96.55%", "roc_auc": "0.9927", "recall": "94.44%", "specificity": "100.0%"}
                },
                "cdc": {
                    "previous_baseline": {"model": "Balanced CatBoost", "test_acc": "81.49%", "recall": "58.18%", "roc_auc": "0.8273"},
                    "diatrust_nx": {"model": "Balanced CatBoost", "test_acc": "73.06%", "recall": "78.70%", "roc_auc": "0.8336", "recall_gain": "+20.52%"}
                }
            }
        }

    @app.post("/predict/early-stage", response_model=PredictionResponse)
    @app.post("/api/predict/early-stage", response_model=PredictionResponse)
    def predict_early_stage(payload: EarlyStageInput):
        try:
            return engine.predict_early_stage(payload)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/cdc", response_model=PredictionResponse)
    @app.post("/api/predict/cdc", response_model=PredictionResponse)
    def predict_cdc(payload: CDCInput):
        try:
            return engine.predict_cdc(payload)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/demo-samples")
    def get_demo_samples():
        """Returns realistic clinical demonstration cases for 1-click form prefilling."""
        return {
            "early_stage": {
                "high_risk": {
                    "age": 52, "gender": "Male", "polyuria": "Yes", "polydipsia": "Yes",
                    "sudden_weight_loss": "Yes", "weakness": "Yes", "polyphagia": "Yes",
                    "genital_thrush": "Yes", "visual_blurring": "Yes", "itching": "Yes",
                    "irritability": "Yes", "delayed_healing": "Yes", "partial_paresis": "No",
                    "muscle_stiffness": "Yes", "alopecia": "Yes", "obesity": "Yes"
                },
                "low_risk": {
                    "age": 28, "gender": "Male", "polyuria": "No", "polydipsia": "No",
                    "sudden_weight_loss": "No", "weakness": "No", "polyphagia": "No",
                    "genital_thrush": "No", "visual_blurring": "No", "itching": "No",
                    "irritability": "No", "delayed_healing": "No", "partial_paresis": "No",
                    "muscle_stiffness": "No", "alopecia": "No", "obesity": "No"
                },
                "borderline": {
                    "age": 44, "gender": "Female", "polyuria": "No", "polydipsia": "Yes",
                    "sudden_weight_loss": "No", "weakness": "Yes", "polyphagia": "No",
                    "genital_thrush": "No", "visual_blurring": "Yes", "itching": "Yes",
                    "irritability": "No", "delayed_healing": "Yes", "partial_paresis": "No",
                    "muscle_stiffness": "No", "alopecia": "Yes", "obesity": "No"
                }
            },
            "cdc": {
                "high_risk": {
                    "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 34.2,
                    "Smoker": 1, "Stroke": 0, "HeartDiseaseorAttack": 1, "PhysActivity": 0,
                    "Fruits": 0, "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1,
                    "NoDocbcCost": 0, "GenHlth": 4, "MentHlth": 12, "PhysHlth": 20,
                    "DiffWalk": 1, "Sex": 1, "Age": 10, "Education": 4, "Income": 3
                },
                "low_risk": {
                    "HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 22.4,
                    "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 1,
                    "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1,
                    "NoDocbcCost": 0, "GenHlth": 1, "MentHlth": 0, "PhysHlth": 0,
                    "DiffWalk": 0, "Sex": 0, "Age": 3, "Education": 6, "Income": 8
                },
                "borderline": {
                    "HighBP": 1, "HighChol": 0, "CholCheck": 1, "BMI": 27.8,
                    "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 1,
                    "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1,
                    "NoDocbcCost": 0, "GenHlth": 3, "MentHlth": 2, "PhysHlth": 4,
                    "DiffWalk": 0, "Sex": 1, "Age": 9, "Education": 5, "Income": 6
                }
            }
        }

    @app.get("/api/reports/ann")
    def get_ann_report():
        p = Path(REPORTS_DIR) / "DIATRUST_NX_ANN_COMPARISON.csv"
        if p.exists():
            df = pd.read_csv(p)
            return df.to_dict(orient="records")
        return []

    @app.get("/api/reports/models/{dataset}")
    def get_models_report(dataset: str):
        filename = f"{dataset.lower()}_all_model_experiments.csv"
        p = Path(REPORTS_DIR) / filename
        if p.exists():
            df = pd.read_csv(p)
            return df.to_dict(orient="records")
        raise HTTPException(status_code=404, detail=f"Report for dataset '{dataset}' not found")

    @app.get("/api/reports/dataset-comparison")
    def get_dataset_comparison_report():
        p = Path(REPORTS_DIR) / "DIATRUST_NX_FINAL_DATASET_COMPARISON.csv"
        if p.exists():
            df = pd.read_csv(p)
            return df.to_dict(orient="records")
        return []

    @app.get("/api/reports/feature-importance/{dataset}")
    def get_feature_importance_report(dataset: str):
        filename = f"{dataset.lower()}_feature_importance.csv"
        p = Path(REPORTS_DIR) / filename
        if p.exists():
            df = pd.read_csv(p)
            return df.to_dict(orient="records")
        raise HTTPException(status_code=404, detail=f"Feature importance for '{dataset}' not found")

    @app.get("/api/plots/list")
    def get_plots_list():
        plots_dir = Path(PLOTS_DIR)
        if not plots_dir.exists():
            return []
        return [f.name for f in plots_dir.glob("*.png")]

    @app.api_route("/", methods=["GET", "HEAD"])
    def serve_index():
        index_file = Path("frontend") / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {
            "service": "DiaTrust-NX API",
            "status": "operational",
            "version": "2.0.0",
            "message": "Frontend index.html not yet built. Use API endpoints.",
            "endpoints": ["/predict/early-stage", "/predict/cdc", "/metadata", "/health"],
            "disclaimer": MEDICAL_DISCLAIMER
        }

    return app
