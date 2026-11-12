"""DiaTrust-NX Data Loader Module
Reproducibly loads and validates UCI Early Stage (529) and UCI CDC (891) datasets.
Maintains separate pipelines, validation schemas, and metadata.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

from configs.config import (
    DATASET_CONFIG,
    EARLY_STAGE_FEATURES,
    CDC_FEATURES,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR
)


def load_early_stage_data(raw_path: Path = None) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Loads and validates the UCI Early Stage Diabetes Risk Prediction Dataset (UCI ID: 529).
    
    Standardizes column names to lowercase snake_case:
    e.g. 'sudden weight loss' -> 'sudden_weight_loss'
    Converts binary target 'Positive'/'Negative' -> 1/0
    
    Returns:
    --------
    X : pd.DataFrame
        Cleaned predictor features dataframe with standardized column names.
    y : pd.Series
        Binary target series (1=Positive, 0=Negative).
    info : Dict[str, Any]
        Dataset summary report.
    """
    if raw_path is None:
        raw_path = DATASET_CONFIG["early_stage"]["raw_path"]
        
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Early Stage dataset not found at: {raw_path}")
        
    df = pd.read_csv(raw_path)
    
    # Standardize column names
    col_mapping = {
        "Age": "age",
        "Gender": "gender",
        "Polyuria": "polyuria",
        "Polydipsia": "polydipsia",
        "sudden weight loss": "sudden_weight_loss",
        "weakness": "weakness",
        "Polyphagia": "polyphagia",
        "Genital thrush": "genital_thrush",
        "visual blurring": "visual_blurring",
        "Itching": "itching",
        "Irritability": "irritability",
        "delayed healing": "delayed_healing",
        "partial paresis": "partial_paresis",
        "muscle stiffness": "muscle_stiffness",
        "Alopecia": "alopecia",
        "Obesity": "obesity",
        "class": "target"
    }
    
    df = df.rename(columns=col_mapping)
    
    # Validate expected features
    missing_cols = set(EARLY_STAGE_FEATURES) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected features in Early Stage dataset: {missing_cols}")
        
    X = df[EARLY_STAGE_FEATURES].copy()
    
    # Target encoding
    y_raw = df["target"].astype(str).str.strip().str.capitalize()
    y = (y_raw == "Positive").astype(int)
    y.name = "diabetes_risk"
    
    info = {
        "dataset_name": "UCI Early Stage Diabetes Risk Prediction",
        "uci_id": 529,
        "n_samples": int(len(df)),
        "n_features": len(EARLY_STAGE_FEATURES),
        "feature_names": EARLY_STAGE_FEATURES,
        "target_name": "diabetes_risk",
        "class_distribution": {
            "Positive (1)": int(y.sum()),
            "Negative (0)": int((y == 0).sum())
        },
        "positive_prevalence": float(round(y.mean(), 4)),
        "missing_values_total": int(df.isnull().sum().sum()),
        "data_types": {col: str(dtype) for col, dtype in X.dtypes.items()}
    }
    
    return X, y, info


def load_cdc_data(raw_path: Path = None) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Loads and validates the UCI CDC Diabetes Health Indicators Dataset (UCI ID: 891).
    
    Features: 21 health indicator and lifestyle features.
    Target: Diabetes_binary (1 = Diabetes/Prediabetes, 0 = No Diabetes).
    
    Returns:
    --------
    X : pd.DataFrame
        Predictor features dataframe.
    y : pd.Series
        Binary target series.
    info : Dict[str, Any]
        Dataset summary report.
    """
    if raw_path is None:
        raw_path = DATASET_CONFIG["cdc"]["raw_path"]
        
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"CDC dataset not found at: {raw_path}")
        
    df = pd.read_csv(raw_path)
    
    # Identify target column
    target_candidates = ["Diabetes_binary", "diabetes_binary", "Diabetes_binary_5050split"]
    target_col = None
    for cand in target_candidates:
        if cand in df.columns:
            target_col = cand
            break
            
    if target_col is None:
        raise ValueError(f"Could not identify target column in CDC dataset. Available: {df.columns.tolist()}")
        
    # Validate expected features
    missing_cols = set(CDC_FEATURES) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected features in CDC dataset: {missing_cols}")
        
    X = df[CDC_FEATURES].copy().astype(float)
    y = df[target_col].astype(int)
    y.name = "diabetes_risk"
    
    info = {
        "dataset_name": "UCI CDC Diabetes Health Indicators",
        "uci_id": 891,
        "n_samples": int(len(df)),
        "n_features": len(CDC_FEATURES),
        "feature_names": CDC_FEATURES,
        "target_name": "diabetes_risk",
        "class_distribution": {
            "Diabetes/Prediabetes (1)": int(y.sum()),
            "No Diabetes (0)": int((y == 0).sum())
        },
        "positive_prevalence": float(round(y.mean(), 4)),
        "missing_values_total": int(df.isnull().sum().sum()),
        "data_types": {col: str(dtype) for col, dtype in X.dtypes.items()}
    }
    
    return X, y, info

