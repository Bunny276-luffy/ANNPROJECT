"""DiaTrust-NX Preprocessing & Feature Engineering Module
Provides scikit-learn compatible pipelines fitted exclusively on training data.
Prevents data leakage and supports serialization for deployment.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from configs.config import EARLY_STAGE_FEATURES, CDC_FEATURES


class EarlyStageFeatureTransformer(BaseEstimator, TransformerMixin):
    """Custom scikit-learn transformer for the Early Stage Diabetes dataset.
    
    Transforms:
    1. Binary symptoms ('Yes'/'No') -> 1/0
    2. Gender ('Male'/'Female') -> 1/0
    3. Handles missing values via SimpleImputer fitted on training data
    4. Engineered clinical features:
       - 'symptom_count': Total number of positive symptom indicators
       - 'cardinal_triad': Polyuria + Polydipsia + Sudden Weight Loss
       - 'polyuria_polydipsia_interaction': Polyuria * Polydipsia
       - 'age_over_45': Clinically significant age threshold indicator
    5. Standardizes continuous age and symptom count using StandardScaler (fitted on train only).
    """
    
    def __init__(self, include_engineered_features: bool = True):
        self.include_engineered_features = include_engineered_features
        self.age_scaler = StandardScaler()
        self.feature_names_out_: List[str] = []
        self._is_fitted = False
        
    def _map_categoricals(self, X_df: pd.DataFrame) -> pd.DataFrame:
        df = X_df.copy()
        for col in df.columns:
            if col == "gender":
                df[col] = df[col].astype(str).str.strip().str.capitalize().map({"Male": 1, "Female": 0}).fillna(0)
            elif col != "age":
                df[col] = df[col].astype(str).str.strip().str.capitalize().map({"Yes": 1, "No": 0}).fillna(0)
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        return df

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X_df = self._map_categoricals(X)
        self.age_median_ = float(X_df["age"].median())
        clean_age = X_df["age"].fillna(self.age_median_).values.reshape(-1, 1)
        self.age_scaler.fit(clean_age)
        
        # Build feature names
        base_features = [c for c in EARLY_STAGE_FEATURES]
        if self.include_engineered_features:
            engineered = [
                "symptom_count",
                "cardinal_triad",
                "polyuria_polydipsia_interaction",
                "age_over_45"
            ]
            self.feature_names_out_ = base_features + engineered
        else:
            self.feature_names_out_ = base_features
            
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Transformer must be fitted before transforming data.")
        X_df = self._map_categoricals(X)
        
        # Fill missing age with training median
        clean_age = X_df["age"].fillna(self.age_median_).values.reshape(-1, 1)
        scaled_age = self.age_scaler.transform(clean_age)
        
        # Binary symptoms list
        symptom_cols = [c for c in EARLY_STAGE_FEATURES if c not in ["age", "gender"]]
        
        output_dfs = []
        for col in EARLY_STAGE_FEATURES:
            if col == "age":
                output_dfs.append(scaled_age.flatten())
            else:
                output_dfs.append(X_df[col].values.astype(float))
                
        if self.include_engineered_features:
            symptoms_matrix = X_df[symptom_cols].values.astype(float)
            symptom_count = symptoms_matrix.sum(axis=1)
            cardinal_triad = (X_df["polyuria"].values * X_df["polydipsia"].values * X_df["sudden_weight_loss"].values)
            poly_poly = X_df["polyuria"].values * X_df["polydipsia"].values
            age_over_45 = (X_df["age"].fillna(self.age_median_).values >= 45).astype(float)
            
            output_dfs.extend([
                symptom_count,
                cardinal_triad,
                poly_poly,
                age_over_45
            ])
            
        return np.column_stack(output_dfs)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return list(self.feature_names_out_)


class CDCFeatureTransformer(BaseEstimator, TransformerMixin):
    """Custom scikit-learn transformer for the CDC Diabetes Health Indicators dataset.
    
    Transforms:
    1. Validates and handles missing numeric entries using training medians.
    2. Engineered clinical features:
       - 'cardiometabolic_risk': HighBP + HighChol + Stroke + HeartDiseaseorAttack
       - 'obese_bmi_flag': BMI >= 30.0
       - 'severe_obesity_flag': BMI >= 35.0
       - 'genhlth_age_interaction': GenHlth * Age
       - 'sedentary_diffwalk_risk': (PhysActivity == 0) & (DiffWalk == 1)
    3. Standardizes continuous and ordinal variables using StandardScaler (fitted on train only).
    """
    
    def __init__(self, include_engineered_features: bool = True):
        self.include_engineered_features = include_engineered_features
        self.scaler = StandardScaler()
        self.feature_names_out_: List[str] = []
        self._is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X_df = X.copy().astype(float)
        self.medians_ = X_df.median().to_dict()
        X_clean = X_df.fillna(self.medians_)
        
        transformed = self._compute_matrix(X_clean)
        self.scaler.fit(transformed)
        
        base_features = [c for c in CDC_FEATURES]
        if self.include_engineered_features:
            engineered = [
                "cardiometabolic_risk",
                "obese_bmi_flag",
                "severe_obesity_flag",
                "genhlth_age_interaction",
                "sedentary_diffwalk_risk"
            ]
            self.feature_names_out_ = base_features + engineered
        else:
            self.feature_names_out_ = base_features
            
        self._is_fitted = True
        return self

    def _compute_matrix(self, X_df: pd.DataFrame) -> np.ndarray:
        arrays = [X_df[c].values.astype(float) for c in CDC_FEATURES]
        if self.include_engineered_features:
            cardio = (
                X_df["HighBP"].values +
                X_df["HighChol"].values +
                X_df["Stroke"].values +
                X_df["HeartDiseaseorAttack"].values
            )
            obese = (X_df["BMI"].values >= 30.0).astype(float)
            severe_obese = (X_df["BMI"].values >= 35.0).astype(float)
            genhlth_age = X_df["GenHlth"].values * X_df["Age"].values
            sedentary = ((X_df["PhysActivity"].values == 0) & (X_df["DiffWalk"].values == 1)).astype(float)
            
            arrays.extend([cardio, obese, severe_obese, genhlth_age, sedentary])
            
        return np.column_stack(arrays)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Transformer must be fitted before transforming data.")
        X_df = X.copy().astype(float).fillna(self.medians_)
        matrix = self._compute_matrix(X_df)
        return self.scaler.transform(matrix)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return list(self.feature_names_out_)

