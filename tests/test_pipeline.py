"""DiaTrust-NX Automated Test Suite
Covers data loading, preprocessing, duplicate profile isolation, feature validation,
probability ranges, model inference, and API endpoint verification.
"""

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.data_loader import load_early_stage_data, load_cdc_data
from src.duplicate_detection import compute_profile_groups, stratified_group_train_val_test_split
from src.preprocessing import EarlyStageFeatureTransformer, CDCFeatureTransformer
from src.prediction import (
    EarlyStageInput,
    CDCInput,
    create_app,
    DiaTrustPredictionEngine
)
from configs.config import EARLY_STAGE_FEATURES, CDC_FEATURES


# ---------------------------------------------------------
# 1. Data Loading Tests
# ---------------------------------------------------------

def test_load_early_stage_data():
    X, y, info = load_early_stage_data()
    assert len(X) == 520, f"Expected 520 rows, got {len(X)}"
    assert len(y) == 520
    assert X.shape[1] == 16
    assert set(X.columns) == set(EARLY_STAGE_FEATURES)
    assert set(np.unique(y)) == {0, 1}
    assert info["uci_id"] == 529
    assert info["missing_values_total"] == 0


def test_load_cdc_data():
    X, y, info = load_cdc_data()
    assert len(X) == 253680, f"Expected 253680 rows, got {len(X)}"
    assert len(y) == 253680
    assert X.shape[1] == 21
    assert set(X.columns) == set(CDC_FEATURES)
    assert set(np.unique(y)) == {0, 1}
    assert info["uci_id"] == 891
    assert info["missing_values_total"] == 0


# ---------------------------------------------------------
# 2. Duplicate Group Isolation & Leakage Tests
# ---------------------------------------------------------

def test_duplicate_group_isolation():
    X, y, _ = load_early_stage_data()
    df = pd.concat([X, y], axis=1)
    
    group_series, summary = compute_profile_groups(df, EARLY_STAGE_FEATURES)
    assert summary["total_rows"] == 520
    assert summary["unique_profiles"] == 251, f"Expected 251 unique profiles, got {summary['unique_profiles']}"
    assert summary["duplicate_rows"] == 376

    train_df, val_df, test_df, report = stratified_group_train_val_test_split(
        df,
        feature_cols=EARLY_STAGE_FEATURES,
        target_col="diabetes_risk",
        train_ratio=0.65,
        val_ratio=0.17,
        test_ratio=0.18,
        random_state=42
    )

    assert report["zero_leakage_verified"] is True

    # Assert rigorous zero group overlap
    train_tuples = set(train_df[EARLY_STAGE_FEATURES].apply(tuple, axis=1))
    val_tuples = set(val_df[EARLY_STAGE_FEATURES].apply(tuple, axis=1))
    test_tuples = set(test_df[EARLY_STAGE_FEATURES].apply(tuple, axis=1))

    assert len(train_tuples & val_tuples) == 0, "Leakage detected between train and val!"
    assert len(train_tuples & test_tuples) == 0, "Leakage detected between train and test!"
    assert len(val_tuples & test_tuples) == 0, "Leakage detected between val and test!"


# ---------------------------------------------------------
# 3. Preprocessing Tests
# ---------------------------------------------------------

def test_early_stage_preprocessing():
    X, y, _ = load_early_stage_data()
    transformer = EarlyStageFeatureTransformer(include_engineered_features=True)
    transformer.fit(X.iloc[:350])
    
    transformed = transformer.transform(X.iloc[350:])
    names = transformer.get_feature_names_out()
    
    assert transformed.shape[1] == len(names)
    assert "symptom_count" in names
    assert "cardinal_triad" in names
    assert not np.isnan(transformed).any(), "NaN values found after transformation"


def test_cdc_preprocessing():
    X, y, _ = load_cdc_data()
    sample = X.iloc[:200].copy()
    transformer = CDCFeatureTransformer(include_engineered_features=True)
    transformer.fit(sample)
    
    transformed = transformer.transform(sample)
    names = transformer.get_feature_names_out()
    
    assert transformed.shape[1] == len(names)
    assert "cardiometabolic_risk" in names
    assert not np.isnan(transformed).any()


# ---------------------------------------------------------
# 4. Pydantic Input Validation Tests
# ---------------------------------------------------------

def test_early_stage_input_validation():
    # Valid input
    valid_data = EarlyStageInput(
        age=50,
        gender="Male",
        polyuria="Yes",
        polydipsia="Yes",
        sudden_weight_loss="No",
        weakness="Yes",
        polyphagia="No",
        genital_thrush="No",
        visual_blurring="No",
        itching="Yes",
        irritability="No",
        delayed_healing="No",
        partial_paresis="No",
        muscle_stiffness="No",
        alopecia="No",
        obesity="No"
    )
    assert valid_data.age == 50

    # Invalid gender
    with pytest.raises(ValueError):
        EarlyStageInput(
            age=50,
            gender="Invalid",
            polyuria="Yes",
            polydipsia="Yes",
            sudden_weight_loss="No",
            weakness="Yes",
            polyphagia="No",
            genital_thrush="No",
            visual_blurring="No",
            itching="Yes",
            irritability="No",
            delayed_healing="No",
            partial_paresis="No",
            muscle_stiffness="No",
            alopecia="No",
            obesity="No"
        )

    # Invalid symptom flag
    with pytest.raises(ValueError):
        EarlyStageInput(
            age=50,
            gender="Male",
            polyuria="Maybe",
            polydipsia="Yes",
            sudden_weight_loss="No",
            weakness="Yes",
            polyphagia="No",
            genital_thrush="No",
            visual_blurring="No",
            itching="Yes",
            irritability="No",
            delayed_healing="No",
            partial_paresis="No",
            muscle_stiffness="No",
            alopecia="No",
            obesity="No"
        )


def test_cdc_input_validation():
    # Valid input
    valid_cdc = CDCInput(
        HighBP=1,
        HighChol=0,
        CholCheck=1,
        BMI=27.5,
        Smoker=0,
        Stroke=0,
        HeartDiseaseorAttack=0,
        PhysActivity=1,
        Fruits=1,
        Veggies=1,
        HvyAlcoholConsump=0,
        AnyHealthcare=1,
        NoDocbcCost=0,
        GenHlth=2,
        MentHlth=0,
        PhysHlth=0,
        DiffWalk=0,
        Sex=1,
        Age=7,
        Education=5,
        Income=7
    )
    assert valid_cdc.BMI == 27.5

    # Out of range BMI
    with pytest.raises(ValueError):
        CDCInput(
            HighBP=1,
            HighChol=0,
            CholCheck=1,
            BMI=5.0,  # Invalid (<10)
            Smoker=0,
            Stroke=0,
            HeartDiseaseorAttack=0,
            PhysActivity=1,
            Fruits=1,
            Veggies=1,
            HvyAlcoholConsump=0,
            AnyHealthcare=1,
            NoDocbcCost=0,
            GenHlth=2,
            MentHlth=0,
            PhysHlth=0,
            DiffWalk=0,
            Sex=1,
            Age=7,
            Education=5,
            Income=7
        )


# ---------------------------------------------------------
# 5. Production Engine & Endpoints Contract Tests
# ---------------------------------------------------------

def test_prediction_engine_early_stage():
    engine = DiaTrustPredictionEngine()
    sample = EarlyStageInput(
        age=45,
        gender="Male",
        polyuria="Yes",
        polydipsia="Yes",
        sudden_weight_loss="Yes",
        weakness="Yes",
        polyphagia="No",
        genital_thrush="No",
        visual_blurring="Yes",
        itching="No",
        irritability="Yes",
        delayed_healing="No",
        partial_paresis="No",
        muscle_stiffness="No",
        alopecia="No",
        obesity="Yes"
    )
    resp = engine.predict_early_stage(sample)
    assert resp.project == "DiaTrust-NX"
    assert 0.0 <= resp.risk_probability <= 1.0
    assert 0.0 <= resp.uncertainty_score <= 1.0
    assert resp.prediction in [0, 1]
    assert len(resp.top_contributing_factors) > 0
    assert "Not a clinically validated diagnostic tool" in resp.medical_disclaimer


def test_prediction_engine_cdc():
    engine = DiaTrustPredictionEngine()
    sample = CDCInput(
        HighBP=1,
        HighChol=1,
        CholCheck=1,
        BMI=32.5,
        Smoker=1,
        Stroke=0,
        HeartDiseaseorAttack=1,
        PhysActivity=0,
        Fruits=1,
        Veggies=1,
        HvyAlcoholConsump=0,
        AnyHealthcare=1,
        NoDocbcCost=0,
        GenHlth=4,
        MentHlth=10,
        PhysHlth=15,
        DiffWalk=1,
        Sex=1,
        Age=9,
        Education=4,
        Income=3
    )
    resp = engine.predict_cdc(sample)
    assert resp.project == "DiaTrust-NX"
    assert 0.0 <= resp.risk_probability <= 1.0
    assert 0.0 <= resp.uncertainty_score <= 1.0
    assert resp.prediction in [0, 1]
    assert len(resp.top_contributing_factors) > 0


def test_fastapi_app_contract():
    app = create_app()
    assert app.title == "DiaTrust-NX Risk Prediction Service"
    assert app.version == "2.0.0"
    route_paths = [route.path for route in app.routes]
    assert "/" in route_paths
    assert "/health" in route_paths
    assert "/metadata" in route_paths
    assert "/predict/early-stage" in route_paths
    assert "/predict/cdc" in route_paths

