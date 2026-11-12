"""DiaTrust-NX Application Integration Tests
Verifies FastAPI endpoints, saved model artifact inference, input validation schemas,
error handling, probability bounds, report serving, and frontend static assets.
"""

from pathlib import Path
import pytest
from pydantic import ValidationError

from src.prediction import (
    create_app,
    DiaTrustPredictionEngine,
    EarlyStageInput,
    CDCInput,
    PredictionResponse
)


@pytest.fixture(scope="module")
def app_and_routes():
    app = create_app()
    routes = {r.path: r.endpoint for r in app.routes if hasattr(r, "endpoint")}
    return app, routes


def test_app_routes_registered(app_and_routes):
    _, routes = app_and_routes
    expected_paths = [
        "/health",
        "/api/health",
        "/metadata",
        "/api/metadata",
        "/predict/early-stage",
        "/api/predict/early-stage",
        "/predict/cdc",
        "/api/predict/cdc",
        "/api/demo-samples",
        "/api/reports/ann",
        "/api/reports/models/{dataset}",
        "/api/reports/dataset-comparison",
        "/api/reports/feature-importance/{dataset}",
        "/api/plots/list",
        "/"
    ]
    for path in expected_paths:
        assert path in routes, f"Expected endpoint '{path}' not found in registered routes"


def test_health_endpoint(app_and_routes):
    _, routes = app_and_routes
    res = routes["/health"]()
    assert res["status"] == "healthy"
    assert res["version"] == "2.0.0"
    assert res["early_stage_loaded"] is True
    assert res["cdc_loaded"] is True
    assert res["early_stage_model"] == "Logistic Regression"
    assert res["cdc_model"] == "CatBoost"


def test_metadata_endpoint(app_and_routes):
    _, routes = app_and_routes
    data = routes["/metadata"]()
    assert "early_stage" in data
    assert "cdc" in data
    assert data["early_stage"]["selected_model"] == "Logistic Regression"
    assert data["cdc"]["selected_model"] == "CatBoost"
    assert "input_features" in data["early_stage"]
    assert "decision_threshold" in data["cdc"]


def test_demo_samples_endpoint(app_and_routes):
    _, routes = app_and_routes
    demos = routes["/api/demo-samples"]()
    assert "early_stage" in demos
    assert "cdc" in demos
    assert "high_risk" in demos["early_stage"]
    assert "low_risk" in demos["early_stage"]
    assert "borderline" in demos["early_stage"]
    assert "high_risk" in demos["cdc"]
    assert "low_risk" in demos["cdc"]


def test_early_stage_valid_prediction(app_and_routes):
    _, routes = app_and_routes
    demos = routes["/api/demo-samples"]()
    
    # 1. Test high risk profile
    sample_high = EarlyStageInput(**demos["early_stage"]["high_risk"])
    res_high = routes["/predict/early-stage"](sample_high)
    assert isinstance(res_high, PredictionResponse)
    assert res_high.project == "DiaTrust-NX"
    assert 0.0 <= res_high.risk_probability <= 1.0
    assert res_high.prediction == 1
    assert "High Risk" in res_high.risk_label
    assert res_high.decision_threshold == pytest.approx(0.1682, rel=1e-2)
    assert 0.0 <= res_high.uncertainty_score <= 1.0
    assert len(res_high.top_contributing_factors) > 0
    assert "Research prototype only" in res_high.medical_disclaimer

    # 2. Test low risk profile
    sample_low = EarlyStageInput(**demos["early_stage"]["low_risk"])
    res_low = routes["/predict/early-stage"](sample_low)
    assert res_low.prediction == 0
    assert "Low Risk" in res_low.risk_label
    assert res_low.risk_probability < res_high.risk_probability


def test_early_stage_invalid_input_rejected():
    # 1. Invalid gender
    with pytest.raises(ValidationError):
        EarlyStageInput(
            age=45, gender="Alien", polyuria="Yes", polydipsia="Yes", sudden_weight_loss="Yes",
            weakness="Yes", polyphagia="No", genital_thrush="No", visual_blurring="Yes",
            itching="No", irritability="Yes", delayed_healing="No", partial_paresis="No",
            muscle_stiffness="No", alopecia="No", obesity="Yes"
        )

    # 2. Invalid symptom value
    with pytest.raises(ValidationError):
        EarlyStageInput(
            age=45, gender="Male", polyuria="Maybe", polydipsia="Yes", sudden_weight_loss="Yes",
            weakness="Yes", polyphagia="No", genital_thrush="No", visual_blurring="Yes",
            itching="No", irritability="Yes", delayed_healing="No", partial_paresis="No",
            muscle_stiffness="No", alopecia="No", obesity="Yes"
        )

    # 3. Invalid age out of bounds
    with pytest.raises(ValidationError):
        EarlyStageInput(
            age=250, gender="Male", polyuria="Yes", polydipsia="Yes", sudden_weight_loss="Yes",
            weakness="Yes", polyphagia="No", genital_thrush="No", visual_blurring="Yes",
            itching="No", irritability="Yes", delayed_healing="No", partial_paresis="No",
            muscle_stiffness="No", alopecia="No", obesity="Yes"
        )


def test_cdc_valid_prediction(app_and_routes):
    _, routes = app_and_routes
    demos = routes["/api/demo-samples"]()
    
    # 1. Test high risk profile
    sample_high = CDCInput(**demos["cdc"]["high_risk"])
    res_high = routes["/predict/cdc"](sample_high)
    assert isinstance(res_high, PredictionResponse)
    assert res_high.project == "DiaTrust-NX"
    assert 0.0 <= res_high.risk_probability <= 1.0
    assert res_high.prediction == 1
    assert "High Risk" in res_high.risk_label
    assert res_high.decision_threshold == pytest.approx(0.5136, rel=1e-2)
    assert 0.0 <= res_high.uncertainty_score <= 1.0
    assert len(res_high.top_contributing_factors) > 0

    # 2. Test low risk profile
    sample_low = CDCInput(**demos["cdc"]["low_risk"])
    res_low = routes["/predict/cdc"](sample_low)
    assert res_low.prediction == 0
    assert "Low Risk" in res_low.risk_label
    assert res_low.risk_probability < res_high.risk_probability


def test_cdc_invalid_input_rejected():
    # 1. Invalid binary flag (HighBP > 1)
    with pytest.raises(ValidationError):
        CDCInput(
            HighBP=9, HighChol=1, CholCheck=1, BMI=30.0, Smoker=0, Stroke=0,
            HeartDiseaseorAttack=0, PhysActivity=1, Fruits=1, Veggies=1,
            HvyAlcoholConsump=0, AnyHealthcare=1, NoDocbcCost=0, GenHlth=3,
            MentHlth=0, PhysHlth=0, DiffWalk=0, Sex=1, Age=5, Education=5, Income=6
        )

    # 2. Negative / out-of-range BMI (< 10)
    with pytest.raises(ValidationError):
        CDCInput(
            HighBP=0, HighChol=1, CholCheck=1, BMI=-5.0, Smoker=0, Stroke=0,
            HeartDiseaseorAttack=0, PhysActivity=1, Fruits=1, Veggies=1,
            HvyAlcoholConsump=0, AnyHealthcare=1, NoDocbcCost=0, GenHlth=3,
            MentHlth=0, PhysHlth=0, DiffWalk=0, Sex=1, Age=5, Education=5, Income=6
        )

    # 3. Out-of-range GenHlth (> 5)
    with pytest.raises(ValidationError):
        CDCInput(
            HighBP=0, HighChol=1, CholCheck=1, BMI=25.0, Smoker=0, Stroke=0,
            HeartDiseaseorAttack=0, PhysActivity=1, Fruits=1, Veggies=1,
            HvyAlcoholConsump=0, AnyHealthcare=1, NoDocbcCost=0, GenHlth=99,
            MentHlth=0, PhysHlth=0, DiffWalk=0, Sex=1, Age=5, Education=5, Income=6
        )


def test_reports_endpoints(app_and_routes):
    _, routes = app_and_routes
    
    # ANN comparison report
    ann_data = routes["/api/reports/ann"]()
    assert len(ann_data) == 16  # 8 ANNs x 2 datasets

    # Early Stage models report
    es_models = routes["/api/reports/models/{dataset}"]("early_stage")
    assert len(es_models) >= 12

    # CDC models report
    cdc_models = routes["/api/reports/models/{dataset}"]("cdc")
    assert len(cdc_models) >= 11

    # Feature importance report
    fi_data = routes["/api/reports/feature-importance/{dataset}"]("early_stage")
    assert len(fi_data) > 0


def test_frontend_serving(app_and_routes):
    _, routes = app_and_routes
    res_index = routes["/"]()
    # Should be FileResponse serving frontend/index.html
    assert Path(res_index.path).name == "index.html"
    assert Path(res_index.path).exists()
