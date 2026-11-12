"""DiaTrust-NX Complete Training & Experiment Pipeline
Orchestrates duplicate-aware splitting, feature preprocessing, ANN benchmarking,
classical ML model comparisons, hyperparameter optimization, threshold tuning,
probability calibration, explainability, test evaluation, and artifact deployment.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

from configs.config import (
    DATASET_CONFIG,
    EARLY_STAGE_FEATURES,
    CDC_FEATURES,
    DEPLOYMENT_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
    RANDOM_SEED,
    MEDICAL_DISCLAIMER
)
from src.data_loader import load_early_stage_data, load_cdc_data
from src.duplicate_detection import stratified_group_train_val_test_split
from src.preprocessing import EarlyStageFeatureTransformer, CDCFeatureTransformer
from src.ann_models import PyTorchANNClassifier
from src.model_factory import (
    get_base_models,
    create_soft_voting_ensemble,
    create_stacking_ensemble,
    tune_lightgbm,
    tune_catboost
)
from src.threshold_optimization import optimize_decision_threshold
from src.calibration import ProbabilityCalibrator
from src.evaluation import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_calibration_curve_fig,
    plot_training_history,
    plot_feature_importance
)
from src.explainability import generate_shap_explanations, compute_permutation_importance


def run_ann_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    dataset_name: str,
    max_epochs: int = 35,
    batch_size: int = 128
) -> Tuple[pd.DataFrame, Dict[str, Any], PyTorchANNClassifier]:
    """Trains and compares all 8 PyTorch ANN architectures on the dataset."""
    arch_list = [
        ("ANN Baseline", "baseline"),
        ("ANN Deep", "deep"),
        ("ANN Wide", "wide"),
        ("ANN Regularized", "regularized"),
        ("ANN Batch Normalization", "batch_norm"),
        ("ANN Dropout", "dropout"),
        ("Residual MLP", "residual_mlp"),
        ("Optimized ANN", "optimized")
    ]

    records = []
    histories = {}
    best_ann = None
    best_val_auc = -1.0

    # For datasets with >35k samples (e.g. CDC), benchmark ANNs on a representative 35,000 stratified sample
    # while validating on the full validation set (37k+ rows) to maximize throughput and stability.
    if len(X_train) > 35000:
        ann_sub_idx = np.random.RandomState(RANDOM_SEED).choice(len(X_train), size=35000, replace=False)
        X_ann_tr = X_train[ann_sub_idx]
        y_ann_tr = y_train[ann_sub_idx]
    else:
        X_ann_tr = X_train
        y_ann_tr = y_train

    print(f"--- Running 8 ANN Architectures Benchmark for {dataset_name} ---", flush=True)
    for display_name, arch_key in arch_list:
        print(f"  -> [{display_name}] Training...", flush=True)
        t0 = time.time()
        clf = PyTorchANNClassifier(
            arch=arch_key,
            lr=1e-3,
            max_epochs=max_epochs,
            batch_size=batch_size,
            patience=6,
            use_class_weight=True,
            use_lr_scheduler=True,
            random_state=RANDOM_SEED
        )
        clf.fit(X_ann_tr, y_ann_tr, eval_set=(X_val, y_val))
        t_elapsed = time.time() - t0

        train_prob = clf.predict_proba(X_ann_tr)[:, 1]
        val_prob = clf.predict_proba(X_val)[:, 1]

        train_metrics = calculate_metrics(y_ann_tr, train_prob, threshold=0.50)
        val_metrics = calculate_metrics(y_val, val_prob, threshold=0.50)

        records.append({
            "Dataset": dataset_name,
            "ANN Model": display_name,
            "Training Accuracy": train_metrics["Accuracy"],
            "Training Loss": clf.history_["train_loss"][-1] if clf.history_["train_loss"] else 0.0,
            "Validation Accuracy": val_metrics["Accuracy"],
            "Validation Balanced Accuracy": val_metrics["Balanced Accuracy"],
            "Validation Recall": val_metrics["Recall"],
            "Validation F1": val_metrics["F1"],
            "Validation ROC-AUC": val_metrics["ROC-AUC"],
            "Validation PR-AUC": val_metrics["PR-AUC"],
            "Best Epoch": clf.best_epoch_,
            "Training Time (s)": round(t_elapsed, 2)
        })

        histories[display_name] = clf.history_

        if val_metrics["ROC-AUC"] > best_val_auc:
            best_val_auc = val_metrics["ROC-AUC"]
            best_ann = clf

    ann_df = pd.DataFrame(records).sort_values(by="Validation ROC-AUC", ascending=False).reset_index(drop=True)
    return ann_df, histories, best_ann


def train_and_evaluate_dataset(
    dataset_key: str,
    tune_hyperparams: bool = True
) -> Dict[str, Any]:
    """Complete end-to-end training and evaluation for a specific dataset ('early_stage' or 'cdc')."""
    is_early_stage = (dataset_key == "early_stage")
    display_title = "Early Stage" if is_early_stage else "CDC"
    print(f"\n=======================================================")
    print(f"      STARTING PIPELINE FOR: {display_title}")
    print(f"=======================================================")

    # 1. Load Data
    if is_early_stage:
        X_raw, y_raw, data_info = load_early_stage_data()
        feature_cols = EARLY_STAGE_FEATURES
        transformer = EarlyStageFeatureTransformer(include_engineered_features=True)
        train_r, val_r, test_r = 0.65, 0.17, 0.18
        batch_size = 32
        ann_epochs = 40
    else:
        X_raw, y_raw, data_info = load_cdc_data()
        feature_cols = CDC_FEATURES
        transformer = CDCFeatureTransformer(include_engineered_features=True)
        train_r, val_r, test_r = 0.70, 0.15, 0.15
        batch_size = 1024
        ann_epochs = 12

    # 2. Stratified Group Splitting (Duplicate-Aware)
    df_combined = pd.concat([X_raw, y_raw], axis=1)
    train_df, val_df, test_df, split_report = stratified_group_train_val_test_split(
        df_combined,
        feature_cols=feature_cols,
        target_col="diabetes_risk",
        train_ratio=train_r,
        val_ratio=val_r,
        test_ratio=test_r,
        random_state=RANDOM_SEED
    )

    X_train_raw = train_df[feature_cols].copy()
    y_train = train_df["diabetes_risk"].values.astype(int)
    X_val_raw = val_df[feature_cols].copy()
    y_val = val_df["diabetes_risk"].values.astype(int)
    X_test_raw = test_df[feature_cols].copy()
    y_test = test_df["diabetes_risk"].values.astype(int)

    print(f"Split completed: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 3. Fit Preprocessing exclusively on Training data
    transformer.fit(X_train_raw)
    X_train = transformer.transform(X_train_raw)
    X_val = transformer.transform(X_val_raw)
    X_test = transformer.transform(X_test_raw)
    feature_names = transformer.get_feature_names_out()

    # Calculate class weighting ratio
    pos_ratio = (np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))

    # 4. ANN Architectures Benchmark
    ann_df, ann_histories, best_ann = run_ann_benchmark(
        X_train, y_train, X_val, y_val,
        dataset_name=display_title,
        max_epochs=ann_epochs,
        batch_size=batch_size
    )

    # Plot ANN curves for best architecture
    best_ann_name = ann_df.iloc[0]["ANN Model"]
    best_history = ann_histories[best_ann_name]
    plot_training_history(
        best_history,
        title=f"{display_title} - {best_ann_name} Learning Trajectory",
        save_path=str(PLOTS_DIR / f"{dataset_key}_ann_learning_curve.png")
    )

    # 5. Classical ML and Ensemble Suite
    base_models = get_base_models(pos_weight=pos_ratio, random_state=RANDOM_SEED, include_ann=False)
    # Inject best ANN
    base_models["Optimized ANN"] = best_ann

    # Optional Optuna Tuning for Tree Models
    if tune_hyperparams:
        print("Tuning LightGBM and CatBoost with Optuna...")
        try:
            tuned_lgbm = tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=12, random_state=RANDOM_SEED)
            base_models["LightGBM (Tuned)"] = tuned_lgbm
        except Exception as e:
            print(f"LightGBM tuning notice: {e}")

        try:
            tuned_cb = tune_catboost(X_train, y_train, X_val, y_val, n_trials=12, random_state=RANDOM_SEED)
            base_models["CatBoost (Tuned)"] = tuned_cb
        except Exception as e:
            print(f"CatBoost tuning notice: {e}")

    # Build Soft Voting Ensemble
    voting_candidates = [m for m in ["CatBoost (Tuned)", "CatBoost", "LightGBM (Tuned)", "LightGBM", "Optimized ANN", "Random Forest"] if m in base_models][:3]
    soft_vote = create_soft_voting_ensemble(base_models, voting_candidates)
    base_models["Soft Voting Ensemble"] = soft_vote

    # Fit all models and evaluate on Validation Set
    model_experiment_records = []
    fitted_models = {}
    val_probs_dict = {}

    print(f"Fitting and evaluating candidate models for {display_title}...", flush=True)
    for model_name, model in base_models.items():
        print(f"  -> Fitting {model_name}...", flush=True)
        t0 = time.time()
        try:
            if "ann" in model_name.lower():
                fitted_model = model
            elif len(X_train) > 20000 and model_name in ["Support Vector Machine", "K-Nearest Neighbors"]:
                sub_idx = np.random.RandomState(RANDOM_SEED).choice(len(X_train), size=3000, replace=False)
                model.fit(X_train[sub_idx], y_train[sub_idx])
                fitted_model = model
            else:
                model.fit(X_train, y_train)
                fitted_model = model
        except Exception as err:
            print(f"Error fitting {model_name}: {err}")
            continue

        fit_time = time.time() - t0
        fitted_models[model_name] = fitted_model

        # Train and Validation probabilities
        if len(X_train) > 30000 and model_name in ["Support Vector Machine", "K-Nearest Neighbors"]:
            train_eval_idx = np.random.RandomState(RANDOM_SEED).choice(len(X_train), size=5000, replace=False)
            train_prob = np.zeros(len(y_train))
            train_prob[train_eval_idx] = fitted_model.predict_proba(X_train[train_eval_idx])[:, 1]
            train_eval_mask = train_eval_idx
        else:
            train_prob = fitted_model.predict_proba(X_train)[:, 1]
            train_eval_mask = None
        val_prob = fitted_model.predict_proba(X_val)[:, 1]
        val_probs_dict[model_name] = val_prob

        # Threshold optimization strictly on Validation data
        opt_thresh, th_summary = optimize_decision_threshold(
            y_val, val_prob,
            objective="f1" if is_early_stage else "balanced_accuracy"
        )

        if train_eval_mask is not None:
            train_metrics = calculate_metrics(y_train[train_eval_mask], train_prob[train_eval_mask], threshold=opt_thresh)
        else:
            train_metrics = calculate_metrics(y_train, train_prob, threshold=opt_thresh)
        val_metrics = calculate_metrics(y_val, val_prob, threshold=opt_thresh)

        model_experiment_records.append({
            "Dataset": display_title,
            "Model": model_name,
            "Training Accuracy": train_metrics["Accuracy"],
            "Training ROC-AUC": train_metrics["ROC-AUC"],
            "Validation Threshold": opt_thresh,
            "Validation Accuracy": val_metrics["Accuracy"],
            "Validation Balanced Accuracy": val_metrics["Balanced Accuracy"],
            "Validation Precision": val_metrics["Precision"],
            "Validation Recall": val_metrics["Recall"],
            "Validation F1": val_metrics["F1"],
            "Validation ROC-AUC": val_metrics["ROC-AUC"],
            "Validation PR-AUC": val_metrics["PR-AUC"],
            "Training Time (seconds)": round(fit_time, 2)
        })

    exp_df = pd.DataFrame(model_experiment_records).sort_values(
        by="Validation Balanced Accuracy" if not is_early_stage else "Validation F1",
        ascending=False
    ).reset_index(drop=True)

    # 6. Model Selection
    selected_model_name = exp_df.iloc[0]["Model"]
    selected_model = fitted_models[selected_model_name]
    selected_threshold = float(exp_df.iloc[0]["Validation Threshold"])
    print(f"Selected Best Model for {display_title}: {selected_model_name} (Threshold: {selected_threshold:.4f})")

    # 7. Probability Calibration on Validation Data
    calibrator = ProbabilityCalibrator(method="sigmoid")
    calibrator.fit(selected_model, X_val, y_val)
    cal_summary = calibrator.get_summary()

    # 8. Unseen Frozen Test Set Final Evaluation
    train_probs_final = selected_model.predict_proba(X_train)[:, 1]
    val_probs_final = selected_model.predict_proba(X_val)[:, 1]
    test_probs_final = selected_model.predict_proba(X_test)[:, 1]

    final_train_m = calculate_metrics(y_train, train_probs_final, threshold=selected_threshold)
    final_val_m = calculate_metrics(y_val, val_probs_final, threshold=selected_threshold)
    final_test_m = calculate_metrics(y_test, test_probs_final, threshold=selected_threshold)

    perf_df = pd.DataFrame([
        {"Split": "Training", **{k: v for k, v in final_train_m.items() if k not in ["Confusion Matrix", "Threshold"]}},
        {"Split": "Validation", **{k: v for k, v in final_val_m.items() if k not in ["Confusion Matrix", "Threshold"]}},
        {"Split": "Test", **{k: v for k, v in final_test_m.items() if k not in ["Confusion Matrix", "Threshold"]}}
    ])

    # 9. Publication Visualizations
    cm_test = confusion_matrix(y_test, (test_probs_final >= selected_threshold).astype(int), labels=[0, 1])
    plot_confusion_matrix(
        cm_test,
        title=f"{display_title} - Test Confusion Matrix ({selected_model_name})",
        save_path=str(PLOTS_DIR / f"{dataset_key}_confusion_matrix.png")
    )

    plot_roc_curve(
        y_test, test_probs_final,
        title=f"{display_title} - Test ROC Curve",
        save_path=str(PLOTS_DIR / f"{dataset_key}_roc.png")
    )

    plot_pr_curve(
        y_test, test_probs_final,
        title=f"{display_title} - Test Precision-Recall Curve",
        save_path=str(PLOTS_DIR / f"{dataset_key}_pr.png")
    )

    plot_calibration_curve_fig(
        y_test, test_probs_final,
        title=f"{display_title} - Reliability Calibration Curve",
        save_path=str(PLOTS_DIR / f"{dataset_key}_calibration.png")
    )

    # 10. Explainability & Feature Importance
    shap_sample = X_val[:min(200, len(X_val))]
    try:
        _, shap_importance_df = generate_shap_explanations(
            selected_model,
            shap_sample,
            feature_names=feature_names,
            save_plot_path=str(PLOTS_DIR / f"{dataset_key}_shap_summary.png")
        )
        shap_importance_df.to_csv(REPORTS_DIR / f"{dataset_key}_feature_importance.csv", index=False)
    except Exception as e:
        print(f"SHAP explanation notice: {e}")
        # Fallback permutation importance
        shap_importance_df = compute_permutation_importance(selected_model, X_val, y_val, feature_names)
        shap_importance_df.to_csv(REPORTS_DIR / f"{dataset_key}_feature_importance.csv", index=False)

    # Feature Importance Plot
    top_features = shap_importance_df.head(15)
    plot_feature_importance(
        importances=top_features["mean_abs_shap"].values if "mean_abs_shap" in top_features else top_features["importance_mean"].values,
        feature_names=top_features["feature"].tolist(),
        title=f"{display_title} - Most Influential Features",
        save_path=str(PLOTS_DIR / f"{dataset_key}_feature_importance.png")
    )

    # 11. Save Reports to CSV
    exp_df.to_csv(REPORTS_DIR / f"{dataset_key}_all_model_experiments.csv", index=False)
    perf_df.to_csv(REPORTS_DIR / f"{dataset_key}_final_performance.csv", index=False)

    # 12. Save Deployment Artifacts
    artifact_payload = {
        "model": selected_model,
        "preprocessor": transformer,
        "calibrator": calibrator,
        "decision_threshold": selected_threshold,
        "feature_names": feature_names,
        "dataset": display_title
    }
    model_save_path = DEPLOYMENT_DIR / f"diatrust_nx_{dataset_key}_final.joblib"
    joblib.dump(artifact_payload, model_save_path)

    metadata_payload = {
        "project": "DiaTrust-NX",
        "dataset": display_title,
        "selected_model": selected_model_name,
        "decision_threshold": float(selected_threshold),
        "input_features": feature_cols,
        "engineered_features": feature_names,
        "training_metrics": final_train_m,
        "validation_metrics": final_val_m,
        "test_metrics": final_test_m,
        "calibration_summary": cal_summary,
        "duplicate_aware_split": True,
        "split_statistics": split_report,
        "medical_notice": MEDICAL_DISCLAIMER
    }
    meta_save_path = DEPLOYMENT_DIR / f"diatrust_nx_{dataset_key}_metadata.json"
    with open(meta_save_path, "w") as f:
        json.dump(metadata_payload, f, indent=4)

    print(f"Artifacts saved to {model_save_path} and {meta_save_path}")

    return {
        "dataset": display_title,
        "ann_df": ann_df,
        "exp_df": exp_df,
        "perf_df": perf_df,
        "selected_model": selected_model_name,
        "threshold": selected_threshold,
        "test_metrics": final_test_m,
        "split_report": split_report
    }


def run_full_training_pipeline() -> Dict[str, Any]:
    """Runs the complete training and experimentation pipeline across both datasets."""
    print("=================================================================")
    print("      DIATRUST-NX ADVANCED MODEL RECONSTRUCTION & TRAINING")
    print("=================================================================")

    # 1. Early Stage
    res_es = train_and_evaluate_dataset("early_stage", tune_hyperparams=True)

    # 2. CDC
    res_cdc = train_and_evaluate_dataset("cdc", tune_hyperparams=True)

    # 3. Save Combined ANN Report
    combined_ann_df = pd.concat([res_es["ann_df"], res_cdc["ann_df"]], axis=0).reset_index(drop=True)
    combined_ann_df.to_csv(REPORTS_DIR / "DIATRUST_NX_ANN_COMPARISON.csv", index=False)

    # 4. Save Final Dataset Comparison Report
    comparison_rows = [
        {
            "Dataset": "UCI Early Stage",
            "Selected Model": res_es["selected_model"],
            "Training Accuracy": res_es["perf_df"].loc[res_es["perf_df"]["Split"] == "Training", "Accuracy"].values[0],
            "Validation Accuracy": res_es["perf_df"].loc[res_es["perf_df"]["Split"] == "Validation", "Accuracy"].values[0],
            "Test Accuracy": res_es["test_metrics"]["Accuracy"],
            "Test Balanced Accuracy": res_es["test_metrics"]["Balanced Accuracy"],
            "Test Precision": res_es["test_metrics"]["Precision"],
            "Test Recall": res_es["test_metrics"]["Recall"],
            "Test F1": res_es["test_metrics"]["F1"],
            "Test ROC-AUC": res_es["test_metrics"]["ROC-AUC"],
            "Test PR-AUC": res_es["test_metrics"]["PR-AUC"]
        },
        {
            "Dataset": "UCI CDC",
            "Selected Model": res_cdc["selected_model"],
            "Training Accuracy": res_cdc["perf_df"].loc[res_cdc["perf_df"]["Split"] == "Training", "Accuracy"].values[0],
            "Validation Accuracy": res_cdc["perf_df"].loc[res_cdc["perf_df"]["Split"] == "Validation", "Accuracy"].values[0],
            "Test Accuracy": res_cdc["test_metrics"]["Accuracy"],
            "Test Balanced Accuracy": res_cdc["test_metrics"]["Balanced Accuracy"],
            "Test Precision": res_cdc["test_metrics"]["Precision"],
            "Test Recall": res_cdc["test_metrics"]["Recall"],
            "Test F1": res_cdc["test_metrics"]["F1"],
            "Test ROC-AUC": res_cdc["test_metrics"]["ROC-AUC"],
            "Test PR-AUC": res_cdc["test_metrics"]["PR-AUC"]
        }
    ]
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(REPORTS_DIR / "DIATRUST_NX_FINAL_DATASET_COMPARISON.csv", index=False)

    # 5. Save Complete Master JSON Report
    master_report = {
        "project": "DiaTrust-NX",
        "title": "Adaptive Neuro-Ensemble Framework for Explainable, Calibrated and Uncertainty-Aware Diabetes Risk Prediction",
        "seed": RANDOM_SEED,
        "datasets": {
            "Early Stage": {
                "selected_model": res_es["selected_model"],
                "threshold": res_es["threshold"],
                "split_details": res_es["split_report"],
                "test_metrics": res_es["test_metrics"]
            },
            "CDC": {
                "selected_model": res_cdc["selected_model"],
                "threshold": res_cdc["threshold"],
                "split_details": res_cdc["split_report"],
                "test_metrics": res_cdc["test_metrics"]
            }
        },
        "evaluation_protocol": (
            "Exact duplicate predictor profiles were isolated into discrete groups. "
            "Zero feature profile leakage across Train, Validation, and Test. "
            "All transformers and scalers fitted strictly on training data. "
            "Thresholds and calibrations tuned exclusively on validation data. "
            "Final metrics computed on untouched test sets."
        ),
        "medical_disclaimer": MEDICAL_DISCLAIMER
    }
    with open(REPORTS_DIR / "DIATRUST_NX_COMPLETE_REPORT.json", "w") as f:
        json.dump(master_report, f, indent=4)

    print("\nTraining pipeline finished successfully! All reports, plots, and models updated.")
    return master_report

