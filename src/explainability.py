"""DiaTrust-NX Explainability & Interpretability Module
Provides global and local model explanations via SHAP and Permutation Importance.
Includes feature rankings, summary plots, and single-patient attribution reports.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import shap


def compute_permutation_importance(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """Computes non-parametric permutation feature importance on validation data."""
    perm = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=-1
    )
    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std
    }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)
    return df


def generate_shap_explanations(
    model: Any,
    X_sample: np.ndarray,
    feature_names: List[str],
    save_plot_path: Optional[str] = None,
    max_display: int = 15
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Generates SHAP values, feature importance rankings, and summary plot for a model.
    
    Parameters:
    -----------
    model : Any
        Trained model (tree-based or general).
    X_sample : np.ndarray
        Background/evaluation feature matrix (e.g. validation subset).
    feature_names : List[str]
        Names of features matching X_sample columns.
    save_plot_path : Optional[str]
        File path to save the SHAP summary plot.
    max_display : int
        Maximum number of top features to display.
        
    Returns:
    --------
    shap_values : np.ndarray
        Array of computed SHAP attribution values.
    importance_df : pd.DataFrame
        Ranked features with mean absolute SHAP scores.
    """
    # Select appropriate explainer
    model_type = type(model).__name__.lower()
    
    try:
        if any(t in model_type for t in ["tree", "forest", "lgbm", "lightgbm", "catboost", "xgb", "gradient"]):
            explainer = shap.TreeExplainer(model)
            shap_obj = explainer(X_sample)
            if hasattr(shap_obj, "values"):
                shap_vals = shap_obj.values
            else:
                shap_vals = shap_obj
        else:
            # Fallback to Kernel or Permutation Explainer with background sample
            background = X_sample[:min(50, len(X_sample))]
            explainer = shap.Explainer(model.predict_proba, background)
            shap_obj = explainer(X_sample[:min(100, len(X_sample))])
            shap_vals = shap_obj.values
    except Exception:
        # Robust fallback using Permutation explainer
        background = X_sample[:min(30, len(X_sample))]
        predict_fn = getattr(model, "predict_proba", model.predict)
        explainer = shap.Explainer(predict_fn, background)
        shap_obj = explainer(X_sample[:min(50, len(X_sample))])
        shap_vals = shap_obj.values

    # If binary classification returns 3D shape (samples, features, 2), extract positive class
    if isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3 and shap_vals.shape[2] == 2:
        shap_vals = shap_vals[:, :, 1]
    elif isinstance(shap_vals, list) and len(shap_vals) == 2:
        shap_vals = shap_vals[1]

    # Compute mean absolute SHAP importance
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    # Save summary plot
    if save_plot_path is not None:
        plt.figure(figsize=(9, 6))
        shap.summary_plot(
            shap_vals,
            X_sample[:len(shap_vals)],
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
        plt.title("SHAP Feature Attribution Summary", fontsize=13, pad=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    return shap_vals, importance_df


def explain_single_patient(
    model: Any,
    x_patient: np.ndarray,
    feature_names: List[str],
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """Generates local feature attribution for a single patient profile.
    
    Returns the top influential factors contributing positively or negatively to the risk estimation.
    """
    x_2d = x_patient.reshape(1, -1)
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer(x_2d).values
        if len(shap_vals.shape) == 3:
            shap_vals = shap_vals[:, :, 1]
    except Exception:
        # Perturbation approximation
        base_prob = model.predict_proba(x_2d)[0, 1]
        shap_vals = np.zeros((1, len(feature_names)))
        for i in range(len(feature_names)):
            x_perturbed = x_2d.copy()
            x_perturbed[0, i] = 0.0
            pert_prob = model.predict_proba(x_perturbed)[0, 1]
            shap_vals[0, i] = base_prob - pert_prob

    vals = shap_vals.flatten()
    sorted_indices = np.argsort(np.abs(vals))[::-1][:top_n]

    explanations = []
    for idx in sorted_indices:
        contrib = float(vals[idx])
        explanations.append({
            "feature": feature_names[idx],
            "value": float(x_patient[idx]),
            "contribution": contrib,
            "direction": "Increases Risk" if contrib > 0 else "Decreases Risk"
        })

    return explanations

