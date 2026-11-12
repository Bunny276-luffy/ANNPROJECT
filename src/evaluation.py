"""DiaTrust-NX Evaluation & Metrics Suite
Computes comprehensive clinical and statistical metrics and generates publication-grade visualizations.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve


# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def calculate_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Calculates comprehensive classification, clinical, and ranking metrics.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1).
    y_pred_prob : np.ndarray
        Predicted probabilities of the positive class (class 1).
    threshold : float
        Decision threshold used for binary classification.
        
    Returns:
    --------
    metrics : Dict[str, Any]
        Dictionary containing all metrics and confusion counts.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_prob = np.clip(np.asarray(y_pred_prob, dtype=float), 0.0, 1.0)
    y_pred = (y_pred_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_pred_prob))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_pred_prob))
    except Exception:
        pr_auc = float(np.mean(y_true))

    brier = float(brier_score_loss(y_true, y_pred_prob))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "Accuracy": acc,
        "Balanced Accuracy": bal_acc,
        "Precision": prec,
        "Recall": rec,
        "Sensitivity": rec,
        "Specificity": spec,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Brier Score": brier,
        "Threshold": float(threshold),
        "Confusion Matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        }
    }


def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: str, labels: List[str] = None):
    """Generates and saves an annotated confusion matrix heatmap."""
    if labels is None:
        labels = ["Negative", "Positive"]
        
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title(title, fontsize=13, pad=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.ylabel("True Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, title: str, save_path: str):
    """Generates and saves an annotated ROC curve with diagonal reference."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"ROC Curve (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="#888888", lw=1.5, linestyle="--", label="Random Classifier")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, pad=12, fontweight="bold")
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, title: str, save_path: str):
    """Generates and saves a Precision-Recall curve with baseline prevalence reference."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_val = average_precision_score(y_true, y_prob)
    prevalence = np.mean(y_true)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="#2ca02c", lw=2.5, label=f"PR Curve (PR-AUC = {pr_auc_val:.4f})")
    plt.axhline(y=prevalence, color="#888888", linestyle="--", lw=1.5, label=f"Prevalence ({prevalence:.3f})")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    plt.ylabel("Precision (PPV)", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, pad=12, fontweight="bold")
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_calibration_curve_fig(y_true: np.ndarray, y_prob: np.ndarray, title: str, save_path: str):
    """Generates and saves a reliability calibration curve."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    brier = brier_score_loss(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", color="#d62728", lw=2, label=f"Model (Brier = {brier:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability", fontsize=11, fontweight="bold")
    plt.ylabel("Fraction of Positives", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, pad=12, fontweight="bold")
    plt.legend(loc="upper left", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_history(history: Dict[str, List[float]], title: str, save_path: str):
    """Plots training vs validation accuracy and loss for ANN models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    if "val_loss" in history and history["val_loss"]:
        ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#d62728", lw=2, linestyle="--")
    ax1.set_title("Loss vs Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.legend(frameon=True)

    ax2.plot(epochs, history["train_acc"], label="Train Accuracy", color="#2ca02c", lw=2)
    if "val_acc" in history and history["val_acc"]:
        ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="#ff7f0e", lw=2, linestyle="--")
    ax2.set_title("Accuracy vs Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.legend(frameon=True)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(importances: np.ndarray, feature_names: List[str], title: str, save_path: str, top_n: int = 15):
    """Generates and saves a horizontal bar chart of top feature importances."""
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices][::-1]
    top_scores = [importances[i] for i in indices][::-1]

    plt.figure(figsize=(8, 6))
    bars = plt.barh(top_features, top_scores, color="#3b528b", edgecolor="#1e2c4d", height=0.65)
    plt.xlabel("Importance Score", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, pad=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

