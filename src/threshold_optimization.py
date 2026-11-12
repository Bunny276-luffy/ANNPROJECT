"""DiaTrust-NX Decision Threshold Optimization Module
Determines optimal classification decision thresholds strictly on validation data.
Addresses class imbalance and maximizes clinical utility without test data leakage.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score


def optimize_decision_threshold(
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray,
    objective: str = "f1",
    min_sensitivity: float = 0.0,
    steps: int = 100
) -> Tuple[float, Dict[str, Any]]:
    """Sweeps candidate decision thresholds on validation data to maximize the selected objective.
    
    Parameters:
    -----------
    y_val_true : np.ndarray
        Ground truth validation labels.
    y_val_prob : np.ndarray
        Predicted validation probabilities for class 1.
    objective : str
        Optimization criterion: 'f1', 'balanced_accuracy', 'youden', or 'clinical_recall'.
    min_sensitivity : float
        Optional minimum sensitivity constraint (e.g. 0.70).
    steps : int
        Number of threshold evaluation points between 0.05 and 0.95.
        
    Returns:
    --------
    best_threshold : float
        Optimal threshold found on validation set.
    optimization_summary : Dict[str, Any]
        Details including best score, metric trajectories, and candidate evaluations.
    """
    thresholds = np.linspace(0.05, 0.95, steps)
    best_score = -1.0
    best_threshold = 0.50
    history: List[Dict[str, float]] = []

    y_val_true = np.asarray(y_val_true, dtype=int)
    y_val_prob = np.asarray(y_val_prob, dtype=float)

    for th in thresholds:
        preds = (y_val_prob >= th).astype(int)
        rec = float(recall_score(y_val_true, preds, zero_division=0))
        prec = float(precision_score(y_val_true, preds, zero_division=0))
        f1 = float(f1_score(y_val_true, preds, zero_division=0))
        bal_acc = float(balanced_accuracy_score(y_val_true, preds))

        # Youden's Index J = Sensitivity + Specificity - 1
        tn = np.sum((preds == 0) & (y_val_true == 0))
        fp = np.sum((preds == 1) & (y_val_true == 0))
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        youden = rec + spec - 1.0

        if objective == "f1":
            score = f1
        elif objective == "balanced_accuracy":
            score = bal_acc
        elif objective == "youden":
            score = youden
        elif objective == "clinical_recall":
            # Joint score penalizing recall drops below 0.70
            score = (f1 + bal_acc) / 2.0 if rec >= min_sensitivity else -1.0
        else:
            raise ValueError(f"Unknown objective: {objective}")

        if min_sensitivity > 0.0 and rec < min_sensitivity:
            score = -1.0

        history.append({
            "threshold": float(th),
            "score": float(score),
            "f1": float(f1),
            "balanced_accuracy": float(bal_acc),
            "recall": float(rec),
            "precision": float(prec),
            "specificity": float(spec)
        })

        if score > best_score:
            best_score = score
            best_threshold = float(th)

    # Fallback to 0.5 if no candidate satisfied constraint
    if best_score == -1.0:
        best_threshold = 0.50
        best_score = float(f1_score(y_val_true, (y_val_prob >= 0.50).astype(int), zero_division=0))

    return best_threshold, {
        "best_threshold": float(best_threshold),
        "best_score": float(best_score),
        "objective": objective,
        "history": history
    }

