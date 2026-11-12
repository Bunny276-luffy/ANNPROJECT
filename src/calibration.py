"""DiaTrust-NX Probability Calibration & Uncertainty Quantification Module
Implements Platt scaling, isotonic regression, Brier score auditing, and entropy-based uncertainty estimation.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


def calculate_entropy_uncertainty(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Calculates normalized binary Shannon entropy as an uncertainty indicator.
    
    Returns a score between 0.0 (maximum confidence) and 1.0 (maximum uncertainty at p=0.5).
    """
    p = np.clip(probabilities, eps, 1.0 - eps)
    entropy = - (p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return np.clip(entropy, 0.0, 1.0)


class ProbabilityCalibrator:
    """Calibrates classifier probabilities using validation data via Platt scaling (sigmoid) or Isotonic regression."""
    
    def __init__(self, method: str = "sigmoid"):
        if method not in ["sigmoid", "isotonic"]:
            raise ValueError(f"Method must be 'sigmoid' or 'isotonic', got: {method}")
        self.method = method
        self.calibrated_model_: Optional[CalibratedClassifierCV] = None
        self.brier_before_: float = 0.0
        self.brier_after_: float = 0.0

    def fit(self, base_estimator, X_val: np.ndarray, y_val: np.ndarray):
        """Fits calibration curves strictly on validation data."""
        y_val_arr = np.asarray(y_val, dtype=int)
        
        # Calculate Brier score before calibration
        raw_probs = base_estimator.predict_proba(X_val)[:, 1]
        self.brier_before_ = float(brier_score_loss(y_val_arr, raw_probs))

        # Fit calibrator on frozen base estimator
        try:
            from sklearn.frozen import FrozenEstimator
            self.calibrated_model_ = CalibratedClassifierCV(
                estimator=FrozenEstimator(base_estimator),
                method=self.method
            )
            self.calibrated_model_.fit(X_val, y_val_arr)
        except Exception:
            # Fallback for older or customized estimators
            self.calibrated_model_ = CalibratedClassifierCV(
                estimator=base_estimator,
                method=self.method,
                cv=2
            )
            self.calibrated_model_.fit(X_val, y_val_arr)

        cal_probs = self.calibrated_model_.predict_proba(X_val)[:, 1]
        self.brier_after_ = float(brier_score_loss(y_val_arr, cal_probs))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.calibrated_model_ is None:
            raise RuntimeError("Calibrator has not been fitted.")
        return self.calibrated_model_.predict_proba(X)

    def predict_with_uncertainty(self, X: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns binary predictions, calibrated probabilities, and uncertainty indicators."""
        probs = self.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)
        uncertainty = calculate_entropy_uncertainty(probs)
        return preds, probs, uncertainty

    def get_summary(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "brier_score_before": self.brier_before_,
            "brier_score_after": self.brier_after_,
            "brier_improvement": self.brier_before_ - self.brier_after_
        }

