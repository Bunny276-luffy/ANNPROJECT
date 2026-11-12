"""DiaTrust-NX Model Factory & Hyperparameter Optimization
Instantiates classical ML algorithms, ensembles, and manages Optuna hyperparameter searches.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import optuna

from src.ann_models import PyTorchANNClassifier


# Suppress Optuna logging noise
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_base_models(
    pos_weight: float = 1.0,
    random_state: int = 42,
    include_ann: bool = True
) -> Dict[str, Any]:
    """Instantiates the candidate model suite with calibrated class weights and hyperparameters."""
    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1500,
            C=1.0,
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            random_state=random_state
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter=150,
            learning_rate=0.08,
            random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=5,
            tree_method="hist",
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=250,
            learning_rate=0.08,
            depth=5,
            auto_class_weights="Balanced",
            verbose=0,
            random_seed=random_state
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=6,
            class_weight="balanced",
            verbose=-1,
            random_state=random_state,
            n_jobs=-1
        ),
        "Support Vector Machine": SVC(
            C=1.5,
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=9,
            weights="distance",
            n_jobs=-1
        )
    }

    if include_ann:
        models["Optimized ANN"] = PyTorchANNClassifier(
            arch="optimized",
            lr=1e-3,
            max_epochs=50,
            patience=8,
            use_class_weight=True,
            random_state=random_state
        )

    return models


def create_soft_voting_ensemble(models: Dict[str, Any], selected_names: List[str]) -> VotingClassifier:
    """Builds a Soft Voting Ensemble combining diverse high-performing models."""
    estimators = [(name.lower().replace(" ", "_"), models[name]) for name in selected_names if name in models]
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


def create_stacking_ensemble(models: Dict[str, Any], selected_names: List[str], random_state: int = 42) -> StackingClassifier:
    """Builds a Stacking Classifier with Logistic Regression meta-learner using out-of-fold probabilities."""
    estimators = [(name.lower().replace(" ", "_"), models[name]) for name in selected_names if name in models]
    meta_learner = LogisticRegression(class_weight="balanced", random_state=random_state)
    return StackingClassifier(estimators=estimators, final_estimator=meta_learner, cv=3, n_jobs=-1)


def tune_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 20,
    random_state: int = 42
) -> LGBMClassifier:
    """Optimizes LightGBM hyperparameters using Optuna strictly on validation ROC-AUC."""
    from sklearn.metrics import roc_auc_score

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "class_weight": "balanced",
            "verbose": -1,
            "random_state": random_state,
            "n_jobs": -1
        }
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, probs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_params.update({"class_weight": "balanced", "verbose": -1, "random_state": random_state, "n_jobs": -1})
    best_model = LGBMClassifier(**best_params)
    best_model.fit(X_train, y_train)
    return best_model


def tune_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 20,
    random_state: int = 42
) -> CatBoostClassifier:
    """Optimizes CatBoost hyperparameters using Optuna strictly on validation ROC-AUC."""
    from sklearn.metrics import roc_auc_score

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 150, 350),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "auto_class_weights": "Balanced",
            "verbose": 0,
            "random_seed": random_state
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, probs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params.update({"auto_class_weights": "Balanced", "verbose": 0, "random_seed": random_state})
    best_model = CatBoostClassifier(**best_params)
    best_model.fit(X_train, y_train)
    return best_model

