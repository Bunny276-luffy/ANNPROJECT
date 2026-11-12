"""DiaTrust-NX Duplicate Profile Detection & Stratified Group Splitting Module
Guarantees that identical predictor profiles never cross train, validation, and test splits.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import StratifiedGroupKFold


def compute_profile_groups(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.Series, Dict[str, Any]]:
    """Identifies exact duplicate feature profiles using highly optimized vectorized grouping."""
    group_series = df.groupby(feature_cols, sort=False).ngroup()
    group_series.name = "profile_group_id"
    
    total_rows = len(df)
    unique_profiles = int(group_series.max() + 1) if total_rows > 0 else 0
    counts = group_series.value_counts()
    duplicate_groups = int((counts > 1).sum())
    duplicate_rows = int(counts[counts > 1].sum())
    
    summary = {
        "total_rows": int(total_rows),
        "unique_profiles": int(unique_profiles),
        "duplicate_groups": int(duplicate_groups),
        "duplicate_rows": int(duplicate_rows),
        "max_group_size": int(counts.max()) if len(counts) > 0 else 0,
        "single_instance_profiles": int((counts == 1).sum())
    }
    
    return group_series, summary


def stratified_group_train_val_test_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Splits a dataset into Train, Validation, and Test sets using fast, exact duplicate-aware grouped stratification.
    
    Guarantees:
    1. Zero predictor leakage: Exact duplicate feature profiles are confined entirely to one split.
    2. Target stratification: Proportions of positive and negative classes are closely preserved.
    3. Scalable O(N) performance on datasets with hundreds of thousands of groups.
    4. Deterministic reproducibility across all experiments.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    # 1. Compute profile group IDs
    group_ids, group_summary = compute_profile_groups(df, feature_cols)
    
    # Ensure target is numeric binary
    y = df[target_col].values
    if not np.issubdtype(y.dtype, np.number):
        unique_vals = sorted(list(set(y)))
        val_map = {val: idx for idx, val in enumerate(unique_vals)}
        y_numeric = np.array([val_map[v] for v in y])
    else:
        y_numeric = y.astype(int)

    # 2. Group-level target summary and stratification
    unique_groups = np.unique(group_ids.values)
    group_df = pd.DataFrame({"group": group_ids.values, "target": y_numeric})
    group_stats = group_df.groupby("group").agg(
        target_mean=("target", "mean"),
        size=("target", "size")
    )

    # Strata: purely negative groups, purely positive groups, and mixed duplicate groups
    def assign_stratum(row):
        tm = row["target_mean"]
        if tm == 0.0:
            return 0
        elif tm == 1.0:
            return 1
        else:
            return 2

    strata = group_stats.apply(assign_stratum, axis=1).values

    rng = np.random.RandomState(random_state)
    train_groups_set = set()
    val_groups_set = set()
    test_groups_set = set()

    for s in np.unique(strata):
        s_groups = unique_groups[strata == s]
        shuffled = rng.permutation(s_groups)
        n = len(shuffled)
        n_tr = int(round(n * train_ratio))
        n_va = int(round(n * val_ratio))

        train_groups_set.update(shuffled[:n_tr])
        val_groups_set.update(shuffled[n_tr:n_tr + n_va])
        test_groups_set.update(shuffled[n_tr + n_va:])

    # Assign row indices based on group membership
    grp_vals = group_ids.values
    train_mask = np.isin(grp_vals, list(train_groups_set))
    val_mask = np.isin(grp_vals, list(val_groups_set))
    test_mask = np.isin(grp_vals, list(test_groups_set))

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    # Leakage Verification: Assert zero group overlap
    overlap_tr_va = train_groups_set & val_groups_set
    overlap_tr_te = train_groups_set & test_groups_set
    overlap_va_te = val_groups_set & test_groups_set

    assert len(overlap_tr_va) == 0, f"Leakage detected between train and val: {len(overlap_tr_va)} groups"
    assert len(overlap_tr_te) == 0, f"Leakage detected between train and test: {len(overlap_tr_te)} groups"
    assert len(overlap_va_te) == 0, f"Leakage detected between val and test: {len(overlap_va_te)} groups"

    def split_metrics(subset: pd.DataFrame, groups: set) -> Dict[str, Any]:
        target_counts = subset[target_col].value_counts().to_dict()
        pos_count = target_counts.get(1, target_counts.get("Positive", 0))
        total = len(subset)
        return {
            "samples": total,
            "sample_percentage": round(total / len(df) * 100, 2),
            "unique_profiles": len(groups),
            "class_counts": {str(k): int(v) for k, v in target_counts.items()},
            "positive_prevalence": round(pos_count / total, 4) if total > 0 else 0.0
        }

    split_report = {
        "dataset_summary": group_summary,
        "zero_leakage_verified": True,
        "splits": {
            "train": split_metrics(train_df, train_groups_set),
            "validation": split_metrics(val_df, val_groups_set),
            "test": split_metrics(test_df, test_groups_set)
        }
    }

    return train_df, val_df, test_df, split_report

