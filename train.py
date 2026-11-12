"""DiaTrust-NX Main Training CLI
Usage:
    python train.py                     # Runs full pipeline for both datasets
    python train.py --dataset early_stage # Runs Early Stage dataset only
    python train.py --dataset cdc       # Runs CDC dataset only
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.training import run_full_training_pipeline, train_and_evaluate_dataset


def main():
    parser = argparse.ArgumentParser(description="DiaTrust-NX Advanced Model Training & Evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "early_stage", "cdc"],
        help="Target dataset to train on ('all', 'early_stage', 'cdc')"
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip Optuna hyperparameter searches to accelerate training"
    )

    args = parser.parse_args()

    if args.dataset == "all":
        run_full_training_pipeline()
    else:
        train_and_evaluate_dataset(args.dataset, tune_hyperparams=not args.no_tune)


if __name__ == "__main__":
    main()

