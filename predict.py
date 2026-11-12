"""DiaTrust-NX Inference CLI & Server Entrypoint
Usage:
    python predict.py --serve               # Starts FastAPI inference service & web dashboard on port 8000
    python predict.py --example early_stage # Runs sample prediction for Early Stage model
    python predict.py --example cdc         # Runs sample prediction for CDC model
"""

import argparse
import sys
import json
from pathlib import Path
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.prediction import (
    create_app,
    DiaTrustPredictionEngine,
    EarlyStageInput,
    CDCInput
)


def run_sample_early_stage():
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
    result = engine.predict_early_stage(sample)
    print("\n--- Early Stage Sample Prediction Result ---")
    print(json.dumps(result.model_dump(), indent=4))


def run_sample_cdc():
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
    result = engine.predict_cdc(sample)
    print("\n--- CDC Sample Prediction Result ---")
    print(json.dumps(result.model_dump(), indent=4))


def main():
    parser = argparse.ArgumentParser(description="DiaTrust-NX Inference & API")
    parser.add_argument("--serve", action="store_true", help="Launch FastAPI server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="API host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    parser.add_argument("--example", type=str, choices=["early_stage", "cdc"], help="Run sample prediction")

    args = parser.parse_args()

    if args.serve:
        print(f"Starting DiaTrust-NX API server & Research Dashboard at http://{args.host}:{args.port} ...")
        app = create_app()
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.example == "early_stage":
        run_sample_early_stage()
    elif args.example == "cdc":
        run_sample_cdc()
    else:
        print("Please specify --serve or --example [early_stage|cdc]. Use --help for options.")


if __name__ == "__main__":
    main()
