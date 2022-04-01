import argparse
import mlflow
import yaml
import pathlib
import sys
import pandas as pd

ROOT = f"{pathlib.Path(__file__).parent.resolve()}/../"
sys.path.insert(0, f"{ROOT}/ubiquant_market_prediction/")

from utils import create_run_directory
from config import ValidationConfig
from loading import load_data
from models import get_model
from preprocessing import get_preprocessor
from validation import TimeCrossValidator


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation_config",
        default=None,
        help="Path to yaml config file.",
        type=str,
    )

    parser.add_argument(
        "--data_path",
        default=f"{ROOT}/../data_pickle/",
        help="Path to data converted to pickle.",
        type=str,
    )

    parser.add_argument(
        "--output_path",
        default=f"{ROOT}/../experiments/",
        help="Path to output directory (will be created if it does not exist).",
        type=str,
    )
    parser.add_argument(
        "--mlflow_runs_path",
        default=f"{ROOT}/../mlruns.db",
        help="Path to sql-lite db to store mlflow runs",
        type=str,
    )
    parser.add_argument(
        "--mlflow_experiment",
        default="run_validation",
        help="Name of mlflow experiment",
        type=str,
    )
    args = parser.parse_args()

    # load config
    if args.validation_config is None:
        config_dict = {}
    else:
        with open(args.validation_config, "r") as f:
            config_dict = yaml.safe_load(f)
    config = ValidationConfig(config_dict)

    # data
    data = load_data(args.data_path, **config["loading"])

    # set model and preprocessor
    model = get_model(config["model"]["model_type"], config["model"]["model_args"])
    preprocessor = get_preprocessor(
        config["preprocessing"]["preprocessor_type"],
        config["preprocessing"]["preprocessor_args"],
    )

    # run validation
    validator = TimeCrossValidator(**config["validator_args"])
    scores, preds = validator.run(data, model, preprocessor)

    # save output
    run_dir = create_run_directory(args.output_path)
    print(f"Saving output at {run_dir}")

    with open(f"{run_dir}/scores.yml", "w") as f:
        yaml.safe_dump(scores, f)

    pd.to_pickle(preds, f"{run_dir}/preds.p")

    with open(f"{run_dir}/validation_config.yml", "w") as f:
        yaml.safe_dump(config_dict, f)

    # log to mlflow
    mlflow.set_tracking_uri(f"sqlite:///{args.mlflow_runs_path}")
    mlflow.set_experiment(args.mlflow_experiment)
    mlflow.log_param("run_dir", run_dir)
    mlflow.log_param("model_type", config["model"]["model_type"])
    mlflow.log_params(config["model"]["model_args"])
    mlflow.log_param("preprocessor_type", config["preprocessing"]["preprocessor_type"])
    mlflow.log_params(config["preprocessing"]["preprocessor_args"])
    for mode in ["valid", "train"]:
        mlflow.log_metric(f"score_{mode}", scores[f"score_{mode}"])
        for step, score in enumerate(scores[f"cv_scores_{mode}"]):
            mlflow.log_metric(f"cv_scores_{mode}", score, step + 1)
