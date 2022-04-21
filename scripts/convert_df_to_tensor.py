import pandas as pd
import numpy as np
import os
import sys
import argparse
import pathlib
from tqdm import tqdm

ROOT = f"{pathlib.Path(__file__).parent.resolve()}/../"


def df_to_tensor(data_path, output_path):
    train = pd.read_pickle(f"{data_path}/train.p")
    timesteps = sorted(train.time_id.unique())
    investment_ids = sorted(train.investment_id.unique())

    train = train.drop(columns="row_id").set_index(["investment_id", "time_id"])

    targets = []
    features = []
    for investment_id in tqdm(investment_ids):
        df = train.loc[investment_id].reindex(timesteps)

        targets.append(np.expand_dims(df.target.values, 0))
        features.append(np.expand_dims(df.drop(columns="target").values, 0))

    targets = np.concatenate(targets, axis=0)
    features = np.concatenate(features, axis=0)

    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    pd.to_pickle(targets, f"{output_path}/targets.p")
    pd.to_pickle(features, f"{output_path}/features.p")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default=f"{ROOT}/../data_pickle/",
        help="Path to pickle data (input).",
        type=str,
    )
    parser.add_argument(
        "--output_path",
        default=f"{ROOT}/../data_tensors/",
        help="Path to tensor data (output).",
        type=str,
    )
    args = parser.parse_args()
    df_to_tensor(args.data_path, args.output_path)
