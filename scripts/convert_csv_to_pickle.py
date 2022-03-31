import os
import pandas as pd
import argparse
import pathlib

ROOT = f"{pathlib.Path(__file__).parent.resolve()}/../"


def csv_to_pickle(data_path, output_path):

    data_files = [v for v in os.listdir(data_path) if ".csv" in v]

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    dtype_train = {
        "row_id": "str",
        "time_id": "uint16",
        "investment_id": "uint16",
        "target": "float32",
    }
    for i in range(300):
        dtype_train[f"f_{i}"] = "float32"

    for file in data_files:

        input_file_path = f"{data_path}/{file}"
        output_file = file.replace(".csv", ".p")
        output_file_path = f"{output_path}/{output_file}"

        print(f"Converting {input_file_path} to {output_file_path}")

        if "train.csv" in file:
            dtype = dtype_train
        else:
            dtype = None

        pd.to_pickle(pd.read_csv(input_file_path, dtype=dtype), output_file_path)

    print("Done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default=f"{ROOT}/../data/",
        help="Path to csv data (input).",
        type=str,
    )
    parser.add_argument(
        "--output_path",
        default=f"{ROOT}/../data_pickle/",
        help="Path to pickle data (output).",
        type=str,
    )
    args = parser.parse_args()
    csv_to_pickle(args.data_path, args.output_path)
