import datetime as dt
import os

def create_run_directory(output_path):
    run_dir = dt.datetime.now().strftime("run_%Y_%m_%d_%Hh_%Mm_%Ss")
    run_dir = f"{output_path}/{run_dir}"

    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    else:
        raise ValueError(f"Directory {run_dir} already exists")
    return run_dir