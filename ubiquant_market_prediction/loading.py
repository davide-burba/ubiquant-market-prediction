import pandas as pd


def load_data(data_path, add_supplemental=False):
    data = pd.read_pickle(f"{data_path}/train.p")
    if add_supplemental:
        supplemental_train = pd.read_pickle(f"{data_path}/supplemental_train.p")
        supplemental_train = supplemental_train[data.columns]
        data = pd.concat([data, supplemental_train])
    return data
