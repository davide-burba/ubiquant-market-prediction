import pandas as pd


def load_data(data_path, add_supplemental=False, tensor_like=False):
    if tensor_like:
        targets = pd.read_pickle(f"{data_path}/targets.p")
        features = pd.read_pickle(f"{data_path}/features.p")
        data = (targets, features)
        if add_supplemental:
            raise NotImplementedError
    else:
        data = pd.read_pickle(f"{data_path}/train.p")
        if add_supplemental:
            supplemental_train = pd.read_pickle(f"{data_path}/supplemental_train.p")
            supplemental_train = supplemental_train[data.columns]
            data = pd.concat([data, supplemental_train])
    return data
