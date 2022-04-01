from abc import abstractclassmethod


def get_preprocessor(preprocessor_type, preprocessor_args):
    preprocessors = {
        "naive": NaivePreprocessor,
    }
    return preprocessors[preprocessor_type.lower()](**preprocessor_args)


class BasePreprocessor:
    @abstractclassmethod
    def run(self, train_data, valid_data):
        return


class NaivePreprocessor(BasePreprocessor):
    def __init__(self, cols_to_drop=[]):
        self.cols_to_drop = cols_to_drop

    def run(self, train_data, valid_data):
        x_train = train_data.drop(columns=["target", "row_id"] + self.cols_to_drop)
        x_valid = valid_data.drop(columns=["target", "row_id"] + self.cols_to_drop)

        timesteps_train = train_data.time_id.values
        timesteps_valid = valid_data.time_id.values

        y_train = train_data.target.values
        y_valid = valid_data.target.values

        return x_train, x_valid, timesteps_train, timesteps_valid, y_train, y_valid