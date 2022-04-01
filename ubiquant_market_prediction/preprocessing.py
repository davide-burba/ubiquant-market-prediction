from abc import abstractclassmethod


def get_preprocessor(preprocessor_type, preprocessor_args):
    preprocessors = {
        "naive": NaivePreprocessor,
    }
    return preprocessors[preprocessor_type.lower()](**preprocessor_args)


class BasePreprocessor:
    @abstractclassmethod
    def run(self, train_data, valid_data):
        """Used for validation"""
        pass

    @abstractclassmethod
    def run_inference(self, valid_data, train_data=None):
        """You may want to use train data to construct x_valid"""
        pass

    @abstractclassmethod
    def run_train(self, train_data):
        """Used for training before inference"""
        pass


class NaivePreprocessor(BasePreprocessor):
    def __init__(self, cols_to_drop=[]):
        self.cols_to_drop = cols_to_drop

    def run(self, train_data, valid_data):

        x_train = self.run_train(train_data)
        x_valid = self.run_inference(valid_data)

        timesteps_train = train_data.time_id.values
        timesteps_valid = valid_data.time_id.values

        y_train = train_data.target.values
        y_valid = valid_data.target.values

        return x_train, x_valid, timesteps_train, timesteps_valid, y_train, y_valid

    def run_inference(self, valid_data, train_data=None):
        return self._run(valid_data)

    def run_train(self, train_data):
        x_train = self._run(train_data)
        y_train = train_data.target.values
        return x_train, y_train

    def _run(self, df):
        cols_to_drop = ["row_id"] + self.cols_to_drop
        if "target" in df.columns:
            cols_to_drop.append("target")
        return df.drop(columns=cols_to_drop)
