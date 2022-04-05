from abc import abstractclassmethod
import numpy as np
import sklearn


def get_preprocessor(preprocessor_type, preprocessor_args):
    preprocessors = {
        "naive": NaivePreprocessor,
        "tensor": TensorPreprocessor,
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

    def inverse_transform(self, y_pred):
        return y_pred


class NaivePreprocessor(BasePreprocessor):
    def __init__(
        self,
        cols_to_drop=[],
        scaler_features=None,
        scaler_features_args={},
        crop_low=None,
        crop_high=None,
    ):
        self.cols_to_drop = cols_to_drop
        self.crop_low = crop_low
        self.crop_high = crop_high
        if scaler_features is not None:
            scaler_featues = getattr(sklearn.preprocessing, scaler_features)(
                **scaler_features_args
            )
        self.scaler_features = scaler_features

    def run(self, train_data, valid_data):

        x_train, y_train = self.run_train(train_data, fit_scaler=True)
        x_valid = self.run_inference(valid_data)

        timesteps_train = train_data.time_id.values
        timesteps_valid = valid_data.time_id.values

        y_valid = valid_data.target.values

        return x_train, x_valid, timesteps_train, timesteps_valid, y_train, y_valid

    def run_inference(self, valid_data, train_data=None):
        return self._run(valid_data)

    def run_train(self, train_data):
        x_train = self._run(train_data, fit_scaler=True)
        y_train = train_data.target.values

        if self.crop_high is not None:
            y_train[y_train > self.crop_high] = self.crop_high
        if self.crop_low is not None:
            y_train[y_train < self.crop_low] = self.crop_low

        return x_train, y_train

    def _run(self, df, fit_scaler=False):
        cols_to_drop = ["row_id"] + self.cols_to_drop
        if "target" in df.columns:
            cols_to_drop.append("target")
        df = df.drop(columns=cols_to_drop)

        if self.scaler_features is not None:
            if fit_scaler:
                self.scaler_features.fit(df)
            df = self.scaler_features.transform(df)
        return df


class TensorPreprocessor(BasePreprocessor):
    def __init__(
        self,
        fill_na_target=True,
        scaler_features=None,
        scaler_targets=None,
        scaler_fit_sample=None,
        scaler_features_args={},
        scaler_targets_args={},
        crop_low=None,
        crop_high=None,
    ):
        self.fill_na_target = fill_na_target
        self.crop_low = crop_low
        self.crop_high = crop_high

        self.ts_scaler = TimeSeriesTensorScaler(
            scaler_features,
            scaler_targets,
            scaler_fit_sample,
            scaler_features_args,
            scaler_targets_args,
        )

    def run(self, train_data, valid_data):

        x_train, y_train = self._run_data(train_data, copy=True)
        x_valid, y_valid = self._run_data(valid_data, copy=True)

        self.ts_scaler.fit(x_train, y_train)
        x_train, y_train = self.ts_scaler.transform(x_train, y_train)
        x_valid, y_valid = self.ts_scaler.transform(x_valid, y_valid)

        # set timesteps,repeat over N axis
        timesteps_train = np.arange(y_train.shape[1])
        timesteps_valid = np.arange(y_valid.shape[1])
        timesteps_train = timesteps_train.reshape(1, -1).repeat(len(y_train), 0)
        timesteps_valid = timesteps_valid.reshape(1, -1).repeat(len(y_valid), 0)

        return x_train, x_valid, timesteps_train, timesteps_valid, y_train, y_valid

    def run_inference(self, x):
        x[np.isnan(x)] = 0
        return self.ts_scaler.transform(x)

    def run_train(self, data):
        x, y = self._run_data(data)
        self.ts_scaler.fit(x, y)
        x, y = self.ts_scaler.transform(x, y)
        return x, y

    def inverse_transform(self, valid_pred):
        return self.ts_scaler.inverse_transform(valid_pred)

    def _run_data(self, data, copy=False):
        y, x = data
        if copy:
            x = x.copy()
            y = y.copy()
        x[np.isnan(x)] = 0
        if self.fill_na_target:
            y[np.isnan(y)] = 0

        if self.crop_high is not None:
            y[y > self.crop_high] = self.crop_high
        if self.crop_low is not None:
            y[y < self.crop_low] = self.crop_low
        return x, y


class TimeSeriesTensorScaler:
    def __init__(
        self,
        scaler_features=None,
        scaler_targets=None,
        scaler_fit_sample=None,
        scaler_features_args={},
        scaler_targets_args={},
    ):

        self.scaler_features = self._get_scaler(scaler_features, scaler_features_args)
        self.scaler_targets = self._get_scaler(scaler_targets, scaler_targets_args)
        self.scaler_fit_sample = scaler_fit_sample

    def fit(self, features, targets):
        if self.scaler_features is not None:
            feat_reshaped = features.reshape(-1, features.shape[2])
            if self.scaler_fit_sample is not None:
                sel_idx = np.random.choice(
                    len(feat_reshaped), size=self.scaler_fit_sample, replace=False
                )
                feat_reshaped = feat_reshaped[sel_idx]

            self.scaler_features.fit(feat_reshaped)

        if self.scaler_targets is not None:
            targ_reshaped = targets.reshape(-1, 1)
            if self.scaler_fit_sample is not None:
                sel_idx = np.random.choice(
                    len(targ_reshaped), size=self.scaler_fit_sample, replace=False
                )
                targ_reshaped = targ_reshaped[sel_idx]

            self.scaler_targets.fit(targ_reshaped)

    def transform(self, features, targets=None):

        scaled_features = features.copy()
        if self.scaler_features is not None:
            for i in range(len(features)):
                scaled_features[i] = self.scaler_features.transform(features[i])

        if targets is None:
            return scaled_features

        scaled_targets = targets.copy()
        if self.scaler_targets is not None:
            for i in range(len(targets)):
                scaled_targets[i : i + 1] = self.scaler_targets.transform(
                    targets[i : i + 1].transpose(1, 0)
                ).transpose(1, 0)

        return scaled_features, scaled_targets

    def _get_scaler(self, scaler, scaler_args):
        if scaler is None:
            return None
        return getattr(sklearn.preprocessing, scaler)(**scaler_args)

    def inverse_transform(self, targets):
        """
        Assume is 1d.
        """
        assert len(targets.shape) == 1 or targets.shape[1] == 1
        if self.scaler_targets is None:
            return targets
        return self.scaler_targets.inverse_transform(targets.reshape(-1, 1)).reshape(
            targets.shape
        )
