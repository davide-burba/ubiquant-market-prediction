from evaluation import compute_avg_pearson_by_timestep
import numpy as np


class TimeCrossValidator:
    def __init__(
        self,
        n_folds=5,
        n_timesteps_per_fold=100,
        n_timesteps_to_train=500,
        tensor_like=False,
    ):
        self.n_folds = n_folds
        self.n_timesteps_per_fold = n_timesteps_per_fold
        self.n_timesteps_to_train = n_timesteps_to_train
        self.tensor_like = tensor_like

    def run(self, data, model, preprocessor):
        """
        if self.tensor_like, data should be a pair of tensors (target,features)
        else, data should be a dataframe.
        The preprocessor must match the type of data.
        """
        scores = {
            "cv_scores_train": [],
            "cv_scores_valid": [],
        }
        preds = {
            "y_train": [],
            "y_train_pred": [],
            "y_valid": [],
            "y_valid_pred": [],
        }
        timesteps = self._get_timesteps(data)

        for fold in range(self.n_folds):
            print(f"Computing fold {fold+1}/{self.n_folds}")

            train_data, valid_data = self._split_train_valid(data, timesteps, fold)
            (
                x_train,
                x_valid,
                timesteps_train,
                timesteps_valid,
                y_train,
                y_valid,
            ) = preprocessor.run(train_data, valid_data)

            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)

            if self.tensor_like:
                y_valid_pred = model.predict(x_valid, x_train)

                # reshape to compute score
                y_train = y_train.reshape(-1)
                y_valid = y_valid.reshape(-1)
                y_train_pred = y_train_pred.reshape(-1)
                y_valid_pred = y_valid_pred.reshape(-1)
                timesteps_train = timesteps_train.reshape(-1)
                timesteps_valid = timesteps_valid.reshape(-1)
            else:
                y_valid_pred = model.predict(x_valid)

            score_train = compute_avg_pearson_by_timestep(
                y_train, y_train_pred, timesteps_train
            )
            score_valid = compute_avg_pearson_by_timestep(
                y_valid, y_valid_pred, timesteps_valid
            )

            print(f"  Score valid: {score_valid:.3f}, score train: {score_train:.3f}")

            scores["cv_scores_train"].append(score_train)
            scores["cv_scores_valid"].append(score_valid)
            preds["y_train"].append(y_train)
            preds["y_train_pred"].append(y_train_pred)
            preds["y_valid"].append(y_valid)
            preds["y_valid_pred"].append(y_valid_pred)

        avg_score_train = np.mean(scores["cv_scores_train"]).item()
        avg_score_valid = np.mean(scores["cv_scores_valid"]).item()
        scores["score_train"] = avg_score_train
        scores["score_valid"] = avg_score_valid

        print(
            f"\n Avg score valid {avg_score_valid:.3f}, avg score train {avg_score_train:.3f}"
        )
        return scores, preds

    def _get_timesteps(self, data):
        if self.tensor_like:
            return np.arange(data[0].shape[1])
        else:
            timesteps = sorted(data.time_id.unique())
        return timesteps

    def _split_train_valid(self, data, timesteps, fold):
        last_train = timesteps[-(self.n_folds - fold) * self.n_timesteps_per_fold - 1]
        last_valid = timesteps[
            -(self.n_folds - (fold + 1)) * self.n_timesteps_per_fold - 1
        ]
        if self.tensor_like:
            train_data = data[0][:, :last_train], data[1][:, :last_train]
            valid_data = (
                data[0][:, last_train:last_valid],
                data[1][:, last_train:last_valid],
            )
            if self.n_timesteps_to_train is not None:
                train_data = (
                    train_data[0][:, -self.n_timesteps_to_train :],
                    train_data[1][:, -self.n_timesteps_to_train :],
                )
        else:
            train_data = data[data.time_id <= last_train]
            valid_data = data[
                (data.time_id > last_train) & (data.time_id <= last_valid)
            ]
            if self.n_timesteps_to_train is not None:

                time_ids = sorted(train_data.time_id.unique())
                start_time_id = time_ids[-self.n_timesteps_to_train]
                train_data = train_data[train_data.time_id > start_time_id]

        return train_data, valid_data
