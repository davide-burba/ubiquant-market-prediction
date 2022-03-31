from evaluation import compute_avg_pearson_by_timestep
import numpy as np


class TimeCrossValidator:
    def __init__(self, n_folds=5, n_timesteps_per_fold=100):
        self.n_folds = n_folds
        self.n_timesteps_per_fold = n_timesteps_per_fold

    def run(self, data, model, preprocessor):
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
        timesteps = sorted(data.time_id.unique())

        for fold in range(self.n_folds):
            print(f"Computing fold {fold+1}/{self.n_folds}")

            last_train = timesteps[
                -(self.n_folds - fold) * self.n_timesteps_per_fold - 1
            ]
            last_valid = timesteps[
                -(self.n_folds - (fold + 1)) * self.n_timesteps_per_fold - 1
            ]

            train_data = data[data.time_id <= last_train]
            valid_data = data[
                (data.time_id > last_train) & (data.time_id <= last_valid)
            ]

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
