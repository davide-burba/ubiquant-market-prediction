import numpy as np
import pandas as pd
import torch


def run_inference(test_df, model, preprocessor, hidden_state_helper):
    (
        test_features,
        test_investment_ids,
        test_timesteps,
    ) = _build_tensor_features(test_df)

    y_pred = _build_prediction_tensor(
        test_features, test_investment_ids, model, preprocessor, hidden_state_helper
    )

    predictions = _pred_tensor_to_list(
        test_df, y_pred, test_investment_ids, test_timesteps
    )
    return predictions


def _build_tensor_features(test_df):
    test_df["time_id"] = test_df.row_id.str.split("_", expand=True)[0].astype(int)
    test_investment_ids = sorted(test_df.investment_id.unique())
    test_timesteps = np.array(sorted(test_df.time_id.unique()))

    test_features = _df_to_tensor(test_df, test_investment_ids, test_timesteps)

    return test_features, test_investment_ids, test_timesteps


def _build_prediction_tensor(
    test_features, test_investment_ids, model, preprocessor, hidden_state_helper
):

    x_test = preprocessor.run_inference(test_features)
    h_state_prev = hidden_state_helper.get_hstate(test_investment_ids)
    y_pred, h_state = model._predict(x_test, h_state=h_state_prev)
    hidden_state_helper.update_hstate(h_state, test_investment_ids)
    return y_pred


def _pred_tensor_to_list(test_df, y_pred, test_investment_ids, test_timesteps):
    predictions = []
    for r in test_df.row_id:
        time_id, investment_id = r.split("_")
        time_id = int(time_id)
        investment_id = int(investment_id)

        inv_idx = np.argwhere(test_investment_ids == investment_id).item()
        time_idx = np.argwhere(test_timesteps == time_id).item()

        predictions.append(float(y_pred[inv_idx, time_idx]))
    return predictions


def _df_to_tensor(df, investment_ids, timesteps):
    df = df.drop(columns="row_id").set_index(["investment_id", "time_id"])

    features = []
    for investment_id in investment_ids:
        df_inv = df.loc[investment_id].reindex(timesteps)
        features.append(np.expand_dims(df_inv.values, 0))

    features = np.concatenate(features, axis=0)
    return features


class HiddenStateHelper:
    def __init__(self, h_state, investment_ids):

        self.d1, _, self.d3 = h_state[0].shape
        self._h1 = {}
        self._h2 = {}
        self.update_hstate(h_state, investment_ids)

    def update_hstate(self, h_state, investment_ids):
        h1, h2 = h_state
        for i, inv_id in enumerate(investment_ids):
            self._h1[inv_id] = h1[:, i : i + 1, :]
            self._h2[inv_id] = h2[:, i : i + 1, :]

    def get_hstate(self, investment_ids):
        h1 = []
        h2 = []
        for inv_id in investment_ids:
            if inv_id in self._h1:
                h1.append(self._h1[inv_id])
                h2.append(self._h2[inv_id])
            else:
                h1.append(self._build_zero_state())
                h2.append(self._build_zero_state())
        h1 = torch.cat(h1, dim=1)
        h2 = torch.cat(h2, dim=1)
        return (h1, h2)

    def _build_zero_state(self):
        return torch.zeros(self.d1, 1, self.d3)
