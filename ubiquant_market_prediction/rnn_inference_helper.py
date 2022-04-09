import numpy as np
import torch
from copy import deepcopy


def run_inference_rnn(
    test_df,
    model,
    preprocessor,
    hidden_state_helper,
    keep_investment_id=False,
    update_state_during_inference=True,
    check_last_timestep=False,
):
    test_df = test_df.fillna(0)
    (
        test_features,
        test_investment_ids,
        test_timesteps,
    ) = _build_tensor_features(test_df, keep_investment_id)
    
    if check_last_timestep:
        last_timestep = min(test_timesteps) - 1
    else:
        last_timestep = None

    y_pred = _build_prediction_tensor(
        test_features,
        test_investment_ids,
        last_timestep,
        model,
        preprocessor,
        hidden_state_helper,
        update_state_during_inference,
    )

    predictions = _pred_tensor_to_list(
        test_df, y_pred, test_investment_ids, test_timesteps
    )
    return predictions


def _build_tensor_features(test_df, keep_investment_id):
    if "time_id" not in test_df.columns:
        test_df["time_id"] = test_df.row_id.str.split("_", expand=True)[0].astype(int)
    test_investment_ids = np.array(sorted(test_df.investment_id.unique()))
    test_timesteps = np.array(sorted(test_df.time_id.unique()))

    test_features = _df_to_tensor(
        test_df, test_investment_ids, test_timesteps, keep_investment_id
    )

    return test_features, test_investment_ids, test_timesteps


def _build_prediction_tensor(
    test_features,
    test_investment_ids,
    last_timestep,
    model,
    preprocessor,
    hidden_state_helper,
    update_state_during_inference,
):
    
    x_test = preprocessor.run_inference(test_features)
    h_state_prev = hidden_state_helper.get_hstate(test_investment_ids,last_timestep)
    y_pred, h_state = model._predict(x_test, h_state=h_state_prev)
    if update_state_during_inference:
        hidden_state_helper.update_hstate(h_state, test_investment_ids,last_timestep+1)
    return y_pred


def _pred_tensor_to_list(test_df, y_pred, test_investment_ids, test_timesteps):
    predictions = []
    for i, r in enumerate(test_df.row_id):
        time_id, investment_id = r.split("_")
        time_id = int(time_id)
        investment_id = test_df.iloc[i].investment_id.item()

        inv_idx = np.argwhere(test_investment_ids == investment_id).item()
        time_idx = np.argwhere(test_timesteps == time_id).item()

        predictions.append(float(y_pred[inv_idx, time_idx]))
    return predictions


def _df_to_tensor(df, investment_ids, timesteps, keep_investment_id):
    df = df.drop(columns="row_id").set_index(["investment_id", "time_id"])

    features = []
    for investment_id in investment_ids:
        df_inv = df.loc[investment_id].reindex(timesteps)
        if keep_investment_id:
            df_inv.insert(0, "investment_id", investment_id)
        features.append(np.expand_dims(df_inv.values, 0))

    features = np.concatenate(features, axis=0)
    return features


class HiddenStateHelper:
    def __init__(
        self, h_state, investment_ids, initialize_inv_id=True, h_state_default=None,last_timestep=0,
    ):

        self.d1, _, self.d3 = h_state[0].shape
        self._h1 = {}
        self._h2 = {}
        self._last_timestep = {}
        
        self.h_state_default = h_state_default
        if initialize_inv_id:
            self.update_hstate(h_state, investment_ids,last_timestep)

    def update_hstate(self, h_state, investment_ids,last_timestep):
        h1, h2 = h_state
        for i, inv_id in enumerate(investment_ids):
            self._h1[inv_id] = h1[:, i : i + 1, :]
            self._h2[inv_id] = h2[:, i : i + 1, :]
            self._last_timestep[inv_id] = last_timestep

    def get_hstate(self, investment_ids, last_timestep=None):
        h1 = []
        h2 = []
        for inv_id in investment_ids:
            
            get_default = True
            if inv_id in self._h1:
                if last_timestep is None or self._last_timestep[inv_id] == last_timestep:
                    get_default = False
                    
            if get_default:
                _h1, _h2 = self.get_default_state()
                h1.append(_h1)
                h2.append(_h2)
            else:
                h1.append(self._h1[inv_id])
                h2.append(self._h2[inv_id])
                
        h1 = torch.cat(h1, dim=1)
        h2 = torch.cat(h2, dim=1)
        return (h1, h2)

    def get_default_state(self):
        if self.h_state_default is None:
            _h1 = self._build_zero_state()
            _h2 = self._build_zero_state()
        else:
            _h1, _h2 = deepcopy(self.h_state_default)
        return _h1, _h2

    def _build_zero_state(self):
        return torch.zeros(self.d1, 1, self.d3)


