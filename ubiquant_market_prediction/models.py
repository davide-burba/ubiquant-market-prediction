from abc import abstractclassmethod
from unittest.mock import Base
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from torch.utils.data import DataLoader
import torch
from nn_commons import (
    RNNArch,
    MLPArch,
    TimeSplitter,
    TensorLoader,
    to_numpy,
    to_tensor,
    corr_loss,
    corr_exp_loss,
)


def get_model(model_type, model_args):
    models = {
        "random": RandomModel,
        "lightgbm": LightGBMModel,
        "rnn": RNNModel,
        "sklearn_mlp": MLPRegressor,
        "mlp": MLPModel,
    }
    return models[model_type.lower()](**model_args)


class BaseModel:
    @abstractclassmethod
    def fit(self, x, y):
        pass

    @abstractclassmethod
    def predict(self, x):
        pass


class RandomModel(BaseModel):
    def fit(self, x, y):
        pass

    def predict(self, x):
        return np.random.normal(size=len(x))


class LightGBMModel(BaseModel):
    def __init__(self, categorical_feature="auto", **lightgbm_params):
        self.categorical_feature = categorical_feature
        self.lightgbm_params = lightgbm_params

    def fit(self, x, y):
        train_set = lgb.Dataset(x, y, categorical_feature=self.categorical_feature)

        # train
        self.engine = lgb.train(
            self.lightgbm_params,
            train_set,
            None,
        )

    def predict(self, x):
        return self.engine.predict(x)


class MLPModel(BaseModel):
    def __init__(
        self,
        mlp_params={},
        objective="mae",
        num_epochs=5,
        batch_size=128,
        learning_rate=1e-3,
        weight_decay=0.01,
        embedding_dim_list=None,
        random_state=123,
    ):
        self.mlp_params = mlp_params
        self.objective = objective
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.embedding_dim_list = embedding_dim_list

        torch.manual_seed(random_state)
        np.random.seed(random_state)

    def _set_architecture(self, x_train):

        # manage embedding
        if self.embedding_dim_list is not None:
            self.categories = []
            self.num_embeddings_list = []
            for i, _ in enumerate(self.embedding_dim_list):
                cat_map = {}
                for cat_idx, cat in enumerate(sorted(np.unique(x_train[:, i]))):
                    cat_map[cat] = cat_idx
                self.categories.append(cat_map)
                self.num_embeddings_list.append(len(cat_map) + 1)

            self.mlp_params["use_embedding"] = True
            self.mlp_params["num_embeddings_list"] = self.num_embeddings_list
            self.mlp_params["embedding_dim_list"] = self.embedding_dim_list

        self.mlp_params["input_size"] = x_train.shape[1]

        # Initialize engine and optimizer
        self.engine = MLPArch(**self.mlp_params)
        self.optimizer = torch.optim.Adam(
            self.engine.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _format_embedding(self, x):
        if self.embedding_dim_list is None:
            return x

        n_embeddings = len(self.embedding_dim_list)
        for i in range(n_embeddings):
            cat_map = self.categories[i]
            last = len(cat_map)
            map_fun = lambda x: cat_map[x] if x in cat_map else last
            map_fun = np.vectorize(map_fun)
            x[:, i] = map_fun(x[:, i])

        return x

    def fit(self, x_train, y_train):

        if isinstance(x_train, pd.DataFrame):
            x_train = x_train.values

        self._set_architecture(x_train)

        x_train = self._format_embedding(x_train)

        # Define the loader using x_train, y_train
        loader = DataLoader(
            dataset=TensorLoader(x_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.training_loss_value = []

        for epoch in range(self.num_epochs):
            print(f"epoch {epoch+1}/{self.num_epochs}")
            epoch_loss_value = []

            for x_batch, y_batch in loader:

                pred = self.engine(x_batch)
                assert y_batch.shape == pred.shape

                if self.objective == "mae":
                    loss_value = torch.mean(torch.abs(y_batch - pred))
                elif self.objective == "mse":
                    loss_value = torch.mean((y_batch - pred) ** 2)
                elif self.objective == "corr":
                    loss_value = corr_loss(y_batch, pred)
                elif self.objective == "corr_exp":
                    loss_value = corr_exp_loss(y_batch, pred)
                else:
                    raise ValueError(f"Unknown objective {self.objective}")

                # Run the optimizer
                self.optimizer.zero_grad()
                loss_value.backward()
                self.optimizer.step()
                epoch_loss_value.append(loss_value.item())

            # store the epoch loss
            epoch_loss_value = np.mean(epoch_loss_value)
            self.training_loss_value.append(epoch_loss_value)
            print(f"Epoch loss: {epoch_loss_value:.4f}")

    def predict(self, x):

        if isinstance(x, pd.DataFrame):
            x = x.values

        self.engine.eval()
        x = self._format_embedding(x)
        x = to_tensor(x)
        pred = to_numpy(self.engine(x))
        self.engine.train()

        return pred


class RNNModel(BaseModel):
    def __init__(
        self,
        rnn_params,
        train_on_sequence=True,
        window_sizes=None,
        objective="mae",
        num_epochs=5,
        batch_size=128,
        learning_rate=1e-3,
        weight_decay=0.01,
        embedding_dim_list=None,
        stateful_pred=True,
        random_state=123,
    ):

        self.train_on_sequence = train_on_sequence
        self.window_sizes = window_sizes
        self.objective = objective
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.embedding_dim_list = embedding_dim_list
        self.rnn_params = rnn_params
        self.stateful_pred = stateful_pred

        torch.manual_seed(random_state)
        np.random.seed(random_state)

    def _set_architecture(self, x_train):

        if self.embedding_dim_list is not None:
            self.categories = []
            self.num_embeddings_list = []
            for i, _ in enumerate(self.embedding_dim_list):
                cat_map = {}
                # Warning: this step assumes that all categories are time invariant!
                for cat_idx, cat in enumerate(sorted(np.unique(x_train[:, 0, i]))):
                    cat_map[cat] = cat_idx
                self.categories.append(cat_map)
                self.num_embeddings_list.append(len(cat_map) + 1)

            self.rnn_params["use_embedding"] = True
            self.rnn_params["num_embeddings_list"] = self.num_embeddings_list
            self.rnn_params["embedding_dim_list"] = self.embedding_dim_list
        
        self.rnn_params["input_size"] = x_train.shape[-1]

        # Initialize engine and optimizer
        self.engine = RNNArch(**self.rnn_params)
        self.optimizer = torch.optim.Adam(
            self.engine.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _format_embedding(self, x):

        if self.embedding_dim_list is None:
            return x

        n_embeddings = len(self.embedding_dim_list)
        mask_all_zeros = x[:, :, n_embeddings:].sum(axis=2).sum(axis=1) == 0
        for i in range(n_embeddings):
            cat_map = self.categories[i]
            last = len(cat_map)
            map_fun = lambda x: cat_map[x] if x in cat_map else last
            map_fun = np.vectorize(map_fun)
            x[:, :, i] = map_fun(x[:, :, i])
            # if all features are zeros, it means we actually never saw
            # values. in this case, we assign the last category
            x[mask_all_zeros] = last

        return x

    def fit(self, x_train, y_train):

        self._set_architecture(x_train)

        x_train = self._format_embedding(x_train)

        # Define the loader using x_train, y_train
        loader = DataLoader(
            dataset=TensorLoader(x_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.training_loss_value = []

        for epoch in range(self.num_epochs):
            print(f"epoch {epoch+1}/{self.num_epochs}")
            epoch_loss_value = []
            if self.window_sizes is not None:
                window_size = self.window_sizes[epoch % len(self.window_sizes)]
            else:
                window_size = x_train.shape[1]  # Use all T

            for x_batch, y_batch in loader:
                timesplitter = TimeSplitter(x_batch, y_batch, window_size)
                for x_batch_t, y_batch_t in timesplitter:

                    pred, _ = self.engine(x_batch_t)

                    if not self.train_on_sequence:
                        pred = pred[:, -1]
                        y_batch_t = y_batch_t[:, -1]

                    assert y_batch_t.shape == pred.shape

                    if self.objective not in {"corr", "corr_exp"}:
                        drop_na_mask = ~y_batch_t.isnan()
                        y_batch_t = y_batch_t[drop_na_mask]
                        pred = pred[drop_na_mask]
                        error = y_batch_t[drop_na_mask] - pred[drop_na_mask]

                    if self.objective == "mae":
                        loss_value = torch.mean(torch.abs(error))
                    elif self.objective == "mse":
                        loss_value = torch.mean((error) ** 2)
                    elif self.objective == "corr":
                        loss_value = corr_loss(y_batch_t, pred)
                    elif self.objective == "corr_exp":
                        loss_value = corr_exp_loss(y_batch_t, pred)
                    else:
                        raise ValueError(f"Unknown objective {self.objective}")

                    # Run the optimizer
                    self.optimizer.zero_grad()
                    loss_value.backward()
                    self.optimizer.step()
                    epoch_loss_value.append(loss_value.item())

            # store the epoch loss
            epoch_loss_value = np.mean(epoch_loss_value)
            self.training_loss_value.append(epoch_loss_value)
            print(f"Epoch loss: {epoch_loss_value:.4f}")

    def predict(self, x, x_past=None):

        if x_past is None or not self.stateful_pred:
            h_state = None
        else:
            _, h_state = self._predict(x_past)
        y_pred, _ = self._predict(x, h_state)
        return y_pred

    def _predict(
        self, x, h_state=None, return_intermediate_pred=True, return_states=True
    ):
        x = self._format_embedding(x)
        self.engine.eval()
        # Convert x to tensor
        x = to_tensor(x)
        # Get the predictions --> convert the predictions to numpy
        with torch.no_grad():
            prediction, h_state = self.engine(x, h_state)
            prediction = to_numpy(prediction)
        self.engine.train()
        if not return_intermediate_pred:
            prediction = prediction[:, -1]
        if return_states:
            return prediction, h_state
        else:
            return prediction
