from abc import abstractclassmethod
import lightgbm as lgb
import numpy as np
from torch.utils.data import DataLoader
import torch
from rnn import RNNArch, TimeSplitter, TensorLoader, to_numpy, to_tensor


def get_model(model_type, model_args):
    models = {
        "random": RandomModel,
        "lightgbm": LightGBMModel,
        "rnn": RNNModel,
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


class LightGBMModel:
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


class RNNModel:
    def __init__(
        self,
        rnn_params,
        train_on_sequence=True,
        window_sizes=None,
        objective="mae",
        num_epochs=5,
        batch_size=1,
        learning_rate=1e-3,
        weight_decay=0.01,
        random_state=123,
    ):

        self.train_on_sequence = train_on_sequence
        self.window_sizes = window_sizes
        self.objective = objective
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Initialize engine and optimizer
        self.engine = RNNArch(**rnn_params)
        self.optimizer = torch.optim.Adam(
            self.engine.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def fit(self, X_train, y_train):
        # Define the loader using X_train, y_train
        loader = DataLoader(
            dataset=TensorLoader(X_train, y_train),
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
                window_size = X_train.shape[1]  # Use all T

            for X_batch, y_batch in loader:
                timesplitter = TimeSplitter(X_batch, y_batch, window_size)
                for X_batch_t, y_batch_t in timesplitter:

                    pred, _ = self.engine(X_batch_t)

                    if not self.train_on_sequence:
                        pred = pred[:, -1]
                        y_batch_t = y_batch_t[:, -1]

                    assert y_batch_t.shape == pred.shape

                    if self.objective == "mae":
                        loss_value = torch.mean(torch.abs(y_batch_t - pred))
                    elif self.objective == "mse":
                        loss_value = torch.mean((y_batch_t - pred) ** 2)

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

        if x_past is not None:
            _, h_state = self._predict(x_past)
        else:
            h_state = None
        y_pred, _ = self._predict(x, h_state)
        return y_pred

    def _predict(
        self, X, h_state=None, return_intermediate_pred=True, return_states=True
    ):
        self.engine.eval()
        # Convert X to tensor
        X = to_tensor(X)
        # Get the predictions --> convert the predictions to numpy
        with torch.no_grad():
            prediction, h_state = self.engine(X, h_state)
            prediction = to_numpy(prediction)
        self.engine.train()
        if not return_intermediate_pred:
            prediction = prediction[:, -1]
        if return_states:
            return prediction, h_state
        else:
            return prediction
