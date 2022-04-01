import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer


class RNNArch(nn.Module):
    """
    input and output tensors are provided as (batch, seq, feature)
    """

    DEFAULTS = {}

    def __init__(
        self,
        input_size,
        hidden_size=32,
        num_layers=1,
        dropout_prob=0.1,
    ):
        super(RNNArch, self).__init__()
        # Initialize RNN
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_prob,
            batch_first=True,
        )

        # Initialize Regressor
        layers = []
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.LeakyReLU(0.02))
        if dropout_prob > 0:
            layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(hidden_size, 1))
        self.regressor = nn.Sequential(*layers)

    def forward(self, X, h_state=None):
        N, T, F = X.shape
        # RNN
        out_rnn, h_state = self.rnn(X, h_state)
        # The following line is doing: N,T,F_hidden --> NT, F_hidden --> N,T,F_out
        out_reg = self.regressor(out_rnn.reshape([N * T, -1])).reshape([N, T, -1])
        return out_reg.squeeze(-1), h_state


def to_numpy(x):
    return x.data.numpy()


def to_tensor(x):
    if type(x) != type(torch.tensor(0)):
        x = torch.tensor(x.astype("float32"))
    return x


class TensorLoader:
    def __init__(self, X, y):
        self.X = X.astype("float32")
        self.y = y.astype("float32")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = to_tensor(self.X[index])
        y = to_tensor(self.y[index])
        return X, y


class TimeSplitter:
    def __init__(self, X, y, window_size):
        N, T, F = X.shape
        self.X_chunked = []
        self.y_chunked = []
        for start_t in range(0, T, window_size):  # this could be rolling!!!!
            end_t = start_t + window_size
            self.X_chunked.append(X[:, start_t:end_t])
            self.y_chunked.append(y[:, start_t:end_t])
        self.order = np.arange(len(self.X_chunked))

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        return self.X_chunked[index], self.y_chunked[index]


class CustomScaler:
    def fit(self, X, y):
        self.y_scaler = MinMaxScaler().fit(y.transpose(1, 0))
        self.X_scaler = [MinMaxScaler().fit(X[i]) for i in range(X.shape[0])]

    def transform(self, X, y):
        y_trans = self.y_scaler.transform(y.transpose(1, 0)).transpose(1, 0)
        X_trans = np.stack(
            [self.X_scaler[i].transform(X[i]) for i in range(X.shape[0])]
        )
        return X_trans, y_trans

    def inverse_transform(self, y):
        """no need to inverse transform X"""
        y_invtrans = self.y_scaler.inverse_transform(y.transpose(1, 0)).transpose(1, 0)
        # X_invtrans = np.stack([self.X_scaler[i].inverse_transform(X[i]) for i in range(X.shape[0])])
        return y_invtrans
