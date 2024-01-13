"""Predict Amazon closing prices with an LSTM, using the same steps as the notebook."""

from copy import deepcopy as dc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

DATA_FILE = "data-amz.csv"
LOOKBACK = 7
TRAIN_FRACTION = 0.95


def load_closing_prices(path=DATA_FILE):
    """Read the csv and keep the Date and Close columns."""
    data = pd.read_csv(path)[["Date", "Close"]]
    data["Date"] = pd.to_datetime(data["Date"])
    return data


def prepare_dataframe_for_lstm(df, n_steps):
    """Add Close(t-1) ... Close(t-n_steps) columns built from shifted closing prices."""
    df = dc(df)
    df.set_index("Date", inplace=True)

    for i in range(1, n_steps + 1):
        df[f"Close(t-{i})"] = df["Close"].shift(i)

    df.dropna(inplace=True)
    return df


def split_data(shifted_df, lookback=LOOKBACK, train_fraction=TRAIN_FRACTION):
    """Scale to [-1, 1] and split into train and test arrays shaped for an LSTM.

    Returns X_train, y_train, X_test, y_test and the fitted scaler.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(shifted_df.to_numpy())

    X = dc(np.flip(scaled[:, 1:], axis=1))
    y = scaled[:, 0]

    split_index = int(len(X) * train_fraction)
    X_train = X[:split_index].reshape((-1, lookback, 1))
    X_test = X[split_index:].reshape((-1, lookback, 1))
    y_train = y[:split_index].reshape((-1, 1))
    y_test = y[split_index:].reshape((-1, 1))
    return X_train, y_train, X_test, y_test, scaler


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size, device=x.device)

        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def train_one_epoch(model, loader, loss_function, optimizer, device, epoch):
    model.train(True)
    print(f"Epoch: {epoch + 1}")
    running_loss = 0.0

    for batch_index, (x_batch, y_batch) in enumerate(loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:  # print every 100 batches
            print("Batch {0}, Loss: {1:.3f}".format(batch_index + 1, running_loss / 100))
            running_loss = 0.0
    print()


def validate_one_epoch(model, loader, loss_function, device):
    model.train(False)
    running_loss = 0.0

    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        with torch.no_grad():
            running_loss += loss_function(model(x_batch), y_batch).item()

    print("Val Loss: {0:.3f}".format(running_loss / len(loader)))
    print("***************************************************")
    print()


def to_price(values, scaler, lookback=LOOKBACK):
    """Undo the scaling of the Close column, which is column 0 of the scaled data."""
    dummies = np.zeros((len(values), lookback + 1))
    dummies[:, 0] = values.flatten()
    return scaler.inverse_transform(dummies)[:, 0]


def main(num_epochs=10, learning_rate=0.001, batch_size=16, show_plots=True):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    shifted_df = prepare_dataframe_for_lstm(load_closing_prices(), LOOKBACK)
    X_train, y_train, X_test, y_test, scaler = split_data(shifted_df)

    X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
    X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()

    train_loader = DataLoader(
        TimeSeriesDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TimeSeriesDataset(X_test, y_test), batch_size=batch_size, shuffle=False
    )

    model = LSTM(1, 4, 1).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, loss_function, optimizer, device, epoch)
        validate_one_epoch(model, test_loader, loss_function, device)

    if show_plots:
        import matplotlib.pyplot as plt

        with torch.no_grad():
            test_predictions = model(X_test.to(device)).cpu().numpy()

        plt.plot(to_price(y_test.numpy(), scaler), label="Actual Close")
        plt.plot(to_price(test_predictions, scaler), label="Predicted Close")
        plt.xlabel("Day")
        plt.ylabel("Close")
        plt.legend()
        plt.show()


def root_mean_squared_price_error(actual_prices, predicted_prices):
    """RMSE in dollars between two price series of the same length."""
    actual = np.asarray(actual_prices, dtype=float)
    predicted = np.asarray(predicted_prices, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("price series must have the same shape")
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


if __name__ == "__main__":
    main()
