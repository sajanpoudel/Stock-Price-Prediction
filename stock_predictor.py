"""Predict Amazon closing prices with an LSTM, using the same steps as the notebook."""
from copy import deepcopy as dc

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATA_FILE = 'data-amz.csv'
LOOKBACK = 7
TRAIN_FRACTION = 0.95


def load_closing_prices(path=DATA_FILE):
    """Read the csv and keep the Date and Close columns."""
    data = pd.read_csv(path)[['Date', 'Close']]
    data['Date'] = pd.to_datetime(data['Date'])
    return data


def prepare_dataframe_for_lstm(df, n_steps):
    """Add Close(t-1) ... Close(t-n_steps) columns built from shifted closing prices."""
    df = dc(df)
    df.set_index('Date', inplace=True)

    for i in range(1, n_steps + 1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

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
