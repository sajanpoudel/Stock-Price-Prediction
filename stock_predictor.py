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
