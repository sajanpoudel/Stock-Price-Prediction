import pandas as pd

from stock_predictor import (
    LOOKBACK,
    load_closing_prices,
    prepare_dataframe_for_lstm,
    split_data,
    to_price,
)


def test_prepare_adds_lag_columns():
    data = load_closing_prices()
    shifted = prepare_dataframe_for_lstm(data, LOOKBACK)
    assert list(shifted.columns) == ["Close"] + [f"Close(t-{i})" for i in range(1, LOOKBACK + 1)]
    assert len(shifted) == len(data) - LOOKBACK


def test_split_shapes_and_fraction():
    shifted = prepare_dataframe_for_lstm(load_closing_prices(), LOOKBACK)
    X_train, y_train, X_test, y_test, _ = split_data(shifted)
    assert X_train.shape[1:] == (LOOKBACK, 1)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert abs(len(X_train) / (len(X_train) + len(X_test)) - 0.95) < 0.01


def test_to_price_restores_original_close():
    shifted = prepare_dataframe_for_lstm(load_closing_prices(), LOOKBACK)
    _, y_train, _, _, scaler = split_data(shifted)
    restored = to_price(y_train, scaler)
    expected = shifted["Close"].to_numpy()[: len(restored)]
    assert abs(restored - expected).max() < 1e-6


def test_split_keeps_train_and_test_in_time_order():
    shifted = prepare_dataframe_for_lstm(load_closing_prices(), LOOKBACK)
    X_train, y_train, X_test, y_test, _ = split_data(shifted)
    assert len(X_train) + len(X_test) == len(shifted)


def test_load_closing_prices_has_sorted_unique_dates():
    data = load_closing_prices()
    assert data['Date'].is_monotonic_increasing
    assert data['Date'].is_unique
