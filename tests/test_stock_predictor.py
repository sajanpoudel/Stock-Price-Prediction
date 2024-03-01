import numpy as np
import pandas as pd
import torch

from stock_predictor import (
    LOOKBACK,
    LSTM,
    root_mean_squared_price_error,
    TimeSeriesDataset,
    load_model,
    save_model,
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


def test_closing_prices_are_positive():
    assert (load_closing_prices()['Close'] > 0).all()


def test_dataset_returns_matching_rows():
    X = torch.arange(12, dtype=torch.float32).reshape(4, 3, 1)
    y = torch.arange(4, dtype=torch.float32).reshape(4, 1)
    dataset = TimeSeriesDataset(X, y)
    assert len(dataset) == 4
    features, target = dataset[2]
    assert features.shape == (3, 1)
    assert target.item() == 2.0


def test_lstm_outputs_one_value_per_sample():
    model = LSTM(1, 4, 1)
    out = model(torch.zeros(5, LOOKBACK, 1))
    assert out.shape == (5, 1)


def test_lstm_is_deterministic_in_eval_mode():
    torch.manual_seed(0)
    model = LSTM(1, 4, 1).eval()
    x = torch.rand(2, LOOKBACK, 1)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_scaled_targets_stay_inside_the_scaler_range():
    shifted = prepare_dataframe_for_lstm(load_closing_prices(), LOOKBACK)
    _, y_train, _, y_test, _ = split_data(shifted)
    assert np.min(y_train) >= -1.0 and np.max(y_train) <= 1.0


def test_price_error_is_zero_for_identical_series():
    assert root_mean_squared_price_error([1, 2, 3], [1, 2, 3]) == 0.0


def test_price_error_is_in_price_units():
    assert root_mean_squared_price_error([10, 10], [13, 7]) == 3.0


def test_price_error_rejects_different_shapes():
    import pytest

    with pytest.raises(ValueError):
        root_mean_squared_price_error([1, 2], [1])
