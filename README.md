# Time Series Prediction with LSTM

This project is about predicting time series data using an LSTM model. The data used in this project is the closing prices of Amazon stocks.

## Dependencies

- numpy
- pandas
- matplotlib
- torch
- sklearn

## Data Preparation

The data is loaded from a CSV file named 'data-amz.csv'. The 'Date' and 'Close' columns are extracted and the 'Date' column is converted to datetime format. The data is then plotted to visualize the closing prices over time.

A function named `prepare_dataframe_for_lstm` is used to prepare the data for the LSTM model. This function shifts the 'Close' column by a specified number of steps and adds these shifted columns to the dataframe.

The data is then scaled using the `MinMaxScaler` from sklearn to a range of -1 to 1.

The data is split into training and testing sets, with 95% of the data used for training and the remaining 5% used for testing.

## Model

The model used in this project is an LSTM model implemented using PyTorch. The model has one LSTM layer and one fully connected layer. The LSTM layer takes in a sequence of data and outputs the hidden state for each element in the sequence. The fully connected layer takes the last hidden state output by the LSTM layer and outputs a single value, which is the prediction for the next time step.

## Training

The model is trained using a function named `train_one_epoch`. This function takes in the training data and the model, and trains the model for one epoch. The loss for each batch is printed every 100 batches.

## Validation

The function `validate_one_epoch` is used to validate the model on the testing data. This function takes in the testing data and the model, and calculates the loss for the testing data.

## Usage

Install the dependencies and run the script. It loads the data, prepares it for the LSTM model, trains the model, validates it on the testing data and plots the predicted closing prices against the actual ones.

```
pip install -r requirements.txt
python stock_predictor.py
```

The same steps are available step by step in `Stock_Predictor.ipynb`.

## Tests

```
pip install pytest
pytest
```
