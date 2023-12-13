from django.http import JsonResponse, HttpResponse
from django.views import View
from django.shortcuts import render
import os
from django.conf import settings
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from .models import Prediction
import json
import requests
import pandas as pd
import datetime


class PredictionView(View):
    def get(self, request):
        # return view
        return render(request, 'forecast/predictions.html')


def get_prediction(request):
    prediction_days = int(request.GET.get('prediction_days'))
    # Load the saved model
    model_path = os.path.join(settings.BASE_DIR, 'complex_model.h5')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(model_path)

    # time_start is 1 January 2022
    time_start = datetime.datetime(2022, 1, 1)
    time_end = datetime.datetime.now()

    # Konversi waktu ke format timestamp Unix
    time_end_unix = int(time_end.timestamp())
    time_start_unix = int(time_start.timestamp())

    # Define the API URL
    api_url = f'https://api.coinmarketcap.com/data-api/v3/cryptocurrency/historical?id=1&convertId=2794&timeStart={time_start_unix}&timeEnd={time_end_unix}'

    r = requests.get(api_url)
    data = []

    # Extract data from the API response
    for item in r.json()['data']['quotes']:
        close = item['quote']['close']
        volume = item['quote']['volume']
        date = item['quote']['timestamp']
        high = item['quote']['high']
        low = item['quote']['low']
        data.append([close, volume, date, high, low])

    # Define column names for the DataFrame
    cols = ["close", "volume", "date", "high", "low"]

    # Create a Pandas DataFrame
    df = pd.DataFrame(data, columns=cols)

    # Convert the date column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Sort the DataFrame by date
    df.sort_values(by='date', inplace=True, ascending=True)

    data = df.copy()

    # Preprocess the data
    df = df[['close', 'volume', 'high', 'low']]

    # Split the data into train and test
    train, test = train_test_split(df, test_size=0.2, shuffle=False)

    # Scale the data
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    # Split the data into X_train, y_train, X_test, y_test
    X_train = train[:, 1:]
    y_train = train[:, 0]
    X_test = test[:, 1:]
    y_test = test[:, 0]

    # Reshape the data
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    # Generate input sequences for the next 7 days
    # Take the last sequence from the test data
    input_sequence = X_test[-1]
    input_sequence = input_sequence.reshape(
        1, 1, input_sequence.shape[1])  # Reshape for model prediction

    # Make predictions for the next 7 days
    predicted_prices = []
    for _ in range(prediction_days):
        next_day_prediction = model.predict(input_sequence)
        # Assuming one-dimensional output
        predicted_prices.append(next_day_prediction[0, 0])
        input_sequence = np.append(input_sequence[:, 0, 1:], next_day_prediction).reshape(
            1, 1, input_sequence.shape[2])

    # Inverse transform the predicted values
    predicted_prices = np.array(predicted_prices).reshape(-1, 1)
    predicted_prices = scaler.inverse_transform(np.concatenate(
        [predicted_prices, np.zeros((predicted_prices.shape[0], 3))], axis=1))[:, 0]

    # assuming the last date in the test data is the last date in the historical data
    last_date = data['date'].iloc[-1]

    # Generate date for the next 7 days
    date = pd.date_range(last_date, periods=prediction_days, freq='D').tolist()

    # Create a DataFrame for the predicted prices
    predictions = pd.DataFrame({
        'date': date,
        'close': predicted_prices
    })

    # Combine historical and predicted data
    combined_data = pd.concat([data[['date', 'close']], predictions])

    # get the last 30 days data + requested prediction days
    combined_data = combined_data.iloc[-(30 +
                                         prediction_days):]

    # remove duplicate data from combined_data
    combined_data = combined_data.drop_duplicates(subset=['date'])

    # for historical data, set the predicted data to null
    historical_data = combined_data.copy()
    historical_data['close'].iloc[-(prediction_days):] = 'null'

    # for predicted data, set the historical data to null
    predictions = combined_data.copy()
    predictions['close'].iloc[:-(1 + prediction_days)] = 'null'

    # Convert combined_data to dictionary for JsonResponse
    combined_data_dict = {
        'labels': combined_data['date'].dt.strftime('%Y-%m-%d').tolist(),
        'historical_data': historical_data['close'].tolist(),
        'predictions': predictions['close'].tolist()
    }

    return JsonResponse(combined_data_dict)
