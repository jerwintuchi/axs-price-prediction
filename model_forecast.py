import joblib
import pandas as pd
import streamlit as st
import os
def forecast():
    # Load the Forecaster object using joblib
    loaded_model = joblib.load('models/lstm_model.pkl')
    
    # Set the estimator to the desired trained model
    loaded_model.set_estimator('lstm')  
    loaded_model.generate_future_dates(2190)

    # Call the manual_forecast method for forecast
    predictions = loaded_model.manual_forecast(
    test_again=False,
    call_me='lstm_loaded'
    )

    return loaded_model

@st.cache_resource()
def save_forecast(_loaded_model):
    forecast_data = _loaded_model.forecast
    future_dates = _loaded_model.future_dates

    forecast_df = pd.DataFrame(forecast_data, index=future_dates, columns=['forecast'])
    forecast_df.to_csv('forecasts/latest_lstm_forecast.csv', index=True, index_label='DATE')


def get_forecast():
    csv_path = 'forecasts/latest_lstm_forecast.csv'
    
    if os.path.exists(csv_path):
        forecast_df = pd.read_csv(csv_path)
        return forecast_df
    else:
        loaded_model = forecast()  # original forecast() function
        save_forecast(loaded_model)
        forecast_df = pd.read_csv(csv_path, parse_dates=['DATE'])
        return forecast_df

