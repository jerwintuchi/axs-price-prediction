import joblib
import pandas as pd
import streamlit as st

def forecast():
    # Load the Forecaster object using joblib
    loaded_model = joblib.load('models/lstm_model.pkl')
    
    # Set the estimator to the desired trained model
    loaded_model.set_estimator('lstm')  
    loaded_model.generate_future_dates(2190)

    # Call the manual_forecast method for forecast
    predictions = loaded_model.manual_forecast(
    call_me='lstm_loaded'
    )

    return loaded_model

@st.cache_resource()
def save_forecast(_loaded_model):
    # call the forecast function
    forecast = _loaded_model

    # Save the predictions to a CSV file
    # Save forecast to CSV
    forecast_df = pd.DataFrame(_loaded_model.forecast, index=_loaded_model.future_dates, columns=['forecast']) # Convert list to DataFrame
    saved = forecast_df.to_csv('forecasts/latest_lstm_forecast.csv', index=True, index_label='DATE')

    return saved

save_forecast(forecast())