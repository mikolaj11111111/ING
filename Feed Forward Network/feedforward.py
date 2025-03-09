import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error, mean_absolute_error

def create_lagged_features(actual_series, district_series, n_lags=3):
    """
    Create features such that for each time t:
      - District features: values at t-n_lags to t-1.
      - Actual features: values at t-n_lags to t-1 plus the current actual at t.
    The target is the district value at time t.
    """
    X, y = [], []
    for i in range(n_lags, len(actual_series)):
        district_features = list(district_series[i - n_lags:i])
        actual_features = list(actual_series[i - n_lags:i]) + [actual_series[i]]
        features = district_features + actual_features
        X.append(features)
        y.append(district_series[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def build_ffn_model(input_dim):
    """
    Build a simple feedforward neural network (MLP) using Keras.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))  # Single output for price prediction
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    # -----------------------
    # 1) Read the Excel file
    # -----------------------
    file_name = 'Dzielnice Warszawy.xlsx'
    sheet_name = 'summary statistics'
    
    df = pd.read_excel(file_name, sheet_name=sheet_name)
    
    # Define the columns we need
    columns_needed = [
        'Date', 'Actual', 'Bemowo', 'Białołęka', 'Bielany', 'Mokotów', 'Ochota',
        'Praga-Południe', 'Praga-Północ', 'Rembertów', 'Targówek', 'Ursus',
        'Ursynów', 'Wawer', 'Wesoła', 'Wilanów', 'Wola', 'Włochy', 'Śródmieście', 'Żoliborz'
    ]
    df = df[columns_needed]
    
    # ----------------------------------------------------------------
    # 2) Filter data to use only 2020Q1–2024Q4 for training/validation
    # ----------------------------------------------------------------
    valid_quarters = [
        "2020Q1","2020Q2","2020Q3","2020Q4",
        "2021Q1","2021Q2","2021Q3","2021Q4",
        "2022Q1","2022Q2","2022Q3","2022Q4",
        "2023Q1","2023Q2","2023Q3","2023Q4",
        "2024Q1","2024Q2","2024Q3","2024Q4"
    ]
    train_df = df[df['Date'].isin(valid_quarters)].copy()
    train_df.sort_values(by='Date', inplace=True)
    
    # -----------------------
    # 3) Define Districts and settings
    # -----------------------
    district_names = [
        'Bemowo','Białołęka','Bielany','Mokotów','Ochota',
        'Praga-Południe','Praga-Północ','Rembertów','Targówek','Ursus',
        'Ursynów','Wawer','Wesoła','Wilanów','Wola','Włochy','Śródmieście','Żoliborz'
    ]
    forecast_results = {}
    n_lags = 3  # Number of lags for district features
    # Naively assume the Actual column remains constant in the future
    last_known_actual = train_df['Actual'].values[-1]
    
    # -----------------------
    # 4) Train a model per district and forecast future quarters
    # -----------------------
    for district in district_names:
        print(f"\nTraining Feedforward Network for District: {district}")
        
        actual_series = train_df['Actual'].values
        district_series = train_df[district].values
        
        X, y = create_lagged_features(actual_series, district_series, n_lags=n_lags)
        
        # Create a training and a small validation set
        if len(X) < 8:
            X_train, y_train = X, y
            X_val, y_val = X[-1:], y[-1:]
        else:
            split_index = len(X) - 4
            X_train, y_train = X[:split_index], y[:split_index]
            X_val, y_val = X[split_index:], y[split_index:]
        
        model = build_ffn_model(input_dim=X_train.shape[1])
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=4,
            verbose=0,
            validation_data=(X_val, y_val)
        )
        
        val_preds = model.predict(X_val).flatten()
        mse = mean_squared_error(y_val, val_preds)
        mae = mean_absolute_error(y_val, val_preds)
        print(f"Validation MSE: {mse:.2f}, MAE: {mae:.2f}")
        
        # -----------------------------------------------------
        # Rolling Forecast for 2025Q1–2025Q4
        # -----------------------------------------------------
        # For each forecast, we construct the input such that:
        # - District features: last n_lags district values.
        # - Actual features: last n_lags actual values + current quarter's Actual (naively set to last_known_actual).
        extended_district = list(district_series)
        extended_actual = list(actual_series)
        
        future_quarters = ["2025Q1", "2025Q2", "2025Q3", "2025Q4"]
        district_forecasts = []
        
        for step in range(4):  # forecasting 4 future quarters
            district_features = extended_district[-n_lags:]
            actual_features = extended_actual[-n_lags:] + [last_known_actual]
            x_input = np.array(district_features + actual_features, dtype=np.float32).reshape(1, -1)
            
            pred = model.predict(x_input)[0, 0]
            district_forecasts.append(pred)
            
            extended_district.append(pred)
            extended_actual.append(last_known_actual)
        
        forecast_results[district] = dict(zip(future_quarters, district_forecasts))
        print(f"Forecast for 2025Q1–Q4: {forecast_results[district]}")
    
    # -----------------------
    # 5) Append predictions to the original DataFrame and save new Excel file
    # -----------------------
    # Create new rows for each future quarter with the same columns as the original file.
    new_rows = []
    for quarter in future_quarters:
        row = {}
        for col in columns_needed:
            if col == 'Date':
                row[col] = quarter
            elif col == 'Actual':
                row[col] = last_known_actual
            elif col in district_names:
                row[col] = forecast_results[col][quarter]
            else:
                row[col] = np.nan  # In case of any extra columns
        new_rows.append(row)
    pred_df = pd.DataFrame(new_rows, columns=columns_needed)
    
    # Combine the original data with the new forecast rows.
    new_df = pd.concat([df, pred_df], ignore_index=True)
    new_df.to_excel("Dzielnice Warszawy_new.xlsx", index=False)
    print("\nNew Excel file 'Dzielnice Warszawy_new.xlsx' has been generated with predictions.")

if __name__ == "__main__":
    main()
