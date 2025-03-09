"""
This script implements a Macroeconomic VAR/VEC Model for Polish real estate price forecasting.
All comments are in English, and references to macroeconomic variables (M1..M6) are provided below.

Variables (M1..M6) Explanation:
- M1: GDP [PLN] for Poland (Gross Domestic Product). 
      A fundamental macroeconomic indicator measuring the total value of goods and services produced.
      Data Source: FRED Economic Data | St. Louis FED

- M2: Lombard Rate (lom [%]) for Poland.
      The cost of borrowing set by the central bank. It influences mortgage rates and credit availability.
      Data Source: National Bank of Poland

- M3: Composite Consumer Confidence (CCC) for Poland.
      Reflects consumer optimism about the economy, influencing willingness to make large purchases.
      Data Source: FRED Economic Data | St. Louis FED

- M4: Private Mortgage Insurance (PMI) for Poland.
      Affects mortgage affordability and lending risk. Changes in PMI can stimulate or deter housing demand.
      Data Source: Bankier.pl

- M5: Inflation (Inflacja do okresu w poprzednim roku).
      Measures the general increase in prices. Often correlated with real estate prices as a hedge against inflation.
      Data Source: Główny Urząd Statystyczny (GUS), Poland

- M6: Registered Unemployment Rate for Poland.
      Influences household income and housing demand. Higher unemployment generally suppresses property prices.
      Data Source: FRED Economic Data | St. Louis FED

Target Variable:
- "Średnia cena transakcyjna za metr w WWA" = Average Transaction Price per Square Meter in Warsaw.
  This is our main dependent variable for forecasting.

Usage:
1. Ensure you have the Excel file "VAR VEC model macro.xlsx" with the expected columns.
2. Run this script in a Python environment with the required libraries installed.
3. The script will perform Johansen cointegration testing, fit either a VECM or VAR model, 
   generate rolling forecasts, and produce plots of actual vs. predicted values.
"""

import warnings
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

warnings.filterwarnings("ignore", message="An unsupported index was provided")

#====================================================
# 1. Data Loading & Basic Setup
#====================================================
filename = "VAR VEC model macro.xlsx"

# Expected columns in the Excel file (Polish naming is retained in the code for continuity)
cols_in_excel = [
    "GDP [PLN]",  # M1
    "lom [%]",    # M2
    "Inflacja do okresu w poprzednim roku",  # M5
    "PMI",        # M4
    "CCC",        # M3
    "Registered Unemployment",  # M6
    "Średnia cena transakcyjna za metr w WWA",  # Target: Avg. price per sqm in Warsaw
    "Średnia cena diff"  # First difference of the target price
]

# Read the Excel file, keeping only the columns of interest
df = pd.read_excel(filename)
df = df[cols_in_excel].copy()
df.ffill(inplace=True)  # Forward fill any missing values

# Create a PeriodIndex from 2010Q1 onward
start_period = pd.Period('2010Q1', freq='Q')
full_periods = len(df)
date_index = pd.period_range(start=start_period, periods=full_periods, freq='Q')
df.index = date_index

# The original price level series (for final comparisons and reintegration)
price_level_series = df["Średnia cena transakcyjna za metr w WWA"]

# Train/Test split (80% training, 20% test)
n = len(df)
train_size = int(0.8 * n)
df_train = df.iloc[:train_size].dropna()
df_test = df.iloc[train_size:].copy()

price_level_train = price_level_series.iloc[:train_size]
price_level_test = price_level_series.iloc[train_size:]

full_dates = df.index
train_dates = full_dates[:train_size]
test_dates = full_dates[train_size:]

#====================================================
# 2. Johansen Cointegration Test (Train Only)
#====================================================
# We test for cointegration among I(1) variables.
# The price level itself is I(2), so we use its first difference ("Średnia cena diff") for the cointegration test.
coint_cols = [
    "GDP [PLN]",  # M1
    "lom [%]",    # M2
    "Inflacja do okresu w poprzednim roku",  # M5
    "PMI",        # M4
    "CCC",        # M3
    "Registered Unemployment",  # M6
    "Średnia cena diff"         # I(1) version of the target price
]
coint_data = df_train[coint_cols].copy()

# Define the lag order in differences for the VECM
k_ar_diff_val = 2

# Johansen test for cointegration
joh_result = coint_johansen(coint_data, det_order=0, k_ar_diff=k_ar_diff_val)
r = 0
for i, (trace_stat, crit_val_5pct) in enumerate(zip(joh_result.lr1, joh_result.cvt[:, 1])):
    if trace_stat > crit_val_5pct:
        r = i + 1
print(f"Johansen Cointegration Test: {r} cointegrating relationship(s) found.")

use_vecm = (r > 0)

#====================================================
# 3. Fit VECM or VAR (Train Only)
#====================================================
if use_vecm:
    print("\nCointegration detected. Estimating VECM on training set.")
    vecm_model = VECM(
        coint_data,
        k_ar_diff=k_ar_diff_val,
        coint_rank=r,
        deterministic="n"
    )
    vecm_res = vecm_model.fit()
    print(vecm_res.summary())
    model_label = "VECM"
else:
    print("\nNo cointegration detected. Estimating VAR on twice-differenced data (training set).")
    model_label = "VAR"
    # For the VAR model, we second-difference the target price and first-difference other variables
    df_diff = pd.DataFrame()
    for col in cols_in_excel:
        if col == "Średnia cena transakcyjna za metr w WWA":
            continue
        elif col == "Średnia cena diff":
            df_diff["Średnia cena diff2"] = df_train[col].diff()
        else:
            df_diff[col + " diff"] = df_train[col].diff()
    df_diff.dropna(inplace=True)
    
    var_model = VAR(df_diff)
    results_aic = var_model.fit(maxlags=6, ic="aic")
    print(f"Selected VAR lag order (AIC): {results_aic.k_ar}")
    var_res = results_aic

#====================================================
# 3.5 Model Stability Check: Eigenvalues and Unit Circle
#====================================================
plt.figure(figsize=(6, 6))
theta = np.linspace(0, 2 * np.pi, 200)
plt.plot(np.cos(theta), np.sin(theta), 'b--', label='Unit Circle')

if use_vecm:
    # For a VECM, the underlying VAR representation has p = k_ar_diff_val + 1 lags.
    k = vecm_res.alpha.shape[0]   # number of endogenous variables
    p = k_ar_diff_val + 1         # total VAR lags
    I_k = np.eye(k)
    
    # Short-run parameters from the fitted VECM
    Gamma_1 = vecm_res.gamma[:, :k]
    # If (p-1) >= 2, then we have a second Gamma_2 block
    Gamma_2 = vecm_res.gamma[:, k:2*k] if (p - 1) >= 2 else np.zeros((k, k))
    
    # Build the VAR coefficient matrices for the companion form
    A1 = I_k + np.dot(vecm_res.alpha, vecm_res.beta.T) + Gamma_1
    A2 = -Gamma_1 + Gamma_2
    
    # If p=3, we might have A3 = -Gamma_2
    if (p - 1) == 2:
        A3 = -Gamma_2
        companion_top = np.hstack([A1, A2, A3])
        companion_bottom = np.hstack([np.eye(k * (p - 1)), np.zeros((k * (p - 1), k))])
        companion = np.vstack([companion_top, companion_bottom])
    else:
        # If p=2, only A1 and A2 are used in the companion matrix
        companion_top = np.hstack([A1, A2])
        companion_bottom = np.hstack([np.eye(k * (p - 1)), np.zeros((k * (p - 1), k))])
        companion = np.vstack([companion_top, companion_bottom])
    
    eigenvalues = np.linalg.eigvals(companion)
else:
    eigenvalues = var_res.roots

plt.plot(eigenvalues.real, eigenvalues.imag, 'ro', label='Eigenvalues')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title(f'Model Stability: Eigenvalues ({model_label})')
plt.legend()
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.savefig("Model_Stability_Eigenvalues.png")
plt.show()
print("Saved stability plot to 'Model_Stability_Eigenvalues.png'")

#====================================================
# 3.6 Test Stationarity of the Error Correction Term (VECM only)
#====================================================
if use_vecm:
    # The Error Correction Term (ECT) is the cointegrating combination of the variables
    ect = coint_data.dot(vecm_res.beta)
    
    # If r=1, we have a single ECT to test
    if r == 1:
        ect_series = ect.iloc[:, 0]
        adf_result = adfuller(ect_series)
        print("\nADF test for Error Correction Term (ECT):")
        print("Test Statistic:", adf_result[0])
        print("p-value:", adf_result[1])
        print("Critical Values:", adf_result[4])
    else:
        # If r>1, test each cointegrating relation separately
        for i in range(r):
            ect_series = ect.iloc[:, i]
            adf_result = adfuller(ect_series)
            print(f"\nADF test for Error Correction Term {i+1}:")
            print("Test Statistic:", adf_result[0])
            print("p-value:", adf_result[1])
            print("Critical Values:", adf_result[4])
    
    # Print the adjustment coefficients (alpha), indicating speed of adjustment to equilibrium
    print("\nAdjustment Coefficients (alpha):")
    alpha_df = pd.DataFrame(
        vecm_res.alpha,
        index=coint_cols,
        columns=[f"ECT {i+1}" for i in range(r)]
    )
    print(alpha_df)

#====================================================
# 4. Rolling Multi-Step Forecast over Test Set
#====================================================
horizons = [1, 2, 3, 4]

def forecast_multi_step_VECM(vecm_fitted, df_history, h):
    """
    Perform iterative multi-step forecast (h steps) for a fitted VECM.
    We forecast first differences (Średnia cena diff).
    """
    hist_copy = df_history.copy()
    preds_diff = []
    for _ in range(h):
        fc = vecm_fitted.predict(steps=1)  # shape (1, #vars)
        fc_series = pd.Series(fc[0], index=hist_copy.columns)
        
        # Append the forecast as the next row
        next_row = hist_copy.iloc[-1].copy()
        for col in hist_copy.columns:
            next_row[col] = hist_copy.iloc[-1][col] + fc_series[col]
        hist_copy = pd.concat([hist_copy, next_row.to_frame().T], ignore_index=True)
        
        preds_diff.append(fc_series["Średnia cena diff"])
    return preds_diff

def forecast_multi_step_VAR(var_fitted, df_history, h):
    """
    Perform iterative multi-step forecast (h steps) for a fitted VAR on differenced data.
    We forecast the second differences (Średnia cena diff2).
    """
    hist_copy = df_history.copy()
    p = var_fitted.k_ar
    preds_second_diff = []
    for _ in range(h):
        last_p = hist_copy.iloc[-p:].values
        fc = var_fitted.forecast(last_p, steps=1)
        fc_series = pd.Series(fc[0], index=hist_copy.columns)
        
        new_row = {col: fc_series[col] for col in hist_copy.columns}
        hist_copy = pd.concat([hist_copy, pd.DataFrame([new_row])], ignore_index=True)
        
        preds_second_diff.append(fc_series["Średnia cena diff2"])
    return preds_second_diff

# Dictionaries to store predictions: horizon -> {date -> predicted_value}
pred_map = {h: {} for h in horizons}
true_map = {h: {} for h in horizons}

# For the VECM, we use the cointegration columns. For VAR, we use differenced columns.
if use_vecm:
    df_full_coint = pd.concat([df_train[coint_cols], df_test[coint_cols]])
else:
    df_full_diff = pd.DataFrame()
    for col in cols_in_excel:
        if col == "Średnia cena transakcyjna za metr w WWA":
            continue
        elif col == "Średnia cena diff":
            df_full_diff["Średnia cena diff2"] = df[col].diff()
        else:
            df_full_diff[col + " diff"] = df[col].diff()
    df_full_diff.dropna(inplace=True)

def reintegrate_VECM_price(pred_diffs, last_price_level):
    """
    Convert predicted first differences back to levels.
    """
    levels = [last_price_level]
    for d in pred_diffs:
        levels.append(levels[-1] + d)
    return levels[1:]

def reintegrate_VAR_price(pred_second_diffs, last_price_level, last_first_diff):
    """
    Convert predicted second differences back to levels via first differences.
    """
    first_diffs = [last_first_diff]
    for sd in pred_second_diffs:
        first_diffs.append(first_diffs[-1] + sd)
    forecasted_first_diffs = np.array(first_diffs[1:])
    
    levels = [last_price_level]
    for fd in forecasted_first_diffs:
        levels.append(levels[-1] + fd)
    return levels[1:]

# Rolling forecast over each point in the test set
test_len = len(df_test)
for t in range(test_len):
    current_date = test_dates[t]
    
    if use_vecm:
        history_i1 = df_full_coint.loc[:current_date].copy()
    else:
        history_diff = df_full_diff.loc[:current_date].copy()
    
    for h in horizons:
        if (t + h) >= test_len:
            continue
        
        future_date = test_dates[t + h]
        last_price_level = price_level_series.loc[current_date]
        
        if use_vecm:
            preds_diff = forecast_multi_step_VECM(vecm_res, history_i1, h)
            forecasted_levels = reintegrate_VECM_price(preds_diff, last_price_level)
            forecast_val = forecasted_levels[-1]
        else:
            last_first_diff = df.loc[current_date, "Średnia cena diff"]
            preds_second_diff = forecast_multi_step_VAR(var_res, history_diff, h)
            forecasted_levels = reintegrate_VAR_price(preds_second_diff, last_price_level, last_first_diff)
            forecast_val = forecasted_levels[-1]
        
        pred_map[h][future_date] = forecast_val
        true_map[h][future_date] = price_level_series.loc[future_date]

#====================================================
# 5. Multi-Horizon Forecast (Test Set): 2×2 Subplots
#====================================================
aggregated_preds = {}
aggregated_trues = {}

for h in horizons:
    sorted_dates = sorted(true_map[h].keys())
    y_pred = [pred_map[h][d] for d in sorted_dates if d in pred_map[h]]
    y_true = [true_map[h][d] for d in sorted_dates if d in true_map[h]]
    aggregated_preds[h] = np.array(y_pred)
    aggregated_trues[h] = np.array(y_true)

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
for idx, h in enumerate(horizons):
    row = idx // 2
    col = idx % 2
    ax = axs[row, col]
    
    sorted_dates = sorted(true_map[h].keys())
    if len(sorted_dates) == 0:
        ax.set_title(f"Multi-Horizon Forecast t+{h} (No data)")
        continue
    
    y_true = [true_map[h][d] for d in sorted_dates]
    y_pred = [pred_map[h][d] for d in sorted_dates]
    x_axis = [d.to_timestamp() for d in sorted_dates]
    
    ax.plot(x_axis, y_true, color='red', label='Actual')
    ax.plot(x_axis, y_pred, '--', color='green', label='Predicted')
    ax.set_title(f"Multi-Horizon Forecast t+{h} (Test Set)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Transaction Price per Square Meter in Warsaw")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.savefig("Multiple_Horizon_Forecast_Comparisons.png")
plt.show()
print("Saved multi-horizon forecast figure to 'Multiple_Horizon_Forecast_Comparisons.png'")

#====================================================
# 5.1 Single-Plot for t+1 Forecast in the 20% Out-of-Sample (2022Q1..2024Q4)
#====================================================
test_horizon = 1
test_start = pd.Period('2022Q1', freq='Q')
test_end = pd.Period('2024Q4', freq='Q')

sorted_test_dates = sorted(d for d in pred_map[test_horizon].keys() if test_start <= d <= test_end)
y_true_test = [price_level_series.loc[d] for d in sorted_test_dates]
y_pred_test = [pred_map[test_horizon][d] for d in sorted_test_dates]

plt.figure(figsize=(10, 6))
plt.plot([d.to_timestamp() for d in sorted_test_dates], y_true_test, color='blue', label='Actual', marker='o')
plt.plot([d.to_timestamp() for d in sorted_test_dates], y_pred_test, color='orange', label='Predicted', marker='x')
plt.title("Actual vs. Predicted PPM in Warsaw (Y1)")
plt.xlabel("Date")
plt.ylabel("Average Transaction Price per Square Meter in Warsaw")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Actual_vs_Predicted_PPM_in_Warsaw_Y1.png")
plt.show()
print("Saved single-plot figure 'Actual_vs_Predicted_PPM_in_Warsaw_Y1.png' for test horizon t+1 (2022Q1..2024Q4).")

#====================================================
# 6. Verification Metrics (MAE, MSE, RMSE, MAPE, R^2, DPA)
#====================================================
for h in horizons:
    y_pred = aggregated_preds[h]
    y_true = aggregated_trues[h]
    if len(y_pred) == 0:
        print(f"Horizon t+{h}: No forecasts generated.")
        continue
    
    errors = y_pred - y_true
    mse = np.mean(errors**2)
    mae = np.mean(np.abs(errors))
    rmse = math.sqrt(mse)
    mape = np.mean(np.abs(errors / y_true)) * 100 if np.all(y_true != 0) else np.nan
    
    # R^2 calculation
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    
    # Directional Prediction Accuracy (DPA)
    def trend_label(diff):
        return 'grows' if diff > 0 else ('goes_down' if diff < 0 else 'none')
    
    actual_labels = []
    pred_labels = []
    for i in range(len(y_true) - 1):
        actual_diff = y_true[i+1] - y_true[i]
        pred_diff = y_pred[i+1] - y_pred[i]
        actual_labels.append(trend_label(actual_diff))
        pred_labels.append(trend_label(pred_diff))
    if actual_labels:
        dpa = (sum(1 for a, p in zip(actual_labels, pred_labels) if a == p) / len(actual_labels)) * 100
    else:
        dpa = np.nan
    
    print(f"Horizon t+{h}: "
          f"MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, "
          f"MAPE={mape:.2f}%, R^2={r2:.4f}, DPA={dpa:.2f}%")

#====================================================
# 7. Forecast for 2025Q1..2025Q4 from the LAST KNOWN data point (2024Q4)
#====================================================
final_date = full_dates[-1]  # Expected to be 2024Q4
if use_vecm:
    df_full_coint_all = pd.concat([df_train[coint_cols], df_test[coint_cols]])
    df_history = df_full_coint_all.loc[:final_date].copy()
else:
    df_full_diff_all = pd.DataFrame()
    for col in cols_in_excel:
        if col == "Średnia cena transakcyjna za metr w WWA":
            continue
        elif col == "Średnia cena diff":
            df_full_diff_all["Średnia cena diff2"] = df[col].diff()
        else:
            df_full_diff_all[col + " diff"] = df[col].diff()
    df_full_diff_all.dropna(inplace=True)
    df_history = df_full_diff_all.loc[:final_date].copy()

forecast_h = 4
last_price_level = price_level_series.loc[final_date]

if use_vecm:
    preds_diff_4 = forecast_multi_step_VECM(vecm_res, df_history, forecast_h)
    forecasted_levels_4 = reintegrate_VECM_price(preds_diff_4, last_price_level)
else:
    last_first_diff = df.loc[final_date, "Średnia cena diff"]
    preds_second_diff_4 = forecast_multi_step_VAR(var_res, df_history, forecast_h)
    forecasted_levels_4 = reintegrate_VAR_price(preds_second_diff_4, last_price_level, last_first_diff)

future_periods = pd.period_range(start=final_date + 1, periods=4, freq='Q')

#====================================================
# 8. Plot the Full Data Sample (2010Q1..2024Q4) + Predictions for 2025Q1..2025Q4
#====================================================
plt.figure(figsize=(12, 6))
plt.plot(full_dates.to_timestamp(), price_level_series.values, 
         color='blue', marker='o', label='Actual (2010Q1..2024Q4)')

forecast_ts = [p.to_timestamp() for p in future_periods]
plt.plot(forecast_ts, forecasted_levels_4, '--x', color='green', label='Forecast (2025Q1..2025Q4)')

plt.title("Full Dataset + Forecast (2025Q1..2025Q4) - No Overlap with 20% Test")
plt.xlabel("Date")
plt.ylabel("Average Transaction Price per Square Meter in Warsaw")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("RealEstate_full_series_with_2025_forecast.png")
plt.show()

print("\nSaved full-sample figure with 2025 forecast to 'RealEstate_full_series_with_2025_forecast.png'.")
print("Done.")
