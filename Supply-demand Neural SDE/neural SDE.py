import os
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchcde
import torchsde
from torch.utils.data import Dataset, DataLoader

###############################################################################
# Variable Explanations:
#
# D1 – Number of Completed Apartments ("mieszkania oddane do użytkowania"):
#      Supply-side metric representing the number of newly completed housing units.
#      Increased supply can lower prices if demand remains constant.
#
# D2 – Annual Growth Rate of New Loans to Households and Non-Financial Corporations:
#      Demand-side metric indicating the growth rate of new loans.
#      Higher growth signals more credit availability and increased demand.
#
# D3 – Value of Newly Signed Loan Agreements ("Wartość nowo podp umów w mld. Zł"):
#      Tracks the total value of new mortgage agreements.
#      Higher values indicate stronger financing activity and rising demand.
#
# D4 – Number of Housing Construction Starts ("Mieszkania, których budowę rozpoczęto"):
#      Reflects the number of new housing projects started.
#      Acts as a leading indicator for future supply.
#
# D5 – Political Stimulus (Binary Variable for Exceptional Events):
#      Flags major government interventions that can temporarily boost housing demand.
#
# D6 – Number of Newly Signed Mortgage Agreements ("Liczba nowo podpisanych umów kredytowych"):
#      Measures the number of new mortgage agreements, indicating buyer interest.
#
# D7 – Number of Building Permits Issued ("pozwolenia wydane na budowę i zgłoszenia budowy z projektem budowlanym"):
#      Early indicator of future supply through building permits.
#
# D8 – Population of Warsaw (Age Groups: 20-24, 25-29, 30-34, 35-39, 40-44):
#      Fundamental demand metric representing key age groups in Warsaw.
#
# PPM (Y1) – Average Transaction Price per Square Meter in Warsaw ("Średnia cena transakcyjna za metr w WWA"):
#      Core dependent variable representing the average price per square meter for new residential properties.
###############################################################################

###############################################################################
# 1) Ensure Reproducibility
###############################################################################
# Set random seeds for reproducibility across different libraries.
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)   # Set Python hash seed
    np.random.seed(seed)                       # Set numpy random seed
    random.seed(seed)                          # Set Python's built-in random seed
    torch.manual_seed(seed)                    # Set PyTorch CPU seed
    torch.random.manual_seed(seed)             # Set PyTorch's random seed
    torch.cuda.manual_seed(seed)               # Set PyTorch GPU seed (if available)
    torch.cuda.manual_seed_all(seed)           # Set all GPU seeds (if using multiple GPUs)
    # For exact reproducibility (may slow down training):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42  # Random seed value
seed_everything(SEED)

###############################################################################
# 2) Device Setup
###############################################################################
# Check if CUDA (GPU) is available and set the device accordingly.
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Using device:", device)

###############################################################################
# 3) Load & Preprocess Excel Data from Sheet2
###############################################################################
def parse_quarter(q_str):
    """
    Convert a quarter string (e.g., "2010 Q1") to a pandas Timestamp.
    The conversion selects the first month of the quarter.
    """
    year, quarter = q_str.split(" ")
    quarter = quarter.replace("Q", "")
    month = {"1": 1, "2": 4, "3": 7, "4": 10}[quarter]
    return pd.Timestamp(year=int(year), month=month, day=1)

def load_data(excel_path):
    """
    Load an Excel file from the sheet named "Sheet2".
    If a 'Date' column exists, convert quarter strings to timestamps and sort the data.
    
    Parameters:
      excel_path: Path to the Excel file.
      
    Returns:
      df: A pandas DataFrame containing the loaded and preprocessed data.
    """
    df = pd.read_excel(excel_path, sheet_name="Sheet2")
    if 'Date' in df.columns:
        # Convert quarter strings (e.g., "2010 Q1") to timestamps.
        df['Date'] = df['Date'].apply(lambda x: parse_quarter(x) if isinstance(x, str) else x)
        df.sort_values(by='Date', inplace=True)
    return df

###############################################################################
# 4) Create Sliding Windows for Forecasting on Differenced Data
###############################################################################
def create_sliding_windows_forecast_diff(data_array, diff_target_array, original_target_array, input_window=20, forecast_horizon=5, step_size=1):
    """
    Create sliding windows for forecasting using differenced target data.
    
    Parameters:
      data_array: Array of explanatory variables of shape [n, num_features] (standardized).
      diff_target_array: Differenced target series (standardized), length = n-1.
      original_target_array: Original target series, length = n.
      input_window: Number of past time points used as history.
      forecast_horizon: Number of future differences to forecast.
      step_size: The stride of the sliding window.
      
    Returns:
      inputs: Array of shape [num_windows, input_window, num_features] for the input windows.
      targets: Array of shape [num_windows, forecast_horizon] containing the forecast differences.
      window_indices: List of tuples (start_idx, end_idx, diff_target_end) for each window.
      baselines: Array of the last observed original target value from each input window.
    """
    inputs = []
    targets = []
    baselines = []
    window_indices = []
    n = len(original_target_array)
    # For the differenced series, note that diff_target_array has length n-1.
    for start_idx in range(0, n - input_window - forecast_horizon + 1, step_size):
        end_idx = start_idx + input_window
        diff_start = end_idx - 1
        diff_end = diff_start + forecast_horizon
        window = data_array[start_idx:end_idx]       # Explanatory variables window.
        targ = diff_target_array[diff_start:diff_end]  # Corresponding forecast differences.
        baseline = original_target_array[end_idx - 1]  # Last observed original target value.
        inputs.append(window)
        targets.append(targ)
        baselines.append(baseline)
        window_indices.append((start_idx, end_idx, diff_end))
    inputs = np.array(inputs)
    targets = np.array(targets)
    baselines = np.array(baselines)
    return inputs, targets, window_indices, baselines

###############################################################################
# 5) Build Cubic Spline Data from Input Windows Only
###############################################################################
def build_cde_data_from_input(sequences_std):
    """
    Build data tensors and compute cubic spline coefficients from input windows.
    
    Parameters:
      sequences_std: Standardized explanatory variable sequences.
                     If multivariate, shape [num_windows, input_window, num_features].
                     
    Returns:
      data_tensor: Tensor of shape [num_windows, input_window, 1 + num_features],
                   where the first channel is the normalized time.
      coeffs: Cubic spline coefficients computed over the input window.
    """
    if sequences_std.ndim == 2:
        num_windows, input_window = sequences_std.shape
        num_features = 1
        times_np = np.linspace(0, 1, input_window)
        all_data = []
        for seq in sequences_std:
            stacked = np.stack([times_np, seq], axis=-1)
            all_data.append(stacked)
    elif sequences_std.ndim == 3:
        num_windows, input_window, num_features = sequences_std.shape
        times_np = np.linspace(0, 1, input_window).reshape(input_window, 1)
        all_data = []
        for seq in sequences_std:
            # Concatenate the time column with the multivariate features.
            stacked = np.concatenate([times_np, seq], axis=1)  # shape: [input_window, 1+num_features]
            all_data.append(stacked)
    else:
        raise ValueError("Input sequences_std must be 2D or 3D")
    all_data = np.array(all_data, dtype=np.float32)
    data_tensor = torch.tensor(all_data, dtype=torch.float32)
    times_torch = torch.linspace(0, 1, sequences_std.shape[1])
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(data_tensor, times_torch)
    return data_tensor, coeffs

###############################################################################
# 6) Dataset & DataLoader for Forecasting
###############################################################################
class RealEstateForecastDataset(Dataset):
    """
    Custom Dataset for real estate forecasting.
    
    Attributes:
      data_tensor: Tensor of shape [num_windows, input_window, 1+num_features] containing input histories (with time channel).
      coeffs: Cubic spline coefficients computed from data_tensor.
      targets: Array of ground-truth standardized differenced target values with shape [num_windows, forecast_horizon].
    """
    def __init__(self, data_tensor, coeffs, targets):
        self.data_tensor = data_tensor
        self.coeffs = coeffs
        self.targets = targets
    def __len__(self):
        return len(self.data_tensor)
    def __getitem__(self, idx):
        target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)
        return self.data_tensor[idx], self.coeffs[idx], target_tensor

###############################################################################
# 7) Neural SDE Classes with External Signal Conditioning
###############################################################################
# Custom activation function: Lipswish.
class LipSwish(nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)

# Multi-layer Perceptron (MLP) used in various network parts.
class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_dim, num_layers, tanh=False, activation='lipswish'):
        """
        Parameters:
          in_size: Input feature dimension.
          out_size: Output feature dimension.
          hidden_dim: Dimension of hidden layers.
          num_layers: Number of hidden layers.
          tanh: If True, apply Tanh activation at the output.
          activation: Type of activation function to use ('lipswish' or other).
        """
        super().__init__()
        if activation == 'lipswish':
            activation_fn = LipSwish()
        else:
            activation_fn = nn.ReLU()
        layers = [nn.Linear(in_size, hidden_dim), activation_fn]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn)
        layers.append(nn.Linear(hidden_dim, out_size))
        if tanh:
            layers.append(nn.Tanh())
        self._model = nn.Sequential(*layers)
    def forward(self, x):
        return self._model(x)

# Neural SDE function with external signal conditioning via a cubic spline.
class NeuralSDEFunc(nn.Module):
    """
    Neural SDE function that incorporates an external signal.
    
    At each time t, it evaluates the external signal and concatenates it with t and the current hidden state.
    """
    def __init__(self, input_dim, hidden_dim, hidden_hidden_dim, num_layers, activation='lipswish'):
        """
        Parameters:
          input_dim: Dimension of the external signal (1 time channel + number of explanatory variables).
          hidden_dim: Dimension of the hidden state.
          hidden_hidden_dim: Dimension for hidden layers within the MLPs.
          num_layers: Number of layers in the MLPs.
          activation: Activation function type.
        """
        super().__init__()
        self.sde_type = "ito"
        self.noise_type = "diagonal"
        self.input_dim = input_dim  # External signal dimension.
        self.hidden_dim = hidden_dim
        # New input dimension: time (1) + hidden state (hidden_dim) + external signal (input_dim).
        self.linear_in = nn.Linear(1 + hidden_dim + input_dim, hidden_dim)
        self.f_net = MLP(hidden_dim, hidden_dim, hidden_hidden_dim, num_layers, activation=activation)
        self.noise_in = nn.Linear(1 + hidden_dim + input_dim, hidden_dim)
        self.g_net = MLP(hidden_dim, hidden_dim, hidden_hidden_dim, num_layers, activation=activation)
    def set_X(self, coeffs, times):
        """
        Set the cubic spline representation of the external signal.
        
        Parameters:
          coeffs: Cubic spline coefficients.
          times: Time grid for evaluation.
        """
        self.coeffs = coeffs
        self.times = times
        self.X = torchcde.CubicSpline(self.coeffs, self.times)
    def f(self, t, y):
        # Evaluate the external signal at time t.
        x_t = self.X.evaluate(t)  # Shape: [B, input_dim]
        t_tensor = torch.full((y.shape[0], 1), t, device=y.device)
        inp = torch.cat((t_tensor, y, x_t), dim=-1)
        h = self.linear_in(inp)
        return self.f_net(h)
    def g(self, t, y):
        x_t = self.X.evaluate(t)
        t_tensor = torch.full((y.shape[0], 1), t, device=y.device)
        inp = torch.cat((t_tensor, y, x_t), dim=-1)
        h = self.noise_in(inp)
        return self.g_net(h)

# Neural Differential Equation (NDE) model that wraps the Neural SDE function.
class NDE_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation='lipswish', vector_field=None):
        """
        Parameters:
          input_dim: Dimension of the external signal (1 time + explanatory variables).
          hidden_dim: Dimension of the hidden state.
          output_dim: Dimension of the prediction output.
          num_layers: Number of layers in the Neural SDE function.
          activation: Activation function type.
          vector_field: A lambda function to create an instance of NeuralSDEFunc.
        """
        super().__init__()
        # vector_field is a lambda that returns a NeuralSDEFunc.
        self.func = vector_field(input_dim, hidden_dim, hidden_dim, num_layers, activation)
        # Initial condition derived from the external signal at t=0.
        self.initial = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
    def forward(self, coeffs, times):
        self.func.set_X(coeffs, times)
        y0_raw = self.func.X.evaluate(times[0])  # [B, input_dim]
        y0 = self.initial(y0_raw)                # [B, hidden_dim]
        dt = times[1] - times[0]
        z = torchsde.sdeint(sde=self.func, y0=y0, ts=times, dt=dt, method='euler')
        z = z.permute(1, 0, 2)  # Rearranging shape to [B, T, hidden_dim]
        return self.decoder(z)  # Final output: [B, T, output_dim]

###############################################################################
# 8) Main Training, Forecasting, & Charting Script (No Data Leakage)
###############################################################################
def main():
    # Update the Excel file name accordingly.
    excel_path = "RealEstate_WWA.xlsx"  # Path to the Excel file containing the dataset.
    df = load_data(excel_path)
    
    # Define the target column and the explanatory variable columns.
    # target_col: The column representing the average transaction price per square meter in Warsaw.
    target_col = "Średnia cena transakcyjna za metr w WWA"
    explanatory_cols = [
        "mieszkania oddane do użytkowania",  # D1: Number of Completed Apartments
        "20-24",                             # D8: Population of Warsaw, Age 20-24
        "25-29",                             # D8: Population of Warsaw, Age 25-29
        "30-34",                             # D8: Population of Warsaw, Age 30-34
        "35-39",                             # D8: Population of Warsaw, Age 35-39
        "40-44",                             # D8: Population of Warsaw, Age 40-44
        "Annual growth rate of new loans to households and non-financial corporations",  # D2: Annual Growth Rate of New Loans
        "political stimulus",                # D5: Political Stimulus
        "Liczba nowo podpisanych umów kredytowych",  # D6: Number of Newly Signed Mortgage Agreements
        "Wartość nowo podp umów w mld. Zł",    # D3: Value of Newly Signed Loan Agreements
        "Mieszkania, których budowę rozpoczęto",  # D4: Number of Housing Construction Starts
        "pozwolenia wydane na budowę i zgłoszenia budowy z projektem budowlanym"  # D7: Number of Building Permits Issued
    ]
    
    # If there is a 'Date' column, use it for plotting; otherwise, use an index array.
    if 'Date' in df.columns:
        dates = df['Date'].values
    else:
        dates = np.arange(len(df))
    
    # Extract raw target values (original property prices).
    target_raw = df[target_col].values.astype(np.float32)
    
    # --------------------------
    # DIFFERENCING THE TARGET
    # --------------------------
    # Compute first differences: D[t] = X[t+1] - X[t]
    target_diff = np.diff(target_raw)  # Length is n-1.
    # Standardize the differenced target.
    diff_mean = target_diff.mean()       # Mean of differences.
    diff_std_dev = target_diff.std() + 1e-8  # Standard deviation of differences (with small epsilon).
    target_diff_std = (target_diff - diff_mean) / diff_std_dev
    
    # --------------------------
    # Extract and standardize explanatory variables.
    # --------------------------
    exog_raw = df[explanatory_cols].values.astype(np.float32)  # Raw explanatory variables.
    exog_std = np.zeros_like(exog_raw)  # Array to store standardized explanatory variables.
    exog_means = {}  # Dictionary to store means for each explanatory variable.
    exog_stds = {}   # Dictionary to store standard deviations for each explanatory variable.
    for i, col in enumerate(explanatory_cols):
        col_mean = exog_raw[:, i].mean()
        col_std = exog_raw[:, i].std() + 1e-8
        exog_std[:, i] = (exog_raw[:, i] - col_mean) / col_std
        exog_means[col] = col_mean
        exog_stds[col] = col_std
    
    # Set hyperparameters for sliding windows.
    input_window = 20    # Number of past time points used as history.
    forecast_horizon = 4 # Number of future steps to forecast (integrated differences yield t+1, t+2, ..., t+4).
    step_size = 1        # Stride for moving the sliding window.
    
    # Create sliding windows.
    # exog_std has length n; target_diff_std has length n-1.
    inputs, targets, window_indices, baselines = create_sliding_windows_forecast_diff(
        exog_std, target_diff_std, target_raw, input_window, forecast_horizon, step_size)
    num_windows = len(inputs)
    print(f"Total windows: {num_windows}")
    
    # Split the windows into training and testing sets (80% training, 20% testing).
    train_wins = int(0.8 * num_windows)
    train_inputs = inputs[:train_wins]
    train_targets = targets[:train_wins]
    test_inputs = inputs[train_wins:]
    test_targets = targets[train_wins:]
    train_idx = window_indices[:train_wins]
    test_idx = window_indices[train_wins:]
    train_baselines = baselines[:train_wins]
    test_baselines = baselines[train_wins:]
    
    # Build cubic spline coefficients using only the input (historical) portion.
    train_data_tensor, train_coeffs = build_cde_data_from_input(train_inputs)
    test_data_tensor, test_coeffs = build_cde_data_from_input(test_inputs)
    
    # Create Dataset objects and DataLoaders for training and testing.
    batch_size = 8  # Adjust batch size if there are fewer windows.
    train_dataset = RealEstateForecastDataset(train_data_tensor, train_coeffs, train_targets)
    test_dataset = RealEstateForecastDataset(test_data_tensor, test_coeffs, test_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # --------------------------
    # Model and Optimizer Setup
    # --------------------------
    num_explanatory = exog_std.shape[1]  # Number of explanatory variables (should be 12 as defined above).
    input_dim = 1 + num_explanatory      # Total input dimension: 1 time channel + explanatory features.
    output_dim = 1                     # Output dimension (predicting the standardized difference).
    hidden_dim = 32                    # Hidden state dimension.
    num_layers = 2                     # Number of layers in the neural networks.
    lr = 1e-3                        # Learning rate.
    num_epochs = 100                 # Number of training epochs (can be adjusted).
    
    # Create the Neural Differential Equation model.
    model = NDE_model(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        activation='lipswish',
        vector_field=lambda i, h1, h2, nl, act='lipswish': NeuralSDEFunc(i, h1, h2, nl, act)
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # Mean Squared Error loss.
    
    # Set up an extended time grid for SDE integration.
    dt = 1 / (input_window - 1)         # Time step based on input window.
    T_extended = 1 + dt * forecast_horizon  # Total time including forecast horizon.
    time_grid_extended = torch.linspace(0, T_extended, input_window + forecast_horizon).to(device)
    
    ############################################################################
    # Early Stopping Setup
    ############################################################################
    patience = 6       # Number of epochs to wait for improvement before stopping.
    best_loss = float('inf')
    no_improvement = 0
    
    ############################################################################
    # Training Loop with Early Stopping
    ############################################################################
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0.0
        for batch_data, batch_coeffs, batch_targets in train_loader:
            batch_data = batch_data.to(device)      # Shape: [B, input_window, 1+num_explanatory]
            batch_coeffs = batch_coeffs.to(device)
            batch_targets = batch_targets.to(device)  # Shape: [B, forecast_horizon]
    
            optimizer.zero_grad()
            # Predict on the extended time grid.
            y_pred_full = model(batch_coeffs, time_grid_extended)  # Shape: [B, input_window+forecast_horizon, 1]
            y_pred_full = y_pred_full.squeeze(-1)  # Shape: [B, input_window+forecast_horizon]
            # Use only the forecast portion (last forecast_horizon steps) which are the predicted standardized differences.
            y_pred_forecast = y_pred_full[:, input_window:]
            loss = criterion(y_pred_forecast, batch_targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
    
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_coeffs, batch_targets in test_loader:
                batch_data = batch_data.to(device)
                batch_coeffs = batch_coeffs.to(device)
                batch_targets = batch_targets.to(device)
                y_pred_full = model(batch_coeffs, time_grid_extended)
                y_pred_full = y_pred_full.squeeze(-1)
                y_pred_forecast = y_pred_full[:, input_window:]
                loss = criterion(y_pred_forecast, batch_targets)
                total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_loader)
        print(f"Epoch [{epoch}/{num_epochs}] Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")
    
        # Early Stopping check: Stop if no improvement in test loss for 'patience' epochs.
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
    
    ############################################################################
    # Prediction & Reconstruction for Test Windows (per forecast horizon)
    ############################################################################
    # Create dictionaries to store predictions and true values for each forecast horizon.
    pred_map = {h: {} for h in range(1, forecast_horizon+1)}
    true_map = {h: {} for h in range(1, forecast_horizon+1)}
    with torch.no_grad():
        for i, (input_data, coeff, target_diff_std_window) in enumerate(zip(test_data_tensor, test_coeffs, test_targets)):
            # Get the window indices for the current test sample.
            start_idx, end_idx, _ = test_idx[i]
            baseline = test_baselines[i]  # The last observed original target for this window.
            # Predict standardized differences on the extended time grid.
            input_data = input_data.unsqueeze(0).to(device)   # Shape: [1, input_window, 1+num_explanatory]
            coeff = coeff.unsqueeze(0).to(device)
            time_grid_extended = torch.linspace(0, T_extended, input_window + forecast_horizon).to(device)
            pred_diff_std_full = model(coeff, time_grid_extended)
            pred_diff_std_full = pred_diff_std_full.squeeze(0).squeeze(-1).cpu().numpy()  # Shape: [input_window+forecast_horizon]
            # Only take the forecast portion.
            pred_diff_std_forecast = pred_diff_std_full[input_window:]
            # Un-standardize predicted differences.
            pred_diff = pred_diff_std_forecast * diff_std_dev + diff_mean
            # Compute cumulative sum to recover forecasted original values.
            pred_forecast_original = baseline + np.cumsum(pred_diff)
            
            # Similarly, compute true original forecast from standardized differences.
            true_diff = target_diff_std_window * diff_std_dev + diff_mean
            true_forecast_original = baseline + np.cumsum(true_diff)
            
            # Store predictions and ground-truth for each forecast horizon.
            for h in range(1, forecast_horizon+1):
                # Global index corresponds to the forecasted original target position in the series.
                global_idx = end_idx - 1 + h
                pred_map[h].setdefault(global_idx, []).append(pred_forecast_original[h-1])
                true_map[h].setdefault(global_idx, []).append(true_forecast_original[h-1])
    
    # Aggregate overlapping predictions for each forecast horizon.
    aggregated_preds = {}
    aggregated_trues = {}
    horizons = range(1, forecast_horizon+1)
    
    # ---- Combined chart with 2x2 subplots for each forecast horizon ----
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    for idx, h in enumerate(horizons):
        row = idx // 2
        col = idx % 2
        ax = axs[row, col]
        sorted_keys = sorted(pred_map[h].keys())
        agg_pred = [np.mean(pred_map[h][k]) for k in sorted_keys]
        agg_true = [np.mean(true_map[h][k]) for k in sorted_keys]
        aggregated_preds[h] = np.array(agg_pred)
        aggregated_trues[h] = np.array(agg_true)
    
        x_axis = np.arange(len(sorted_keys))
        custom_positions = np.linspace(0, len(sorted_keys)-1, 5, dtype=int)
        if len(dates) > max(sorted_keys):
            custom_labels = [pd.to_datetime(dates[sorted_keys[i]]).strftime("%d.%m.%Y") for i in custom_positions]
        else:
            custom_labels = [str(sorted_keys[i]) for i in custom_positions]
        ax.plot(x_axis, aggregated_trues[h], color='red', label='Actual')
        ax.plot(x_axis, aggregated_preds[h], '--', color='green', label='Predicted')
        ax.set_xticks(custom_positions)
        ax.set_xticklabels(custom_labels, rotation=45)
        ax.set_title(f"Forecast t+{h}")
        ax.set_xlabel("Index")
        ax.set_ylabel("Average transaction price per square meter in WWA")
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig("RealEstate_prediction_all_continuous.png")
    print("Saved combined forecast figure to RealEstate_prediction_all_continuous.png")
    
    # ---- Verification Metrics ----
    for h in horizons:
        errors = aggregated_preds[h] - aggregated_trues[h]
        mse  = np.mean(errors**2)
        mae  = np.mean(np.abs(errors))
        rmse = math.sqrt(mse)
        mape = np.mean(np.abs((aggregated_trues[h] - aggregated_preds[h]) / aggregated_trues[h])) * 100
        # Compute R² as a measure of goodness-of-fit.
        r2 = 1 - np.sum((aggregated_trues[h] - aggregated_preds[h])**2) / np.sum((aggregated_trues[h] - np.mean(aggregated_trues[h]))**2)
        # Directional Prediction Accuracy (DPA)
        def trend_label(diff):
            return 'grows' if diff > 0 else ('goes_down' if diff < 0 else 'none')
        actual_labels = []
        pred_labels = []
        for i in range(len(aggregated_trues[h]) - 1):
            actual_diff = aggregated_trues[h][i+1] - aggregated_trues[h][i]
            pred_diff = aggregated_preds[h][i+1] - aggregated_preds[h][i]
            actual_labels.append(trend_label(actual_diff))
            pred_labels.append(trend_label(pred_diff))
        dpa = (sum(1 for a, p in zip(actual_labels, pred_labels) if a == p) / len(actual_labels)) * 100 if actual_labels else np.nan
        print(f"Horizon t+{h}: MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%, R²={r2:.4f}, DPA={dpa:.2f}%")
    
    # ---- Full Time Series Chart with Forecast for T+1 to T+4 ----
    # Use the last input window from the full dataset to generate the forecast.
    last_window_exog = exog_std[-input_window:]  # Latest historical explanatory variables.
    last_window_exog = np.expand_dims(last_window_exog, axis=0)  # Shape: [1, input_window, num_explanatory]
    last_data_tensor, last_coeffs = build_cde_data_from_input(last_window_exog)
    baseline_full = target_raw[-1]  # Last observed original target value.
    time_grid_extended = torch.linspace(0, T_extended, input_window + forecast_horizon).to(device)
    model.eval()
    with torch.no_grad():
        pred_diff_std_full = model(last_coeffs.to(device), time_grid_extended)
        pred_diff_std_full = pred_diff_std_full.squeeze(0).squeeze(-1).cpu().numpy()  # Shape: [input_window+forecast_horizon]
    pred_diff_std_forecast = pred_diff_std_full[input_window:]
    pred_diff_full = pred_diff_std_forecast * diff_std_dev + diff_mean
    pred_full_forecast = baseline_full + np.cumsum(pred_diff_full)
    
    # Generate forecast dates for the next 4 quarters.
    last_date = pd.to_datetime(dates[-1])
    forecast_dates = [last_date + pd.DateOffset(months=3*i) for i in range(1, forecast_horizon+1)]
    
    # Plot full series: actual data and the forecast.
    plt.figure(figsize=(12,6))
    plt.plot(dates, target_raw, color='blue', label='Actual Data')
    plt.plot(forecast_dates, pred_full_forecast, 'o--', color='green', label='Forecast')
    plt.title("Full Time Series with Forecast for T+1 to T+4")
    plt.xlabel("Date")
    plt.ylabel("Average transaction price per square meter in WWA")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("RealEstate_full_series_with_forecast.png")
    print("Saved full time series forecast figure to RealEstate_full_series_with_forecast.png")
        
if __name__ == "__main__":
    main()
