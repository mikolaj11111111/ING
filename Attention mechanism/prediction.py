import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

class SimpleEnsemble(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(input_dim) / input_dim)

    def forward(self, x):
        weights_norm = torch.softmax(self.weights, dim=0)
        return torch.sum(weights_norm * x, dim=1, keepdim=True)

input_df = pd.read_excel("model_predictions.xlsx", sheet_name=2)

train_df = pd.read_excel("model_predictions.xlsx", sheet_name='Training Predictions')

X_columns = ['VEC', 'SDE', 'Linear']
y_column = ['Actual Value']

scaler_X = MinMaxScaler().fit(train_df[X_columns])
scaler_y = MinMaxScaler().fit(train_df[y_column])

input_df_scaled = scaler_X.transform(input_df[X_columns])
input_tensor = torch.tensor(input_df_scaled, dtype=torch.float32)

model = SimpleEnsemble(input_dim=3)
model.load_state_dict(torch.load('simple_ensemble.pth'))
model.eval()

with torch.no_grad():
    predictions_scaled = model(input_tensor).numpy()

predictions = scaler_y.inverse_transform(predictions_scaled).flatten()

input_df['Predicted Actual Value'] = predictions
input_df.to_excel("predictions_2025.xlsx", index=False)

print("Predykcje zapisane do pliku predictions_2025.xlsx")
