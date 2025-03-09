import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

class SimpleEnsemble(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(input_dim) / input_dim)

    def forward(self, x):
        weights_norm = torch.softmax(self.weights, dim=0)
        return torch.sum(weights_norm * x, dim=1, keepdim=True)

train_df = pd.read_excel("model_predictions.xlsx", sheet_name='Training Predictions').dropna()

X_columns = ['VEC', 'SDE', 'Linear']
y_column = ['Actual Value']

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_X.fit_transform(train_df[X_columns])
y_train = scaler_y.fit_transform(train_df[y_column])

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

model = SimpleEnsemble(input_dim=3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

torch.save(model.state_dict(), 'simple_ensemble.pth')
