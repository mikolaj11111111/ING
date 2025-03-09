import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class SimpleEnsemble(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(input_dim) / input_dim)

    def forward(self, x):
        weights_norm = torch.softmax(self.weights, dim=0)
        return torch.sum(weights_norm * x, dim=1, keepdim=True)

train_df = pd.read_excel("model_predictions.xlsx", sheet_name='Training Predictions')
X_columns = ['VEC', 'SDE', 'Linear']

model = SimpleEnsemble(input_dim=3)
model.load_state_dict(torch.load('simple_ensemble.pth'))
model.eval()

with torch.no_grad():
    attention_weights = torch.softmax(model.weights, dim=0).numpy()


plt.figure(figsize=(6, 4))
plt.bar(X_columns, attention_weights, color=['blue', 'green', 'red'])
plt.xlabel("Features")
plt.ylabel("Attention Weights")
plt.title("Attention Weights Distribution")
plt.show()

print("Wagi atencji:")
for feature, weight in zip(X_columns, attention_weights):
    print(f"{feature}: {weight:.6f}")
