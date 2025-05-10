import pandas as pd
import torch
from torch.utils.data import DataLoader
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
import numpy as np

# Import QuantumLayer from your quantum_layer.py
from quantum_layer import QuantumLayer

# Load and prepare returns data
returns = pd.read_csv('portfolio_prices.csv', index_col='Date', parse_dates=True)
returns = np.log(returns / returns.shift(1)).dropna()

# Data split
split_date = '2023-01-01'
train_returns = returns[returns.index < split_date]
val_returns = returns[returns.index >= split_date]

# Convert data to PyTorch tensors
train_tensor = torch.tensor(train_returns.values, dtype=torch.float32)
val_tensor = torch.tensor(val_returns.values, dtype=torch.float32)

# Define classical Normalizing Flow architecture
num_features = train_returns.shape[1]

flow = Flow(
    transform=MaskedAffineAutoregressiveTransform(features=num_features, hidden_features=32),
    distribution=StandardNormal([num_features])
)

# Initialize Quantum Layer
quantum_layer = QuantumLayer()

# Optimizers for both quantum and classical parameters
optimizer = torch.optim.Adam(
    list(flow.parameters()) + list(quantum_layer.parameters()), lr=1e-3
)

# DataLoader
train_loader = DataLoader(train_tensor, batch_size=32, shuffle=True)

# Training loop with Quantum Integration
epochs = 50
for epoch in range(epochs):
    flow.train()
    quantum_layer.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        # Quantum layer preprocessing
        quantum_batch = torch.stack([quantum_layer(x[:4]) for x in batch])
        quantum_batch = quantum_batch.view(batch.size(0), -1)  # ensure correct shape


        # Combine quantum features with remaining classical features if necessary
        if quantum_batch.shape[1] < num_features:
            combined_input = torch.cat([quantum_batch, batch[:, quantum_batch.shape[1]:]], dim=1)
        else:
            combined_input = quantum_batch[:, :num_features]

        # Train flow on quantum-enhanced inputs
        loss = -flow.log_prob(combined_input).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{epochs}] - Quantum-Enhanced Training Loss: {avg_loss:.4f}')

# Validation
flow.eval()
quantum_layer.eval()

with torch.no_grad():
    quantum_val = torch.stack([quantum_layer(x[:4]) for x in val_tensor])

    if quantum_val.shape[1] < num_features:
        combined_val_input = torch.cat([quantum_val, val_tensor[:, quantum_val.shape[1]:]], dim=1)
    else:
        combined_val_input = quantum_val[:, :num_features]

    val_loss = -flow.log_prob(combined_val_input).mean().item()
    print(f'/nQuantum-Enhanced Validation Loss: {val_loss:.4f}')

# Generate synthetic samples
with torch.no_grad():
    synthetic_samples = flow.sample(5)
    print("/nQuantum-Enhanced Synthetic Samples:/n", synthetic_samples.numpy())

# Save trained model weights clearly
torch.save(flow.state_dict(), "trained_flow.pth")
torch.save(quantum_layer.state_dict(), "trained_quantum.pth")

print("✅ Models saved successfully!")
