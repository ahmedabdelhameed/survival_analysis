"""
This script implements a neural network-based Cox Proportional Hazards (CoxPH) model using PyTorch to analyze survival data. 
The code generates a synthetic dataset with two features, survival durations, and event indicators, which represent whether an event (e.g., failure or death) occurred. 
The features are normalized, and both features and targets are converted into PyTorch tensors for model training. 
The model architecture consists of three fully connected layers with ReLU activations and dropout for regularization. 
A custom loss function is defined to compute the negative log-likelihood, which is minimized during training using the Adam optimizer. 
The training process includes early stopping to prevent overfitting, and the model is evaluated to make predictions on the input data.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set random seed for reproducibility
torch.manual_seed(42)

# Generate a synthetic dataset with 100 samples, each having two features,
# a duration (survival time), and an event indicator (binary).
data_size = 100
features = np.random.normal(size=(data_size, 2))
durations = np.random.exponential(scale=10, size=(data_size, 1))
events = np.random.binomial(1, p=0.5, size=(data_size, 1))
data = np.hstack([features, durations, events])

# Create a DataFrame and separate features and targets
df = pd.DataFrame(data, columns=['feature1', 'feature2', 'duration', 'event'])
X = df[['feature1', 'feature2']].values
y = df[['duration', 'event']].values

# Normalize features to have a mean of 0 and a standard deviation of 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert normalized features and targets into PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Prepare DataLoader for batch processing, with a batch size of 10
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=10, shuffle=True)

# Define a neural network model for the Cox Proportional Hazards model
class CoxPHModel(nn.Module):
    def __init__(self):
        super(CoxPHModel, self).__init__()
        self.fc1 = nn.Linear(2, 32)  # Input layer to hidden layer
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Dropout layer to prevent overfitting
        self.fc2 = nn.Linear(32, 16)  # Hidden layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)  # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Custom loss function for Cox model to calculate the negative log-likelihood
def cox_loss(y_true, y_pred):
    durations, events = y_true[:, 0], y_true[:, 1]
    
    # Calculate the log hazard ratio
    log_hazard = y_pred.squeeze()

    # Sort the durations and accordingly adjust log_hazard and events
    sorted_indices = torch.argsort(durations)
    durations = durations[sorted_indices]
    events = events[sorted_indices]
    log_hazard = log_hazard[sorted_indices]

    # Calculate the cumulative sum of exp(log_hazard) for the risk set
    risk = torch.exp(log_hazard)
    risk_sum = torch.zeros_like(risk)
    
    for i in range(len(risk)):
        risk_sum[i] = torch.sum(risk[i:])
    
    # Calculate the log-likelihood components and consider only the events
    log_likelihood = log_hazard - torch.log(risk_sum)
    log_likelihood = log_likelihood * events
    
    # Return the negative log-likelihood to minimize
    return -torch.sum(log_likelihood)

# Initialize the model, optimizer (Adam with L2 regularization), and loss function
model = CoxPHModel()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Training loop with early stopping to avoid overfitting
def train_model(num_epochs, patience):
    best_loss = np.inf
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()  # Reset gradients
            inputs, labels = batch
            outputs = model(inputs)
            loss = cox_loss(labels, outputs)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss}')
        
        # Implement early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print('Early stopping!')
                break

# Train the model with 100 epochs and early stopping patience of 10
train_model(num_epochs=100, patience=10)

# Make predictions on the original dataset
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    predictions = model(X_tensor)
    print("Predictions:", predictions[:10])
