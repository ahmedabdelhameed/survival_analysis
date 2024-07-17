#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:51:25 2024

@author: abdelhameed.ahmed
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

# Generate a synthetic dataset
data_size = 100
features = np.random.normal(size=(data_size, 2))
durations = np.random.exponential(scale=10, size=(data_size, 1))
events = np.random.binomial(1, p=0.5, size=(data_size, 1))
data = np.hstack([features, durations, events])

df = pd.DataFrame(data, columns=['feature1', 'feature2', 'duration', 'event'])
X = df[['feature1', 'feature2']].values
y = df[['duration', 'event']].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert arrays to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Prepare DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=10, shuffle=True)

# Define the model
class CoxPHModel(nn.Module):
    def __init__(self):
        super(CoxPHModel, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Custom loss function for Cox model
def cox_loss(y_pred,y_true):
    # Extract durations and events from y_true
    durations, events = y_true[:, 0], y_true[:, 1]
    
    # Calculate the log hazard ratio
    log_hazard = y_pred.squeeze()
    
    #One implementation

    #Sort the durations and accordingly adjust log_hazard and events
    sorted_indices1 = torch.argsort(durations, descending=True)
    durations1 = durations[sorted_indices1]
    events1 = events[sorted_indices1]
    log_hazard1 = log_hazard[sorted_indices1]

    # Calculate the cumulative sum of exp(log_hazard1) for the risk set
    #risk_cumsum = torch.cumsum(torch.exp(log_hazard1), dim=0)
    risk_cumsum = torch.flip(torch.cumsum(torch.exp(log_hazard1), dim=0), dims=[0])
    
    # Calculate the log-likelihood components
    log_likelihood = log_hazard1 - torch.log(risk_cumsum)
    
    #Another implementation
    # sorted_indices2 = torch.argsort(durations)
    # durations2 = durations[sorted_indices2]
    # events2 = events[sorted_indices2]
    # log_hazard2 = log_hazard[sorted_indices2]
    
    # # Calculate the cumulative sum of exp(log_hazard) for the risk set
    # risk = torch.exp(log_hazard2)
    # risk_sum = torch.zeros_like(risk)
    
    # for i in range(len(risk)):
    #     risk_sum[i] = torch.sum(risk[i:])
        
    
    
    # # Calculate the log-likelihood components
    # log_likelihood = log_hazard2 - torch.log(risk_sum)
   
    
    # Consider only the events
    log_likelihood = log_likelihood * events
    
    # Return the negative log-likelihood to minimize
    return -torch.sum(log_likelihood)

# Initialize the model, optimizer, and loss function
model = CoxPHModel()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # L2 regularization

# Training loop with early stopping
def train_model(num_epochs, patience):
    best_loss = np.inf
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            inputs, labels = batch
            outputs = model(inputs)
            loss = cox_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss}')
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print('Early stopping!')
                break

# Train the model
train_model(num_epochs=100, patience=10)

# Predictions
model.eval()
with torch.no_grad():
    predictions = model(X_tensor)
    print("Predictions (log hazard ratios):", predictions[:10])

    # Calculate the hazard ratios by exponentiating the log hazard ratios
    hazard_ratios = torch.exp(predictions)
    print("Hazard Ratios:", hazard_ratios[:10])

# Convert events to boolean
events_bool = events.astype(bool).flatten()

# Calculate the C-index
durations = df['duration'].values
#events = df['event'].values
c_index = concordance_index_censored(events_bool, durations, -predictions)[0]
print(f'Concordance Index: {c_index}')



### For extimating time to event

#Compute the baseline survival function using Kaplan-Meier estimator
km_time, km_survival_prob = kaplan_meier_estimator(events_bool, durations)

#Compute hazard ratios
hazard_ratios = np.exp(predictions)


#Estimate individual survival functions
def estimate_survival_function(hazard_ratios, baseline_survival, times):
    return np.power(baseline_survival[:, np.newaxis], hazard_ratios)

baseline_survival = np.interp(durations, km_time, km_survival_prob, left=1.0, right=0.0)
individual_survival_functions = estimate_survival_function(hazard_ratios, baseline_survival, durations)


#Estimate median survival time for each individual

median_survival_times = []
for survival_prob in individual_survival_functions.T:
    median_survival_time = np.interp(0.5, np.flip(survival_prob), np.flip(km_time))
    median_survival_times.append(median_survival_time)

print("Median Survival Times:", median_survival_times[:10])
