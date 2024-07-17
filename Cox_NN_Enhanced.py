#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:20:26 2024

@author: abdelhameed.ahmed
"""

import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class CoxPHNet(nn.Module):
    def __init__(self, input_dim):
        super(CoxPHNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Negative partial log-likelihood loss function
def cox_ph_loss(hazards, events, durations):
    # hazards: predicted log hazards
    # events: 1 if the event occurred, 0 for censoring
    # durations: time to event or censoring

    # Sort by durations in descending order
    order = torch.argsort(durations, descending=True)
    hazards = hazards[order]
    events = events[order]

    # Calculate log cumulative hazards
    #log_cum_hazards = torch.logcumsumexp(hazards, dim=0)
    
    
    # Calculate log cumulative hazards (reverse cumulative sum)
    # it is still ok if not flipped
    log_cum_hazards = torch.flip(torch.logcumsumexp(hazards, dim=0), dims=[0])

    # Calculate the log partial likelihood
    log_likelihood = hazards - log_cum_hazards

    # Only consider uncensored events
    log_likelihood = log_likelihood * events

    # Negative average log-likelihood
    return -torch.mean(log_likelihood)

# Generate synthetic data (for example purposes)
torch.manual_seed(0)
input_dim = 10
num_samples = 100
X = torch.randn(num_samples, input_dim)
durations = torch.randint(1, 100, (num_samples,), dtype=torch.float32)
events = torch.randint(0, 2, (num_samples,), dtype=torch.float32)

# Initialize the model, optimizer and loss function
model = CoxPHNet(input_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    hazards = model(X).squeeze()
    loss = cox_ph_loss(hazards, events, durations)

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")

# Model evaluation (example)
model.eval()
with torch.no_grad():
    hazards = model(X).squeeze()
    print(f"Final hazards: {hazards[:5]}")
