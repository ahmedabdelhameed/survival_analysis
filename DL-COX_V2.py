import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from lifelines.utils import concordance_index
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic data with integer times in the range 0 to 800
num_samples = 10000
num_features = 10
X = np.random.randn(num_samples, num_features)
times = np.random.randint(0, 801, size=num_samples)  # Integer times in the range 0 to 800
events = np.random.binomial(1, p=0.5, size=num_samples)

# Standardize features
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / std

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
times = torch.tensor(times, dtype=torch.float32)
events = torch.tensor(events, dtype=torch.float32)

# Define a neural network model for the Cox Proportional Hazards model
class CoxModel(nn.Module):
    def __init__(self, input_dim):
        super(CoxModel, self).__init__()
        # First hidden layer with 100 units
        self.fc1 = nn.Linear(input_dim, 100)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
        # Second hidden layer with 50 units
        self.fc2 = nn.Linear(100, 50)
        self.relu2 = nn.ReLU()
        # Output layer with a single unit (log hazard ratio)
        self.fc3 = nn.Linear(50, 1)

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

# Initialize model, optimizer, and loss function
input_dim = num_features
model = CoxModel(input_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Training loop with gradient clipping and early stopping
n_epochs = 1000
patience = 10
best_loss = np.inf
epochs_no_improve = 0

for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    risk_scores = model(X)
    loss = cox_loss(torch.stack((times, events), dim=1), risk_scores)
    
    # Stop training if loss becomes NaN
    if torch.isnan(loss):
        print(f"NaN loss encountered at epoch {epoch+1}")
        break
    
    # Backpropagation and optimization
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}')
    
    # Early stopping mechanism
    if loss.item() < best_loss:
        best_loss = loss.item()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping!")
            break

# Function to calculate the survival function over a range of time points for new data
def calculate_survival_function(model, X, time_points):
    model.eval()
    with torch.no_grad():
        risk_scores = model(X).squeeze()
        # Calculate hazard function
        hazard_function = torch.exp(risk_scores).unsqueeze(1).expand(-1, time_points.size(0))
        # Calculate cumulative hazard
        cumulative_hazard = torch.cumsum(hazard_function, dim=1) * (time_points[1] - time_points[0])
        # Calculate survival function
        survival_function = torch.exp(-cumulative_hazard)
    return survival_function

# Define time points for survival function calculation
time_points = torch.tensor(np.arange(0, 731, 1), dtype=torch.float32)  # Time points from 0 to 720 days

# Generate new synthetic data for prediction (10 samples)
new_X = np.random.randn(10, num_features)
new_X = (new_X - mean) / std  # Standardize new data
new_X = torch.tensor(new_X, dtype=torch.float32)

# Calculate the survival function for the new data
survival_function = calculate_survival_function(model, new_X, time_points)

print("Survival function shape:", survival_function.shape)

# Predict survival within one year (365 days)
if 365 in time_points:
    idx_365 = (time_points == 365).nonzero(as_tuple=True)[0].item()
    survival_within_one_year = survival_function[:, idx_365] if survival_function.dim() > 1 else survival_function
else:
    survival_within_one_year = torch.zeros(new_X.size(0))

# Predict survival between one year (365 days) and two years (730 days)
if 730 in time_points:
    idx_730 = (time_points == 730).nonzero(as_tuple=True)[0].item()
    survival_between_one_and_two_years = survival_function[:, idx_730] if survival_function.dim() > 1 else survival_function
else:
    survival_between_one_and_two_years = torch.zeros(new_X.size(0))

print(f"Survival within one year: {survival_within_one_year}")
print(f"Survival between one and two years: {survival_between_one_and_two_years}")

# Plot the survival function for the first data point over 2 years
plt.figure(figsize=(10, 6))
plt.plot(time_points.numpy(), survival_function[0].numpy(), label='Survival Function (First Data Point)')

plt.xlabel('Time (days)')
plt.ylabel('Survival Probability')
plt.title('Survival Function over 2 Years for New Data Point')
plt.grid(True)
plt.legend()
plt.show()


# Calculate and print risk scores and hazard for the new data
risk_scores = model(new_X).squeeze()
hazard_function = torch.exp(risk_scores)
print("Risk scores for new data:", risk_scores)
print("Hazard function for new data:", hazard_function)

# Evaluate the model using C-Index on the training data
model.eval()
with torch.no_grad():
    predictions = model(X).squeeze()
c_index = concordance_index(times.numpy(), -predictions.numpy(), events.numpy())
print(f'C-Index: {c_index}')

# Calculate the survival function for the original training data
survival_function_train = calculate_survival_function(model, X, time_points)

# Predict survival within one year (365 days) for the training data
if 365 in time_points:
    idx_365_train = (time_points == 365).nonzero(as_tuple=True)[0].item()
    predicted_survival_prob_train = survival_function_train[:, idx_365_train].numpy()
else:
    predicted_survival_prob_train = torch.zeros(X.size(0)).numpy()

# Calculate Brier Score for one year survival prediction on the training data
observed_survival = (times.numpy() > 365).astype(int)  # Ensure this is a 1D array of length 1000
assert observed_survival.shape == predicted_survival_prob_train.shape, \
    f"Inconsistent shapes: observed_survival shape {observed_survival.shape}, predicted_survival_prob shape {predicted_survival_prob_train.shape}"

brier_score = brier_score_loss(observed_survival, predicted_survival_prob_train)
print(f'Brier Score for 1-year survival prediction: {brier_score}')

# Calibration plot (predicted vs. observed survival probabilities)
plt.plot(predicted_survival_prob_train, np.repeat(np.mean(observed_survival), len(predicted_survival_prob_train)), 'o')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('Predicted Survival Probability')
plt.ylabel('Actual Survival Probability')
plt.title(f'Calibration Plot for 1-Year Survival')
plt.show()
