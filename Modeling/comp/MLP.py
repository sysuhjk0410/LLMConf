import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score

# Read the CSV file
data = pd.read_csv('/LLMConf/Modeling/data.csv')

# Columns 1 to 10 are configuration parameters, the remaining columns are 7 evaluation metrics
X = data.iloc[:, :10].values  # Independent variables: configuration parameters

# Names of the evaluation metrics
metrics = ['latency_average', 'latency_p99',
           'tokens_per_second_average', 
           'time_to_first_token_average', 'time_to_first_token_p99',
           'time_per_output_token_average', 'time_per_output_token_p99']

# Data normalization
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Parameter configuration
input_dim = 10  # Input dimension: 10 configuration parameters
hidden_dim = 64  # Hidden layer dimension
output_dim = 1   # Output dimension: evaluation metric

# Training and prediction function construction
for i, metric in enumerate(metrics):
    print(f"Training MLP model for {metric}...")

    # Get the dependent variable for the current evaluation metric 
    y = data.iloc[:, 10 + i].values

    # Normalize evaluation metric data
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).reshape(-1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Create MLP model
    model = MLP(input_dim, hidden_dim, output_dim)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Train the model
    epochs = 150
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

    # After model training, evaluate performance
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        test_predictions = model(X_test_tensor).numpy()

    # Inverse transform the predicted results and actual values
    test_predictions_inverse = scaler_y.inverse_transform(test_predictions)
    y_test_inverse = scaler_y.inverse_transform(y_test_tensor.numpy())

    # Calculate Mean Squared Error (MSE) and R² score
    mse = mean_squared_error(y_test_inverse, test_predictions_inverse)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inverse, test_predictions_inverse)

    print(f"Model for {metric}:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}\n")

    # Save the model and prediction function
    globals()[f"model_{metric}"] = model  # Dynamically create model variable

    # Define the prediction function
    def make_predict_function(metric_name, model, scaler_X, scaler_y):
        def predict(max_num_batched_tokens, max_num_seqs, swap_space, block_size,
                    scheduler_delay_factor, gpu_memory_utilization, enable_chunked_prefill,
                    enable_prefix_caching, disable_custom_all_reduce, use_v2_block_manager):

            model.eval()  # Set the model to evaluation mode
            config_params = np.array([max_num_batched_tokens, max_num_seqs, swap_space, block_size,
                                      scheduler_delay_factor, gpu_memory_utilization, 
                                      int(enable_chunked_prefill), int(enable_prefix_caching),
                                      int(disable_custom_all_reduce), int(use_v2_block_manager)])
            config_params_scaled = scaler_X.transform([config_params])
            config_params_tensor = torch.tensor(config_params_scaled, dtype=torch.float32)

            with torch.no_grad():
                prediction_scaled = model(config_params_tensor)
                prediction = scaler_y.inverse_transform(prediction_scaled.numpy().reshape(-1, 1))
                return prediction.item()
        return predict

    # Dynamically create prediction function
    globals()[f"predict_{metric}"] = make_predict_function(metric, model, scaler_X, scaler_y)

print("All MLP models trained and functions created.")
