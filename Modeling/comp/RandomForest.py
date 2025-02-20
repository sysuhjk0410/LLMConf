import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Read the CSV file
data = pd.read_csv('/LLMConf/Modeling/data.csv')

# The first 10 columns are configuration parameters, the remaining columns are 7 performance metrics
X = data.iloc[:, :10].values  # Independent variables: configuration parameters

# List of performance metric names
metrics = ['latency_average', 'latency_p99',
           'tokens_per_second_average', 
           'time_to_first_token_average', 'time_to_first_token_p99',
           'time_per_output_token_average', 'time_per_output_token_p99']

# Data standardization
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Training and prediction function construction
for i, metric in enumerate(metrics):
    print(f"Training Random Forest model for {metric}...")

    # Get the dependent variable for the current performance metric 
    y = data.iloc[:, 10 + i].values

    # Standardize the performance metric data
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).reshape(-1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Create a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # After training, evaluate the model's performance
    test_predictions = model.predict(X_test)

    # Inverse transform the predictions and true values to undo standardization
    test_predictions_inverse = scaler_y.inverse_transform(test_predictions.reshape(-1, 1)).reshape(-1)
    y_test_inverse = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    # Calculate Mean Squared Error (MSE) and R² score
    mse = mean_squared_error(y_test_inverse, test_predictions_inverse)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inverse, test_predictions_inverse)

    print(f"Model for {metric}:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}\n")

    # Save the model and prediction function
    globals()[f"model_{metric}"] = model  # Dynamically create model variables

    # Define prediction function
    def make_predict_function(metric_name, model, scaler_X, scaler_y):
        def predict(max_num_batched_tokens, max_num_seqs, swap_space, block_size,
                    scheduler_delay_factor, gpu_memory_utilization, enable_chunked_prefill,
                    enable_prefix_caching, disable_custom_all_reduce, use_v2_block_manager):

            config_params = np.array([max_num_batched_tokens, max_num_seqs, swap_space, block_size,
                                      scheduler_delay_factor, gpu_memory_utilization, 
                                      int(enable_chunked_prefill), int(enable_prefix_caching),
                                      int(disable_custom_all_reduce), int(use_v2_block_manager)])
            config_params_scaled = scaler_X.transform([config_params])

            # Use Random Forest for prediction
            prediction_scaled = model.predict(config_params_scaled)
            prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))
            return prediction.item()
        return predict

    # Dynamically create prediction function
    globals()[f"predict_{metric}"] = make_predict_function(metric, model, scaler_X, scaler_y)

print("All Random Forest models trained and functions created.")
