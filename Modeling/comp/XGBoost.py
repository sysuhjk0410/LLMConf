import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Read CSV file
data = pd.read_csv('/LLMConf/Modeling/data.csv')

# Columns 1 to 10 are configuration parameters, the rest are 7 evaluation metrics
X = data.iloc[:, :10].values  # Independent variables: configuration parameters

# Evaluation metrics column names
metrics = ['latency_average', 'latency_p99',
           'tokens_per_second_average', 
           'time_to_first_token_average', 'time_to_first_token_p99',
           'time_per_output_token_average', 'time_per_output_token_p99']

# Data standardization
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Constructing training and prediction functions
for i, metric in enumerate(metrics):
    print(f"Training XGBoost model for {metric}...")

    # Get the current evaluation metric as the dependent variable 
    y = data.iloc[:, 10 + i].values

    # Standardize the evaluation metric data
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).reshape(-1)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Create XGBoost model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # After training, evaluate model performance
    test_predictions = model.predict(X_test)

    # Inverse transform the predicted and true values
    test_predictions_inverse = scaler_y.inverse_transform(test_predictions.reshape(-1, 1)).reshape(-1)
    y_test_inverse = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    # Calculate Mean Squared Error (MSE) and R-squared (R²)
    mse = mean_squared_error(y_test_inverse, test_predictions_inverse)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inverse, test_predictions_inverse)

    print(f"Model for {metric}:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}\n")

    # Save the model and prediction function
    globals()[f"model_{metric}"] = model  # Dynamically create model variables

    # Define the prediction function
    def make_predict_function(metric_name, model, scaler_X, scaler_y):
        def predict(max_num_batched_tokens, max_num_seqs, swap_space, block_size,
                    scheduler_delay_factor, gpu_memory_utilization, enable_chunked_prefill,
                    enable_prefix_caching, disable_custom_all_reduce, use_v2_block_manager):

            config_params = np.array([max_num_batched_tokens, max_num_seqs, swap_space, block_size,
                                      scheduler_delay_factor, gpu_memory_utilization, 
                                      int(enable_chunked_prefill), int(enable_prefix_caching),
                                      int(disable_custom_all_reduce), int(use_v2_block_manager)])
            config_params_scaled = scaler_X.transform([config_params])

            # Use XGBoost to make predictions
            prediction_scaled = model.predict(config_params_scaled)
            prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))
            return prediction.item()
        return predict

    # Dynamically create prediction functions
    globals()[f"predict_{metric}"] = make_predict_function(metric, model, scaler_X, scaler_y)

print("All XGBoost models trained and functions created.")
