import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tpot import TPOTRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import joblib
import json  # Used to save MSE, RMSE, and R² Score

# Read the CSV file
data = pd.read_csv('/LLMConf/Modeling/data.csv')

# The first 10 columns are configuration parameters, and the remaining columns are evaluation metrics
X = data.iloc[:, :10].values  # Independent variable: configuration parameters

# Define the selected 7 performance metrics
metrics = ['latency_average', 'latency_p99', 
           'tokens_per_second_average', 
           'time_to_first_token_average', 'time_to_first_token_p99',
           'time_per_output_token_average', 'time_per_output_token_p99']

# Data standardization
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Define save path
save_dir = '/LLMConf/Modeling/Functions'
os.makedirs(save_dir, exist_ok=True)  # Ensure the target directory exists

# Create a Python file to store prediction functions
with open(os.path.join(save_dir, 'predict_functions.py'), 'w') as f:
    f.write("import numpy as np\n")
    f.write("from sklearn.preprocessing import StandardScaler\n")
    f.write("import joblib\n\n")

    f.write("# Load the scaler\n")
    f.write("scaler_X = joblib.load('/LLMConf/Modeling/Functions/scaler_X.pkl')\n\n")

    # Create a dictionary to store MSE, RMSE, and R² Score for all evaluation metrics
    metrics_scores = {}

    for i, metric in enumerate(metrics):
        print(f"Training TPOT model for {metric}...")

        # Get the dependent variable for the current evaluation metric 
        y = data.iloc[:, 10 + i].values

        # Standardize the evaluation metric data
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).reshape(-1)

        # Split the training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

        # Create the TPOT model (set generations and population_size to control the search process)
        tpot = TPOTRegressor(generations=150, population_size=50, verbosity=2, random_state=42)

        # Train the TPOT model
        tpot.fit(X_train, y_train)

        # Make predictions
        test_predictions = tpot.predict(X_test)

        # Inverse transform the predicted and true values
        test_predictions_inverse = scaler_y.inverse_transform(test_predictions.reshape(-1, 1))
        y_test_inverse = scaler_y.inverse_transform(y_test.reshape(-1, 1))

        # Calculate Mean Squared Error (MSE) and R² Score
        mse = mean_squared_error(y_test_inverse, test_predictions_inverse)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_inverse, test_predictions_inverse)

        print(f"Model for {metric}:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R² Score: {r2:.4f}\n")

        # Save the metric values to the dictionary
        metrics_scores[metric] = {
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }

        # Save the model
        # Save the final model (fitted_pipeline_)
        model_filename = os.path.join(save_dir, f'tpot_model_{metric}.pkl')
        joblib.dump(tpot.fitted_pipeline_, model_filename)

        # Save the scaler for the evaluation metric
        scaler_filename = os.path.join(save_dir, f'scaler_y_{metric}.pkl')
        joblib.dump(scaler_y, scaler_filename)

        # Write the prediction function to the Python file
        f.write(f"# Prediction function for {metric}\n")
        f.write(f"def predict_{metric}(max_num_batched_tokens, max_num_seqs, swap_space, block_size,\n")
        f.write("                  scheduler_delay_factor, gpu_memory_utilization, enable_chunked_prefill,\n")
        f.write("                  enable_prefix_caching, disable_custom_all_reduce, use_v2_block_manager):\n")
        f.write("    config_params = np.array([max_num_batched_tokens, max_num_seqs, swap_space, block_size,\n")
        f.write("                              scheduler_delay_factor, gpu_memory_utilization,\n")
        f.write("                              int(enable_chunked_prefill), int(enable_prefix_caching),\n")
        f.write("                              int(disable_custom_all_reduce), int(use_v2_block_manager)])\n")
        f.write("    config_params_scaled = scaler_X.transform([config_params])\n")
        f.write(f"    model = joblib.load('{model_filename}')\n")
        f.write(f"    scaler_y = joblib.load('{scaler_filename}')\n")
        f.write("    prediction_scaled = model.predict(config_params_scaled)\n")
        f.write("    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))\n")
        f.write("    return prediction.item()\n\n")

# Save the overall scaler for X
joblib.dump(scaler_X, os.path.join(save_dir, 'scaler_X.pkl'))

# Save MSE, RMSE, and R² values to a JSON file
with open(os.path.join(save_dir, 'metrics_scores.json'), 'w') as json_file:
    json.dump(metrics_scores, json_file, indent=4)

print("All TPOT models trained, prediction functions, and metrics scores saved.")
