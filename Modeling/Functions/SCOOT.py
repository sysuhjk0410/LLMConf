from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from predict_functions import (
    predict_latency_average, predict_latency_p99,
    predict_time_to_first_token_average, predict_time_to_first_token_p99,
    predict_time_per_output_token_average, predict_time_per_output_token_p99,
    predict_tokens_per_second_average
)

# Define the rf_cv_lgb function
def rf_cv_lgb(max_num_batched_tokens, max_num_seqs, swap_space, block_size, scheduler_delay_factor, gpu_memory_utilization, enable_chunked_prefill, enable_prefix_caching, disable_custom_all_reduce, use_v2_block_manager):
    # Map block_size to 8, 16, 32
    if block_size <= 1:
        block_size = 8
    elif block_size <= 2:
        block_size = 16
    else:
        block_size = 32

    # Ensure binary parameters are mapped to 0 or 1 based on the 0.5 rule
    enable_chunked_prefill = 0 if enable_chunked_prefill <= 0.5 else 1
    enable_prefix_caching = 0 if enable_prefix_caching <= 0.5 else 1
    disable_custom_all_reduce = 0 if disable_custom_all_reduce <= 0.5 else 1
    use_v2_block_manager = 0 if use_v2_block_manager <= 0.5 else 1

    # Calculate the objective function
    return (
        predict_tokens_per_second_average(max_num_batched_tokens, max_num_seqs, swap_space, block_size, scheduler_delay_factor, gpu_memory_utilization, enable_chunked_prefill, enable_prefix_caching, disable_custom_all_reduce, use_v2_block_manager) -
        predict_latency_average(max_num_batched_tokens, max_num_seqs, swap_space, block_size, scheduler_delay_factor, gpu_memory_utilization, enable_chunked_prefill, enable_prefix_caching, disable_custom_all_reduce, use_v2_block_manager) -
        predict_latency_p99(max_num_batched_tokens, max_num_seqs, swap_space, block_size, scheduler_delay_factor, gpu_memory_utilization, enable_chunked_prefill, enable_prefix_caching, disable_custom_all_reduce, use_v2_block_manager) -
        predict_time_to_first_token_average(max_num_batched_tokens, max_num_seqs, swap_space, block_size, scheduler_delay_factor, gpu_memory_utilization, enable_chunked_prefill, enable_prefix_caching, disable_custom_all_reduce, use_v2_block_manager) -
        predict_time_to_first_token_p99(max_num_batched_tokens, max_num_seqs, swap_space, block_size, scheduler_delay_factor, gpu_memory_utilization, enable_chunked_prefill, enable_prefix_caching, disable_custom_all_reduce, use_v2_block_manager) -
        predict_time_per_output_token_average(max_num_batched_tokens, max_num_seqs, swap_space, block_size, scheduler_delay_factor, gpu_memory_utilization, enable_chunked_prefill, enable_prefix_caching, disable_custom_all_reduce, use_v2_block_manager) -
        predict_time_per_output_token_p99(max_num_batched_tokens, max_num_seqs, swap_space, block_size, scheduler_delay_factor, gpu_memory_utilization, enable_chunked_prefill, enable_prefix_caching, disable_custom_all_reduce, use_v2_block_manager)
    )

# Random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Define parameter boundaries
pbounds = {
    'max_num_batched_tokens': (4000, 8192),
    'max_num_seqs': (64, 2048),
    'swap_space': (1, 8),
    'block_size': (0, 3),
    'scheduler_delay_factor': (0.0, 2.0),
    'gpu_memory_utilization': (0.7, 1.0),
    'enable_chunked_prefill': (0, 1),
    'enable_prefix_caching': (0, 1),
    'disable_custom_all_reduce': (0, 1),
    'use_v2_block_manager': (0, 1)
}

# Define optimization parameters
bayes_lgb = BayesianOptimization(
    f=rf_cv_lgb,
    pbounds=pbounds,
    verbose=2,
    random_state=42
)

# Start optimization
for i in range(100):
    # Get the current optimal parameters
    if i > 0:
        # Predict the next parameter point using the random forest model
        results = bayes_lgb.res
        X = np.array([list(res['params'].values()) for res in results])
        y = np.array([res['target'] for res in results])

        rf_model.fit(X, y)

        # Parallelly recommend multiple parameter configurations
        num_suggestions = 5  # Number of recommended parameter combinations
        next_params = []

        for _ in range(num_suggestions):
            # Generate random parameters
            new_params = {
                'max_num_batched_tokens': np.random.uniform(4000, 8192),
                'max_num_seqs': np.random.uniform(64, 2048),
                'swap_space': np.random.uniform(1, 8),
                'block_size': np.random.randint(0, 4),
                'scheduler_delay_factor': np.random.uniform(0.0, 2.0),
                'gpu_memory_utilization': np.random.uniform(0.7, 1.0),
                'enable_chunked_prefill': np.random.choice([0, 1]),
                'enable_prefix_caching': np.random.choice([0, 1]),
                'disable_custom_all_reduce': np.random.choice([0, 1]),
                'use_v2_block_manager': np.random.choice([0, 1]),
            }
            next_params.append(new_params)

        # Submit all recommended parameter combinations
        for params in next_params:
            bayes_lgb.probe(params=params, lazy=True)

    # Perform maximization
    bayes_lgb.maximize(n_iter=1)

# Output optimized results and map parameter values
def map_params(params):
    # Map block_size parameter
    if params['block_size'] <= 1:
        params['block_size'] = 8
    elif params['block_size'] <= 2:
        params['block_size'] = 16
    else:
        params['block_size'] = 32

    # Map binary parameters
    params['enable_chunked_prefill'] = 0 if params['enable_chunked_prefill'] <= 0.5 else 1
    params['enable_prefix_caching'] = 0 if params['enable_prefix_caching'] <= 0.5 else 1
    params['disable_custom_all_reduce'] = 0 if params['disable_custom_all_reduce'] <= 0.5 else 1
    params['use_v2_block_manager'] = 0 if params['use_v2_block_manager'] <= 0.5 else 1

    return params

# Get the optimized parameters
optimized_params = bayes_lgb.max['params']
mapped_params = map_params(optimized_params)

# Output the mapped optimized results
print("Mapped Optimized Parameters:", mapped_params)
