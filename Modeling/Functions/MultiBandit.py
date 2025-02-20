import numpy as np
import matplotlib.pyplot as plt
# Import prediction functions
from predict_functions import (
    predict_latency_average, predict_latency_p99,
    predict_time_to_first_token_average, predict_time_to_first_token_p99,
    predict_time_per_output_token_average, predict_time_per_output_token_p99,
    predict_tokens_per_second_average
)

# Define the range of configuration parameters
config_ranges = {
    "max_num_batched_tokens": (4000, 8192),
    "max_num_seqs": (64, 2048),
    "swap_space": (1, 8),
    "block_size": [8, 16, 32],
    "scheduler_delay_factor": (0.0, 2.0),
    "gpu_memory_utilization": (0.7, 1.0),
    "enable_chunked_prefill": [0, 1],
    "enable_prefix_caching": [0, 1],
    "disable_custom_all_reduce": [0, 1],
    "use_v2_block_manager": [0, 1]
}

class MultiArmedBandit:
    def __init__(self, config_ranges):
        self.config_ranges = config_ranges
        self.K = 10  # 10 types of configuration parameters

    def evaluate(self, config):
        # Evaluate the performance of the configuration
        latencies = [
            predict_latency_average(**config),
            predict_latency_p99(**config),
            predict_time_to_first_token_average(**config),
            predict_time_to_first_token_p99(**config),
            predict_time_per_output_token_average(**config),
            predict_time_per_output_token_p99(**config),
            -predict_tokens_per_second_average(**config)
        ]
        return np.sum(latencies)

    def random_config(self):
        # Randomly generate configuration parameters
        return {
            "max_num_batched_tokens": np.random.randint(*self.config_ranges["max_num_batched_tokens"]),
            "max_num_seqs": np.random.randint(*self.config_ranges["max_num_seqs"]),
            "swap_space": np.random.randint(*self.config_ranges["swap_space"]),
            "block_size": np.random.choice(self.config_ranges["block_size"]),
            "scheduler_delay_factor": np.random.uniform(*self.config_ranges["scheduler_delay_factor"]),
            "gpu_memory_utilization": np.random.uniform(*self.config_ranges["gpu_memory_utilization"]),
            "enable_chunked_prefill": np.random.choice(self.config_ranges["enable_chunked_prefill"]),
            "enable_prefix_caching": np.random.choice(self.config_ranges["enable_prefix_caching"]),
            "disable_custom_all_reduce": np.random.choice(self.config_ranges["disable_custom_all_reduce"]),
            "use_v2_block_manager": np.random.choice(self.config_ranges["use_v2_block_manager"])
        }

# Define the Epsilon-Greedy algorithm
class EpsilonGreedySolver:
    def __init__(self, bandit, epsilon=0.1):
        self.bandit = bandit
        self.epsilon = epsilon
        self.best_config = None
        self.best_value = float('inf')  # Because we want to minimize the value
        self.configs = []
        self.values = []

    def run(self, num_steps):
        for _ in range(num_steps):
            if np.random.random() < self.epsilon:
                config = self.bandit.random_config()  # Exploration
            else:
                if self.best_config is None:
                    config = self.bandit.random_config()  # Randomly choose configuration during initialization
                else:
                    config = self.best_config  # Exploitation
            value = self.bandit.evaluate(config)  # Evaluate configuration

            # Update the best configuration
            if value < self.best_value:
                self.best_value = value
                self.best_config = config
            
            self.configs.append(config)
            self.values.append(value)

# Initialize the multi-armed bandit
bandit = MultiArmedBandit(config_ranges)
solver = EpsilonGreedySolver(bandit, epsilon=0.1)

# Run the Epsilon-Greedy algorithm
num_steps = 100
solver.run(num_steps)

# Output the best configuration
print("Best configuration parameters:", solver.best_config)
