import random

# Define the parameter ranges and types
param_space = {
    "max-num-batched-tokens": {"type": "Integer", "range": [4000, 8192]},
    "max-num-seqs": {"type": "Integer", "range": [64, 2048]},
    "swap-space": {"type": "Integer", "range": [1, 8]},
    "block-size": {"type": "Enumeration", "values": [8, 16, 32]},
    "scheduler-delay-factor": {"type": "Float", "range": [0.0, 2.0]},
    "gpu-memory-utilization": {"type": "Float", "range": [0.7, 1.0]},
    "enable-chunked-prefill": {"type": "Boolean", "values": [True, False]},
    "enable-prefix-caching": {"type": "Boolean", "values": [True, False]},
    "disable-custom-all-reduce": {"type": "Boolean", "values": [True, False]},
    "use-v2-block-manager": {"type": "Boolean", "values": [True, False]}
}

# Randomly generate a set of parameter values
random_params = {
    "max-num-batched-tokens": random.randint(4000, 8192),
    "max-num-seqs": random.randint(64, 2048),
    "swap-space": random.randint(1, 8),
    "block-size": random.choice([8, 16, 32]),
    "scheduler-delay-factor": random.uniform(0.0, 2.0),
    "gpu-memory-utilization": random.uniform(0.7, 1.0),
    "enable-chunked-prefill": random.choice([True, False]),
    "enable-prefix-caching": random.choice([True, False]),
    "disable-custom-all-reduce": random.choice([True, False]),
    "use-v2-block-manager": random.choice([True, False])
}

# Output the randomly generated parameter values
print(random_params)
