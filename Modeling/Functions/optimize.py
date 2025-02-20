import numpy as np
from collections import Counter
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy.linalg import LinAlgError

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

# Convert configuration to a vector
def config_to_vector(config):
    return [
        config["max_num_batched_tokens"],
        config["max_num_seqs"],
        config["swap_space"],
        config["block_size"],
        config["scheduler_delay_factor"],
        config["gpu_memory_utilization"],
        config["enable_chunked_prefill"],
        config["enable_prefix_caching"],
        config["disable_custom_all_reduce"],
        config["use_v2_block_manager"],
    ]

# Objective calculation: Use the input configuration for prediction functions and return 7 objective values
def cal_obj(pop, nobj):
    objs = np.zeros((pop.shape[0], nobj))
    for i in range(pop.shape[0]):
        config = {
            "max_num_batched_tokens": int(pop[i, 0]),
            "max_num_seqs": int(pop[i, 1]),
            "swap_space": int(pop[i, 2]),
            "block_size": int(pop[i, 3]),
            "scheduler_delay_factor": pop[i, 4],
            "gpu_memory_utilization": pop[i, 5],
            "enable_chunked_prefill": int(pop[i, 6]),
            "enable_prefix_caching": int(pop[i, 7]),
            "disable_custom_all_reduce": int(pop[i, 8]),
            "use_v2_block_manager": int(pop[i, 9]),
        }

        # Unpack and pass the values from the configuration dictionary to the prediction functions
        objs[i, 0] = predict_latency_average(**config)
        objs[i, 1] = predict_latency_p99(**config)
        objs[i, 2] = predict_time_to_first_token_average(**config)
        objs[i, 3] = predict_time_to_first_token_p99(**config)
        objs[i, 4] = predict_time_per_output_token_average(**config)
        objs[i, 5] = predict_time_per_output_token_p99(**config)
        objs[i, 6] = -predict_tokens_per_second_average(**config)


    return objs

# Define the function to generate reference points
def reference_points(npop, nobj):
    ref_points = np.random.rand(npop, nobj)
    return ref_points

# Environmental selection
def environmental_selection(pop, objs, zmin, npop, ref_points):
    # Use Chebyshev distance
    distances = np.max(np.abs(objs - zmin), axis=1).reshape(-1, 1)
    selected_indices = np.argpartition(distances.flatten(), npop)[:npop]
    return pop[selected_indices], objs[selected_indices], selected_indices

# Non-dominated sorting
def nd_sort(objs):
    S = [[] for _ in range(objs.shape[0])]
    front = [[]]
    n = np.zeros(objs.shape[0])
    rank = np.zeros(objs.shape[0])

    for p in range(objs.shape[0]):
        for q in range(objs.shape[0]):
            if np.all(objs[p] <= objs[q]) and np.any(objs[p] < objs[q]):
                S[p].append(q)
            elif np.all(objs[q] <= objs[p]) and np.any(objs[q] < objs[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            front[0].append(p)

    i = 0
    while front[i]:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        front.append(Q)

    fronts = [np.array(front[i]) for i in range(len(front) - 1)]
    return fronts, rank

# Selection operation
def selection(pop, pc, rank):
    selected_indices = np.random.choice(np.arange(pop.shape[0]), size=int(pc * pop.shape[0]), replace=False)
    return pop[selected_indices]

# Simulated binary crossover
def crossover(pop, lb, ub, pc, eta_c):
    offsprings = np.zeros_like(pop)
    for i in range(0, pop.shape[0], 2):
        p1, p2 = pop[i], pop[(i + 1) % pop.shape[0]]
        for j in range(pop.shape[1]):
            if np.random.rand() < pc:
                beta = 1.0 + (2 * min(p1[j] - lb[j], ub[j] - p1[j]) / (p2[j] - p1[j] if p2[j] != p1[j] else 1e-9))
                beta_q = (2 * np.random.rand()) ** (1 / (eta_c + 1)) if np.random.rand() < 0.5 else 1 - (2 * np.random.rand()) ** (1 / (eta_c + 1))
                offsprings[i, j] = 0.5 * ((1 + beta_q) * p1[j] + (1 - beta_q) * p2[j])
                offsprings[(i + 1) % pop.shape[0], j] = 0.5 * ((1 - beta_q) * p1[j] + (1 + beta_q) * p2[j])
            else:
                offsprings[i, j], offsprings[(i + 1) % pop.shape[0], j] = p1[j], p2[j]

    # Ensure discrete parameters are within valid range
    for j in range(pop.shape[1]):
        if isinstance(config_ranges[list(config_ranges.keys())[j]], list):
            offsprings[:, j] = np.random.choice(config_ranges[list(config_ranges.keys())[j]], pop.shape[0])

    return offsprings

# Coevolutionary mutation
def coevolutionary_mutation(pop, lb, ub, pm):
    mutated_pop = np.copy(pop)
    n = pop.shape[0]
    for i in range(n):
        if np.random.rand() < pm:
            # Randomly select another individual for mutation
            j = np.random.randint(0, n)
            for k in range(pop.shape[1]):
                # Use some parameters from another individual
                if np.random.rand() < 0.5:
                    mutated_pop[i, k] = pop[j, k] + np.random.uniform(-0.1, 0.1) * (ub[k] - lb[k])
                    mutated_pop[i, k] = np.clip(mutated_pop[i, k], lb[k], ub[k])
    
    # Ensure discrete parameters are within valid range
    for j in range(mutated_pop.shape[1]):
        if isinstance(config_ranges[list(config_ranges.keys())[j]], list):
            mutated_pop[:, j] = np.random.choice(config_ranges[list(config_ranges.keys())[j]], mutated_pop.shape[0])

    return mutated_pop

# Greedy initialization of the population
def greedy_initialization(npop, nvar, lb, ub):
    pop = []
    for _ in range(npop):
        # Randomly generate a configuration
        config = [
            np.random.randint(lb[0], ub[0] + 1),  # max_num_batched_tokens
            np.random.randint(lb[1], ub[1] + 1),  # max_num_seqs
            np.random.randint(lb[2], ub[2] + 1),  # swap_space
            np.random.choice(config_ranges["block_size"]),  # block_size
            np.random.uniform(lb[4], ub[4]),  # scheduler_delay_factor
            np.random.uniform(lb[5], ub[5]),  # gpu_memory_utilization
            np.random.choice(config_ranges["enable_chunked_prefill"]),  # enable_chunked_prefill
            np.random.choice(config_ranges["enable_prefix_caching"]),  # enable_prefix_caching
            np.random.choice(config_ranges["disable_custom_all_reduce"]),  # disable_custom_all_reduce
            np.random.choice(config_ranges["use_v2_block_manager"]),  # use_v2_block_manager
        ]
        
        # Calculate objective values and insert into the population
        pop.append(config)

    pop = np.array(pop)
    
    # Use greedy selection to find optimal configurations
    objs = cal_obj(pop, nobj)
    best_indices = np.argsort(objs.sum(axis=1))[:npop]  # Sort by objective values and select the top npop
    return pop[best_indices]

# Main function (NSGA-III optimization algorithm)
def nsga3(npop, ngen, nobj, nvar, lb, ub, ref_points):
    pc = 0.9
    pm = 1 / nvar
    eta_c = 20

    # Initialize population using greedy algorithm
    pop = greedy_initialization(npop, nvar, lb, ub)

    objs = cal_obj(pop, nobj)
    zmin = np.min(objs, axis=0)

    # Evolution generations
    for gen in range(ngen):
        # Non-dominated sorting
        pfs, rank = nd_sort(objs)

        # Selection operation
        mating_pool = selection(pop, pc, rank)

        # Crossover operation
        offsprings = crossover(mating_pool, lb, ub, pc, eta_c)

        # Coevolutionary mutation operation
        offsprings = coevolutionary_mutation(offsprings, lb, ub, pm)

        # Calculate objective function values for offspring
        new_objs = cal_obj(offsprings, nobj)

        # Combine populations
        combined_pop = np.vstack((pop, offsprings))
        combined_objs = np.vstack((objs, new_objs))

        # Environmental selection
        pop, objs, _ = environmental_selection(combined_pop, combined_objs, zmin, npop, ref_points)

        # Update minimum values
        zmin = np.min(objs, axis=0)

    return pop, objs

# Execute optimization with specified parameters
nobj = 7  # Number of objectives
nvar = len(config_ranges)  # Number of configuration parameters
lb = np.array([config_ranges[key][0] if isinstance(config_ranges[key], tuple) else 0 for key in config_ranges])
ub = np.array([config_ranges[key][1] if isinstance(config_ranges[key], tuple) else len(config_ranges[key]) - 1 for key in config_ranges])
npop = 100  # Population size
ngen = 200  # Number of generations
ref_points = reference_points(npop, nobj)  # Generate reference points

# Execute NSGA-III optimization
final_pop, final_objs = nsga3(npop, ngen, nobj, nvar, lb, ub, ref_points)

# Return the final population and their objective function values
print("Final population configuration:", final_pop)
print("Final objective function values:", final_objs)
