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

# Convert configuration ranges to a list
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

# Objective calculation: Use the passed configuration for the prediction function and return 7 objective values
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

        # Use unpacking to pass the values from the configuration dictionary to the prediction function
        objs[i, 0] = predict_latency_average(**config)
        objs[i, 1] = predict_latency_p99(**config)
        objs[i, 2] = predict_time_to_first_token_average(**config)
        objs[i, 3] = predict_time_to_first_token_p99(**config)
        objs[i, 4] = predict_time_per_output_token_average(**config)
        objs[i, 5] = predict_time_per_output_token_p99(**config)
        objs[i, 6] = -predict_tokens_per_second_average(**config)


    return objs

# Define reference point generation function
def reference_points(npop, nobj):
    ref_points = np.random.rand(npop, nobj)
    return ref_points

# Environmental selection
def environmental_selection(pop, objs, zmin, npop, ref_points):
    distances = cdist(objs - zmin, ref_points)
    selected_indices = np.argpartition(distances.sum(axis=1), npop)[:npop]
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

# Polynomial mutation
def mutation(pop, lb, ub, pm, eta_m):
    for i in range(pop.shape[0]):
        for j in range(pop.shape[1]):
            if np.random.rand() < pm:
                delta = np.random.rand()
                delta_q = (2 * delta) ** (1 / (eta_m + 1)) - 1 if delta < 0.5 else 1 - (2 * (1 - delta)) ** (1 / (eta_m + 1))
                pop[i, j] += delta_q * (ub[j] - lb[j])
                pop[i, j] = np.clip(pop[i, j], lb[j], ub[j])
    
    # Ensure discrete parameters are within valid range
    for j in range(pop.shape[1]):
        if isinstance(config_ranges[list(config_ranges.keys())[j]], list):
            pop[:, j] = np.random.choice(config_ranges[list(config_ranges.keys())[j]], pop.shape[0])

    return pop

# Main function (NSGA-III optimization algorithm)
def nsga3(npop, ngen, nobj, nvar, lb, ub, ref_points):
    pc = 0.9
    pm = 1 / nvar
    eta_c = 20
    eta_m = 20

    # Initialize population
    pop = np.zeros((npop, nvar))
    pop[:, 0] = np.random.randint(lb[0], ub[0] + 1, size=npop)  # max_num_batched_tokens
    pop[:, 1] = np.random.randint(lb[1], ub[1] + 1, size=npop)  # max_num_seqs
    pop[:, 2] = np.random.randint(lb[2], ub[2] + 1, size=npop)  # swap_space
    pop[:, 3] = np.random.choice(config_ranges["block_size"], size=npop)  # block_size
    pop[:, 4] = np.random.uniform(lb[4], ub[4], size=npop)  # scheduler_delay_factor
    pop[:, 5] = np.random.uniform(lb[5], ub[5], size=npop)  # gpu_memory_utilization
    pop[:, 6] = np.random.choice(config_ranges["enable_chunked_prefill"], size=npop)  # enable_chunked_prefill
    pop[:, 7] = np.random.choice(config_ranges["enable_prefix_caching"], size=npop)  # enable_prefix_caching
    pop[:, 8] = np.random.choice(config_ranges["disable_custom_all_reduce"], size=npop)  # disable_custom_all_reduce
    pop[:, 9] = np.random.choice(config_ranges["use_v2_block_manager"], size=npop)  # use_v2_block_manager

    objs = cal_obj(pop, nobj)
    zmin = np.min(objs, axis=0)

    # Evolutionary generations
    for gen in range(ngen):
        # Non-dominated sorting
        pfs, rank = nd_sort(objs)

        # Selection operation
        mating_pool = selection(pop, pc, rank)

        # Crossover operation
        offsprings = crossover(mating_pool, lb, ub, pc, eta_c)

        # Mutation operation
        offsprings = mutation(offsprings, lb, ub, pm, eta_m)

        # Calculate objective values for offspring
        new_objs = cal_obj(offsprings, nobj)

        # Merge populations
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
ngen = 200  # Generations
ref_points = reference_points(npop, nobj)  # Generate reference points

# Execute NSGA-III optimization
final_pop, final_objs = nsga3(npop, ngen, nobj, nvar, lb, ub, ref_points)

# Return final population and their objective values
print("Final population configuration:", final_pop)
print("Final objective function values:", final_objs)
