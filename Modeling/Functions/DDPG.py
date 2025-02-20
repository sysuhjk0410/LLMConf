import argparse
import os
import random
from itertools import product

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

# Import prediction function
from predict_functions import (
    predict_latency_average, 
    predict_latency_p99,
    predict_time_to_first_token_average, 
    predict_time_to_first_token_p99,
    predict_time_per_output_token_average, 
    predict_time_per_output_token_p99,
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

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)
parser.add_argument("--env_name", default="Pendulum-v1")  # Change the environment name to Pendulum-v1
parser.add_argument('--tau',  default=0.005, type=float)
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int)
parser.add_argument('--capacity', default=1000000, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
parser.add_argument('--sample_frequency', default=2000, type=int)
parser.add_argument('--render', default=False, type=bool)
parser.add_argument('--log_interval', default=50, type=int)
parser.add_argument('--load', default=False, type=bool)
parser.add_argument('--render_interval', default=100, type=int)
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=100000, type=int)
parser.add_argument('--print_log', default=5, type=int)
parser.add_argument('--update_iteration', default=200, type=int)
args = parser.parse_args()

# Set up device and environment
device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
env = gym.make(args.env_name)

# Set random seed
if args.seed:
    env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

# Environment dimension settings
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_val = torch.tensor(1e-7).float().to(device)

# Save path
directory = '/LLMConf/Modeling/Functions/Model/'

# Replay buffer
class ReplayBuffer:
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        return [np.array(self.storage[i], copy=False) for i in ind]

# Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))

# Critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        return self.l3(x)

# DDPG algorithm
class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = ReplayBuffer()
        self.writer = SummaryWriter(directory)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):
        for it in range(args.update_iteration):
            x, y, u, r, d = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Calculate target Q-value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * args.gamma * target_Q).detach()

            # Calculate current Q-value
            current_Q = self.critic(state, action)

            # Calculate Critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Calculate Actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target network
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))

# Evaluate the performance of a single configuration
def evaluate_config(config):
    # Generate predictions using configuration parameters
    latency_average = predict_latency_average(
        max_num_batched_tokens=config["max_num_batched_tokens"],
        max_num_seqs=config["max_num_seqs"],
        swap_space=config["swap_space"],
        block_size=config["block_size"],
        scheduler_delay_factor=config["scheduler_delay_factor"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        enable_chunked_prefill=config["enable_chunked_prefill"],
        enable_prefix_caching=config["enable_prefix_caching"],
        disable_custom_all_reduce=config["disable_custom_all_reduce"],
        use_v2_block_manager=config["use_v2_block_manager"]
    )

    latency_p99 = predict_latency_p99(
        max_num_batched_tokens=config["max_num_batched_tokens"],
        max_num_seqs=config["max_num_seqs"],
        swap_space=config["swap_space"],
        block_size=config["block_size"],
        scheduler_delay_factor=config["scheduler_delay_factor"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        enable_chunked_prefill=config["enable_chunked_prefill"],
        enable_prefix_caching=config["enable_prefix_caching"],
        disable_custom_all_reduce=config["disable_custom_all_reduce"],
        use_v2_block_manager=config["use_v2_block_manager"]
    )

    time_to_first_token_average = predict_time_to_first_token_average(
        max_num_batched_tokens=config["max_num_batched_tokens"],
        max_num_seqs=config["max_num_seqs"],
        swap_space=config["swap_space"],
        block_size=config["block_size"],
        scheduler_delay_factor=config["scheduler_delay_factor"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        enable_chunked_prefill=config["enable_chunked_prefill"],
        enable_prefix_caching=config["enable_prefix_caching"],
        disable_custom_all_reduce=config["disable_custom_all_reduce"],
        use_v2_block_manager=config["use_v2_block_manager"]
    )

    time_to_first_token_p99 = predict_time_to_first_token_p99(
        max_num_batched_tokens=config["max_num_batched_tokens"],
        max_num_seqs=config["max_num_seqs"],
        swap_space=config["swap_space"],
        block_size=config["block_size"],
        scheduler_delay_factor=config["scheduler_delay_factor"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        enable_chunked_prefill=config["enable_chunked_prefill"],
        enable_prefix_caching=config["enable_prefix_caching"],
        disable_custom_all_reduce=config["disable_custom_all_reduce"],
        use_v2_block_manager=config["use_v2_block_manager"]
    )

    time_per_output_token_average = predict_time_per_output_token_average(
        max_num_batched_tokens=config["max_num_batched_tokens"],
        max_num_seqs=config["max_num_seqs"],
        swap_space=config["swap_space"],
        block_size=config["block_size"],
        scheduler_delay_factor=config["scheduler_delay_factor"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        enable_chunked_prefill=config["enable_chunked_prefill"],
        enable_prefix_caching=config["enable_prefix_caching"],
        disable_custom_all_reduce=config["disable_custom_all_reduce"],
        use_v2_block_manager=config["use_v2_block_manager"]
    )

    time_per_output_token_p99 = predict_time_per_output_token_p99(
        max_num_batched_tokens=config["max_num_batched_tokens"],
        max_num_seqs=config["max_num_seqs"],
        swap_space=config["swap_space"],
        block_size=config["block_size"],
        scheduler_delay_factor=config["scheduler_delay_factor"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        enable_chunked_prefill=config["enable_chunked_prefill"],
        enable_prefix_caching=config["enable_prefix_caching"],
        disable_custom_all_reduce=config["disable_custom_all_reduce"],
        use_v2_block_manager=config["use_v2_block_manager"]
    )

    tokens_per_second_average = predict_tokens_per_second_average(
        max_num_batched_tokens=config["max_num_batched_tokens"],
        max_num_seqs=config["max_num_seqs"],
        swap_space=config["swap_space"],
        block_size=config["block_size"],
        scheduler_delay_factor=config["scheduler_delay_factor"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        enable_chunked_prefill=config["enable_chunked_prefill"],
        enable_prefix_caching=config["enable_prefix_caching"],
        disable_custom_all_reduce=config["disable_custom_all_reduce"],
        use_v2_block_manager=config["use_v2_block_manager"]
    )

    # Calculate score
    epsilon = 1e-6
    score = (
        latency_average + latency_p99 +
        time_to_first_token_average + time_to_first_token_p99 +
        time_per_output_token_average + time_per_output_token_p99
    ) / 6 - (
        tokens_per_second_average 
    ) 

    return score

# Generate configuration and evaluate performance
def generate_and_evaluate_configs(num_configs):
    best_config = None
    best_score = float('inf')

    for _ in range(num_configs):
        config = {
            "max_num_batched_tokens": random.randint(*config_ranges["max_num_batched_tokens"]),
            "max_num_seqs": random.randint(*config_ranges["max_num_seqs"]),
            "swap_space": random.randint(*config_ranges["swap_space"]),
            "block_size": random.choice(config_ranges["block_size"]),
            "scheduler_delay_factor": random.uniform(*config_ranges["scheduler_delay_factor"]),
            "gpu_memory_utilization": random.uniform(*config_ranges["gpu_memory_utilization"]),
            "enable_chunked_prefill": random.choice(config_ranges["enable_chunked_prefill"]),
            "enable_prefix_caching": random.choice(config_ranges["enable_prefix_caching"]),
            "disable_custom_all_reduce": random.choice(config_ranges["disable_custom_all_reduce"]),
            "use_v2_block_manager": random.choice(config_ranges["use_v2_block_manager"]),
        }
        
        score = evaluate_config(config)

        if score < best_score:
            best_score = score
            best_config = config

    return best_config, best_score

if __name__ == '__main__':
    num_configs_to_test = 100  # Number of configurations generated and evaluated
    best_config, best_score = generate_and_evaluate_configs(num_configs_to_test)
    print(f"Best config: {best_config} with score: {best_score}")
    # Call model save function
    ddpg_model = DDPG(state_dim, action_dim, max_action)
    ddpg_model.save()  # Save model
