import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation
import warnings
import tensorflow as tf
import csv
import time

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SLURM_ID = os.getenv("SLURM_JOB_ID")
saved_checkpoint_path = "0.pth"


# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Rainbow DQN components (NoisyLinear & Network)
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)

class RainbowDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RainbowDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.advantage_layer = NoisyLinear(64, output_dim)
        self.value_layer = NoisyLinear(64, 1)
    
    def forward(self, x):
        features = self.feature_layer(x)
        advantage = self.advantage_layer(features)
        value = self.value_layer(features)
        return value + (advantage - advantage.mean())

class MarioAgent:
    def __init__(self, input_dim, action_dim, lr=0.0001, gamma=0.99, min_epsilon=0.01, decay=0.997):
        self.q_net = RainbowDQN(input_dim, action_dim).to(device)
        self.target_q_net = RainbowDQN(input_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = 1.0
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.memory = deque(maxlen=5000)
        self.batch_size = 32
        self.action_dim = action_dim

        # Logging setup
        self.log_dir = "logs_rainbowdqn"
        self.writer = tf.summary.create_file_writer(self.log_dir)

        self.log_file = "mario_rainbowdqn_log.csv"
        with open(self.log_file, "w") as f:
            w = csv.writer(f)
            w.writerow(["Episode", "X_Pos"])

    def log_reward(self, total_reward, episode):
        with self.writer.as_default():  # This ensures the summary is logged to the writer
            tf.summary.scalar('Total Reward', total_reward, step=episode)  # Log total reward
            self.writer.flush()

    def save_checkpoint(self, path):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_checkpoint(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.q_net.load_state_dict(checkpoint)
            self.target_q_net.load_state_dict(checkpoint)
            print(f"Checkpoint loaded from {path}")
        else:
            print(f"No checkpoint found at {path}")
            

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                return torch.argmax(self.q_net(state)).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_q_net(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        loss = torch.nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

env = gym.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
state_shape = np.prod(env.observation_space.shape)
action_size = env.action_space.n
agent = MarioAgent(state_shape, action_size)
agent.load_checkpoint(saved_checkpoint_path)
num_episodes = 100001
for episode in range(num_episodes):
    start_time = time.time()
    state = np.array(env.reset()).flatten()
    total_reward = 0
    max_x_pos = 0

    data = {
                "last_x": 0,
                "max_x": 0,
                "max_reward": 0,
                "total_reward": 0,
                "max_x_frame": None,
                "event": None,
                "jumps": 0
            }
    for t in range(10000):
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_experience(state, action, reward, next_state, done)
        agent.train()
        x_increase = max(0, info['x_pos'] - data["last_x"])

        # if info["flag_get"]:
        #     reward += 1000
        #     reward += math.exp(-0.005 * info["time"]) * 1000
        #     done = True
        #     print("GOAL")
        #     data["event"] = "finished"

        #     # Punish dying
        #     if info["life"] < 2:
        #         reward -= 50
        #         done = True
        #         data["event"] = "death"

        # data["last_x"] = info['x_pos']
        if info['x_pos'] > max_x_pos:
            max_x_pos = info['x_pos']
            max_x_frame = state.copy()
        
        max_x_pos = max(max_x_pos, info['x_pos'])
        # data["total_reward"] += reward

        state = next_state.flatten()
        total_reward += reward
        if done:
            break
    elapsed_time = time.time() - start_time

    with open(agent.log_file, "a") as f: #log max_x_pos csv
        w = csv.writer(f)
        w.writerow([episode, max_x_pos])
    
    agent.log_reward(total_reward, episode)
        
    if episode % 10 == 0:
        checkpoint_path = f"RainbowDQN/{SLURM_ID}/{episode}.pth"
        agent.save_checkpoint(checkpoint_path)

    print(f"Episode {episode+1} finished in {elapsed_time:.2f} seconds.")
    print(f"Episode {episode+1}, Total Reward: {total_reward}, Max X-Position: {max_x_pos}, Epsilon: {agent.epsilon:.4f}", flush=True)

env.close()
