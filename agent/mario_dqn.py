import gym_super_mario_bros
import gym
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym import wrappers
from gym.wrappers.record_video import RecordVideo
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import tensorflow as tf
import time
import os

# Define the Q-Network model
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, 128),  # Reduced from 256
        nn.ReLU(),
        nn.Linear(128, 64),  # Reduced from 128
        nn.ReLU(),
        nn.Linear(64, output_dim)
    )

    def forward(self, x):
        return self.fc(x)

# Define the Mario agent for Q-learning
class MarioAgent:
    def __init__(self, input_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, min_epsilon=0.01, decay=0.995):
        self.q_net = QNetwork(input_dim, action_dim).float()
        self.target_q_net = QNetwork(input_dim, action_dim).float()
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.memory = deque(maxlen=5000)
        self.batch_size = 32
        self.action_dim = action_dim

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                return torch.argmax(self.q_net(state)).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a random batch from the memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert data to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.int64)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)

        # Q-values for the current states
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Q-values for the next states
        next_q_values = self.target_q_net(next_states).max(1)[0]

        # Compute the target Q-values
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = torch.nn.functional.mse_loss(q_values, targets)

        # Perform gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon (for exploration vs exploitation)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

        # Periodically update the target Q-network
        if random.random() < 0.01:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

# Logging setup
log_dir = "logs"
writer = tf.summary.create_file_writer(log_dir)

# Video recording setup
video_folder = "recorded_videos"
os.makedirs(video_folder, exist_ok=True)

# Create the Mario environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)  # Discretize controls
env = RecordVideo(
    env,
    video_folder=video_folder,
    episode_trigger=lambda episode_id: episode_id % 10000 == 0,  # Record every 10,000 episodes
    name_prefix='mario-video-'
)

# State and action space sizes
state_shape = np.prod(env.observation_space.shape)
action_size = env.action_space.n

# Initialize agent
agent = MarioAgent(state_shape, action_size)

# Training parameters
num_episodes = 3

print("--------------------------Begin Episodes-------------------------", flush=True)
for episode in range(num_episodes):
    start_time = time.time()
    print(f"Start of Episode {episode+1}", flush=True)

    state = env.reset()  # Gym 0.26+ returns (obs, info)
    state = np.array(state).flatten()
    # env.render()

    total_reward = 0
    skip_frames = 4  # Process every 4th frame
    for t in range(10000):
        if t % skip_frames == 0:
            action = agent.select_action(state)

        next_state, reward, done, info = env.step(action)

        next_state = np.array(next_state).flatten()

        agent.store_experience(state, action, reward, next_state, done)
        agent.train()

        state = next_state
        total_reward += reward

        if done:  # If episode ends, break and reset
            break

    elapsed_time = time.time() - start_time
    print(f"Episode {episode+1} finished in {elapsed_time:.2f} seconds.")
    print(f"Episode {episode+1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}", flush=True)

    with writer.as_default():
        tf.summary.scalar('Total Reward', total_reward, step=episode)

    # Ensure reset after each episode
    state = np.array(state).flatten()

# Cleanup
env.close()
save_path = "saved_model.pth"
torch.save(agent.q_net.state_dict(), save_path)
print(f"Model saved to {save_path}", flush=True)
