#Ignore Log Warnings
import warnings
import os
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gym_super_mario_bros
import gym
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation
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
import csv
import shutil
from PIL import Image

SLURM_ID = os.getenv("SLURM_JOB_ID")
verbose = False #set to true if you want to save each episode + more details
jump_actions = [2, 4, 5]
speed_actions = [3, 4]

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
    def __init__(self, input_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, min_epsilon=0.01, decay=0.997):
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
        self.last_action = None

    def save_checkpoint(self, path):
        checkpoint = {
            'q_net_state_dict': self.q_net.state_dict(),
            'target_q_net_state_dict': self.target_q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': list(self.memory),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
            self.target_q_net.load_state_dict(checkpoint['target_q_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.memory = deque(checkpoint['memory'], maxlen=5000)
        else:
            print(f"No checkpoint found at {path}")

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                return torch.argmax(self.q_net(state)).item()
        self.last_action = action

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

class DQN():
    def __init__(self):
        # Logging setup
        self.log_dir = "logs"
        self.writer = tf.summary.create_file_writer(self.log_dir)

        self.log_file = "mario_dqn_log.csv"
        with open(self.log_file, "w") as f:
            w = csv.writer(f)
            w.writerow(["Episode", "Event", "X_Pos"])

        self.img_dir = "imgs"
        os.makedirs(self.img_dir, exist_ok=True)

        # Video recording setup
        self.video_folder = "recorded_videos"
        os.makedirs(self.video_folder, exist_ok=True)

        # Set up episode models folder
        if verbose:
            os.makedirs(SLURM_ID, exist_ok=True) 

        # Create the Mario environment
        self.env = gym_super_mario_bros.make('SuperMarioBros-v0')
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)  # Discretize controls

        # Record
        self.env = RecordVideo(
            self.env,
            video_folder=self.video_folder,
            episode_trigger=lambda episode_id: episode_id % 10 == 0,  # Record every 10,000 episodes
            name_prefix='mario-video-'
        )

        # Try to speed it up
        self.env = GrayScaleObservation(self.env)
        self.env = ResizeObservation(self.env, shape=84)

    def save(self, episode):
        if verbose:
            # Save model checkpoint after each episode
            save_path = f"{SLURM_ID}/saved_model_episode_{episode+1}.pth"
            torch.save(agent.q_net.state_dict(), save_path)
            print(f"Model saved to {save_path}", flush=True)

    def log(self, episode, data):
        '''Logs Image, X position, and Reward'''
        if data["max_x_frame"] is not None:
            img = Image.fromarray(data["max_x_frame"].reshape(84, 84))  # Assuming your state is grayscale 84x84
            img.save(f"{self.img_dir}/episode{episode+1}_pos{data['max_x']}.png")
            print(f"Saved max x-position frame for Episode {episode+1}")

        with open(self.log_file, "a") as f: #log max_x_pos csv
            w = csv.writer(f)
            w.writerow([episode, data['event'], data['max_x']])

        with self.writer.as_default(): #log reward tf
            tf.summary.scalar('Total Reward', data["total_reward"], step=episode)

    def run(self, num_episodes=3, checkpoint_path="saved_model.pth"):
        # State and action space sizes
        state_shape = np.prod(self.env.observation_space.shape)
        action_size = self.env.action_space.n

        # Initialize agent
        agent = MarioAgent(state_shape, action_size)

        # This is for loading a model if you already have one
        agent.load_checkpoint(checkpoint_path)

        print("--------------------------Begin Episodes-------------------------", flush=True)
        for episode in range(num_episodes):
            start_time = time.time()
            print(f"Start of Episode {episode+1}", flush=True)

            state = self.env.reset()  # Gym 0.26+ returns (obs, info)
            state = np.array(state).flatten()
            # env.render()

            data = {
                "last_x": 0,
                "max_x": 0,
                "max_reward": 0,
                "total_reward": 0,
                "max_x_frame": None,
                "event": None,
                "jumps": 0
            }

            steps = 10000
            skip_frames = 4  # Process every 4th frame
            for t in range(steps):
                if t % skip_frames == 0:
                    action = agent.select_action(state)
                # else:
                #     action = agent.last_action

                next_state, reward, done, info = self.env.step(action)
                next_state = np.array(next_state).flatten()

                agent.store_experience(state, action, reward, next_state, done)
                agent.train()

                x_increase = max(0, info['x_pos'] - data["last_x"])

                # # Custom Reward
                # # Increase penalty for standing still
                # if action == 0:
                #     reward -= 200
                
                # # Reward moving right more aggressively
                # if action in speed_actions:
                #     reward += x_increase * 2.0  # Increased reward for moving right
                # else:
                #     reward += x_increase * 1.5  # Slightly lower reward for other actions

                # # Encourage being speedy
                # if action in speed_actions:
                #     reward += max(0, info['x_pos'] - data["max_x"]) * 1.2
                # else:
                #     reward += max(0, info['x_pos'] - data["max_x"])

                # if x_increase == 0:
                #     if action in jump_actions:
                #         jumps += 1
                #         reward += 10 * min(jumps, 5)  # Increase reward for sustained jumping
                #     else:
                #         jumps = 0
                #         reward -= 5

                if info["flag_get"]:
                    reward += 1000
                    reward += math.exp(-0.005 * info["time"]) * 1000
                    done = True
                    print("GOAL")
                    data["event"] = "finished"

                # Punish dying
                if info["life"] < 2:
                    reward -= 50
                    done = True
                    data["event"] = "death"
                
                data["last_x"] = info['x_pos']
                if info['x_pos'] > data["max_x"]:
                    data["max_x"] = info['x_pos']
                    data["max_x_frame"] = state.copy()
                data["total_reward"] += reward
                data["max_reward"] = max(data["max_reward"], reward)

                state = next_state

                if done:  # If episode ends, break and reset
                    break

            elapsed_time = time.time() - start_time
            print(f"Episode {episode+1} finished in {elapsed_time:.2f} seconds.")
            print(f"Episode {episode+1}, Total Reward: {data['total_reward']}, Max X-Position: {data['max_x']}, Epsilon: {agent.epsilon:.4f}", flush=True)
            
            if episode % 10 == 0:
                agent.save_checkpoint(checkpoint_path)
            self.save(episode)
            self.log(episode, data)

            # Ensure reset after each episode
            state = np.array(state).flatten()

        # Cleanup
        self.env.close()
        save_path = f"{SLURM_ID}_saved_model.pth" if not verbose else f"{SLURM_ID}/saved_model.pth"
        torch.save(agent.q_net.state_dict(), save_path)
        print(f"Model saved to {save_path}", flush=True)

run = DQN()
run.run()