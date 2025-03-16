# Import necessary libraries
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from gym.wrappers import GrayScaleObservation, ResizeObservation
import csv
import os
from typing import Callable

OPTIMAL_MOVEMENT = [['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A']]
jump_actions = [1, 3, 4]
speed_actions = [2, 3]

# Custom callback to save the model periodically
class SaveCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super(SaveCheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f'mario_model_{self.n_calls}')
            self.model.save(model_path)
            if self.verbose > 0:
                print(f'Saving model checkpoint to {model_path}')
        return True

def linear_schedule(initial_value: float, total_timesteps: int, trained_timesteps: int = 0) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        # Adjust progress_remaining to account for prior training
        adjusted_progress = max(0.0, (total_timesteps - trained_timesteps) / total_timesteps) * progress_remaining
        return max(initial_value * adjusted_progress, 1e-6)
    return func

class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env=None):
        super(CustomEnvWrapper, self).__init__(env)
        self.last_x = 0
        self.max_x = 0
        self.episode = 0

    def reset(self, **kwargs):
        self.last_x = 0
        self.max_x = 0
        self.jump_counter = 0
        return self.env.reset(**kwargs)
        
    def log_x_pos(self, x_pos, event_type):
        with open(log_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([self.episode, event_type, x_pos])
        self.episode += 1
        
    def step(self, action):
        state, reward, done, info = self.env.step(action)

        # Reward moving right more aggressively
        x_increase = max(0, info['x_pos'] - self.last_x)
        if action in speed_actions:
            reward += x_increase * 2.0
        else:
            reward += x_increase * 1.5

        # Encourage sustained jumps when stuck
        if (info['x_pos'] - self.last_x) == 0:
            if action in jump_actions:
                self.jump_counter += 1
                reward += min(self.jump_counter, 5)
            else:
                self.jump_counter = 0
                reward -= 5

        # Reward finishing quickly
        if info["flag_get"]:
            reward += math.exp(-0.005 * info["time"]) * 1000
            done = True
            self.log_x_pos(self.last_x, "finished")

        # Punish dying
        if info["life"] < 2:
            reward -= 250
            done = True
            self.log_x_pos(self.last_x, "death")

        self.max_x = max(self.max_x, self.last_x)
        self.last_x = info["x_pos"]
        
        return state, reward / 10., done, info

def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, OPTIMAL_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=84)
    global log_file
    log_file = "mario_ppo_log.csv"
    with open(log_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Event", "X_Pos"])
    env = CustomEnvWrapper(env)
    return env

def train_mario_agent(total_timesteps=2000000, trained_timesteps=0):
    save_dir = './mario_ppo_checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    model_path = "mario_ppo_model_dne.zip"

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = PPO.load(model_path, env=env)
        # Update learning rate schedule with trained_timesteps
        model.learning_rate = linear_schedule(3e-4, total_timesteps, trained_timesteps)
    else:
        print("No existing model found. Training from scratch.")
        model = PPO(
            policy='CnnPolicy',
            env=env,
            learning_rate=linear_schedule(3e-4, total_timesteps, trained_timesteps),
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log="./mario_tensorboard_ppo/"
        )
    
    checkpoint_callback = SaveCheckpointCallback(save_freq=200000, save_path=save_dir, verbose=1)
    print("Starting training...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save("mario_ppo_model")

def test_mario_agent():
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    model = PPO.load("mario_ppo_model.zip")
    state = env.reset()
    done = False
    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, reward, done, info = env.step(action)
        env.render()
    env.close()

if __name__ == '__main__':
    train_mario_agent(total_timesteps=2000000, trained_timesteps=0)
    # test_mario_agent()
