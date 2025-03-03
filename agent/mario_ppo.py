# Import necessary libraries
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import VecVideoRecorder
import csv
import os

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

# Create and preprocess the environment
def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # Speed up
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=84)

    # Initialize TensorFlow summary writer
    log_file = "mario_ppo_log.csv"
    with open(log_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Event", "X_Pos"])

    # Tweak rewards
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
            
            jump_actions = [2, 4, 5]
            speed_actions = [3, 4]

            # Increase penalty for standing still
            if action == 0:
                reward -= 200
            
            # Reward moving right more aggressively
            x_increase = max(0, info['x_pos'] - self.last_x)
            if action in speed_actions:
                reward += x_increase * 2.0  # Increased reward for moving right
            else:
                reward += x_increase * 1.5  # Slightly lower reward for other actions
            
            # Encourage being speedy
            if action in speed_actions:
                reward += max(0, info['x_pos'] - self.max_x) * 1.2
            else:
                reward += max(0, info['x_pos'] - self.max_x)

            # Encourage sustained jumps when stuck
            if (info['x_pos'] - self.last_x) == 0:
                if action in jump_actions:
                    self.jump_counter += 1
                    reward += 10 * min(self.jump_counter, 5)  # Increase reward for sustained jumping
                else:
                    self.jump_counter = 0
                    reward -= 5
            
            # Reward finishing quickly
            if info["flag_get"]:
                reward += 1000
                reward += info["time"]
                done = True
                print("GOAL")
                self.log_x_pos(self.last_x, "finished")

            # Punish dying
            if info["life"] < 2:
                reward -= 50
                done = True
                self.log_x_pos(self.last_x, "death")
                
            self.max_x = max(self.max_x, self.last_x)
            self.last_x = info["x_pos"]
            return state, reward / 10., done, info
        
    env = CustomEnvWrapper(env)
    return env

# Main training function
def train_mario_agent():
    # Create save directory
    save_dir = './mario_ppo_checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4, channels_order='last')

    # Video setup
    video_folder = "./ppo_videos"
    os.makedirs(video_folder, exist_ok=True)
    env = VecVideoRecorder(
        env,
        video_folder=video_folder,
        record_video_trigger=lambda x: x % 10000 == 0,  # Record every 10,000 steps
        video_length=10000,  # Length of recorded video
        name_prefix="mario-ppo-video"
    )

    # Continuing training a previous model or start fresh
    # model_path = "mario_ppo_model.zip"
    model_path = "mario_ppo_model.zip"
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = PPO.load(model_path, env=env)
    else:
        print("No existing model found. Training from scratch.")
        model = PPO(
            policy='CnnPolicy',
            env=env,
            learning_rate=2.5e-4,
            n_steps=128,
            batch_size=32,
            n_epochs=4,
            gamma=0.95,
            ent_coef=0.15,
            verbose=1,
            tensorboard_log="./mario_tensorboard/"
        )

    # Create the callback for saving checkpoints
    checkpoint_callback = SaveCheckpointCallback(
        # Change this to control how often it is saved
        save_freq=150000,
        save_path=save_dir,
        verbose=1
    )

    # Train the model
    print("Starting training...")
    model.learn(
        total_timesteps=1500000,  # timesteps (adjust as needed)
        callback=checkpoint_callback
    )

    # Save the final model
    model.save("mario_ppo_model")

# Function to test the trained agent
def test_mario_agent():
    # Create the environment
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4, channels_order='last')

    # Load the trained model
    model = PPO.load("mario_ppo_checkpoints/mario_model_500000.zip")

    # Test the agent
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        env.render()

    print(f"Total reward: {total_reward}")
    env.close()

if __name__ == '__main__':
    # Train the agent
    train_mario_agent()
    # test_mario_agent()
