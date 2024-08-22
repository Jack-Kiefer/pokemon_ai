import gymnasium as gym
import numpy as np
import hashlib
from PIL import ImageGrab, Image
import time
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import pydirectinput
import imagehash

# Custom Environment for the Game
class PokemonEnv(gym.Env):
    def __init__(self, bbox, interval=1, pixel_size=(32, 32), hash_size=8, hash_threshold=5):
        super(PokemonEnv, self).__init__()
        self.bbox = bbox
        self.interval = interval
        self.pixel_size = pixel_size  # The size to which the image will be resized
        self.hash_size = hash_size  # Size of the hash (for perceptual hashing)
        self.hash_threshold = hash_threshold  # Hamming distance threshold for considering images as similar
        self.action_space = gym.spaces.Discrete(6)  # 8 possible actions
        # Observation space will be the pixelated image dimensions with 3 color channels
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(pixel_size[1], pixel_size[0], 3), dtype=np.uint8)
        self.image_hashes = []  # List to store image hashes
        self.recent_hashes = deque(maxlen=50)  # Stores the last 50 perceptual hashes
        self.last_hash = None  # To store the last perceptual hash
        self.episode_length = 0
        self.max_steps = 1000  # Arbitrary max steps per episode
        self.seed()
        print(f"Environment initialized with bbox: {self.bbox}, interval: {self.interval}, pixel_size: {self.pixel_size}")

    def seed(self, seed=None):
        np.random.seed(seed)
        print(f"Seed set to: {seed}")

    def perceptual_hash(self, image):
        return imagehash.average_hash(image, hash_size=self.hash_size)

    def is_similar(self, new_hash):
        for existing_hash in self.image_hashes:
            if new_hash - existing_hash < self.hash_threshold:
                return True
        return False

    def step(self, action):
        self.episode_length += 1
        print(f"\nStep {self.episode_length}: Action taken: {action}")

        # Take the action (input to the game)
        self.perform_action(action)
        
        # Capture the screenshot
        screenshot = ImageGrab.grab(bbox=self.bbox)
        print("Screenshot captured.")

        # Pixelate the screenshot by resizing it to a smaller size
        pixelated_screenshot = screenshot.resize(self.pixel_size, Image.BILINEAR)

        # Convert the pixelated screenshot to a NumPy array for observation
        observation = np.array(pixelated_screenshot)
        print("Pixelated observation created.")

        # Calculate the perceptual hash of the image
        perceptual_hash = self.perceptual_hash(pixelated_screenshot)
        print(f"Perceptual hash calculated: {perceptual_hash}")

        # Determine the reward
        if self.last_hash and perceptual_hash == self.last_hash:
            reward = 0  # Negative reward for no change
            print("No change detected. Negative reward: -1")
        elif not self.is_similar(perceptual_hash):
            reward = 100  # Positive reward for completely new or significantly different image
            self.image_hashes.append(perceptual_hash)
            print("New image detected. Full reward: 1")
        elif perceptual_hash not in self.recent_hashes:
            reward = 0.5  # Smaller reward for new-ish image
            print("Image not in recent hashes. Partial reward: 0.5")
        else:
            reward = 0  # No reward for similar image but not new
            print("Similar image detected. No reward.")

        # Update the deque of recent hashes
        self.recent_hashes.append(perceptual_hash)
        self.last_hash = perceptual_hash  # Store the current hash for comparison in the next step

        # Check if the episode is terminated
        terminated = False
        truncated = self.episode_length >= self.max_steps  # Truncated if max steps are reached
        if truncated:
            print("Max steps reached. Episode truncated.")

        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        # Set the seed for reproducibility
        self.seed(seed)
        
        self.episode_length = 0  # Reset episode length
        print("\nEnvironment reset. Episode length reset.")

        # Start with a new screenshot
        initial_state = ImageGrab.grab(bbox=self.bbox)
        print("Initial screenshot captured.")

        # Pixelate the initial screenshot
        pixelated_initial_state = initial_state.resize(self.pixel_size, Image.BILINEAR)
        
        # Convert to observation
        observation = np.array(pixelated_initial_state)

        # Calculate the perceptual hash of the initial image
        perceptual_hash = self.perceptual_hash(pixelated_initial_state)
        print(f"Initial perceptual hash: {perceptual_hash}")

        # Clear the recent hashes deque at the start of each episode
        self.recent_hashes.clear()
        self.last_hash = perceptual_hash  # Initialize last_hash with the starting state

        return observation, {}

    def perform_action(self, action):
        # Map action numbers to specific game inputs
        actions = ['a', 'b', 'up', 'down', 'left', 'right']
        
        # Mapping these to actual DeSmuME controls
        key_mapping = {
            'a': 'a',  # A button
            'b': 'b',  # B button
            'x': 'x',  # X button
            'y': 'y',  # Y button
            'up': 'up',  # D-Pad Up
            'down': 'down',  # D-Pad Down
            'left': 'left',  # D-Pad Left
            'right': 'right'  # D-Pad Right
        }

        # Get the action string
        action_str = actions[action]
        
        # Print which action is being performed
        print(f"Performing action: {action_str}")

        # Check if the action is valid and press the corresponding key
        if action_str in key_mapping:
            pydirectinput.keyDown(key_mapping[action_str])
            time.sleep(0.01)
            print(f"Pressed {key_mapping[action_str]}")
            pydirectinput.keyUp(key_mapping[action_str])

        # Add a delay between actions
        time.sleep(self.interval)
