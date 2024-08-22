from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import env

# Define the bounding box of the game window
bbox = (1250, 110, 2600, 1100)  # Set your correct coordinates
print(f"Bounding box set to: {bbox}")

# Initialize the environment
env = env.PokemonEnv(bbox=bbox, interval=1)
print("Environment initialized.")

# Ensure the environment is valid
print("Checking environment validity...")
check_env(env)
print("Environment is valid.")

# Create the model using MlpPolicy
print("Creating the PPO model with MlpPolicy...")
model = PPO("MlpPolicy", env, verbose=1)
print("Model created.")

# Start continuous training and evaluatiyon
print("Starting continuous training and evaluation...")

while True:
    # Train the model indefinitely
    model.learn(total_timesteps=1000)  # You can adjust the number of timesteps per training loop if desired
    print("Training step completed.")

    # Reset the environment to start a new evaluation
    obs, _ = env.reset()

    # Run the model for a set number of steps or indefinitely
    for step in range(1000):  # You can adjust or remove the loop condition for continuous evaluation
        print(f"Step {step + 1}:")
        action, _states = model.predict(obs)
        print(f"Action predicted: {action}")
        obs, rewards, terminated, truncated, info = env.step(action)
        print(f"Reward received: {rewards}")
        if terminated or truncated:
            print("Episode ended or truncated. Resetting environment...")
            obs, _ = env.reset()
            break  # Restart evaluation after the episode ends

    # Optionally, save the model at intervals
    model.save("ppo_pokemon_model")
    print("Model saved.")

print("Continuous training and evaluation completed.")
