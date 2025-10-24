import gymnasium as gym
import crt_car_env
print("Environment package imported successfully")
env = gym.make('CRTCar-v0')
print("Environment created successfully")
observation, info = env.reset()
print("Environment reset successfully")
print(f"Observation space: {observation}")
env.close()