import crt_car_env
print("Available environments in Gymnasium:")
from gymnasium.envs.registration import registry
for env_id in registry:
    print(f"- {env_id}")
print("\nTrying to import environment directly:")
from crt_car_env.envs.grid_world import CRTCarEnv
print("CRTCarEnv imported successfully")