from gymnasium.envs.registration import register

register(
    id='CRTCar-v0',
    entry_point='crt_car_env.envs.car_env:CRTCarEnv',
)