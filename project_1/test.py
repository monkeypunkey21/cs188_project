import numpy as np
import robosuite as suite
from policies import *


# create environment instance
env = suite.make(
    env_name="Door", # replace with other tasks "Stack" and "NutAssembly"
    robots="Panda",  
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    horizon=5000,
)

# reset the environment
for _ in range(5):
    obs = env.reset()
    policy = DoorPolicy(obs)
    
    print(obs.keys())  # print observation keys

    while True:
        action = policy.get_action(obs)
        obs, reward, done, info = env.step(action)  # take action in the environment
        
        env.render()  # render on display
        if reward == 1.0 or done: break
