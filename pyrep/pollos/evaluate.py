import sys, os
sys.path.append("/home/pbustos/software/")
from stable_baselines import DDPG
from environment import EnvPollos

# Create envs.
env = EnvPollos()

# Instantiate the agent
log_dir = "log/"
model = DDPG.load(log_dir + "best_model.pkl", env)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()


