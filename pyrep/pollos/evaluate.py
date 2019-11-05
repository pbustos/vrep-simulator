import sys, os
sys.path.append("/home/pbustos/software/")
from stable_baselines import DDPG
#from environment import EnvPollos
from envcartesian import EnvPollos

# Create envs.
env = EnvPollos()

# Instantiate the agent
log_dir = "log2/"
model = DDPG.load(log_dir + "best_model.pkl", env)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    #env.render()
    if done:
        env.reset()


