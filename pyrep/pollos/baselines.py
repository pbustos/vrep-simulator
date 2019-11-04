import sys, os
sys.path.append("/home/pbustos/software/")
import numpy as np
import gym
import tensorflow as tf
from mpi4py import MPI
from stable_baselines import logger, bench
from stable_baselines.common.misc_util import set_global_seeds, boolean_flag
from stable_baselines.ddpg.policies import MlpPolicy, LnMlpPolicy
#from stable_baselines.sac.policies import MlpPolicy, LnMlpPolicy, CnnPolicy
from stable_baselines.ddpg import DDPG
from stable_baselines import SAC
from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec, OrnsteinUhlenbeckActionNoise, NormalActionNoise
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan

#from envtraj import EnvPollos
from environment import EnvPollos

import time

best_mean_reward, n_steps = -np.inf, 0
log_dir = "log/"
os.makedirs(log_dir, exist_ok=True)

def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  global n_steps, best_mean_reward, log_dir
  # Print stats every 300 calls
  if (n_steps + 1) % 300 == 0:
      # Evaluate policy training performance
      x, y = ts2xy(load_results(log_dir),'timesteps')
      if len(x) > 0:
          mean_reward = np.mean(y[-100:])
          print(x[-1], 'timesteps')
          print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))
          # New best model, you could save the agent here
          if mean_reward > best_mean_reward:
              best_mean_reward = mean_reward
              # Example for saving best model
              print("Saving new best model")
              _locals['self'].save(log_dir + 'best_model.pkl')
  n_steps += 1
  return True


# Configure things.
rank = 0
rank = MPI.COMM_WORLD.Get_rank()
if rank != 0:
    logger.set_level(logger.DISABLED)

# Create envs.
env = EnvPollos()
env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
env = Monitor(env, log_dir, allow_early_resets=True)
eval_env = None

# Parse noise_type
action_noise = None
param_noise = None
nb_actions = env.action_space.shape[-1]
param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(0.2), desired_action_stddev=float(0.2))

# Seed everything to make things reproducible.
seed = 0
seed = seed + 1000000 * rank
logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
tf.reset_default_graph()
set_global_seeds(seed)
env.seed(seed)
if eval_env is not None:
    eval_env.seed(seed)

# Disable logging for rank != 0 to avoid noise.
start_time = 0
if rank == 0:
    start_time = time.time()

policy = LnMlpPolicy

num_timesteps = 1e6
#model = SAC(MlpPolicy, env, verbose=1)
#model.learn(total_timesteps=100000, log_interval=10, callback=callback)
model = DDPG(policy=policy, env=env, param_noise=param_noise, action_noise=action_noise, buffer_size=int(1e6), verbose=1)
model.learn(total_timesteps=num_timesteps, callback=callback)
logger.info('total runtime: {}s'.format(time.time() - start_time))
env.close()
