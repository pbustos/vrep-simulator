from stable_baselines import results_plotter
from matplotlib import pyplot as plt
import time

while(True):
    results_plotter.plot_results(["./log2"], 10e6, results_plotter.X_TIMESTEPS, "Breakout")
    plt.pause(10)
    plt.close()
    