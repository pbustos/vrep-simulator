from stable_baselines import results_plotter
from matplotlib import pyplot as plt


results_plotter.plot_results(["./log2"], 10e6, results_plotter.X_TIMESTEPS, "Breakout")
plt.show()