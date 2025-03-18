import matplotlib.pyplot as plt
import numpy as np


def rolling_avg(nums, window):
    cumsum = np.cumsum(np.insert(nums, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def plot_eps(eps, path):
    step = max(1, len(eps) // 1000)
    sample = eps[::step]
    rewards, turns, wons = zip(*sample)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].plot(rewards)
    axs[0].grid(True)

    axs[1].plot(rolling_avg(np.array(turns), 10))
    axs[1].grid(True)

    axs[2].plot(rolling_avg(np.array(wons, 10)))
    axs[2].grid(True)

    plt.savefig(path)
    plt.close()

