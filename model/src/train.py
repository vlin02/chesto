import matplotlib.pyplot as plt
import numpy as np


def rolling_avg(nums, window):
    cumsum = np.cumsum(np.insert(nums, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def plot_eps(eps, path):
    rewards, turns, wons = zip(*eps)
    
    # Create figure with GridSpec layout
    fig = plt.figure(figsize=(20, 20))
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1])
    
    # Top plot - rewards (spanning full width)
    ax_rewards = fig.add_subplot(gs[0, :])
    ax_rewards.plot(rolling_avg(np.array(rewards), 1000))
    ax_rewards.grid(True)
    
    # Bottom left - turns
    ax_turns = fig.add_subplot(gs[1, 0])
    ax_turns.plot(rolling_avg(np.array(turns), 1000))
    ax_turns.grid(True)
    
    # Bottom right - wins
    ax_wins = fig.add_subplot(gs[1, 1])
    ax_wins.plot(rolling_avg(np.array(wons), 1000))
    ax_wins.grid(True)
    
    plt.savefig(path)
    plt.close()
