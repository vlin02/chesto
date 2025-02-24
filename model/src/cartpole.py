import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)

def train_cartpole(num_episodes=1000):
    env = gym.make('CartPole-v1')
    policy = Policy(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    
    max_reward = 0
    for episode in range(num_episodes):
        # Collect trajectory
        states, actions, rewards = [], [], []
        state, _ = env.reset()
        done = False
        
        # Run one episode
        while not done:
            state_tensor = torch.FloatTensor(state)
            # Get action probabilities
            probs = policy(state_tensor)
            # Sample action
            distribution = Categorical(probs)
            action = distribution.sample()
            
            # Step environment
            next_state, reward, done, truncated, _ = env.step(action.item())
            done = done or truncated
            
            # Store trajectory
            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
        
        # Calculate episode quantities
        states = torch.stack(states)
        actions = torch.stack(actions)
        
        # Calculate discounted rewards
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        returns = normalize(returns)
        
        # Get action probabilities and log probs
        probs = policy(states)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        
        # Calculate loss (simpler PPO loss)
        loss = -(log_probs * returns).mean()
        
        # Update policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track progress
        episode_reward = sum(rewards)
        max_reward = max(max_reward, episode_reward)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Reward: {episode_reward}, Best: {max_reward}")
            
        if max_reward >= 495:  # Solved
            print(f"Solved in {episode + 1} episodes!")
            break

if __name__ == "__main__":
    train_cartpole()