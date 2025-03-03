import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Made networks slightly bigger
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.actor(x)
    
    def get_value(self, x):
        return self.critic(x)

def train_lunarlander():
    env = gym.make('LunarLander-v3', render_mode=None)
    policy = Policy(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.002)  # Lower learning rate
    
    max_reward = -float('inf')
    for episode in range(2000):
        states, actions, rewards = [], [], []
        state, _ = env.reset()
        
        # Run episode
        while True:
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad():  # Don't track gradients during rollout
                probs = policy(state_tensor)
                action = Categorical(probs).sample()
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            
            if done:
                break
                
            state = next_state
        
        # Process episode data
        states = torch.stack(states)
        actions = torch.stack(actions)
        
        # Calculate returns with higher discount
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.995 * R  # Higher discount factor
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Get values and advantages
        values = policy.get_value(states).squeeze()
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantages
        
        # Calculate losses
        probs = policy(states)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        
        policy_loss = -(log_probs * advantages).mean()
        value_loss = 0.5 * (returns - values).pow(2).mean()
        
        loss = policy_loss + value_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)  # Add gradient clipping
        optimizer.step()
        
        episode_reward = sum(rewards)
        max_reward = max(max_reward, episode_reward)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Reward: {episode_reward:.1f}, Best: {max_reward:.1f}")

if __name__ == "__main__":
    train_lunarlander()