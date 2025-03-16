import os
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from trial import run_experiments

t = time.time()
exp_name = f"{os.path.basename(__file__).split(".")[0]}-{int(t)}"

torch.set_num_threads(1)


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_dist(self, x):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return Normal(action_mean, action_std)


def make_env(env_id):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env

    return thunk


def train(
    update,
    seed=None,
    env_id="Pendulum-v1",
    n_steps=200,
    n_iters=50,
    n_envs=10,
    gamma=0.9,
    clip_coef=0.1,
    n_epochs=5,
    gae_lambda=0,
    minibatch_size=32,
    lr=5e-4,
    vf_coef=.5,
    device="cpu",
):
    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)

    envs = gym.vector.SyncVectorEnv([make_env(env_id) for _ in range(n_envs)])

    agent = Agent().to(device)

    optimizer = optim.AdamW(agent.parameters(), lr=lr)

    n_samples = n_steps * n_envs
    obs = torch.zeros((n_steps, n_envs, 3), device=device)
    next_obss = torch.zeros((n_steps, n_envs, 3), device=device)
    actions = torch.zeros((n_steps, n_envs, 1), device=device)
    logprobs = torch.zeros((n_steps, n_envs), device=device)
    rewards = torch.zeros((n_steps, n_envs), device=device)
    dones = torch.zeros((n_steps, n_envs), device=device)
    values = torch.zeros((n_steps, n_envs), device=device)
    advantages = torch.zeros((n_steps, n_envs), device=device)
    returns = torch.zeros((n_steps, n_envs), device=device)

    def process_obs(obs):
        obs = torch.Tensor(obs).to(device)
        obs = obs[:, 2] / 8
        return obs

    next_obs, _ = envs.reset(seed=seed)
    next_obs = process_obs(next_obs)
    next_done = torch.zeros(n_envs).to(device)

    tot_eps = 0
    episode_rewards = []

    for it in range(n_iters):
        for step in range(0, n_steps):
            obs[step] = next_obs
            dones[step] = next_done

            # Action logic using dist
            with torch.no_grad():
                dist = agent.get_action_dist(next_obs)
                action = dist.sample()
                logprob = dist.log_prob(action).sum(1)
                value = agent.get_value(next_obs).flatten()

                values[step] = value
                actions[step] = action
                logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)

            rewards[step] = ((torch.tensor(reward).to(device) + 8) / 10).view(-1)
            next_obs = process_obs(next_obs)
            
            next_obss[step] = next_obs
            next_done = torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episodic_return = info["episode"]["r"]
                        update(episodic_return)
                        tot_eps += 1

        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()

            # Calculate GAE advantages (keep this the same)
            gae = 0
            for t in reversed(range(n_steps)):
                not_done = 1.0 - dones[t]
                delta = rewards[t] + not_done * gamma * next_value - values[t]
                advantages[t] = gae = delta + not_done * gae_lambda * gamma * gae
                next_value = values[t]
            
            next_return = agent.get_value(next_obs).flatten()
            for t in reversed(range(n_steps)):
                not_done = 1.0 - dones[t]
                returns[t] = rewards[t] + not_done * gamma * next_return
                next_return = returns[t]

        # Flatten the batch
        b_obs = obs.reshape(-1, 3)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1, 1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        for _ in range(n_epochs):
            b_inds = torch.randperm(n_samples)

            for start in range(0, n_samples, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                dist = agent.get_action_dist(b_obs[mb_inds])
                newlogprob = dist.log_prob(b_actions[mb_inds]).sum(1)
                newvalue = agent.get_value(b_obs[mb_inds]).view(-1)

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                # Policy loss calculation
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Simple value loss
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Total loss
                loss = pg_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

    envs.close()

    return agent, episode_rewards



if __name__ == "__main__":
    run_experiments(
        train, f"runs/d/{exp_name}.png", [dict(minibatch_size=x) for x in [16, 32, 64, 128]]
    )