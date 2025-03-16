import asyncio
import aiohttp
from net import NN
import torch
from torch import optim
from input import INPUT_KEYS, load_lookup, vectorize_state, batch_states
from env import Environment
from pymongo import MongoClient


def to_choice(state, idx):
    opt = state["option"]
    if idx < 8:
        i, j = idx // 2, idx % 2
        return dict(type="move", move=opt["move"][i], tera=j == 1)

    i = idx - 8
    return dict(type="switch", species=opt["switches"][i])


async def train(
    lookup,
    device,
    update,
    create_env,
    n_iters=100,
    n_envs=10,
    clip_coef=0.1,
    gamma=0.99,
    vf_coef=0.75,
    n_steps=256,
    n_epochs=5,
    gae_lambda=0.9,
    minibatch_size=32,
    lr=0.003,
):
    nn = NN(lookup).to(device)
    torch.compile(nn)
    
    envs = [create_env() for _ in range(n_envs)]
    optimizer = optim.AdamW(nn.parameters(), lr=lr)

    n_samples = n_steps * n_envs

    states = [None] * n_steps
    values = torch.zeros((n_steps, n_envs), device=device)
    log_probs = torch.zeros((n_steps, n_envs), device=device)
    actions = torch.zeros((n_steps, n_envs), device=device, dtype=torch.long)
    rewards = torch.zeros((n_steps, n_envs), device=device)
    dones = torch.zeros((n_steps, n_envs), device=device)
    advantages = torch.zeros((n_steps, n_envs), device=device)

    next_states = batch_states([vectorize_state(x, lookup, device) for x in asyncio.gather(*(env.reset() for env in envs))])
    tot_rewards = torch.zeros(n_envs, device=device)

    for iter in range(n_iters):
        for t in range(0, n_steps):
            curr_states = next_states
            
            with torch.no_grad():
                states[t] = curr_states
                logits, values[t] = nn(curr_states)
                
                dist = torch.distributions.Categorical(logits)
                action_idxs = dist.sample()

                log_probs[t] = dist.log_prob(action_idxs)
                actions[t] = action_idxs

                steps = asyncio.gather(
                    env.step(to_choice(state, idx)) for env, state, idx in zip(envs, curr_states, action_idxs)
                )

                curr_rewards, curr_dones, next_states =  zip(*steps)
                curr_rewards = torch.tensor(curr_rewards, device=device)
                curr_dones = torch.tensor(curr_dones, device=device)
                next_states = batch_states([vectorize_state(x, device) for x in next_states])

                tot_rewards += curr_rewards
                for env_i in torch.nonzero(curr_dones).flatten().tolist():
                    update(tot_rewards[env_i])
                tot_rewards *= 1 - curr_dones

                rewards[t] = curr_rewards
                dones[t] = curr_dones

        with torch.no_grad():
            _, next_values = nn(curr_states)

            gae = 0
            not_dones = 1.0 - dones
            for t in reversed(range(n_steps)):
                not_done = not_dones[t]
                delta = rewards[t] + not_done * gamma * next_values - values[t]
                advantages[t] = gae = delta + not_done * gae_lambda * gamma * gae
                next_values = values[t]

            returns = advantages + values

        b_states = {k: torch.cat([x[k] for x in states]) for k in INPUT_KEYS}
        b_log_probs = log_probs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        x = {"batch_idx": torch.arange(minibatch_size, device=device)}

        for _ in range(n_epochs):
            idxs = torch.randperm(n_samples)
            cnt = n_samples // minibatch_size

            for start in range(0, cnt):
                mb_idxs = idxs[start * minibatch_size : (start + 1) * minibatch_size]
                
                for k in INPUT_KEYS:
                    x[k] = torch.cat(b_states[k][mb_idxs])
                logits, value = nn(x)

                dist = torch.distributions.Categorical(logits)
                new_log_probs = dist.log_prob(b_actions[mb_idxs])
                log_ratio = new_log_probs - b_log_probs[mb_idxs]
                ratio = log_ratio.exp()

                mb_advantages = b_advantages[mb_idxs]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((value - b_returns[mb_idxs]) ** 2).mean()

                loss = pg_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()


DB_URL = "mongodb://admin:4wj62MDCv%25X%5ErU3F@172.31.30.235:27017/"

async def main():
    async with aiohttp.ClientSession() as session:
        def create_env():
            return Environment(session, "http://172.31.50.187:3000")

        def update(r):
            print(r)

        device = torch.device("cpu")
        client = MongoClient(DB_URL)
        lookup = load_lookup(client["chesto"], device)

        await train(lookup, device, update, create_env)


if __name__ == "__main__":
    asyncio.run(main())
