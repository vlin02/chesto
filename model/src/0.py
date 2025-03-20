import asyncio
import aiohttp
from net import NN
import torch
from torch import optim
from input import STATE_FIELDS, load_lookup, decode_states
from env import BatchEnv
from pymongo import MongoClient
import torch.nn.functional as F
from train import plot_eps
from concurrent.futures import ThreadPoolExecutor

import os
import time

t = time.time()
exp_name = f"__tmp/0/{os.path.basename(__file__).split('.')[0]}-{int(t)}"


async def train(
    env: BatchEnv,
    lookup,
    device,
    update,
    n_iters=500,
    clip_coef=0.15,
    gamma=1,
    vf_coef=0.5,
    n_steps=50,
    n_epochs=10,
    ent_coef=0.01,
    gae_lambda=0.8,
    minibatch_size=512,
    lr=0.0001,
    target_kl=0.01,
):
    n_envs = env.size
    nn = NN(lookup).to(device)
    # nn = torch.compile(nn, mode="reduce-overhead")

    optimizer = optim.AdamW(nn.parameters(), lr=lr)

    n_samples = n_steps * n_envs

    states = [None] * n_steps
    values = torch.zeros((n_steps, n_envs), device=device)
    log_probs = torch.zeros((n_steps, n_envs), device=device)
    actions = torch.zeros((n_steps, n_envs), device=device, dtype=torch.long)
    rewards = torch.zeros((n_steps, n_envs), device=device)
    dones = torch.zeros((n_steps, n_envs), device=device)
    advantages = torch.zeros((n_steps, n_envs), device=device)
    true_probs = torch.zeros((n_steps, n_envs, 14), device=device)

    curr_states = decode_states(await env.reset(), device)
    tot_rewards = torch.zeros(n_envs)

    for it in range(n_iters):
        for t in range(0, n_steps):
            print("it:", it, "t:", t)

            with torch.no_grad():
                states[t] = curr_states
                logits, values[t], (move_logits, switch_logits) = nn(curr_states)

                true_probs[t] = F.softmax(
                    torch.cat([move_logits.flatten(1), switch_logits], dim=-1), dim=1
                )

                dist = torch.distributions.Categorical(F.softmax(logits, dim=1))
                action_ids = dist.sample()

                log_probs[t] = dist.log_prob(action_ids)
                actions[t] = action_ids

                action_ids.tolist()
                trns, curr_dones = await env.step(action_ids.cpu().tolist())

                curr_rewards, curr_states = zip(*trns)

                curr_rewards = torch.tensor(curr_rewards)
                tot_rewards += curr_rewards
                rewards[t] = curr_rewards.to(device)

                for i, turn, won in curr_dones:
                    dones[t][i] = 1
                    update((tot_rewards[i].item(), turn, max(won, 0)))
                    tot_rewards[i] = 0

                curr_states = decode_states(curr_states, device)
        print(rewards.cpu().tolist())
        return

        with torch.no_grad():
            _, next_values, _ = nn(curr_states)

            gae = 0
            not_dones = 1 - dones
            for t in reversed(range(n_steps)):
                not_done = not_dones[t]
                delta = rewards[t] + not_done * gamma * next_values - values[t]
                advantages[t] = gae = delta + not_done * gae_lambda * gamma * gae
                next_values = values[t]

            returns = advantages + values

        b_states = {k: torch.cat([x[k] for x in states]) for k in STATE_FIELDS}
        b_log_probs = log_probs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        print("it:", it, "probs:", true_probs.mean(dim=(0, 1)))

        mb_x = {}
        for epoch in range(n_epochs):
            idxs = torch.randperm(n_samples)
            cnt = n_samples // minibatch_size
            kl = 0

            for start in range(0, cnt):
                mb_i = idxs[start * minibatch_size : (start + 1) * minibatch_size]

                for k in STATE_FIELDS:
                    mb_x[k] = b_states[k][mb_i]

                logits, value, _ = nn(mb_x)

                dist = torch.distributions.Categorical(F.softmax(logits, dim=1))
                new_log_probs = dist.log_prob(b_actions[mb_i])
                log_ratio = new_log_probs - b_log_probs[mb_i]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean()

                entropy = dist.entropy().mean()

                mb_advantages = b_advantages[mb_i]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((value - b_returns[mb_i]) ** 2).mean()

                loss = pg_loss + vf_coef * v_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print("epoch:", epoch, "kl:", kl)
            if target_kl is not None and kl > target_kl:
                print("Early stopped")
                break

        if it % 5 == 4:
            torch.save(nn.state_dict(), f"{exp_name}.pt")


async def main():
    DB_URL = "mongodb://admin:4wj62MDCv%25X%5ErU3F@172.31.30.235:27017/"
    torch.set_float32_matmul_precision("high")

    connector = aiohttp.TCPConnector(limit=20000)
    plotter = ThreadPoolExecutor(max_workers=1, thread_name_prefix="plot_thread")

    async with aiohttp.ClientSession(connector=connector) as session:
        eps = []

        def update(ep):
            eps.append(ep)

            if len(eps) % 100 == 0:
                plotter.submit(plot_eps, eps, exp_name)

        env = BatchEnv(session, "http://172.31.50.187:3001", 500, 100)

        device = torch.device("cuda")
        client = MongoClient(DB_URL)
        lookup = load_lookup(client["chesto"], device)

        await train(env, lookup, device, update)


if __name__ == "__main__":
    asyncio.run(main())
