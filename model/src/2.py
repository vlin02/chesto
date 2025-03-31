import asyncio
from collections import defaultdict
import aiohttp
import torch
from torch import optim
from state import STATE_FIELDS, load_lookup, decode_states
from env import BatchEnv
import torch.nn.functional as F
from trial import plot_eps
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from net import Agent, Config

import os
import time

t = time.time()
exp_name = f"__tmp/5/{os.path.basename(__file__).split('.')[0]}-{int(t)}"


async def main():
    torch.set_float32_matmul_precision("high")
    manager = Manager()
    shared_eps = manager.list()
    eps = []
    plotter = ProcessPoolExecutor(max_workers=1)

    def update(ep):
        nonlocal eps
        eps.append(ep)
        if len(eps) % 500 == 0:
            shared_eps.extend(eps)
            plotter.submit(plot_eps, shared_eps, exp_name)
            eps = []

    await train(update)


async def train(
    update,
    n_envs=60,
    n_iters=500,
    minibatch_size=256,
    n_epochs=10,
    n_steps=60,

    ent_coef=0.005,
    gamma=1,
    gae_lambda=1,
    hidden_dim=512,
    clip_coef=0.05,
    lr=5e-5,
):
    c = Config(hidden_dim=hidden_dim)

    api = aiohttp.ClientSession(base_url="http://172.31.50.187:3001")

    env = BatchEnv(api=api, size=n_envs, upkeep_freq=n_envs)

    device = torch.device("cuda")
    lookup = await load_lookup(c=c, api=api, device=device)

    agent = Agent(lookup, c).to(device)
    agent = torch.compile(agent, mode="reduce-overhead", fullgraph=True, dynamic=False)
    agent.load_state_dict(torch.load("__tmp/5/2-1743380973.pt"))

    actor_opt = optim.AdamW(agent.actor.parameters(), lr=lr)
    critic_opt = optim.AdamW(agent.critic.parameters(), lr=1e-3)

    n_samples = n_steps * n_envs

    states = [None] * n_steps
    values = torch.zeros((n_steps, n_envs), device=device)
    log_probs = torch.zeros((n_steps, n_envs), device=device)
    actions = torch.zeros((n_steps, n_envs), device=device, dtype=torch.long)
    rewards = torch.zeros((n_steps, n_envs), device=device)
    dones = torch.zeros((n_steps, n_envs), device=device)
    advs = torch.zeros((n_steps, n_envs), device=device)

    raw_move_logits = torch.zeros((n_steps, n_envs, 4, 2), device=device)
    raw_switch_logits = torch.zeros((n_steps, n_envs, 6), device=device)
    tot_rewards = torch.zeros((n_envs,), device=device)

    next_states = decode_states(c, await env.reset(), device)

    for it in range(n_iters):
        dones *= 0

        print("it:", it)

        for t in range(0, n_steps):
            if t % 10 == 0:
                print("t:", t)
            
            with torch.no_grad():
                curr_states = next_states
                states[t] = curr_states

                dist, values[t], (move_logits, switch_logits) = agent(curr_states)
                raw_move_logits[t] = move_logits
                raw_switch_logits[t] = switch_logits

                action_ids = dist.sample()
                log_probs[t] = dist.log_prob(action_ids)
                actions[t] = action_ids

                trns, curr_dones = await env.step(action_ids.cpu().tolist())

                curr_rewards, next_states = zip(*trns)
                next_states = decode_states(c, next_states, device)

                rewards[t] = torch.tensor(curr_rewards).to(device)
                tot_rewards += rewards[t]

                for i, turn, won in curr_dones:
                    won = max(0, won)
                    update((tot_rewards[i].cpu().item(), turn, won))

                    dones[t][i] = 1
                    tot_rewards[i] = 0

        with torch.no_grad():
            _, next_values, _ = agent(next_states)

            gae = 0
            not_dones = 1 - dones
            for t in reversed(range(n_steps)):
                not_done = not_dones[t]
                delta = rewards[t] + not_done * gamma * next_values - values[t]
                advs[t] = gae = delta + not_done * gae_lambda * gamma * gae
                next_values = values[t]

            returns = advs + values

        b_states = {k: torch.cat([x[k] for x in states]) for k in STATE_FIELDS}
        b_log_probs = log_probs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advs.reshape(-1)
        b_returns = returns.reshape(-1)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        true_probs = F.softmax(torch.cat([raw_move_logits.flatten(-2), raw_switch_logits], dim=-1), dim=-1).mean(
            dim=(0, 1)
        )
        print("probs:", [f"{i}: {p:.4f}" for i,p in enumerate(true_probs)])

        idxs = torch.randperm(n_samples)
        mbs = []
        for i in range(0, n_samples, minibatch_size):
            mb_i = idxs[i : i + minibatch_size]
            mb_x = {k: b_states[k][mb_i] for k in STATE_FIELDS}

            mbs.append(
                (
                    mb_x,
                    b_actions[mb_i],
                    b_log_probs[mb_i],
                    b_advantages[mb_i],
                    b_returns[mb_i],
                )
            )

        for epoch in range(n_epochs):
            idxs = torch.randperm(n_samples)
            tot_v_loss = 0
            tot_entropy = 0
            tot_pg_loss = 0
            tot_ratio = 0

            for mb_x, mb_actions, mb_log_probs, mb_adv, mb_returns in mbs:
                dist, value, _ = agent(mb_x)

                curr_log_probs = dist.log_prob(mb_actions)
                ratio = (curr_log_probs - mb_log_probs).exp()

                tot_ratio += (1 - ratio).abs().mean()

                entropy = dist.entropy().mean()

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((value - mb_returns) ** 2).mean()

                loss = pg_loss - ent_coef * entropy + v_loss
                
                tot_v_loss += v_loss
                tot_entropy += entropy
                tot_pg_loss += pg_loss

                actor_opt.zero_grad()
                critic_opt.zero_grad()
                loss.backward()
                actor_opt.step()
                critic_opt.step()

            print(
                "epoch:",
                epoch,
                "r:",
                f"{tot_ratio.item() / len(mbs):.4f}",
                "v:",
                f"{tot_v_loss.item() / len(mbs):.4f}",
                "e:",
                f"{tot_entropy.item() / len(mbs):.4f}",
                "pg:",
                f"{tot_pg_loss.item() / len(mbs):.4f}",
            )

        if it % 20 == 4:
            print("checkpoint")
            torch.save(agent.state_dict(), f"{exp_name}.pt")

        print()


if __name__ == "__main__":
    asyncio.run(main())
