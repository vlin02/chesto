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
exp_name = f"__tmp/4/{os.path.basename(__file__).split('.')[0]}-{int(t)}"


async def main():
    torch.set_float32_matmul_precision("high")
    manager = Manager()
    shared_eps = manager.list()
    plotter = ProcessPoolExecutor(max_workers=1)

    def update(ep):
        shared_eps.append(ep)

        if len(shared_eps) % 500 == 0:
            plotter.submit(plot_eps, shared_eps, exp_name)

    await train(update, hidden_dim=256)


async def train(
    update,
    n_envs=200,
    hidden_dim=128,
    n_iters=500,
    gamma=1,
    n_steps=50,
    n_epochs=10,
    ent_coef=0.01,
    gae_lambda=1,
    minibatch_size=128,
    clip_coef=0.2,
    lr=1e-3,
):
    c = Config(hidden_dim=hidden_dim)

    api = aiohttp.ClientSession(base_url="http://172.31.50.187:3001")

    env = BatchEnv(api=api, size=n_envs, upkeep_freq=100)

    device = torch.device("cuda")
    lookup = await load_lookup(c=c, api=api, device=device)

    agent = Agent(lookup, c).to(device)
    agent = torch.compile(agent, mode="reduce-overhead", fullgraph=True)
    # nn.load_state_dict(torch.load("__tmp/4/2-1743373770.pt"))

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

        x = defaultdict(int)

        for t in range(0, n_steps):
            start = time.process_time()

            def stop(k):
                nonlocal start
                end = time.process_time()
                x[k] += end - start
                start = end

            if t % 10 == 0:
                print("t:", t)
            with torch.no_grad():
                curr_states = next_states
                stop(0)
                states[t] = curr_states
                stop(1)

                dist, values[t], (move_logits, switch_logits) = agent(curr_states)
                stop(2)
                raw_move_logits[t] = move_logits
                stop(3)
                raw_switch_logits[t] = switch_logits
                stop(4)

                action_ids = dist.sample()
                stop(5)
                log_probs[t] = dist.log_prob(action_ids)
                stop(6)
                actions[t] = action_ids
                stop(7)

                trns, curr_dones = await env.step(action_ids.cpu().tolist())
                stop(8)

                curr_rewards, next_states = zip(*trns)
                stop(9)
                next_states = decode_states(c, next_states, device)
                stop(10)

                rewards[t] = torch.tensor(curr_rewards).to(device)
                stop(11)
                tot_rewards += rewards[t]
                stop(12)

                for i, turn, won in curr_dones:
                    won = max(0, won)
                    update((tot_rewards[i].cpu().item(), turn, won))

                    dones[t][i] = 1
                    tot_rewards[i] = 0
                stop(13)

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

        true_probs = torch.cat([raw_move_logits.flatten(-2), raw_switch_logits], dim=-1).mean(dim=0)
        print("probs:", true_probs.tolist())

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
            t = 0
            tot_entropy = 0
            tot_pg_loss = 0
            tot_ratio = 0

            for mb_x, mb_actions, mb_log_probs, mb_adv, mb_returns in mbs:
                logits, value, _ = agent(mb_x)

                dist = torch.distributions.Categorical(F.softmax(logits, dim=1))
                new_log_probs = dist.log_prob(mb_actions)
                log_ratio = new_log_probs - mb_log_probs
                ratio = log_ratio.exp()

                tot_ratio += (1 - ratio).abs().mean()

                entropy = dist.entropy().mean()

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((value - mb_returns) ** 2).mean()

                loss = pg_loss - ent_coef * entropy + v_loss
                t += v_loss
                tot_entropy += ent_coef * entropy
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
                f"{t.item() / len(mbs):.4f}",
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
