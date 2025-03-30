import asyncio
import aiohttp
import torch
from torch import optim
from state import STATE_FIELDS, load_lookup, decode_states
from env import BatchEnv
import torch.nn.functional as F
from trial import plot_eps
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from net import ActorCritic, Config

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
    vf_coef=1,
    n_steps=100,
    n_epochs=10,
    ent_coef=0.01,
    gae_lambda=1,
    minibatch_size=256,
    clip_coef=0.15,
    lr=1e-4,
):
    c = Config(hidden_dim=hidden_dim)

    api = aiohttp.ClientSession(base_url="http://172.31.50.187:3001")

    env = BatchEnv(api=api, size=n_envs, upkeep_freq=100)

    device = torch.device("cuda")
    lookup = await load_lookup(c=c, api=api, device=device)

    nn = ActorCritic(lookup, c).to(device)
    nn = torch.compile(nn, mode="reduce-overhead")
    # nn.load_state_dict(torch.load("__tmp/4/2-1743325331.pt"))

    actor_opt = optim.AdamW(nn.actor.parameters(), lr=lr)
    critic_opt = optim.AdamW(nn.critic.parameters(), lr=1e-3)

    n_samples = n_steps * n_envs

    states = [None] * n_steps
    values = torch.zeros((n_steps, n_envs), device=device)
    true_probs = torch.zeros((n_steps, n_envs, 14), device=device)
    log_probs = torch.zeros((n_steps, n_envs), device=device)
    actions = torch.zeros((n_steps, n_envs), device=device, dtype=torch.long)
    rewards = torch.zeros((n_steps, n_envs), device=device)
    dones = torch.zeros((n_steps, n_envs), device=device)
    advs = torch.zeros((n_steps, n_envs), device=device)

    next_states = decode_states(c, await env.reset(), device)
    tot_rewards = torch.zeros(n_envs)

    for it in range(n_iters):
        dones *= 0

        print("it:", it)

        for tot_v_loss in range(0, n_steps):
            if tot_v_loss % 10 == 0:
                print("t:", tot_v_loss)
            with torch.no_grad():
                curr_states = next_states
                states[tot_v_loss] = curr_states
                logits, values[tot_v_loss], (move_logits, switch_logits) = nn(curr_states)

                true_probs[tot_v_loss] = F.softmax(torch.cat([move_logits.flatten(1), switch_logits], dim=-1), dim=1)

                dist = torch.distributions.Categorical(F.softmax(logits, dim=1))
                action_ids = dist.sample()

                log_probs[tot_v_loss] = dist.log_prob(action_ids)
                actions[tot_v_loss] = action_ids

                action_ids.tolist()

                trns, curr_dones = await env.step(action_ids.cpu().tolist())

                curr_rewards, next_states = zip(*trns)
                next_states = decode_states(c, next_states, device)

                curr_rewards = torch.tensor(curr_rewards)
                tot_rewards += curr_rewards
                rewards[tot_v_loss] = curr_rewards.to(device)

                for i, turn, won in curr_dones:
                    win = 0 if won == -1 else 1
                    dones[tot_v_loss][i] = 1
                    update((tot_rewards[i].item(), turn, win))
                    tot_rewards[i] = 0

        with torch.no_grad():
            _, next_values, _ = nn(next_states)

            gae = 0
            not_dones = 1 - dones
            for tot_v_loss in reversed(range(n_steps)):
                not_done = not_dones[tot_v_loss]
                delta = rewards[tot_v_loss] + not_done * gamma * next_values - values[tot_v_loss]
                advs[tot_v_loss] = gae = delta + not_done * gae_lambda * gamma * gae
                next_values = values[tot_v_loss]

            returns = advs + values

        b_states = {k: torch.cat([x[k] for x in states]) for k in STATE_FIELDS}
        b_log_probs = log_probs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advs.reshape(-1)
        b_returns = returns.reshape(-1)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        print("probs:", true_probs.mean(dim=(0, 1)).tolist())

        idxs = torch.randperm(n_samples)
        mbs = []
        for i in range(0, n_samples, minibatch_size):
            mb_i = idxs[i : i + minibatch_size]
            mb_x = {}
            for k in STATE_FIELDS:
                mb_x[k] = b_states[k][mb_i]

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
                logits, value, _ = nn(mb_x)

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
                tot_v_loss += v_loss
                tot_entropy += ent_coef * entropy
                tot_pg_loss += pg_loss

                actor_opt.zero_grad()
                critic_opt.zero_grad()
                loss.backward()
                actor_opt.step()
                critic_opt.step()

            with torch.no_grad():
                print(
                    "epoch:",
                    epoch,
                    "ratio:",
                    tot_ratio.item() / len(mbs),
                    "v_loss:",
                    tot_v_loss.item() / len(mbs),
                    "entropy_loss:",
                    tot_entropy.item() / len(mbs),
                    "pg_loss:",
                    tot_pg_loss.item() / len(mbs),
                )

        if it % 20 == 4:
            print("checkpoint")
            torch.save(nn.state_dict(), f"{exp_name}.pt")
        
        print()


if __name__ == "__main__":
    asyncio.run(main())
