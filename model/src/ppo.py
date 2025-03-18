import asyncio
import aiohttp
from net import NN
import torch
from torch import optim
from input import STATE_FIELDS, load_lookup, decode_state
from env import Environment
from pymongo import MongoClient
import torch.nn.functional as F
import matplotlib.pyplot as plt
from train import plot_eps


@profile
async def train(
    lookup,
    device,
    update,
    create_env,
    n_iters=500,
    n_envs=100,
    clip_coef=0.1,
    gamma=0.8,
    vf_coef=0.75,
    n_steps=70,
    n_epochs=15,
    gae_lambda=0.9,
    minibatch_size=256,
    lr=0.001,
):
    nn = NN(lookup).to(device)
    nn = torch.compile(nn, mode="reduce-overhead")

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

    def process_states(states):
        return decode_state(states, device)

    next_states = process_states(await asyncio.gather(*(env.reset() for env in envs)))
    tot_rewards = torch.zeros(n_envs, device=device)

    for iter in range(n_iters):
        for t in range(0, n_steps):
            print(t)
            curr_states = next_states

            with torch.no_grad():
                states[t] = curr_states
                logits, values[t] = nn(curr_states)

                dist = torch.distributions.Categorical(F.softmax(logits, dim=1))
                action_ids = dist.sample()

                log_probs[t] = dist.log_prob(action_ids)
                actions[t] = action_ids

                steps = await asyncio.gather(
                    *(
                        env.step(action_id)
                        for env, action_id in zip(envs, action_ids.tolist())
                    )
                )

                curr_rewards, statuses, next_states = zip(*steps)
                curr_rewards = torch.tensor(curr_rewards, device=device)
                next_states = process_states(next_states)

                tot_rewards += curr_rewards

                for i, (done, turn, won) in enumerate(statuses):
                    if done:
                        dones[t][i] = 1
                        update((tot_rewards[i], turn, won))
                        tot_rewards[i] = 0

                rewards[t] = curr_rewards

        with torch.no_grad():
            _, next_values = nn(curr_states)

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

        unique, counts = torch.unique(b_actions, return_counts=True)
        print("iter actions:", zip(unique, counts))

        mb_x = {}
        for epoch in range(n_epochs):
            print(epoch)
            idxs = torch.randperm(n_samples)
            cnt = n_samples // minibatch_size

            for start in range(0, cnt):
                mb_i = idxs[start * minibatch_size : (start + 1) * minibatch_size]

                for k in STATE_FIELDS:
                    mb_x[k] = b_states[k][mb_i]

                logits, value = nn(mb_x)

                dist = torch.distributions.Categorical(F.softmax(logits, dim=1))
                new_log_probs = dist.log_prob(b_actions[mb_i])
                log_ratio = new_log_probs - b_log_probs[mb_i]
                ratio = log_ratio.exp()

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

                loss = pg_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


DB_URL = "mongodb://admin:4wj62MDCv%25X%5ErU3F@172.31.30.235:27017/"


async def main():
    torch.set_float32_matmul_precision("high")

    connector = aiohttp.TCPConnector(limit=200)

    async with aiohttp.ClientSession(connector=connector) as session:

        def create_env():
            return Environment(session, "http://172.31.50.187:3000")

        eps = []
        def update(ep):
            eps.append(ep)

            if len(eps) % 100 == 0:
                fig = plot_eps(eps)
                fig.savefig("rewards_plot_1.png")
                plt.close()

        device = torch.device("cuda")
        client = MongoClient(DB_URL)
        lookup = load_lookup(client["chesto"], device)

        await train(lookup, device, update, create_env)


if __name__ == "__main__":
    asyncio.run(main())
