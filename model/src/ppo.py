import asyncio
import aiohttp
from net import INPUT_KEYS, NN, batch_inputs, optim, vectorize_state
import torch
from env import Environment


def to_choice(state, n):
    opt = state["option"]
    if n < 8:
        i, j = n // 2, n % 2
        return dict(type="move", move=opt["move"][i], tera=j == 1)

    i = n - 8
    return dict(type="switch", species=opt["switches"][i])


async def step(envs, batch_actions, device):
    steps = asyncio.gather(
        [env.step(actions) for env, actions in zip(envs, batch_actions)]
    )
    states, rewards, dones = zip(*steps)

    return (
        batch_inputs(list(map(vectorize_state, states))),
        torch.tensor(rewards, device=device),
        torch.tensor(dones, device=device),
    )


def combine_batches(batches):
    return {k: torch.cat([b[k] for b in batches]) for k in INPUT_KEYS}


def slice(x, idx):
    x = {k: x[k][idx] for k in INPUT_KEYS}
    return x


async def train(
    update,
    create_env,
    n_iters=100,
    n_envs=10,
    device=torch.device("cpu"),
    clip_coef=0.1,
    gamma=0.99,
    vf_coef=0.75,
    n_steps=256,
    n_epochs=5,
    gae_lambda=0.9,
    minibatch_size=32,
    lr=0.003,
):
    envs = [create_env() for _ in range(n_envs)]
    nn = NN().to(device)
    torch.compile(nn)

    optimizer = optim.AdamW(nn.parameters(), lr=lr)

    n_samples = n_steps * n_envs

    states = [None] * n_steps
    actions = torch.zeros((n_steps, n_envs), dtype=torch.long, device=device)
    logprobs = torch.zeros((n_steps, n_envs), device=device)
    dones = torch.zeros((n_steps, n_envs), device=device)
    values = torch.zeros((n_steps, n_envs), device=device)
    rewards = torch.zeros((n_steps, n_envs), device=device)
    advantages = torch.zeros((n_steps, n_envs), device=device)

    next_state, reward, done = step(envs, [] * n_envs, device)

    tot_rewards = torch.zeros(n_envs, device=device)

    for iter_i in range(n_iters):
        for step_i in range(0, n_steps):
            state = next_state
            
            with torch.no_grad():
                logits, value = nn(state)
                dist = torch.distributions.Categorical(logits)
                choice_ids = dist.sample()
                logprob = dist.log_prob(choice_ids)

                values[step_i] = value
                logprobs[step_i] = logprob
                actions[step_i] = choice_ids
                state[step_i] = state

                next_state, reward, done = await step(
                    envs,
                    [
                        dict(side=x["side"], choice=to_choice(x, y))
                        for x, y in zip(state, choice_ids)
                    ],
                    device,
                )

                tot_rewards += reward
                for env_i in torch.nonzero(done).flatten().tolist():
                    update(tot_rewards[env_i])
                tot_rewards *= 1 - done

                dones[step_i] = done
                rewards[step_i] = reward

        with torch.no_grad():
            _, next_v = nn(state)

            gae = 0
            not_dones = 1.0 - dones
            for t in reversed(range(n_steps)):
                not_done = not_dones[t]
                delta = rewards[t] + not_done * gamma * next_v - values[t]
                advantages[t] = gae = delta + not_done * gae_lambda * gamma * gae
                next_v = values[t]

            returns = advantages + values

        b_states = combine_batches(states)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        batch_idx = torch.arange(minibatch_size)

        for _ in range(n_epochs):
            b_inds = torch.randperm(n_samples)

            cnt = n_samples // minibatch_size
            for start in range(0, cnt):
                mb_inds = b_inds[start * minibatch_size : (start + 1) * minibatch_size]
                x = slice(b_states, mb_inds)
                x["batch_idx"] = batch_idx

                logits, value = nn(x)
                dist = torch.distributions.Categorical(logits)
                newlogprob = dist.log_prob(b_actions[mb_inds])

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

                v_loss = 0.5 * ((value - b_returns[mb_inds]) ** 2).mean()

                loss = pg_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()


async def main():
    async with aiohttp.ClientSession() as session:

        def create_env():
            return Environment(session, "http://172.31.50.187:3000")

        def update(r):
            print(r)

        train(update, create_env)


if __name__ == "__main__":
    asyncio.run(main)
