import random

SIDES = ["p1", "p2"]
OPP = {"p1": "p2", "p2": "p1"}


class Environment:
    def __init__(self, session, url, t):
        self.session = session
        self.url = url

        self.id = None
        self.side = None
        self.envs = []
        self.t = t
        self.with_reset = None

    async def reset(self):
        to_make = 2 * self.t - len(self.envs)
        if to_make <= 0:
            return

        sides = [random.choice(SIDES) for _ in range(max(to_make, self.t))]

        async with self.session.post(
            f"{self.url}/start", json=[[OPP[side]] for side in sides]
        ) as res:
            envs = await res.json()
            self.envs.extend(zip(sides, envs))

    async def step(self, actions):
        all_actions = []
        for env, action_id in zip(self.envs[: self.t], actions):
            all_actions.append([env.id, [dict(side=env.side, id=action_id)]])

        with_step = self.session.post(f"{self.url}/step", json=all_actions)
        
        updates = await (await with_step).json()
        if self.with_reset:
            await self.with_reset

        results = []

        for i, (env, update) in enumerate(zip(self.envs[: self.t], updates)):
            done = update["done"]
            turn = update["turn"]
            winner = update["winner"]
            curr_update = update[env.side]

            won = 0
            if winner == self.side:
                won = 1
            elif winner == OPP[self.side]:
                won = -1

            status = (done, turn, won)

            if done:
                side, next_env = self.envs.pop()
                self.envs[i] = dict(id=next_env["id"], side=side)
                results.append(
                    curr_update["reward"], status, next_env["update"][side]["state"]
                )
            else:
                results.append(curr_update["reward"], status, curr_update["state"])

        self.with_reset = await self.reset()
