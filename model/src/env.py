import random

SIDES = ["p1", "p2"]
OPP = {"p1": "p2", "p2": "p1"}


class Environment:
    def __init__(self, session, url, t):
        self.session = session
        self.url = url

        self.envs = []
        self.new_envs = []
        self.t = t
        self.with_reset = None
        self.delete_ids = []

    async def reset(self):
        self.envs = []
        self.with_reset = None

        await self.upkeep()

        states = []
        for _ in range(self.t):
            side, env = self.new_envs.pop()
            self.envs.append(dict(id=env["id"], side=side))
            states.append(env["update"][side]["state"])

        return states

    async def upkeep(self):
        deletes = None
        if len(self.delete_ids) > self.t:
            deletes = self.session.delete(f"{self.url}/", json=self.delete_ids)
            self.delete_ids = []

        to_make = self.t - len(self.new_envs)
        if to_make <= 0:
            return

        sides = [random.choice(SIDES) for _ in range(max(to_make, self.t))]

        async with self.session.post(
            f"{self.url}/start", json=[[OPP[side]] for side in sides]
        ) as res:
            envs = await res.json()
            self.new_envs.extend(zip(sides, envs))

        if deletes:
            await deletes

    async def step(self, actions):
        all_actions = []

        for env, action_id in zip(self.envs[: self.t], actions):
            all_actions.append([env["id"], [dict(side=env["side"], id=action_id)]])

        with_step = self.session.post(f"{self.url}/step", json=all_actions)

        updates = await (await with_step).json()
        if self.with_reset:
            await self.with_reset

        results = []

        for i, (env, update) in enumerate(zip(self.envs, updates)):
            side = env["side"]
            done = update["done"]
            turn = update["turn"]
            winner = update["winner"]
            curr_update = update[side]

            won = 0
            if winner == side:
                won = 1
            elif winner == OPP[side]:
                won = -1

            status = (done, turn, won)

            if done:
                self.delete_ids.append(env["id"])
                next_side, next_env = self.new_envs.pop()
                self.envs[i] = dict(id=next_env["id"], side=next_side)
                results.append(
                    (
                        curr_update["reward"],
                        status,
                        next_env["update"][next_side]["state"],
                    )
                )
            else:
                results.append((curr_update["reward"], status, curr_update["state"]))

        self.with_reset = self.upkeep()
        return results
