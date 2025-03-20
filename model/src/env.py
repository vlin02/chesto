import random
from bson import BSON

SIDES = ["p1", "p2"]
OPP = {"p1": "p2", "p2": "p1"}


class BatchEnv:
    def __init__(self, session, url, size, upkeep_freq):
        self.session = session
        self.url = url

        self.size = size
        self.upkeep_freq = upkeep_freq
        self.buffers = []
        self.envs = []
        self.done_ids = []
        self.upkeep_task = None
        self.upkeep_i = upkeep_freq

    async def _new_seeds(self, n: int):
        sides = [random.choice(SIDES) for _ in range(n)]

        res = await self.session.post(
            f"{self.url}/start", json=[[OPP[side]] for side in sides]
        )
        seeds = BSON.decode(await res.read())["results"]

        return zip(sides, seeds)

    async def reset(self):
        self.envs = []

        self.buffers.extend(await self._new_seeds(2 * self.size))

        states = []
        while len(states) < self.size:
            side, seed = self.buffers.pop()
            self.envs.append((seed["id"], side))
            states.append(seed["update"][side]["state"])

        return states

    async def upkeep(self):
        for_delete = self.session.delete(f"{self.url}/", json=self.done_ids)
        for_buffers = await self._new_seeds(self.upkeep_freq)

        self.done_ids = []

        self.buffers.extend(for_buffers)
        await for_delete

    async def step(self, choice_ids: list[int]):
        batch_step = [
            {"id": env_id, "action": {side: choice_id}}
            for (env_id, side), choice_id in zip(self.envs, choice_ids)
        ]
        res = await self.session.post(f"{self.url}/step", json=batch_step)
        updates = BSON.decode(await res.read())["results"]

        transitions = []
        for i, ((env_id, side), update) in enumerate(zip(self.envs, updates)):
            trn = update[side]

            if "done" in update:
                done = update["done"]
                turn = done["turn"]
                winner = done["winner"]

                won = 0
                if winner == side:
                    won = 1
                elif winner == OPP[side]:
                    won = -1

                self.upkeep_i -= 1
                if self.upkeep_i == 0:
                    if self.upkeep_task:
                        await self.upkeep_task
                    self.upkeep_task = self.upkeep()
                    self.upkeep_i = self.upkeep_freq

                self.done_ids.append(env_id)
                side, seed = self.buffers.pop()
                self.envs[i] = (seed["id"], side)

                transitions.append(
                    (
                        trn["reward"],
                        (turn, won),
                        seed["update"][side]["state"],
                    )
                )
            else:
                transitions.append((trn["reward"], None, trn["state"]))

        self.with_reset = self.upkeep()
        return transitions
