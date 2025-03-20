import random
from bson import BSON

SIDES = ["p1", "p2"]
OPP = {"p1": "p2", "p2": "p1"}


class BacthEnv:
    def __init__(self, session, url, size, buffer_size):
        self.session = session
        self.url = url

        self.size = size
        self.buffer_size = buffer_size
        self.buffers = []
        self.envs = []
        self.done_ids = []
        self.for_upkeep = None
        self.n_left = buffer_size

    async def _fetch_buffers(self, n: int):
        sides = [random.choice(SIDES) for _ in range(n)]

        for_start = self.session.post(
            f"{self.url}/start", json=[[OPP[side]] for side in range(n)]
        )
        return zip(sides, await for_start)

    async def reset(self):
        self.envs = []

        self.buffers.extend(await self._fetch_buffers(self.size + 2 * self.upkeep_freq))

        states = []
        while len(states) < self.size:
            side, buf = self.buffers.pop()
            self.envs.append(dict(id=buf["id"], side=side))
            states.append(buf[side]["state"])

        return states

    async def upkeep(self):
        for_delete = self.session.delete(f"{self.url}/", json=self.done_ids)
        for_buffers = await self._fetch_buffers(self.buffer_size)
        self.done_ids = []

        self.buffers.extend(for_buffers)
        await for_delete

    async def step(self, actions):
        batch_step = [
            dict(id=env["id"], action=action) for env, action in zip(self.envs, actions)
        ]
        res = await self.session.post(f"{self.url}/step", json=batch_step)
        updates = BSON.decode(await res)["updates"]

        transitions = []
        for i, ((env_id, side), update) in enumerate(zip(self.envs, updates)):
            done = update["done"]
            trn = update[side]

            if done:
                turn = done["turn"]
                winner = done["winner"]

                won = 0
                if winner == side:
                    won = 1
                elif winner == OPP[side]:
                    won = -1

                self.n_left -= 1
                if self.n_left == 0:
                    if self.for_upkeep:
                        await self.for_upkeep
                    self.for_upkeep = self.upkeep()
                    self.n_left = self.buffer_size

                self.done_ids.append(env_id)
                buf_side, buf = self.new_envs.pop()
                self.envs[i] = (buf["id"], buf_side)

                transitions.append(
                    (
                        trn["reward"],
                        (turn, won),
                        buf["update"][buf_side]["state"],
                    )
                )
            else:
                transitions.append((trn["reward"], None, trn["state"]))

        self.with_reset = self.upkeep()
        return transitions
