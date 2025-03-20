from asyncio import create_task
import random
from aiohttp import ClientSession
from bson import BSON

SIDES = ["p1", "p2"]
OPP = {"p1": "p2", "p2": "p1"}


class BatchEnv:
    session: ClientSession

    def __init__(self, session, url, size, upkeep_freq):
        self.session = session
        self.url = url

        self.size = size
        self.upkeep_freq = upkeep_freq
        self._buffers = []
        self._envs = []
        self._done_ids = []
        self._upkeep_task = None

    async def _get_buffers(self, n: int):
        sides = [random.choice(SIDES) for _ in range(n)]

        res = await self.session.post(
            f"{self.url}/start", json=[[OPP[side]] for side in sides]
        )
        seeds = BSON.decode(await res.read())["results"]

        return [
            ((seed["id"], side), seed["update"][side]["state"])
            for side, seed in zip(sides, seeds)
        ]

    async def reset(self):
        self._envs = []

        self._buffers.extend(await self._get_buffers(self.size + 2 * self.upkeep_freq))

        states = []
        while len(states) < self.size:
            env, state = self._buffers.pop()
            self._envs.append(env)
            states.append(state)

        return states

    async def upkeep(self, done_ids):
        for_delete = self.session.delete(f"{self.url}/", json=done_ids)
        for_buffers = self._get_buffers(self.upkeep_freq)

        buffers = await for_buffers
        self._buffers.extend(buffers)
        await for_delete

    async def step(self, choice_ids: list[int]):
        batch_step = [
            {"id": env_id, "action": {side: choice_id}}
            for (env_id, side), choice_id in zip(self._envs, choice_ids)
        ]
        res = await self.session.post(f"{self.url}/step", json=batch_step)
        updates = BSON.decode(await res.read())["results"]

        transitions = []
        dones = []
        for i, ((env_id, side), update) in enumerate(zip(self._envs, updates)):
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

                dones.append((i, turn, won))
                self._done_ids.append(env_id)
                env, state = self._buffers.pop()

                if len(self._done_ids) == self.upkeep_freq:
                    if self._upkeep_task:
                        await self._upkeep_task
                    self._upkeep_task = create_task(self.upkeep(self._done_ids))
                    self._done_ids = []

                self._envs[i] = env

                transitions.append((trn["reward"], state))
            else:
                transitions.append((trn["reward"], trn["state"]))

        return transitions, dones

    async def close(self):
        if self._upkeep_task:
            await self._upkeep_task

        await self.session.delete(
            f"{self.url}/",
            json=[
                *self._done_ids,
                *[id for id, _ in self._envs],
                *[id for (id, _), _ in self._buffers],
            ],
        )
