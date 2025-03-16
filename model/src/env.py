import random

SIDES = ["p1", "p2"]
OPP = {"p1": "p2", "p2": "p1"}


class Environment:
    def __init__(self, session, url):
        self.session = session
        self.url = url
        self.done = True
        self.id = None
        self.side = None

    def _process_update(self, update):
        side_update = update[self.side]
        return (side_update["state"], side_update["reward"])

    async def reset(self):
        self.done = False
        self.side = random.choice(SIDES)
        async with self.session.post(
            f"{self.url}/start", json=dict(auto=[OPP[self.side]])
        ) as res:
            res = await res.json()
            self.id = res["id"]

            return self._process_update(res["update"])

    async def step(self, choice):
        async with self.session.post(
            f"{self.url}/{self.id}/step/", json=dict(side=self.side, choice=choice)
        ) as res:
            res = await res.json()
            done = res["done"]

            if done:
                x = await self.reset()
            else:
                x = self._process_update(res)

            return (*x, done)
