import random

SIDES = ["p1", "p2"]
OPP = {"p1": "p2", "p2": "p1"}


class Environment:
    def __init__(self, session, url):
        self.session = session
        self.url = url

        self.id = None
        self.side = None

    async def reset(self):
        if self.id:
            await self.session.delete(f"{self.url}/{self.id}")
        
        self.done = False
        self.side = random.choice(SIDES)

        async with self.session.post(
            f"{self.url}/start", json=dict(auto=[OPP[self.side]])
        ) as res:
            res = await res.json()
            self.id = res["id"]

            return res["update"][self.side]["state"]

    async def step(self, choice):
        async with self.session.post(
            f"{self.url}/{self.id}/step", json=[dict(side=self.side, choice=choice)]
        ) as res:
            res = await res.json()
            done = res["done"]
            side = res[self.side]

            if done:
                next_state = await self.reset()
                return side["reward"], True, next_state
            else:
                return side["reward"], False, side["state"]
