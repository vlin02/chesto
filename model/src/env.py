import random

# device = torch.device("cpu")

# DB_URL = "mongodb://admin:4wj62MDCv%25X%5ErU3F@172.31.30.235:27017/"
# client = MongoClient(DB_URL)
# lookup = get_lookup(client["chesto"], device)
# nn = NN(lookup).to(device)

# update = json.loads(S)
# print(
#     nn(
#         batch_inputs(
#             [process_input(lookup, update["p1"]["state"], device=torch.device("cpu"))]
#         )
#     )
# )

SIDES = ["p1", "p2"]
OPP = {"p1": "p2", "p2": "p1"}


class Environment:
    def __init__(self, session, url):
        self.session = session
        self.url = url
        self.done = True
        self.id = None
        self.side = None

    def process_update(self, update):
        side_update = update[self.side]
        step = (side_update["state"], side_update["reward"], update["done"])

        if update["done"]:
            self.done = True

        return step

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
            return self.process_update(res["update"])

    async def step(self, actions):
        if self.done:
            return await self.reset()

        async with self.session.post(f"{self.url}/step", json=actions) as res:
            return self.process_update(res["update"])


# async def main():
#     async with aiohttp.ClientSession() as session:
#         env = Environment(session, "http://172.31.50.187:3000")
#         print(await env.step([]))
#         print(env.done)
#         print(await env.step([]))
#         print(env.id, env.side, env.done)


# if __name__ == "__main__":
#     asyncio.run(main())
