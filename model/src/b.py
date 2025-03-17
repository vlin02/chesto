import torch
import aiohttp 
import asyncio
from env import Environment
from pymongo import MongoClient
from input import load_lookup, vectorize_state

DB_URL = "mongodb://admin:4wj62MDCv%25X%5ErU3F@172.31.30.235:27017/"

async def main():
    async with aiohttp.ClientSession() as session:
        device = torch.device("cuda")
        client = MongoClient(DB_URL)
        lookup = load_lookup(client["chesto"], device)

        env = Environment(session, "http://172.31.50.187:3000")
        state = await env.reset()
        print(env.id, env.side, state)

        for i in range(1000):
            print(i)
            vectorize_state(state)

        # env.id = "fbfb2554-4e31-4b23-b1ff-093cd9fef36a"
        # env.side = "p1"
        # print(await env.step(dict(type="move", tera=False, move="Giga Drain")))


if __name__ == "__main__":
    asyncio.run(main())
