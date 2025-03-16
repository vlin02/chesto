import aiohttp 
import asyncio
from env import Environment

async def main():
    async with aiohttp.ClientSession() as session:
        env = Environment(session, "http://172.31.50.187:3000")
        # state = await env.reset()
        # print(env.id, env.side, state)

        env.id = "fbfb2554-4e31-4b23-b1ff-093cd9fef36a"
        env.side = "p1"
        print(await env.step(dict(type="move", tera=False, move="Giga Drain")))


if __name__ == "__main__":
    asyncio.run(main())
