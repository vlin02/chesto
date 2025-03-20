from asyncio import run
from aiohttp import ClientSession

from env import Environment
async def main():
    async with ClientSession() as session:
        env = Environment(session, "http://localhost:3001", 10)
        await env.reset()
        actions = [0] * 10
        for _ in range(200):
            await env.step(actions)
        print(env.delete_ids)
        


if __name__ == "__main__":
    run(main())
