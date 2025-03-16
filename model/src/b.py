import aiohttp 
import asyncio
from env import Environment

async def main():
    async with aiohttp.ClientSession() as session:
        env = Environment(session, "http://172.31.50.187:3000")
        state = env.reset()
        print(state)


if __name__ == "__main__":
    asyncio.run(main)
