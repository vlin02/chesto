from asyncio import run
from aiohttp import ClientSession
import time 
from env import Environment
async def main():
    N = 60
    async with ClientSession() as session:
        env = Environment(session, "http://172.31.50.187:3001", N)
        await env.reset()
        start = time.perf_counter()
        actions = [0] * N
        for _ in range(1000):
            await env.step(actions)
        
        end = time.perf_counter()
        tot = end - start
        print (tot, tot / (N * 10000))
        


if __name__ == "__main__":
    run(main())
