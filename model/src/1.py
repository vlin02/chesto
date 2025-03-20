from asyncio import run
from aiohttp import ClientSession
import time 
from env import BatchEnv
async def main():
    N = 80
    async with ClientSession() as session:
        env = BatchEnv(session, "http://172.31.50.187:3001", N, 80)
        
        await env.reset()
        start = time.perf_counter()
        actions = [0] * N
        for i in range(1000):
            await env.step(actions)
        
        end = time.perf_counter()
        tot = end - start
        print(tot, tot / (N * 1000))
    
        await env.close()
        


if __name__ == "__main__":
    run(main())
