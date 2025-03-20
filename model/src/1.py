from asyncio import run
from aiohttp import ClientSession
import time 
from env import BatchEnv
async def main():
    N = 1
    async with ClientSession() as session:
        env = BatchEnv(session, "http://172.31.50.187:3001", 1, 1)
        
        await env.reset()
        start = time.perf_counter()
        actions = [0] * N
        for i in range(100):
            _, done = await env.step(actions)
            print(i, done)
        
        end = time.perf_counter()
        tot = end - start
        print(tot, tot / (N * 1000))
    
        await env.close()
        


if __name__ == "__main__":
    run(main())
