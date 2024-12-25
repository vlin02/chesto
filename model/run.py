import asyncio
import websockets
from json import loads, dumps

async def connect_to_server():
    uri = "ws://localhost:8080"
    async with websockets.connect(uri) as websocket:
        while True:
            data = await websocket.recv()
            pkt = loads(data)
            
            if pkt["type"] == "end":
                break
            else:
                await websocket.send(dumps({"side": pkt["side"], "choice": {"type": "auto"}}))

# Run the connect_to_server coroutine
asyncio.get_event_loop().run_until_complete(connect_to_server())