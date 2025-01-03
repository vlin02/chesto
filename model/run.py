import asyncio
import websockets
from json import loads, dumps


async def connect_to_server():
    uri = "ws://localhost:8080"
    async with websockets.connect(uri) as websocket:
        while True:
            data = await websocket.recv()
            pkts = loads(data)
            for pkt in pkts:
                print(pkt)
                if pkt[0] == "choice":
                    await websocket.send(
                        dumps({"side": pkt[1]["side"], "decision": {"type": "auto"}})
                    )
                
                if pkt[0] == "end":
                    return


# Run the connect_to_server coroutine
asyncio.get_event_loop().run_until_complete(connect_to_server())
