import cluster from "cluster"
import { WebSocketServer } from "ws"
import { toID } from "@pkmn/sim"
import { Arena } from "./arena.js"

if (cluster.isPrimary) {
  for (let i = 0; i < 5; i++) {
    cluster.fork()
  }
} else {
  const wss = new WebSocketServer({ port: 8080 })

  wss.on("connection", async (ws) => {
    const arena = new Arena({
      formatId: toID("gen9randombattle")
    })

    ws.on("message", (data) => {
      const action = JSON.parse(data.toString())
      arena.send(action)
    })

    ws.on("close", () => {
      arena.close()
    })

    arena.emitter.on("event", (event) => {
      if (event.type === "end") {
        arena.close()
      }

      ws.send(JSON.stringify(event))
    })

    arena.start()
  })
}
