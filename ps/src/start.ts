import cluster from "cluster"
import { WebSocketServer } from "ws"
import { toID } from "@pkmn/sim"
import { Environment, EventStream } from "./env.js"

if (cluster.isPrimary) {
  for (let i = 0; i < 5; i++) {
    cluster.fork()
  }
} else {
  const wss = new WebSocketServer({ port: 8080 })

  wss.on("connection", async (ws) => {
    const env = new Environment({
      formatId: toID("gen9randombattle")
    })

    ws.on("message", (data) => {
      const action = JSON.parse(data.toString())
      env.send(action)
    })

    ws.on("close", () => {
      env.close()
    })

    const events = new EventStream(env.emitter)
    env.start()

    for await (const event of events) {
      ws.send(JSON.stringify(event))
    }

    env.close()
  })
}
