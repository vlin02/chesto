import cluster from "cluster"
import { WebSocketServer } from "ws"
import { toID } from "@pkmn/sim"
import { Battle } from "@pkmn/sim"
import { Client } from "./client.js"
import { Decision, make, Side } from "./protocol.js"

type Message = {
  side: Side
  decision: Decision
}

if (cluster.isPrimary) {
  for (let i = 0; i < 5; i++) {
    cluster.fork()
  }
} else {
  const wss = new WebSocketServer({ port: 8080 })

  wss.on("connection", async (ws) => {
    const listener = new Client()

    const battle = new Battle({
      formatid: toID("gen9randombattle"),
      send: listener.receive,
      p1: {},
      p2: {}
    })

    function sync() {
      const events = listener.consume()
      console.log(events)
      ws.send(JSON.stringify(events))
    }

    ws.on("message", (data) => {
      const {side, decision}: Message = JSON.parse(data.toString())
      battle.choose(side, make(decision))
      sync()
    })

    ws.on("close", () => {
      battle.destroy()
      console.log("closed")
    })

    sync()
  })
}
