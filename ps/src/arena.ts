import cluster from "cluster"
import { WebSocketServer } from "ws"
import { toID } from "@pkmn/sim"
import { Battle } from "@pkmn/sim"
import { Decision, make, Side } from "./battle.js"
import { Observer } from "./observer.js"
import { Log } from "./replay.js"

type Message = {
  side: Side
  decision: Decision
}

const CORES = 7

if (cluster.isPrimary) {
  for (let i = 0; i < CORES; i++) {
    cluster.fork()
  }
} else {
  const wss = new WebSocketServer({ port: 8080 })

  wss.on("connection", async (ws) => {
    const observer = new Observer()
    let logs: Log[] = []

    const battle = new Battle({
      formatid: toID("gen9randombattle"),
      p1: {},
      p2: {},
      send: (...log) => logs.push(log as Log)
    })

    function update() {
      battle.sendUpdates()
      ws.send(JSON.stringify(observer.consume(logs)))
      logs = []
    }

    ws.on("message", (data) => {
      const { side, decision }: Message = JSON.parse(data.toString())
      battle.choose(side, make(decision))

      update()
    })

    ws.on("close", () => {
      battle.destroy()
      console.log("closed")
    })

    update()
  })
}
