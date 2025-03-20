import { parentPort } from "worker_threads"
import { Side } from "../battle.js"
import { Battle, Teams, toID } from "@pkmn/sim"
import { Generations } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
import { Action, Environment } from "./env.js"
import { TeamGenerators } from "@pkmn/randoms"
import { Log } from "../log.js"

Teams.setGeneratorFactory(TeamGenerators)

export type WorkerRequest = [
  string,
  { type: "start"; auto: Side[] } | { type: "step"; actions: Action[] } | { type: "close" }
]

const envs = new Map<string, Environment>()
const gen = new Generations(Dex).get(9)

const main = parentPort!

const TURN_LIMIT = 100

let battles: [Battle, Log[]][] = []

function refresh() {
  for (let i = 0; i < 200; i++) {
    const logs: Log[] = []
    battles.push(
      [
      new Battle({
        formatid: toID("gen9randombattle"),
        p1: { name: "p1" },
        p2: { name: "p2" },
        send: (...log) => {
          logs.push(log as Log)
        }
      }),
      logs]
    )
  }
}
refresh()

main!.on("message", ([id, body]: WorkerRequest) => {
  switch (body.type) {
    case "start": {
      const { auto } = body
      
      if (battles.length === 0) {
        refresh()
      }
      const env = new Environment(...battles.pop()!, gen, auto, TURN_LIMIT)
      console.log(battles.length)
      envs.set(id, env)

      main.postMessage([id, env.step([])])

      break
    }
    case "step": {
      const { actions } = body
      const env = envs.get(id)!

      main.postMessage([id, env.step(actions)])
      break
    }
    case "close": {
      envs.delete(id)
      break
    }
  }
})
