import { parentPort } from "worker_threads"
import { Side } from "../battle.js"
import { Observer } from "../parser/observer.js"
import { Battle, Teams, toID } from "@pkmn/sim"
import { Generations } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
import { Log } from "../log.js"
import { Action, Environment } from "./env.js"
import { TeamGenerators } from "@pkmn/randoms"

Teams.setGeneratorFactory(TeamGenerators)

export type Request = [
  string,
  { type: "start"; auto: Side[] } | { type: "step"; actions: Action[] } | { type: "close" }
]

const envs = new Map<string, Environment>()
const gen = new Generations(Dex).get(9)

const main = parentPort!

main!.on("message", ([id, body]: Request) => {
  switch (body.type) {
    case "start": {
      const { auto } = body

      const env = {
        p1: new Observer(gen),
        p2: new Observer(gen),
        auto,
        logs: []
      } as any

      env.battle = new Battle({
        formatid: toID("gen9randombattle"),
        p1: { name: "p1" },
        p2: { name: "p2" },
        send: (...log) => {
          env.logs.push(log as Log)
        }
      })
      env.battle.sendUpdates()

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
