import { parentPort } from "worker_threads"
import { Teams } from "@pkmn/sim"
import { Generations } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
import { Environment, Action, Auto } from "./env.js"
import { TeamGenerators } from "@pkmn/randoms"
import { BattleSeed } from "../sim.js"
import { heuristic } from "../model/transports/heuristic.js"

const { packBattle, getBattleFeat: encodeBattle, decodeChoice } = heuristic

Teams.setGeneratorFactory(TeamGenerators)

export type WorkerRequest = [
  string,
  (
    | { type: "start"; auto: Auto; seed?: BattleSeed }
    | { type: "step"; action: Action }
    | { type: "close" }
  )
]

const envs = new Map<string, Environment>()
const gen = new Generations(Dex).get(9)

const main = parentPort!

const TURN_LIMIT = 100

main!.on("message", ([id, body]: WorkerRequest) => {
  switch (body.type) {
    case "start": {
      const { auto, seed } = body

      const env = new Environment(gen, {
        auto,
        seed,
        turnLimit: TURN_LIMIT,
        transport: heuristic
      })
      envs.set(id, env)
      main.postMessage([id, env.step({})])
      break
    }
    case "step": {
      const env = envs.get(id)!
      const { action } = body

      main.postMessage([id, env.step(action)])
      break
    }
    case "close": {
      envs.delete(id)
      break
    }
  }
})
