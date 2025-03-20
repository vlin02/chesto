import { parentPort } from "worker_threads"
import { Side } from "../battle.js"
import { Teams } from "@pkmn/sim"
import { Generations } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
import { Environment, Action } from "./env.js"
import { TeamGenerators } from "@pkmn/randoms"
import { packBinary } from "./transport.js"

Teams.setGeneratorFactory(TeamGenerators)

export type WorkerRequest = [
  string,
  { type: "start"; auto: Side[] } | { type: "step"; action: Action } | { type: "close" }
]

const envs = new Map<string, Environment>()
const gen = new Generations(Dex).get(9)

const main = parentPort!

const TURN_LIMIT = 100

main!.on("message", ([id, body]: WorkerRequest) => {
  switch (body.type) {
    case "start": {
      const { auto } = body

      const env = new Environment(gen, { auto, turnLimit: TURN_LIMIT, pack: packBinary })
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
