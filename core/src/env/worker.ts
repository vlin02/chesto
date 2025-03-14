import { parentPort } from "worker_threads"
import { Side } from "../battle.js"
import { Teams } from "@pkmn/sim"
import { Generations } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
import { Action, Environment } from "./env.js"
import { TeamGenerators } from "@pkmn/randoms"

Teams.setGeneratorFactory(TeamGenerators)

export type WorkerRequest = [
  string,
  { type: "start"; auto: Side[] } | { type: "step"; actions: Action[] } | { type: "close" }
]

const envs = new Map<string, Environment>()
const gen = new Generations(Dex).get(9)

const main = parentPort!

main!.on("message", ([id, body]: WorkerRequest) => {
  switch (body.type) {
    case "start": {
      const { auto } = body

      const env = new Environment(gen, auto)
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
