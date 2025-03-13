import { parentPort } from "worker_threads"
import { Sim } from "./sim.js"
import { Side, SIDES } from "../battle.js"
import { Action } from "../parser/action.js"
import { FOE } from "../parser/protocol.js"
import { calcBattleValue } from "../enc/reward.js"
import { encodeState, BattleState } from "../enc/state.js"

type Environment = {
  sim: Sim
  side: Side
}

const envs = new Map<string, Environment>()

export type Request = [
  string,
  { type: "start" } | { type: "step"; actions: Action[] } | { type: "close" }
]

export type Update = {
  done: boolean
  reward?: number
  state?: BattleState
}

const main = parentPort!

function step({ side, sim }: Environment, actions: Action[]): Update {
  const { obs } = sim[side]

  const vPrev = calcBattleValue(obs)
  const status = sim.step(actions)
  const vCurr = calcBattleValue(obs)
  const reward = vCurr - vPrev

  switch (status.type) {
    case "end":
      return {
        done: true,
        reward
      }
    case "request": {
      const state = encodeState(obs)
      return {
        done: false,
        reward,
        state
      }
    }
  }
}

main!.on("message", ([id, body]: Request) => {
  switch (body.type) {
    case "start": {
      const side = SIDES[Math.floor(Math.random() * 2)]
      const fixed = [FOE[side]]

      const sim = new Sim(fixed)
      const env = { side, sim: new Sim(fixed) }
      envs.set(id, env)

      sim.step([])
      main.postMessage([id, { done: false, state: encodeState(sim[side].obs) }])
      break
    }
    case "step": {
      const { actions } = body
      const env = envs.get(id)!

      main.postMessage([id, step(env, actions)])
      break
    }
    case "close": {
      envs.delete(id)
      break
    }
  }
})
