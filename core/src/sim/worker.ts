import { parentPort } from "worker_threads"
import { Sim } from "./sim.js"
import { Side, SIDES } from "../battle.js"
import { Action } from "../parser/action.js"
import { evalBattle } from "../enc/reward.js"
import { encodeObserver, ObserverF as BattleState } from "../enc/state.js"

type Environment = {
  sim: Sim
}

const envs = new Map<string, Environment>()

export type Request = [
  string,
  { type: "start" } | { type: "step"; actions: Action[] } | { type: "close" }
]

type SideUpdate = {
  reward: number
  state: BattleState
}

export type Update = {
  done: boolean
  p1: SideUpdate
  p2: SideUpdate
}

const main = parentPort!

function step({ sim }: Environment, actions: Action[]): Update {
  const evalSides = () => Object.fromEntries(SIDES.map((side) => [side, evalBattle(sim[side].obs)]))
  const vPrev = evalSides()
  const status = sim.step(actions)
  const vCurr = evalSides()

  const sides: { [k in Side]: SideUpdate } = {} as any
  for (const side of SIDES) {
    const reward = vCurr[side] - vPrev[side]
    sides[side] = {
      reward,
      state: encodeObserver(sim[side].obs)
    }
  }

  const done = status.type === "end"
  return { done, ...sides }
}

main!.on("message", ([id, body]: Request) => {
  switch (body.type) {
    case "start": {
      const side = SIDES[Math.floor(Math.random() * 2)]

      const sim = new Sim([...SIDES])
      const env = { side, sim }
      envs.set(id, env)

      main.postMessage([id, step(env, [])])
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
