import { parentPort } from "worker_threads"
import { Sim } from "./sim.js"
import { Side, SIDES } from "../battle.js"
import { Action } from "../parser/action.js"
import { evalBattle } from "../enc/reward.js"
import { encodeObserver, ObserverF as BattleState } from "../enc/state.js"

const sims = new Map<string, Sim>()

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

function step(sim: Sim, actions: Action[]): Update {
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
      const sim = new Sim()
      sims.set(id, sim)

      main.postMessage([id, step(sim, [])])
      break
    }
    case "step": {
      const { actions } = body
      const sim = sims.get(id)!

      main.postMessage([id, step(sim, actions)])
      break
    }
    case "close": {
      sims.delete(id)
      break
    }
  }
})
