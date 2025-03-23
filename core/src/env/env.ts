import { Side, SIDES, Winner } from "../battle.js"
import { Log, split } from "../log.js"
import { Observer } from "../parser/observer.js"
import { BattleF, encodeBattle } from "./model/state.js"
import { Battle } from "@pkmn/sim"
import { Generation } from "@pkmn/data"
import { evalBattle } from "./model/reward.js"
import { Choice, toMoves } from "../parser/option.js"
import { resolveChoice } from "./model/option.js"
import { startBattle } from "../sim.js"
import { chooseRandom } from "../agents/random.js"

export type Action = { [k in Side]?: number }

export type Transition = {
  reward?: number
  state?: any
}

export type Update = {
  done?: {
    winner: Winner | null
    turn: number
  }
  p1?: Transition
  p2?: Transition
}

export function toChoice(obs: Observer, id: number): Choice {
  const opt = obs.getOption()!
  if (id < 8) {
    const j = id % 2
    const i = (id - j) / 2
    return {
      type: "move",
      move: toMoves(opt.select!)[i],
      tera: j === 1
    }
  }

  id -= 8
  return { type: "switch", species: [...Object.keys(obs.ally.team)][id] }
}

type Player = { obs: Observer; v?: number }

export class Environment {
  battle: Battle
  p1: Player
  p2: Player
  auto: Side[]
  logs: Log[]
  turnLimit: number
  pack: (x: BattleF) => void

  constructor(
    gen: Generation,
    { auto, turnLimit, pack }: { auto: Side[]; turnLimit: number; pack: (x: BattleF) => void }
  ) {
    this.p1 = { obs: new Observer(gen) }
    this.p2 = { obs: new Observer(gen) }
    this.auto = auto
    this.logs = []
    this.battle = startBattle({
      formatId: "gen9randombattle",
      p1: "p1",
      p2: "p2",
      send: (x) => this.logs.push(x)
    })

    this.battle.sendUpdates()
    this.turnLimit = turnLimit
    this.pack = pack
  }

  private choose(side: Side, choice: Choice) {
    this.battle.choose(side, this[side].obs.toInput(choice))
    this.battle.sendUpdates()
  }

  private stepReward(side: Side) {
    const p = this[side]
    const v = evalBattle(p.obs)
    const r = p.v == null ? undefined : v - p.v
    p.v = v
    return r
  }

  start() {
    return this.step({})
  }

  step(action: Action): Update {
    for (const k in action) {
      const side = k as Side
      const choiceId = action[side]!
      this.choose(side, resolveChoice(this[side].obs, choiceId))
    }

    while (true) {
      let winner: Winner | null = null
      const pending: Side[] = []

      for (const log of this.logs) {
        const ch = split(log)

        for (const side of SIDES) {
          const { obs } = this[side]

          for (const line of ch[side]) {
            const e = obs.read(line)

            if (e.error?.startsWith("[Invalid choice]")) {
              console.log(JSON.stringify(this.battle.inputLog))

              throw e.error
            }
            if (e.winner) winner = e.winner
            if (e.pending) pending.push(side)
          }
        }
      }

      this.logs = []

      const { turn } = this.p1.obs
      if ((turn && turn > this.turnLimit) || winner) {
        return {
          done: {
            turn,
            winner
          },
          p1: { reward: this.stepReward("p1")! },
          p2: { reward: this.stepReward("p2")! }
        }
      }

      const deferred: Side[] = []

      for (const side of pending) {
        if (this.auto.includes(side)) {
          const { obs } = this[side]
          this.choose(side, chooseRandom(obs))
        } else {
          deferred.push(side)
        }
      }

      if (deferred.length) {
        const update: Update = {}
        for (const side of deferred) {
          update[side] = {
            reward: this.stepReward(side),
            state: this.pack(encodeBattle(this[side].obs))
          }
        }
        return update
      }
    }
  }
}
