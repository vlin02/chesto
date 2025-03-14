import { Side, SIDES } from "../battle.js"
import { Log, split } from "../log.js"
import { Observer } from "../parser/observer.js"
import { RandomAgent } from "../agents.js"
import { Choice } from "../parser/option.js"
import { BattleF, encodeBattle } from "../model/state.js"
import { Battle, toID } from "@pkmn/sim"
import { Generation } from "@pkmn/data"
import { evalBattle } from "../model/reward.js"

export type Action = { side: Side; choice: Choice }

type SideUpdate = {
  reward: number
  state: BattleF
}

export type EnvUpdate =
  | {
      done: true
      p1: SideUpdate
      p2: SideUpdate
    }
  | {
      done: false
      p1?: SideUpdate
      p2?: SideUpdate
    }

type Player = { obs: Observer; v: number }

export class Environment {
  battle: Battle
  p1: Player
  p2: Player
  auto: Side[]
  logs: Log[]
  turnLimit: number

  constructor(gen: Generation, auto: Side[], turnLimit: number) {
    this.p1 = { obs: new Observer(gen), v: 0 }
    this.p2 = { obs: new Observer(gen), v: 0 }
    this.auto = auto
    this.logs = []
    this.battle = new Battle({
      formatid: toID("gen9randombattle"),
      p1: { name: "p1" },
      p2: { name: "p2" },
      send: (...log) => {
        this.logs.push(log as Log)
      }
    })

    this.battle.sendUpdates()
    this.turnLimit = turnLimit
  }

  private choose({ side, choice }: Action) {
    this.battle.choose(side, this[side].obs.formatChoice(choice))
    this.battle.sendUpdates()
  }

  private getSideUpdate(side: Side) {
    const p = this[side]
    const v = evalBattle(p.obs)
    const update: SideUpdate = { reward: v - p.v, state: encodeBattle(p.obs) }
    p.v = v
    return update
  }

  step(actions: Action[]): EnvUpdate {
    for (const action of actions) {
      this.choose(action)
    }

    while (true) {
      let winner = false
      let turn: number | undefined 
      const pending: Side[] = []

      for (const log of this.logs) {
        const ch = split(log)

        for (const side of SIDES) {
          const { obs } = this[side]
          for (const line of ch[side]) {
            const e = obs.read(line)

            if (e.error?.startsWith("[Invalid choice]")) {
              throw e.error
            }
            if (e.winner) winner = true
            if (e.pending) pending.push(side)
            if (e.turn) turn = e.turn
          }
        }
      }
      this.logs = []

      if ((turn && turn > this.turnLimit) || winner) {
        return {
          done: true,
          p1: this.getSideUpdate("p1"),
          p2: this.getSideUpdate("p2")
        }
      }

      const deferred: Side[] = []

      for (const side of pending) {
        if (this.auto.includes(side)) {
          const { obs } = this[side]
          const choice = new RandomAgent(obs).choose()
          this.choose({ side, choice })
        } else {
          deferred.push(side)
        }
      }

      if (deferred.length) {
        const update: EnvUpdate = { done: false }
        for (const side of deferred) {
          update[side] = this.getSideUpdate(side)
        }

        return update
      }
    }
  }
}
