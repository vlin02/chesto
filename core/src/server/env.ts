import { Side, SIDES, Winner } from "../battle.js"
import { Log, split } from "../log.js"
import { Observer } from "../parser/observer.js"
import { Battle } from "@pkmn/sim"
import { Generation } from "@pkmn/data"
import { evalBattle } from "../model/reward.js"
import { Choice } from "../parser/option.js"
import { BattleSeed, startBattle } from "../sim.js"
import { chooseHeuristic } from "../agent/heuristic.js"
import { chooseRandom } from "../agent/random.js"
import { Transport } from "../model/transport.js"

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

export type Auto = { [K in Side]?: "random" | "heuristic" }
type Player = { obs: Observer; v?: number }

export class Environment {
  battle: Battle
  p1: Player
  p2: Player
  auto: Auto
  logs: Log[]
  turnLimit: number
  tp: Transport<any>
  choiceLog: [Side, Choice][]

  constructor(
    gen: Generation,
    {
      auto,
      seed,
      turnLimit,
      transport
    }: {
      auto: Auto
      seed?: BattleSeed
      turnLimit: number
      transport: Transport<any>
    }
  ) {
    this.p1 = { obs: new Observer(gen) }
    this.p2 = { obs: new Observer(gen) }
    this.auto = auto
    this.logs = []
    this.battle = startBattle({
      formatId: "gen9randombattle",
      p1: "p1",
      p2: "p2",
      seed,
      send: (x) => this.logs.push(x)
    })
    this.choiceLog = []

    this.battle.sendUpdates()
    this.turnLimit = turnLimit
    this.tp = transport
  }

  private choose(side: Side, choice: Choice) {
    this.choiceLog.push([side, choice])
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
      this.choose(side, this.tp.decodeChoice(this[side].obs, choiceId))
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
              console.log(this.choiceLog)
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
        if (this.auto[side]) {
          const { obs } = this[side]
          this.choose(
            side,
            { heuristic: chooseHeuristic, random: chooseRandom }[this.auto[side]](obs)
          )
        } else {
          deferred.push(side)
        }
      }

      if (deferred.length) {
        const update: Update = {}
        for (const side of deferred) {
          update[side] = {
            reward: this.stepReward(side),
            state: this.tp.packBattle(this.tp.encodeBattle(this[side].obs))
          }
        }
        return update
      }
    }
  }
}
