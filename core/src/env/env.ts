import { Side, SIDES, Winner } from "../battle.js"
import { Log, split } from "../log.js"
import { Observer } from "../parser/observer.js"
import { RandomAgent } from "../eval/agents.js"
import { BattleF, encodeBattle } from "../model/state.js"
import { Battle, toID } from "@pkmn/sim"
import { Generation } from "@pkmn/data"
import { evalBattle } from "../model/reward.js"
import { Choice, toMoves } from "../parser/option.js"

function toBuf(x: any, type: "float" | "int"): Buffer {
  x = x.flat(Infinity)
  if (type === "float") x = new Float32Array(x)
  else x = new Int32Array(x)
  return Buffer.from(x.buffer)
}

export type PackedBattle = {
  partyEnc: Buffer
  userEnc: Buffer
  activeIdx: Buffer
  moveChoiceIdx: Buffer
  moveMask: Buffer
  switchMask: Buffer
}

export function packBattle({
  partyEnc,
  userEnc,
  activeIdx,
  moveChoiceIdx,
  moveMask,
  switchMask
}: BattleF) {
  return {
    partyEnc: toBuf(partyEnc, "float"),
    userEnc: toBuf(userEnc, "float"),
    activeIdx: toBuf(activeIdx, "int"),
    moveChoiceIdx: toBuf(moveChoiceIdx, "int"),
    moveMask: toBuf(moveMask, "int"),
    switchMask: toBuf(switchMask, "int")
  }
}

export type Action = { side: Side; id: number }

type SideUpdate = {
  reward?: number
  state?: PackedBattle
}

export type EnvUpdate = {
  done: boolean
  winner: Winner | null
  turn: number
  p1?: SideUpdate
  p2?: SideUpdate
}

type Player = { obs: Observer; v?: number }

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

export class Environment {
  battle: Battle
  p1: Player
  p2: Player
  auto: Side[]
  logs: Log[]
  turnLimit: number

  constructor(gen: Generation, auto: Side[], turnLimit: number) {
    this.p1 = { obs: new Observer(gen) }
    this.p2 = { obs: new Observer(gen) }
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

  private choose(side: Side, choice: Choice) {
    this.battle.choose(side, this[side].obs.formatChoice(choice))
    this.battle.sendUpdates()
  }

  private stepReward(side: Side) {
    const p = this[side]
    const v = evalBattle(p.obs)
    const r = p.v == null ? undefined : v - p.v
    p.v = v
    return r
  }

  step(actions: Action[]): EnvUpdate {
    for (const action of actions) {
      const { side, id } = action
      // this.choose(side, toChoice(this[side].obs, id))
      this.choose(side, new RandomAgent(this[side].obs).choose())
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
          done: true,
          turn,
          winner,
          p1: { reward: this.stepReward("p1")! },
          p2: { reward: this.stepReward("p2")! }
        }
      }

      const deferred: Side[] = []

      for (const side of pending) {
        if (this.auto.includes(side)) {
          const { obs } = this[side]
          const choice = new RandomAgent(obs).choose()
          this.choose(side, choice)
        } else {
          deferred.push(side)
        }
      }

      if (deferred.length) {
        const update: EnvUpdate = { done: false, turn, winner }
        for (const side of deferred) {
          update[side] = {
            reward: this.stepReward(side),
            state: packBattle(encodeBattle(this[side].obs))
          }
        }

        return update
      }
    }
  }
}
