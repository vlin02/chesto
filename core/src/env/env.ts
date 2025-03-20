import { Side, SIDES, Winner } from "../battle.js"
import { Log, split } from "../log.js"
import { Observer } from "../parser/observer.js"
import { RandomAgent } from "../eval/agents.js"
import { PackedBattle, encodeBattle, packBattle } from "../model/state.js"
import { Battle, Teams } from "@pkmn/sim"
import { Generation, Generations } from "@pkmn/data"
import { evalBattle } from "../model/reward.js"
import { Choice, toMoves } from "../parser/option.js"
import { Dex } from "@pkmn/dex"
import { TeamGenerators } from "@pkmn/randoms"

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

  constructor(battle: Battle, logs: Log[], gen: Generation, auto: Side[], turnLimit: number) {
    this.p1 = { obs: new Observer(gen) }
    this.p2 = { obs: new Observer(gen) }
    this.auto = auto
    this.logs = logs
    this.battle = battle
    //@ts-ignore
    this.battle.send = (...log: Log) => {
      this.logs.push(log)
    }

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
Teams.setGeneratorFactory(TeamGenerators)
function run() {
  const logs: Log[] = []
  const env = new Environment(
    ...[
      new Battle({
        formatid: "gen9randombattle" as any,
        p1: { name: "p1" },
        p2: { name: "p2" },
        send: (...log) => {
          logs.push(log as Log)
        }
      }),
      logs
    ],
    new Generations(Dex).get(9),
    ["p1"],
    100
  )
  env.step([])
  const start = process.hrtime()
  while (true) {
    const update = env.step([{ side: "p2", id: 1 }])

    if (update.done) break
  }
  const [seconds, nanoseconds] = process.hrtime(start)
  console.log(seconds + nanoseconds / 1e9)
}

run()
run()
