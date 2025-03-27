import { Battle } from "@pkmn/sim"
import { Side, Winner, SIDES } from "../battle.js"
import { Log, split } from "../log.js"
import { Observer } from "../parser/observer.js"
import { BattleSeed, startBattle } from "../sim.js"
import { Generation } from "@pkmn/data"

type Transition = {
  pending: Side[]
  logs: Log[]
}

export class Trial {
  logs: Log[]
  p1: Observer
  p2: Observer
  battle: Battle
  winner: Winner | null

  constructor(gen: Generation, seed?: BattleSeed) {
    this.logs = []
    this.winner = null
    this.battle = undefined as any
    this.p1 = new Observer(gen)
    this.p2 = new Observer(gen)

    this.battle = startBattle({
      seed,
      formatId: "gen9randombattle",
      p1: "p1",
      p2: "p2",
      send: (log) => {
        this.logs.push(log as Log)
      }
    })
  }

  transition() {
    this.battle.sendUpdates()

    const pending: Side[] = []
    for (const log of this.logs) {
      const ch = split(log)

      for (const side of SIDES) {
        const obs = this[side]
        for (const line of ch[side]) {
          const e = obs.read(line)

          if (e.error?.startsWith("[Invalid choice]")) {
            throw e.error
          }
          if (e.winner) this.winner = e.winner
          if (e.pending) pending.push(side)
        }
      }
    }

    const trn: Transition = {
      pending,
      logs: this.logs
    }
    this.logs = []

    return trn
  }
}
