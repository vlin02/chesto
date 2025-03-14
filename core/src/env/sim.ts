import { Battle, toID, Teams } from "@pkmn/sim"
import { Log, split } from "../log.js"
import { Observer } from "../parser/observer.js"
import { Generations } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
import { Side, SIDES } from "../battle.js"
import { Action, Choice } from "../parser/action.js"

import { TeamGenerators } from "@pkmn/randoms"
import { RandomAgent } from "../arena/agents.js"

Teams.setGeneratorFactory(TeamGenerators)

export type Status =
  | {
      type: "end"
      winner: Side | "tie"
    }
  | {
      type: "request"
      sides: Side[]
    }

interface Agent {
  choose(): Choice
}

function parseSeeds(lines: string[]): string[] {
  return lines.map((line) => {
    const jsonStr = line.includes(">start") ? line.substring(7) : line.substring(line.indexOf("{"))

    const data = JSON.parse(jsonStr)
    return data.seed[1]
  })
}
export class Sim {
  battle: Battle
  p1: { obs: Observer; agent?: Agent }
  p2: { obs: Observer; agent?: Agent }
  logs: Log[]
  choices: any[]

  constructor(seed?: string[]) {
    const gen = new Generations(Dex).get(9)

    this.p1 = { obs: new Observer(gen) }
    this.p2 = { obs: new Observer(gen) }
    this.logs = []
    this.choices = []

    const opt: any = {
      formatid: toID("gen9randombattle"),
      p1: { name: "p1" },
      p2: { name: "p2" },
      send: (...log: any) => {
        // console.log(log)
        this.logs.push(log as Log)
      }
    }

    if (seed) {
      opt.seed = ["sodium", seed[0]]
      opt.p1.seed = ["sodium", seed[1]]
      opt.p2.seed = ["sodium", seed[2]]
    }

    this.battle = new Battle(opt)

    this.battle.sendUpdates()
  }

  choose(side: Side, choice: Choice) {
    this.battle.choose(side, this[side].obs.formatChoice(choice))
    this.battle.sendUpdates()
  }

  step(actions: Action[]): Status {
    actions.forEach(({ side, choice }) => this.choose(side, choice))

    let i = 0
    while (true) {
      i += 1
      if (i > 2000) {
        console.log(JSON.stringify(this.battle.inputLog))
        console.log(parseSeeds(this.battle.inputLog.slice(0, 3)))
        throw "limit"
      }
      let winner: Side | "tie" | undefined
      const pending: Side[] = []

      for (const log of this.logs) {
        const ch = split(log)
        for (const side of SIDES) {
          const { obs } = this[side]
          for (const line of ch[side]) {
            const e = obs.read(line)

            if (e.error?.startsWith("[Invalid choice]")) {
              console.log(JSON.stringify(this.battle.inputLog))
              console.log(parseSeeds(this.battle.inputLog.slice(0, 3)))
              console.log(this.choices)
              throw e.error
            }
            if (e.winner) winner = e.winner
            if (e.pending) pending.push(side)
          }
        }
      }
      this.logs = []

      const deferred: Side[] = []

      if (winner) {
        return {
          type: "end",
          winner
        }
      }

      for (const side of pending) {
        if (this[side].agent) {
          const choice = this[side].agent.choose()
          // console.log(choice)
          this.choose(side, choice)
          this.choices.push(side, choice)
        } else {
          deferred.push(side)
        }
      }

      if (deferred.length) {
        return {
          type: "request",
          sides: deferred
        }
      }
    }
  }
}
