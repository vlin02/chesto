import { Battle, toID, Teams } from "@pkmn/sim"
import { Log, split } from "../log.js"
import { Observer } from "../parser/observer.js"
import { Generations } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
import { Side, SIDES } from "../battle.js"
import { Action, Choice, formatChoice } from "../parser/action.js"

import { TeamGenerators } from "@pkmn/randoms"
import { RandomAgent } from "../arena/agents.js"

Teams.setGeneratorFactory(TeamGenerators)

export type Status =
  | {
      type: "end"
      winner: Side | null
    }
  | {
      type: "request"
      sides: Side[]
    }

export class Sim {
  battle: Battle
  p1: { obs: Observer; agent: RandomAgent }
  p2: { obs: Observer; agent: RandomAgent }
  logs: Log[]
  fixed: Side[]

  constructor(fixed: Side[]) {
    this.fixed = fixed

    const gen = new Generations(Dex).get(9)

    const initSide = () => {
      const obs = new Observer(gen)
      const agent = new RandomAgent(obs)
      return { obs, agent }
    }

    this.p1 = initSide()
    this.p2 = initSide()
    this.logs = []

    this.battle = new Battle({
      formatid: toID("gen9randombattle"),
      p1: {},
      p2: {},
      send: (...log) => {
        this.logs.push(log as Log)
      }
    })
    this.battle.sendUpdates()
  }

  choose(side: Side, choice: Choice) {
    this.battle.choose(side, formatChoice(choice))
    this.battle.sendUpdates()
  }

  step(actions: Action[]): Status {
    actions.forEach(({ side, choice }) => this.choose(side, choice))

    while (true) {
      let winner: Side | null | undefined
      const requests: Side[] = []

      for (const log of this.logs) {
        const ch = split(log)
        for (const side of SIDES) {
          const { obs } = this[side]
          for (const line of ch[side]) {
            switch (obs.read(line)) {
              case "end":
                winner = obs.winner!
                break
              case "request":
                requests.push(side)
                break
            }
          }
        }
      }
      this.logs = []

      if (winner !== undefined) {
        return {
          type: "end",
          winner
        }
      }

      const pending: Side[] = []

      for (const side of requests) {
        if (this.fixed.includes(side)) {
          const choice = this[side].agent.choose()
          this.choose(side, choice)
        } else {
          pending.push(side)
        }
      }

      if (pending.length) {
        return {
          type: "request",
          sides: pending
        }
      }
    }
  }
}
