import { Battle, toID, Teams, PRNGSeed, PRNG } from "@pkmn/sim"
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

interface Agent {
  choose(): Choice
}

type Seed = {
  battle: PRNGSeed
  p1: PRNGSeed
  p2: PRNGSeed
}

export function generateSeed(): Seed {
  return {
    battle: PRNG.generateSeed(),
    p1: PRNG.generateSeed(),
    p2: PRNG.generateSeed()
  }
}

export class Sim {
  battle: Battle
  p1: { obs: Observer; agent?: Agent }
  p2: { obs: Observer; agent?: Agent }
  logs: Log[]
  seed?: Seed

  constructor(seed?: Seed) {
    const gen = new Generations(Dex).get(9)
    this.seed = seed

    this.p1 = { obs: new Observer(gen) }
    this.p2 = { obs: new Observer(gen) }
    this.logs = []

    this.battle = new Battle({
      formatid: toID("gen9randombattle"),
      p1: { name: "p1" },
      p2: { name: "p2" },
      send: (...log) => {
        console.log(log)
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

    let i = 0
    while (true) {
      i += 1
      if (i > 200) {
        console.log(i, JSON.stringify(this.battle.inputLog))
        throw "limit"
      }
      let winner: Side | null | undefined
      const requested: Side[] = []

      for (const log of this.logs) {
        const ch = split(log)
        for (const side of SIDES) {
          const { obs } = this[side]
          for (const line of ch[side]) {
            const e = obs.read(line)
            if (e.error) {
              console.log(this.battle.inputLog)
              throw e.error
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

      const deferred: Side[] = []

      for (const side of requested) {
        if (this[side].agent) {
          const choice = this[side].agent.choose()
          this.choose(side, choice)
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

// console.log(i)
const sim = new Sim()
sim.p1.agent = new RandomAgent(sim.p1.obs)
sim.p2.agent = new RandomAgent(sim.p2.obs)
sim.step([])

// function replayInput(inputs: string[]) {
//   for (const input of )
// }

// const gen = new Generations(Dex).get(9)
// const obs = { p1: new Observer(gen), p2: new Observer(gen) }

// const b = new Battle({
//   formatid: toID("gen9randombattle"),
//   seed: ["sodium", "400ad5bcbe7a70be8d9354e7e8b12db6"],
//   p1: { name: "p1", seed: ["sodium", "69b78e19c6b30a8f08cd574f974f42b2"] },
//   p2: { name: "p2", seed: ["sodium", "8b1648294a2502ac31fe5de18d4173ed"] },
//   send: (...log) => {
//     console.log(log)
//     const ch = split(log as Log)
//     for (const side of SIDES) {
//       for (const line of ch[side]) {
//         obs[side].read(line)
//       }
//     }
//   }
// })
// b.sendUpdates()
// console.log(obs.p1.getOption())
// console.log(obs.p2.getOption())

// const inputs = [
//   '>p1 move earthquake terastallize',
//   '>p2 move thunderbolt terastallize',
//   '>p1 move spikes',
//   '>p2 switch 6',
//   '>p1 move earthquake',
//   '>p2 switch 4',
//   '>p1 switch 5',
//   '>p2 move energyball',
//   '>p1 switch 5',
//   '>p2 move energyball',
//   '>p1 move spikes',
//   '>p2 move icebeam',
//   '>p1 switch 4',
//   '>p2 switch 4',
//   '>p1 switch 5',
//   '>p2 switch 2',
//   '>p1 switch 6',
//   '>p2 switch 4',
//   '>p1 switch 3',
//   '>p2 move energyball',
//   '>p1 switch 2',
//   '>p2 move icebeam',
//   '>p1 switch 5',
//   '>p2 switch 6',
//   '>p1 switch 3',
//   '>p2 switch 2',
//   '>p1 switch 2',
//   '>p2 move psyblade',
//   '>p1 switch 4',
//   '>p2 switch 5',
//   '>p1 switch 2',
//   '>p2 switch 5',
//   '>p1 switch 4',
//   '>p2 switch 5',
//   '>p1 switch 6',
//   '>p2 move rapidspin',
//   '>p1 switch 6',
//   '>p2 move bulkup'
// ]

// for (const input of inputs) {
//   const side = input.slice(1, 3) as Side
//   const msg = input.slice(4)
//   console.log(side, msg)
//   b.choose(side, msg)
//   b.sendUpdates()
// }

// console.log(obs.p1.req)
// console.log(obs.p2.req)

// function replayInput() {}
