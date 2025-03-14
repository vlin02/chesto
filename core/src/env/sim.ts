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

// const v = [
//   '>start {"formatid":"gen9randombattle","seed":["sodium","15ec3dbaa1bebb8d624499ac4102c905"]}',
//   '>player p1 {"name":"p1","seed":["sodium","c42751cebdf17071c6f2577ea7191fb8"]}',
//   '>player p2 {"name":"p2","seed":["sodium","56eb526ea12bbea19fd780d08e96c760"]}'
// ]

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
      const requested: Side[] = []

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
            if (e.req && e.req?.type !== "wait") requested.push(side)
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

      for (const side of requested) {
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

for (let i = 0; i < 10000; i++) {
  console.log(i)
  const sim = new Sim()
  sim.p1.agent = new RandomAgent(sim.p1.obs)
  sim.p2.agent = new RandomAgent(sim.p2.obs)
  sim.step([])
}

// function replay(inputs: string[]) {
//   const seed = parseSeeds(inputs.slice(0, 3))
//   const gen = new Generations(Dex).get(9)
//   const obs = { p1: new Observer(gen), p2: new Observer(gen) }

//   const b = new Battle({
//     formatid: toID("gen9randombattle"),
//     seed: ["sodium", seed[0]],
//     p1: { name: "p1", seed: ["sodium", seed[1]] },
//     p2: { name: "p2", seed: ["sodium", seed[2]] },
//     send: (...log) => {
//       console.log(log)
//       const ch = split(log as Log)
//       for (const side of SIDES) {
//         for (const line of ch[side]) {
//           obs[side].read(line)
//         }
//       }
//     }
//   })
//   b.sendUpdates()

//   for (const input of inputs.slice(3)) {
//     const side = input.slice(1, 3) as Side
//     const msg = input.slice(4)
//     console.log(side, msg)
//     b.choose(side, msg)
//     b.sendUpdates()
//   }
//   console.log(obs.p1.getOption(), obs.p2.getOption())
// }

// replay([
//   '>start {"formatid":"gen9randombattle","seed":["sodium","9d005ac43ecad79e1aaf157292464d6c"]}',
//   '>player p1 {"name":"p1","seed":["sodium","9062523199be9d37697b66c1ae8162bc"]}',
//   '>player p2 {"name":"p2","seed":["sodium","4dbfed1464e1e10f053cda58e117a301"]}',
//   ">p1 switch 5",
//   ">p2 switch 3",
//   ">p1 switch 2",
//   ">p2 switch 4",
//   ">p1 switch 5",
//   ">p2 switch 2",
//   ">p1 switch 6",
//   ">p2 switch 4",
//   ">p1 switch 2",
//   ">p2 move roost",
//   ">p1 switch 3",
//   ">p2 switch 6",
//   ">p1 switch 6",
//   ">p2 move psyshock"
// ])
