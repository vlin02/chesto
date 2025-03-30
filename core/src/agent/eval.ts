import { Battle, Teams } from "@pkmn/sim"
import { ChoiceInput, Log, parseInput, split } from "../log.js"
import { Side, SIDES, Winner } from "../battle.js"
import { Observer } from "../parser/observer.js"
import { Generations } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
import { TeamGenerators } from "@pkmn/randoms"
import { BattleSeed, startBattle } from "../sim.js"
import { encode, tagBattle } from "../model/transports/heuristic.js"

Teams.setGeneratorFactory(TeamGenerators)
const gen = new Generations(Dex).get(9)

type Transition = {
  pending: Side[]
  logs: Log[]
}

// function parseSeed(lines: string[]): BattleSeed {
//   const all = []
//   for (let i = 0; i < 3; i++) {
//     all.push(JSON.parse(lines[i].slice(spaced(lines[i], 0, i === 0 ? 1 : 2).i)).seed as PRNGSeed)
//   }

//   const [battle, p1, p2] = all
//   return { battle, p1, p2 }
// }

class Trial {
  logs: Log[]
  p1: Observer
  p2: Observer
  battle: Battle
  winner: Winner | null

  constructor(seed?: BattleSeed) {
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


function replay(inputs: string[]) {
  const seed: BattleSeed = {} as any
  for (const line of inputs.slice(0, 3)) {
    const input = parseInput(line)
    if (input.type === "start") seed.battle = input.seed
    if (input.type === "player") seed[input.side] = input.seed
  }

  const trial = new Trial(seed)

  const choices = inputs.slice(3).map((line) => {
    const { side, choice } = parseInput(line) as ChoiceInput
    return [side, choice] as const
  })

  return [trial, choices] as const
}

// function r1() {
//   const trial = new Trial()

//   let pending: Side[] = []
//   let updates: any[] = [[[], trial.snapshot()]]
//   function update(trn: Transition) {
//     updates.push([trn.logs, trial.snapshot()])
//     pending.push(...trn.pending)
//   }

//   update(trial.transition())
//   while (!trial.winner) {
//     const side = pending.shift()!
//     trial.step(side, trial[side].toInput(chooseHeuristic(trial[side])))
//     update(trial.transition())
//   }

//   for (let i = 1; i < updates.length; i++) {
//     const [_, prev] = updates[i - 1]
//     const [logs, curr] = updates[i]

//     for (const log of logs) {
//       if (log[0] === "update") {
//         console.log(log[1])
//       }
//     }
//     console.log(diffString(prev.p2, curr.p2))
//   }
// }

function updatesOnly(logs: Log[]) {
  return logs.map((log) => (log[0] === "update" ? log[1] : [])).flat()
}
function r2() {
  const [trial, choices] = replay([
    '>start {"formatid":"gen9randombattle","seed":["sodium","3a95e60f380c22463535d86fb879dd6a"]}',
    '>player p1 {"name":"p1","seed":["sodium","7af567ef2efd85d82f78ccf6b7ec336c"]}',
    '>player p2 {"name":"p2","seed":["sodium","c010f4b1a2ac9baa31c676a6b439084e"]}',
    ">p1 move dragondance",
    ">p2 move earthquake terastallize",
    ">p1 move heatcrash",
    ">p2 switch 2",
    ">p1 move outrage terastallize",
    ">p2 switch 2",
    ">p1 move outrage",
    ">p2 move slackoff",
    ">p2 switch 2",
    ">p1 move outrage",
    ">p2 switch 6",
    ">p1 move heatcrash",
    ">p2 move dynamaxcannon",
    ">p1 switch 4",
    ">p1 move flareblitz",
    ">p2 switch 6",
    ">p2 switch 4",
    ">p1 move flareblitz",
    ">p2 move icebeam",
    ">p1 move flareblitz",
    ">p2 switch 5",
    ">p1 move flareblitz",
    ">p2 move shadowclaw",
    ">p1 move flareblitz",
    ">p2 move shadowclaw",
    ">p1 switch 5",
    ">p2 switch 5",
    ">p1 move liquidation",
    ">p2 move leafstorm",
    ">p1 move recover",
    ">p2 switch 3",
    ">p1 move recover",
    ">p2 move focusblast",
    ">p1 move liquidation",
    ">p2 move focusblast",
    ">p1 move liquidation",
    ">p2 switch 6",
    ">p1 move liquidation",
    ">p2 move dynamaxcannon",
    ">p1 move liquidation",
    ">p2 move recover",
    ">p1 move liquidation",
    ">p2 switch 6",
    ">p1 move recover",
    ">p2 move sludgebomb",
    ">p1 move liquidation",
    ">p2 switch 6",
    ">p1 move liquidation",
    ">p2 move dynamaxcannon",
    ">p1 move recover",
    ">p2 move flamethrower",
    ">p1 move recover",
    ">p2 move dynamaxcannon",
    ">p1 move liquidation",
    ">p2 switch 3",
    ">p1 move liquidation",
    ">p2 switch 3",
    ">p1 move recover",
    ">p2 move flamethrower",
    ">p1 move toxic",
    ">p2 move toxic",
    ">p1 move recover",
    ">p2 move flamethrower",
    ">p1 move recover",
    ">p2 move dynamaxcannon",
    ">p1 move toxic",
    ">p2 move flamethrower",
    ">p1 move toxic",
    ">p2 switch 6",
    ">p1 move toxic",
    ">p2 move focusblast",
    ">p1 move toxic",
    ">p2 move focusblast",
    ">p2 switch 6",
    ">p1 move toxic",
    ">p2 move recover",
    ">p1 move toxic",
    ">p2 move recover",
    ">p1 move toxic",
    ">p2 move toxic",
    ">p1 move toxic",
    ">p2 move dynamaxcannon",
    ">p1 move toxic",
    ">p2 move flamethrower",
    ">p1 move toxic"
  ])

  let snap = {}
  let i = 0
  for (const choice of [null, ...choices]) {
    if (choice) trial.battle.choose(...choice)
    const trn = trial.transition()

    console.log("trn", ++i)
    if (i > 69) {
      console.dir(trial.p1.snapshot(), { depth: null, maxArrayLength: null })
      console.log(trial.p1.getOption())
      const b = encode(trial.p1)
      const { userEnc, ...rest } = b
      console.log(rest)
      console.dir(tagBattle(b), {depth: null, maxArrayLength: null})
      console.log(choice)
      console.log(updatesOnly(trn.logs))
      // for (const log of trn.logs) {
      //   console.log(log)

      // }

      console.log(trial.p1.ally.active.moveSet)

      const _snap = trial.p1.snapshot()
      // if (choice) console.log(diffString(snap, _snap))
      snap = _snap

      console.log()
    }
  }
}

