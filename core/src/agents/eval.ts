import { Battle, Teams, toID } from "@pkmn/sim"
import { Log, split } from "../log.js"
import { Side, SIDES, Winner } from "../battle.js"
import { Observer } from "../parser/observer.js"
import { Generations } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
import { Pool } from "undici"
import { TeamGenerators } from "@pkmn/randoms"
import { Choice } from "../parser/option.js"

Teams.setGeneratorFactory(TeamGenerators)
const gen = new Generations(Dex).get(9)

type Seed = {
  battle: any
  p1: any
  p2: any
}

type Trial = {
  logs: Log[]
  fullLog: any[]
  results: any[]
  p1: { obs: Observer; snapshot: any }
  p2: { obs: Observer; snapshot: any }
  battle: Battle
  winner: Winner | null
}

function start(seed?: Seed) {
  const trial: Trial = {
    logs: [],
    winner: null,
    fullLog: [],
    results: [],
    battle: undefined as any,
    p1: { obs: new Observer(gen), snapshot: {} },
    p2: { obs: new Observer(gen), snapshot: {} }
  }

  trial.battle = new Battle({
    seed: seed?.battle,
    formatid: toID("gen9randombattle"),
    p1: { name: "p1", seed: seed?.p1 },
    p2: { name: "p2", seed: seed?.p2 },
    send: (...log) => {
      trial.logs.push(log as Log)
      trial.fullLog.push(log)
    }
  })
  trial.battle.sendUpdates()
  return trial
}

type Step = {
  action: string
  logs: Log[]
  snapshots: { [k in Side]: any }
  probs?: number[]
  value?: number
}

function choose(trial: Trial, side: Side, choice: Choice) {}

const envs = [...new Array(1)].map(() => start())
const pool = new Pool("http://172.31.57.228:8000")

while (true) {
  let all: Trial[] = []

  for (const env of envs) {
    if (env.winner) continue
    const pending: Side[] = []

    for (const log of env.logs) {
      const ch = split(log)

      for (const side of SIDES) {
        const { obs } = env[side]
        for (const line of ch[side]) {
          const e = obs.read(line)

          if (e.error?.startsWith("[Invalid choice]")) {
            throw e.error
          }
          if (e.winner) env.winner = e.winner
          if (e.pending) pending.push(side)
        }
      }
      for (const side of SIDES) {
        if (env[side].obs.ready() && env[side].obs.turn === 1) {
          console.log(env[side].obs.snapshot())
        }
        env[side].snapshot
      }
    }
    env.logs = []

    for (const side of pending) {
      if (side === "p1") {
        choose(env, side, simpleHeuristic(env.p1.obs))
      } else {
        choose(env, side, simpleHeuristic(env.p2.obs))
        // all.push(env)
      }
    }
  }
  if (envs.every((x) => x.winner)) break

  // if (all.length) {
  //   const results = (await predict(
  //     pool,
  //     all.map((env) => {
  //       return env.p2
  //     }),
  //     "3/0-1742688539.pt"
  //   )) as { action_id: number; probs: number[] }[]

  //   for (let i = 0; i < all.length; i++) {
  //     const env = all[i]
  //     choose(env, "p2", resolveChoice(env.p2, results[i]["action_id"]))
  //     env.results.push(results[i])
  //   }
  // }
}

// const cnt = { p1: 0, p2: 0, tie: 0 }
// let hi = 0
// let best: Env | null = null
// for (const env of envs) {
//   const { winner, p1 } = env
//   cnt[winner!] += 1
//   const t = Object.values(p1.ally.team).reduce((acc, x) => acc + x.hp[0] / x.hp[1], 0)
//   // if (winner === "p1" && t > hi) {
//   hi = t
//   best = env
//   // }
// }

// console.log(cnt)

// if (best) {
//   const { battle, results } = best!
//   console.log(hi)
//   console.log(JSON.stringify({ input: battle.inputLog, results }))
// }
