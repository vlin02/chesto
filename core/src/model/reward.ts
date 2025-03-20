import { Boosts, HAZARDS, SCREENS } from "../battle.js"
import { Observer } from "../parser/observer.js"
import { User } from "../parser/user.js"

const blueRunsOutFirstMatrix = [
  [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  [1.0, 0.5, 0.3333, 0.25, 0.2, 0.1667, 0.1429],
  [1.0, 0.6667, 0.5, 0.4, 0.3333, 0.2857, 0.25],
  [1.0, 0.75, 0.6, 0.5, 0.4286, 0.375, 0.3333],
  [1.0, 0.8, 0.6667, 0.5714, 0.5, 0.4444, 0.4],
  [1.0, 0.8333, 0.7143, 0.625, 0.5556, 0.5, 0.4545],
  [1.0, 0.8571, 0.75, 0.6667, 0.6, 0.5455, 0.5]
]

export function evalUser(users: User[]) {
  let t = 6 * 5
  for (const { hp, status, boosts } of users) {
    if (hp[0] === 0) {
      t -= 5
    } else {
      // const ratio = hp[0] / hp[1]
      // if (status) t -= 2
      // t -= (1 - ratio) * 5
    }

    // t += sumBoosts(boosts)
  }
  return t
}

export function evalBattle(obs: Observer) {
  let t = 0
  if (obs.winner && obs.winner !== "tie") t += obs.winner === obs.side ? 1 : -1

  const { ally, foe } = obs
  const [tAlly, tFoe] = [ally, foe].map(({ team }) => {
    let totAlive = 6
    for (const k in team) {
      if (team[k].hp[0] === 0) totAlive -= 1
    }
    return totAlive
  })

  // return tAlly - tFoe
  return t
}

export function sumBoosts(boosts: Boosts) {
  return Object.values(boosts).reduce<number>((acc, x) => acc + (x ?? 0), 0)
}

export function sumSideEffects(effects: string[]) {
  let totHazards = 0
  let totScreens = 0

  for (const effect of effects) {
    if (HAZARDS.includes(effect as any)) totHazards++
    if (SCREENS.includes(effect as any)) totScreens++
  }
  return { hazards: totHazards, screens: totScreens }
}
