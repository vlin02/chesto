import { Boosts, HAZARDS, SCREENS } from "../battle.js"
import { Observer } from "../parser/observer.js"
import { User } from "../parser/user.js"

export function evalUser(users: User[]) {
  let t = 6 * 8
  for (const { hp, status, boosts } of users) {
    const ratio = hp[0] / hp[1]
    if (ratio === 0) t -= 2
    if (status) t -= 1
    t -= (1 - ratio) * 5

    t += sumBoosts(boosts)
  }
  return t
}

export function evalBattle(obs: Observer) {
  if (!obs.ally) return 0
  let t = 0

  if (obs.winner) t += obs.winner === obs.side ? 20 : -20

  const { ally, foe } = obs
  const [tAlly, tFoe] = [ally, foe].map(({ team, teraUsed }) => {
    return evalUser(Object.values(team)) + (teraUsed ? 0 : 5)
  })

  t += tAlly - tFoe

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
