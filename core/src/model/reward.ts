import { Boosts, HAZARDS, SCREENS } from "../battle.js"
import { Observer } from "../parser/observer.js"

export function evalBattle(obs: Observer) {
  let t = 0

  const { ally, foe } = obs
  const [tAlly, tFoe] = [ally, foe].map(({ team }) => {
    let tothp = 6
    for (const k in team) {
      const { hp } = team[k]
      tothp += hp[0] / hp[1] - 1
    }
    return tothp
  })
  if (obs.winner && obs.winner !== "tie") return tAlly - tFoe
  return 0
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
