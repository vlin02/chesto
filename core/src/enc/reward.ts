import { Observer } from "../parser/observer.js"
import { User } from "../parser/user.js"

export function calcUserValue(users: User[]) {
  let t = 6 * 8
  for (const { hp, status, boosts } of users) {
    const ratio = hp[0] / hp[1]
    if (ratio === 0) t -= 2
    if (status) t -= 1
    t -= (1 - ratio) * 5

    t +=
      2 *
      Object.values(boosts)
        .flat()
        .reduce<number>((t, n) => t + (n ?? 0), 0)
  }
  return t
}

export function calcBattleValue(obs: Observer) {
  let t = 0

  if (obs.winner) t += obs.winner === obs.side ? 20 : -20

  const { ally, foe } = obs
  const [tAlly, tFoe] = [ally, foe].map(({ team, teraUsed }) => {
    return calcUserValue(Object.values(team)) + (teraUsed ? 0 : 5)
  })

  t += tAlly - tFoe

  return t
}
