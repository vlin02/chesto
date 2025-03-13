import { Generation } from "@pkmn/data"
import { MOVE_CATEGORIES, STAT_IDS, Stats } from "../battle.js"
import { User } from "../parser/user.js"
import { Observer } from "../parser/observer.js"

function inferStats(gen: Generation, forme: string, lvl: number): Stats {
  const { baseStats } = gen.species.get(forme)!

  const stats: any = {}
  for (const id of STAT_IDS) {
    stats[id] = gen.stats.calc(id, baseStats[id], 31, 85, lvl)
  }

  return stats
}

function encodeMove(gen: Generation, move: string) {
  const { basePower, priority, accuracy, type, category } = gen.moves.get(move)!
  let x: number[] = []

  x.push(basePower / 250)
  x.push(...MOVE_CATEGORIES.map(k => category === k ? 1 : 0))
  x.push(priority)
  x.push(accuracy === true ? 1 : accuracy)

  return {
    x,
    type
  }
}

function encodeUser(gen: Generation, user: User) {
  const { revealed, hp, lvl, types, forme } = user
  let x: number[] = []

  x.push(revealed ? 1 : 0)

  const stats = user.stats ? { ...user.stats, hp: hp[1] } : inferStats(gen, forme, lvl)
  const hpRatio = hp[0] / hp[1]

  x.push((hpRatio * stats.hp) / 600)
  x.push(...STAT_IDS.map((k) => stats[k] / 600))

  return {
    x,
    types
  }
}

function encode(obs: Observer) {
  
}
