import { Generation } from "@pkmn/data"
import { MOVE_CATEGORIES, STAT_IDS, Stats } from "../battle.js"
import { User } from "../parser/user.js"
import { Observer } from "../parser/observer.js"
import { toMoves } from "../parser/action.js"
import { Party } from "../parser/side.js"

function inferStats(gen: Generation, forme: string, lvl: number): Stats {
  const { baseStats } = gen.species.get(forme)!

  const stats: any = {}
  for (const id of STAT_IDS) {
    stats[id] = gen.stats.calc(id, baseStats[id], 31, 85, lvl)
  }

  return stats
}

export type MoveF = {
  x: number[]
  type: string
}

export function encodeMove(gen: Generation, move: string) {
  const { basePower, priority, accuracy, type, category } = gen.moves.get(move)!
  let x: number[] = []

  x.push(basePower / 250)
  x.push(...MOVE_CATEGORIES.map((k) => (category === k ? 1 : 0)))
  x.push(priority)
  x.push(accuracy === true ? 1 : accuracy)

  return {
    x,
    type
  }
}

type UserF = {
  x: number[]
  types: string[]
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

type PartyF = {
  team: { [k: string]: UserF }
  active: string
}

function encodeParty(gen: Generation, { active, team }: Party) {
  const f: PartyF = { team: {}, active: active.species }
  for (const k in team) {
    f.team[k] = encodeUser(gen, team[k])
  }

  return f
}

export type BattleState = {
  ally: PartyF
  foe: PartyF
  option: {
    tera: boolean | undefined
    moves: string[]
    switches: string[]
  }
}

export function encodeState(obs: Observer) {
  const { ally, foe, gen } = obs

  const { select, switches } = obs.getOption()

  return {
    ally: encodeParty(gen, ally),
    foe: encodeParty(gen, foe),
    option: {
      tera: !!select?.tera,
      moves: select ? toMoves(select) : [],
      switches
    }
  }
}
