import { Generation } from "@pkmn/data"
import { MOVE_CATEGORIES, Side, STAT_IDS, Stats, TYPE_NAMES } from "../battle.js"
import { User } from "../parser/user.js"
import { Observer } from "../parser/observer.js"
import { Party } from "../parser/side.js"
import { toMoves } from "../parser/option.js"
import { sumBoosts, sumSideEffects } from "./reward.js"

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
  x.push((accuracy === true ? 100 : accuracy) / 100)
  x.push(priority)
  x.push(...TYPE_NAMES.map((k) => (type === k ? 1 : 0)))

  return x
}
type UserF = number[]

function encodeUser(gen: Generation, user: User) {
  const { revealed, hp, lvl, types, forme } = user
  let x: number[] = []

  x.push(revealed ? 1 : 0)

  const stats = user.stats ? { ...user.stats, hp: hp[1] } : inferStats(gen, forme, lvl)
  const hpRatio = hp[0] / hp[1]

  x.push((hpRatio * stats.hp) / 600)
  x.push(...STAT_IDS.map((k) => stats[k] / 600))
  x.push(...TYPE_NAMES.map((k) => (types.includes(k) ? 1 : 0)))

  return x
}

type PartyF = {
  team: { [k: string]: UserF }
  active: string
  x: number[]
}

function encodeParty(gen: Generation, { active, team, effects, teraUsed }: Party) {
  const f: PartyF = { team: {}, active: active.species, x: [] }
  let totHp = 6
  let totAlive = 6
  let totBoosts = 0
  let totStatus = 0
  for (const k in team) {
    f.team[k] = encodeUser(gen, team[k])
    const { hp, boosts, status } = team[k]
    totHp - 1 + hp[0] / hp[1]
    totBoosts += sumBoosts(boosts)
    if (status) totStatus++
    if (hp[0] === 0) totAlive -= 1
  }
  const { hazards, screens } = sumSideEffects([...Object.keys(effects)])

  f.x.push(totHp)
  f.x.push(totBoosts)
  f.x.push(hazards)
  f.x.push(screens)
  f.x.push(totStatus)
  f.x.push(totAlive)
  f.x.push(teraUsed ? 1 : 0)

  return f
}

export type OptionF = {
  tera: boolean | undefined
  moves: string[]
  switches: string[]
}

export function encodeOption(obs: Observer): OptionF {
  const opt = obs.getOption()!
  const { select, switches } = opt

  return {
    tera: !!select?.tera,
    moves: select ? toMoves(select) : [],
    switches
  }
}

export type BattleF = {
  side: Side
  ally: PartyF
  foe: PartyF
  option: OptionF
}

export function encodeBattle(obs: Observer): BattleF {
  const { ally, foe, gen, side } = obs

  return {
    side,
    ally: encodeParty(gen, ally),
    foe: encodeParty(gen, foe),
    option: encodeOption(obs)
  }
}
