import { Generation } from "@pkmn/data"
import { MOVE_CATEGORIES, STAT_IDS, Stats, TYPE_NAMES } from "../battle.js"
import { User } from "../parser/user.js"
import { Observer } from "../parser/observer.js"
import { Party, POVS } from "../parser/side.js"
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

  const stats = user.pov === "ally" ? { ...user.stats!, hp: hp[1] } : inferStats(gen, forme, lvl)
  const hpRatio = hp[0] / hp[1]

  x.push((hpRatio * stats.hp) / 600)
  x.push(...STAT_IDS.map((k) => stats[k] / 600))
  x.push(...TYPE_NAMES.map((k) => (types.includes(k) ? 1 : 0)))

  return x
}

function encodeParty({ team, effects, teraUsed }: Party) {
  let x: number[] = []
  let totHp = 6
  let totAlive = 6
  let totBoosts = 0
  let totStatus = 0
  for (const k in team) {
    const { hp, boosts, status } = team[k]
    totHp - 1 + hp[0] / hp[1]
    totBoosts += sumBoosts(boosts)
    if (status) totStatus++
    if (hp[0] === 0) totAlive -= 1
  }
  const { hazards, screens } = sumSideEffects([...Object.keys(effects)])

  x.push(totHp)
  x.push(totBoosts)
  x.push(hazards)
  x.push(screens)
  x.push(totStatus)
  x.push(totAlive)
  x.push(teraUsed ? 1 : 0)

  return x
}

export type BattleState = {
  partyEnc: number[][]
  userEnc: number[][][]
  activeIdx: number[]
  moveChoiceIdx: number[]
  moveMask: number[][]
  switchMask: number[]
}

function zeros(dims: number[]): any {
  if (dims.length === 1) return Array(dims[0]).fill(0)

  return Array(dims[0])
    .fill(null)
    .map(() => zeros(dims.slice(1)))
}

export function encodeBattle(obs: Observer) {
  const { ally, gen } = obs

  const activeIdx: number[] = zeros([2])
  const partyEnc: number[][] = zeros([2, 7])
  const userEnc: number[][][] = zeros([2, 6, 28])
  const moveChoiceIdx: number[] = zeros([4])
  const moveMask: number[][] = zeros([4, 2])
  const switchMask: number[] = zeros([6])

  for (let i = 0; i < 2; i++) {
    const pov = POVS[i]
    const party = obs[pov]
    const { team, active } = party

    partyEnc[i] = encodeParty(party)

    const users = Object.values(team)
    for (let j = 0; j < users.length; j++) {
      userEnc[i][j] = encodeUser(gen, users[j])
      if (users[j] === active) activeIdx[i] = j
    }
  }

  const { select, switches } = obs.getOption()!

  if (select) {
    const { tera } = select
    const moves = toMoves(select)

    for (let i = 0; i < moves.length; i++) {
      for (let j = 0; j < 2; j++) {
        if (j === 1 && !tera) continue
        moveMask[i][j] = 1
      }

      moveChoiceIdx[i] = gen.moves.get(moves[i])!.num
    }
  }

  const teamK = Object.keys(ally.team)
  for (let i = 0; i < 6; i++) {
    if (switches.includes(teamK[i])) switchMask[i] = 1
  }

  return {
    partyEnc,
    userEnc,
    activeIdx,
    moveChoiceIdx,
    moveMask,
    switchMask
  }
}

function toB64(x: any, type: "float" | "int"): string {
  x = x.flat(Infinity)
  if (type === "float") x = new Float32Array(x)
  else x = new Int32Array(x)
  return Buffer.from(x.buffer).toString("base64")
}

export function serializeBattle({
  partyEnc,
  userEnc,
  activeIdx,
  moveChoiceIdx,
  moveMask,
  switchMask
}: BattleState) {
  return {
    partyEnc: toB64(partyEnc, "float"),
    userEnc: toB64(userEnc, "float"),
    activeIdx: toB64(activeIdx, "int"),
    moveChoiceIdx: toB64(moveChoiceIdx, "int"),
    moveMask: toB64(moveMask, "int"),
    switchMask: toB64(switchMask, "int")
  }
}
