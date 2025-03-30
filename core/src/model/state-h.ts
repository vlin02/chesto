import { Generation, Move } from "@pkmn/data"
import { Observer } from "../parser/observer.js"
import { User } from "../parser/user.js"
import {
  Boosts,
  HAZARDS,
  MOD_STAT_IDS,
  MOVE_CATEGORIES,
  SCREENS,
  STAT_IDS,
  TYPE_NAMES
} from "../battle.js"
import { inferStats, zeros } from "./state.js"
import { POVS, Team } from "../parser/side.js"
import { toMoves } from "../parser/option.js"
import { toArrayBuffer, Transport } from "./transport.js"

function netBoosts(boosts: Boosts) {
  let t = 0
  for (const k in boosts) {
    t += boosts[k]!
  }
  return t
}

const ENTRY_HAZARDS = ["Spikes", "Stealth Rock", "Sticky Web", "Toxic Spikes"]
const ANTI_HAZARDS = ["Rapid Spin", "Defog"]

export function encodeMove({ basePower, accuracy, multihit, category, type, self, name }: Move) {
  const x: number[] = []

  x.push(basePower / 100)
  x.push((accuracy === true ? 100 : accuracy) / 100)
  x.push(netBoosts(self?.boosts ?? {}))
  MOVE_CATEGORIES.forEach((cat) => x.push(category === cat ? 1 : 0))

  x.push(...(Array.isArray(multihit) ? multihit : [multihit ?? 1, multihit ?? 1]))

  x.push(ENTRY_HAZARDS.some((k) => name === k) ? 1 : 0)
  x.push(ANTI_HAZARDS.some((k) => name === k) ? 1 : 0)
  x.push(...TYPE_NAMES.map((k) => (type === k ? 1 : 0)))

  return x
}

export function tagBattle({ userEnc }: BattleF) {
  const tags = [
    "hpLeft",
    ...TYPE_NAMES.map((k) => `off-${k}`),
    ...TYPE_NAMES.map((k) => `def-${k}`),
    ...STAT_IDS.map((k) => `stat-${k}`),
    ...MOD_STAT_IDS.map((k) => `boost-${k}`)
  ]

  const userTagged = zeros([2, 6])
  for (let t = 0; t < 2; t++) {
    for (let n = 0; n < 6; n++) {
      const x = userEnc[t][n]
      const tagged: [string, number][] = []
      for (let i = 0; i < x.length; i++) {
        tagged.push([tags[i], x[i]])
      }
      userTagged[t][n] = tagged
    }
  }

  return userTagged
}

export function encodeUser(
  gen: Generation,
  { offensiveTyping, defensiveTyping, hp, stats, boosts, forme, lvl }: User
) {
  const x: number[] = []

  const baseStats = stats ? { ...stats, hp: hp[1] } : inferStats(gen, forme, lvl)
  x.push((baseStats.hp * hp[0]) / hp[1] / 100)
  TYPE_NAMES.forEach((t) => x.push(t in offensiveTyping ? 1 : 0))
  TYPE_NAMES.forEach((t) => x.push(defensiveTyping.includes(t) ? 1 : 0))

  STAT_IDS.forEach((stat) => x.push(baseStats[stat] / 100))

  MOD_STAT_IDS.forEach((stat) => x.push(boosts[stat] ?? 0))

  return x
}

export function encodeTeam(gen: Generation, { team, effects, teraUsed }: Team) {
  const x: number[] = []

  let nAlive = 6
  for (const k in team) {
    if (team[k].hp[0] === 0) nAlive -= 1
  }
  x.push(nAlive)
  x.push(teraUsed ? 1 : 0)
  x.push(...[...HAZARDS, ...SCREENS].map((k) => (k in effects ? 1 : 0)))

  return x
}

export type BattleF = {
  activeIdx: any
  moveMask: any
  switchMask: any
  moveChoiceIdx: any
  teamEnc: any
  userEnc: any
}

export function encodeBattle(obs: Observer) {
  const { gen, ally } = obs

  const { select, switches } = obs.getOption()!

  const activeIdx = zeros([2])
  const moveMask = zeros([4, 2])
  const switchMask = zeros([6])
  const moveChoiceIdx = zeros([4])
  const teamEnc = zeros([2, 9])
  const userEnc = zeros([7, 52])

  for (let i = 0; i < 2; i++) {
    const pov = POVS[i]
    const party = obs[pov]
    teamEnc[i] = encodeTeam(gen, party)
  }

  for (let i = 0; i < 6; i++) {
    userEnc[i] = encodeUser(gen, obs.ally.slots[i])
  }
  userEnc[6] = encodeUser(gen, obs.foe.active)

  if (select) {
    const { tera } = select
    const moves = toMoves(select)

    for (let i = 0; i < moves.length; i++) {
      for (let j = 0; j < 2; j++) {
        if (j === 1 && !tera) continue
        moveMask[i][j] = 1
      }

      moveChoiceIdx[i] = moves[i] === "Recharge" ? 0 : gen.moves.get(moves[i])!.num
    }
  }

  {
    const species = ally.slots.map((x) => x.species)
    for (let i = 0; i < 6; i++) {
      if (!switches.includes(species[i])) continue
      switchMask[i] = 1
    }
  }

  return {
    activeIdx,
    moveMask,
    switchMask,
    moveChoiceIdx,
    teamEnc,
    userEnc
  }
}

export const transportH: Transport<BattleF> = {
  encodeMove,
  encodeBattle,
  packBattle: ({ teamEnc, userEnc, activeIdx, moveChoiceIdx, moveMask, switchMask }: BattleF) => {
    return {
      teamEnc: toArrayBuffer(teamEnc, "float"),
      userEnc: toArrayBuffer(userEnc, "float"),
      activeIdx: toArrayBuffer(activeIdx, "int"),
      moveChoiceIdx: toArrayBuffer(moveChoiceIdx, "int"),
      moveMask: toArrayBuffer(moveMask, "int"),
      switchMask: toArrayBuffer(switchMask, "int")
    }
  },
  decodeChoice: (obs: Observer, id: number) => {
    const opt = obs.getOption()!
    if (id < 8) {
      const j = id % 2
      const i = (id - j) / 2
      return {
        type: "move",
        move: toMoves(opt.select!)[i],
        tera: j === 1
      }
    }

    id -= 8
    return { type: "switch", species: obs.ally.slots[id].species }
  }
}
