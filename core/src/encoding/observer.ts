import { Generation } from "@pkmn/data"
import { Fields, Observer, Weather } from "../client/observer.js"
import {
  BOOST_IDS,
  Boosts,
  HAZARDS,
  PSEUDO_WEATHER_NAMES,
  SCREENS,
  STAT_IDS,
  Stats,
  TERRAIN_NAMES,
  WEATHER_NAMES
} from "../battle.js"
import { Flags, FoeUser, MoveSet, Status, User, Volatiles } from "../client/user.js"
import { DelayedAttack, Side, SideEffects } from "../client/side.js"
import { getPotentialPresets, matchesPreset } from "../version.js"
import { Format } from "../run.js"
import { encodeDelayedAttack, encodeStats, encodeStatus, encodeVolatiles } from "./features.js"
import { INTERIM_FORMES } from "./forme.js"
import { scalePP, scaleStat } from "./norm.js"
import { inferMaxPP } from "../client/move.js"

export type FMoveSlot = {
  move: string
  x: number[]
}

type UserLookup = {
  disabled: string | undefined
  choice: string | undefined
  encore: string | undefined
  locked: string | undefined
  lastMove: string | undefined
  lastBerry: string | undefined
}

type SideLookup = {
  active: string
}

type FUser = {
  x: number[]
  lookup: UserLookup
  moveSet: FMoveSlot[]
  movePool: FMoveSlot[]
  abilities: string[]
  items: string[] | null
  types: string[]
  teraTypes: string[]
}

type FAlly = {
  x: number[]
  team: { [k: string]: FUser }
  lookup: SideLookup
}

type FFoe = {
  x: number[]
  team: { [k: string]: FUser }
  lookup: SideLookup
}

export type FObserver = {
  x: number[]
  ally: FAlly
  foe: FFoe
}

export const DECISION_MODES = ["move", "switch", "revive", "wait"]
export type DecisionMode = (typeof DECISION_MODES)[number]

export function encodeMoveSlot(moveSet: MoveSet, move?: string): FMoveSlot | undefined {
  if (!move) return undefined

  if (move in moveSet) {
    const { used, max } = moveSet[move]
    return { move, x: [scalePP(Math.max(0, max - used)), scalePP(max)] }
  }

  return {
    move,
    x: [1, 1]
  }
}

export function encodeMoveSet(moveSet: MoveSet) {
  return Object.keys(moveSet).map((k) => encodeMoveSlot(moveSet, k)!)
}

function getUserLookup({ volatiles, lastBerry, lastMove }: User) {
  return {
    disabled: volatiles["Disable"]?.move,
    choice: volatiles["Choice Locked"]?.move,
    encore: volatiles["Encore"]?.move,
    locked: volatiles["Locked Move"]?.move,
    lastMove: lastMove,
    lastBerry: lastBerry?.name
  }
}

function encodeUser({
  revealed,
  hpLeft,
  stats,
  status,
  flags,
  forme,
  volatiles,
  boosts,
  tera
}: {
  revealed: boolean
  hpLeft: number
  stats: Stats
  status?: Status
  flags: Flags
  forme: string
  volatiles: Volatiles
  boosts: Boosts
  tera: boolean
}) {
  const feats: number[] = []

  feats.push(revealed ? 1 : 0)
  feats.push(tera ? 1 : 0)
  feats.push(scaleStat("hp", hpLeft))
  feats.push(...encodeStats(stats))
  feats.push(...encodeStatus(status))

  feats.push(
    ...(["battleBond", "intrepidSword", "illusionRevealed"] as const).map((k) => (flags[k] ? 1 : 0))
  )

  feats.push(...INTERIM_FORMES.map((k) => (k === forme ? 1 : 0)))
  feats.push(...BOOST_IDS.map((id) => boosts[id] ?? 0))

  feats.push(...encodeVolatiles(volatiles))

  return feats
}

function inferStats(gen: Generation, forme: string, lvl: number): Stats {
  const { baseStats } = gen.species.get(forme)!

  const stats: any = {}
  for (const id of STAT_IDS) {
    stats[id] = gen.stats.calc(id, baseStats[id], 31, 85, lvl)
  }

  return stats
}

function encodeSide({
  mode,
  effects,
  wish,
  delayedAttack,
  teraUsed
}: {
  mode: DecisionMode
  effects: SideEffects
  wish?: number
  delayedAttack?: DelayedAttack
  teraUsed?: boolean
}) {
  const x: number[] = []

  x.push(wish ? 2 - wish : 0)
  x.push(...encodeDelayedAttack(delayedAttack))
  x.push(teraUsed ? 1 : 0)

  x.push(...DECISION_MODES.map((x) => (mode === x ? 1 : 0)))
  x.push(...HAZARDS.map((name) => effects[name]?.layers ?? 0))
  x.push(...SCREENS.map((name) => effects[name]?.turn ?? 0))

  return x
}

function getSideLookup({ active }: Side) {
  return {
    active: active.species
  }
}

function encodeBattle({ fields, weather }: { fields: Fields; weather?: Weather }) {
  return [
    ...WEATHER_NAMES.map((name) => (weather?.name === name ? 5 - weather.turn : 0)),
    ...[...TERRAIN_NAMES, ...PSEUDO_WEATHER_NAMES].map((name) => {
      const turn = fields[name]
      return turn ? 5 - turn : 0
    })
  ]
}

export function encodeObserver(format: Format, obs: Observer): FObserver {
  const { gen } = format

  const { ally, foe, fields, weather, req } = obs

  let fAlly: FAlly
  {
    const { team, delayedAttack, effects, teraUsed, wish } = ally

    let fTeam: {
      [k: string]: FUser
    } = {}

    for (const species in team) {
      const user = team[species]

      const {
        revealed,
        forme,
        hp,
        types,
        ability,
        stats,
        item,
        status,
        moveSet,
        teraType,
        boosts,
        tera,
        flags,
        volatiles
      } = user

      fTeam[species] = {
        x: encodeUser({
          revealed,
          stats: { ...stats, hp: hp[1] },
          hpLeft: hp[0],
          flags,
          volatiles,
          boosts,
          status,
          forme,
          tera
        }),
        lookup: getUserLookup(user),
        moveSet: encodeMoveSet(moveSet),
        movePool: [],
        abilities: [ability],
        items: item ? [item] : null,
        types,
        teraTypes: [teraType]
      }
    }

    fAlly = {
      x: encodeSide({
        mode: ally.isReviving ? "revive" : req.type,
        delayedAttack,
        teraUsed,
        effects,
        wish
      }),
      lookup: getSideLookup(ally),
      team: fTeam
    }
  }

  let fFoe: FFoe
  {
    const { team, delayedAttack, effects, teraUsed, wish } = foe

    let fTeam: {
      [k: string]: FUser
    } = {}

    for (const species in team) {
      const user = team[species]
      let {
        lvl,
        hp,
        item,
        ability,
        moveSet,
        types,
        status,
        teraType,
        flags,
        tera,
        boosts,
        forme,
        volatiles
      } = user

      let presets = getPotentialPresets(format, forme)
      let filtered = presets.filter((x) => matchesPreset(x, user))
      if (filtered.length) {
        presets = filtered
      } else {
        // check if its due to illusion
        filtered = getPotentialPresets(format, "Zoroark").filter((x) => matchesPreset(x, user))

        if (filtered) {
          forme = "Zoroark"
          presets = filtered
          lvl = format.patch["Zoroark"].level
        } else {
          console.warn("neither matches forme nor zoroark")
        }
      }

      const validItems = new Set<string>()
      const validAbilities = new Set<string>()
      const validMoves = new Set<string>()
      const validTeraTypes = new Set<string>()

      for (const {
        agg: { items, abilities, moves, teraTypes }
      } of presets) {
        for (const item of items) validItems.add(item)
        for (const ability of abilities) validAbilities.add(ability)
        for (const type of teraTypes) validTeraTypes.add(type)
        for (const move of moves) validMoves.add(move)
      }

      fTeam[species] = {
        x: encodeUser({
          revealed: true,
          stats: { ...inferStats(gen, forme, lvl), hp: hp[1] },
          hpLeft: hp[0],
          flags,
          volatiles,
          boosts,
          status,
          forme,
          tera
        }),
        lookup: getUserLookup(user),
        moveSet: Object.keys(moveSet).map((k) => encodeMoveSlot(moveSet, k)!),
        movePool: [...validMoves]
          .filter((move) => !(move in moveSet))
          .map((move) => {
            const pp = scalePP(inferMaxPP(gen, move))
            return {
              move,
              x: [pp, pp]
            }
          }),
        items: item === null ? null : item ? [item] : [...validItems],
        abilities: ability ? [ability] : [...validAbilities],
        types,
        teraTypes: teraType ? [teraType] : [...validTeraTypes]
      }
    }

    let mode: DecisionMode
    {
      switch (req.type) {
        case "move":
          mode = "move"
          break
        case "wait":
          mode = foe.isReviving ? "revive" : "switch"
          break
        case "switch":
          mode = "wait"
          break
      }
    }

    fFoe = {
      x: encodeSide({
        mode: mode,
        delayedAttack,
        teraUsed,
        effects,
        wish
      }),
      lookup: getSideLookup(foe),
      team: fTeam
    }
  }

  return {
    x: encodeBattle({ fields, weather }),
    ally: fAlly,
    foe: fFoe
  }
}
