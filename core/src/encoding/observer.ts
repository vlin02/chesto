import { Stats as Calc, Generation } from "@pkmn/data"
import { Fields, Observer, Weather } from "../client/observer.js"
import {
  BOOST_IDS,
  Boosts,
  HAZARDS,
  SCREENS,
  STAT_IDS,
  Stats,
  TERRAIN_NAMES,
  WEATHER_NAMES
} from "../battle.js"
import { Flags, FoeUser, MoveSet, Status, User, Volatiles } from "../client/user.js"
import { DelayedAttack, Side, SideEffects } from "../client/side.js"
import { getPotentialPresets, matchesPreset } from "../version.js"
import { Format } from "../format.js"
import { encodeDelayedAttack, encodeStats, encodeStatus, encodeVolatiles } from "./features.js"
import { INTERIM_FORMES } from "./forme.js"
import { scalePP, scaleStat } from "./norm.js"
import { inferMaxPP } from "../client/move.js"

export type XMoveSlot = {
  move?: string
  f: number[]
}

type UserLookup = {
  disabled: XMoveSlot | undefined
  choice: XMoveSlot | undefined
  encore: XMoveSlot | undefined
  locked: XMoveSlot | undefined
  lastMove: XMoveSlot | undefined
  lastBerry: string | undefined
}

type SideLookup = {
  active: string
  tera: string | undefined
}

type XAllyUser = {
  f: number[]
  lookup: UserLookup
  moveSet: XMoveSlot[]
  item: string | null
  ability: string | null
  types: string[]
  teraType: string
}

type XAlly = {
  f: number[]
  team: { [k: string]: XAllyUser }
  lookup: SideLookup
}

type XFoeUser = {
  f: number[]
  lookup: UserLookup
  moveSet: XMoveSlot[]
  movePool: XMoveSlot[]
  abilities: string[]
  items: string[]
  types: string[]
  teraTypes: string[]
}

type XFoe = {
  f: number[]
  team: { [k: string]: XFoeUser }
  lookup: SideLookup
}

export type XObserver = {
  f: number[]
  ally: XAlly
  foe: XFoe
}

export const DECISION_MODES = ["move", "switch", "revive", "wait"]
export type DecisionMode = (typeof DECISION_MODES)[number]

export function encodeMoveSlot(moveSet: MoveSet, move?: string): XMoveSlot | undefined {
  if (!move) return undefined

  if (move in moveSet) {
    const { used, max } = moveSet[move]
    return { move, f: [scalePP(Math.max(0, max - used)), scalePP(max)] }
  }

  return {
    move: move === "Struggle" ? move : undefined,
    f: [0, 0]
  }
}

export function encodeMoveSet(moveSet: MoveSet) {
  return Object.keys(moveSet).map((k) => encodeMoveSlot(moveSet, k)!)
}

function encodeUserLookup({ moveSet, volatiles, lastBerry, lastMove }: User) {
  return {
    disabled: encodeMoveSlot(moveSet, volatiles["Disable"]?.move),
    choice: encodeMoveSlot(moveSet, volatiles["Choice Locked"]?.move),
    encore: encodeMoveSlot(moveSet, volatiles["Encore"]?.move),
    locked: encodeMoveSlot(moveSet, volatiles["Locked Move"]?.move),
    lastMove: encodeMoveSlot(moveSet, lastMove),
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
  boosts
}: {
  revealed: boolean
  hpLeft: number
  stats: Stats
  status?: Status
  flags: Flags
  forme: string
  volatiles: Volatiles
  boosts: Boosts
}) {
  const feats: number[] = []

  feats.push(revealed ? 1 : 0)
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

function inferStats(gen: Generation, { forme, lvl }: FoeUser): Stats {
  const calc = new Calc(gen.dex)
  const { baseStats } = gen.species.get(forme)!

  const stats: any = {}
  for (const id of STAT_IDS) {
    stats[id] = calc.calc(id, baseStats[id], 31, 85, lvl)
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
  const f: number[] = []

  f.push(wish ? 2 - wish : 0)
  f.push(...encodeDelayedAttack(delayedAttack))
  f.push(teraUsed ? 1 : 0)

  f.push(...DECISION_MODES.map((x) => (mode === x ? 1 : 0)))
  f.push(...HAZARDS.map((name) => effects[name]?.layers ?? 0))
  f.push(...SCREENS.map((name) => effects[name]?.turn ?? 0))

  return f
}

function encodeSideLookup({ active, team }: Side) {
  return {
    active: active.species,
    tera: Object.keys(team).find((k) => team[k].tera)
  }
}

function encodeBattle({ fields, weather }: { fields: Fields; weather?: Weather }) {
  return [
    ...WEATHER_NAMES.map((name) => (weather?.name === name ? 5 - weather.turn : 0)),
    ...[...TERRAIN_NAMES, "Trick Room"].map((name) => {
      const turn = fields[name]
      return turn ? 5 - turn : 0
    })
  ]
}

export function encodeObserver(format: Format, obs: Observer): XObserver {
  const { gen } = format

  const { ally, foe, fields, weather, req } = obs

  let xAlly: XAlly
  {
    const { team, delayedAttack, effects, teraUsed, wish } = ally

    let xTeam: {
      [k: string]: XAllyUser
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
        flags,
        volatiles
      } = user

      xTeam[species] = {
        f: encodeUser({
          revealed,
          stats: { ...stats, hp: hp[1] },
          hpLeft: hp[0],
          flags,
          volatiles,
          boosts,
          status,
          forme
        }),
        lookup: encodeUserLookup(user),
        moveSet: encodeMoveSet(moveSet),
        ability,
        item,
        types,
        teraType
      }
    }

    xAlly = {
      f: encodeSide({
        mode: ally.isReviving ? "revive" : req.type,
        delayedAttack,
        teraUsed,
        effects,
        wish
      }),
      lookup: encodeSideLookup(ally),
      team: xTeam
    }
  }

  let xFoe: XFoe
  {
    const { team, delayedAttack, effects, teraUsed, wish } = foe

    let xTeam: {
      [k: string]: XFoeUser
    } = {}

    for (const species in team) {
      const user = team[species]
      const {
        hp,
        item,
        ability,
        moveSet,
        types,
        status,
        teraType,
        flags,
        boosts,
        forme,
        volatiles
      } = user

      let presets = getPotentialPresets(format, user)
      presets = presets.filter((x) => matchesPreset(x, user))

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

      xTeam[species] = {
        f: encodeUser({
          revealed: true,
          stats: { ...inferStats(gen, user), hp: hp[1] },
          hpLeft: hp[0],
          flags,
          volatiles,
          boosts,
          status,
          forme
        }),
        lookup: encodeUserLookup(user),
        moveSet: Object.keys(moveSet).map((k) => encodeMoveSlot(moveSet, k)!),
        movePool: [...validMoves]
          .filter((move) => !(move in moveSet))
          .map((move) => {
            const pp = scalePP(inferMaxPP(gen, move))
            return {
              move,
              f: [pp, pp]
            }
          }),
        items: item ? [item] : [...validItems],
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

    xFoe = {
      f: encodeSide({
        mode: mode,
        delayedAttack,
        teraUsed,
        effects,
        wish
      }),
      lookup: encodeSideLookup(foe),
      team: xTeam
    }
  }

  return {
    f: encodeBattle({ fields, weather }),
    ally: xAlly,
    foe: xFoe
  }
}
