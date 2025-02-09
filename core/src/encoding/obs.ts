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
import { getPresetForme, getPotentialPresets, matchesPreset } from "../version.js"
import { Format } from "../format.js"
import { encodeDelayedAttack, encodeStats, encodeStatus, encodeVolatiles } from "./features.js"
import { INTERIM_FORMES } from "./forme.js"
import { scalePP, scaleStat } from "./norm.js"

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
  initialForme: string
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
  movepool: string[]
  abilities: string[]
  items: string[]
  types: string[]
  teraTypes: string[]
  initialForme: string
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

export function encodeMove(moveSet: MoveSet, move?: string): XMoveSlot | undefined {
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
  return Object.keys(moveSet).map((k) => encodeMove(moveSet, k)!)
}

function encodeUserLookup({ moveSet, volatiles, lastBerry, lastMove }: User) {
  return {
    disabled: encodeMove(moveSet, volatiles["Disable"]?.move),
    choice: encodeMove(moveSet, volatiles["Choice Locked"]?.move),
    encore: encodeMove(moveSet, volatiles["Encore"]?.move),
    locked: encodeMove(moveSet, volatiles["Locked Move"]?.move),
    lastMove: encodeMove(moveSet, lastMove),
    lastBerry: lastBerry?.name
  }
}

function encodeUser({
  revealed,
  hpLeft,
  stats,
  status,
  flags,
  interimForme,
  volatiles,
  boosts
}: {
  revealed: boolean
  hpLeft: number
  stats: Stats
  status?: Status
  flags: Flags
  interimForme?: string
  volatiles: Volatiles
  boosts: Boosts
}) {
  const feats: number[] = []

  feats.push(revealed ? 1 : 0)
  feats.push(scaleStat("hp", hpLeft))
  feats.push(...encodeStats(stats))
  feats.push(...encodeStatus(status))

  for (const k of ["battleBond", "intrepidSword", "illusionRevealed"] as const) {
    feats.push(flags[k] ? 1 : 0)
  }

  for (const k of INTERIM_FORMES) {
    feats.push(k === interimForme ? 1 : 0)
  }

  for (const id of BOOST_IDS) {
    feats.push(boosts[id] ?? 0)
  }

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
  teraUsed,
  isReviving
}: {
  mode: DecisionMode
  effects: SideEffects
  wish?: number
  delayedAttack?: DelayedAttack
  isReviving?: boolean
  teraUsed?: boolean
}) {
  const feats: number[] = []

  for (const x of DECISION_MODES) {
    feats.push(mode === x ? 1 : 0)
  }

  feats.push(teraUsed ? 1 : 0)
  feats.push(isReviving ? 1 : 0)

  for (const name of HAZARDS) {
    feats.push(effects[name]?.layers ?? 0)
  }
  for (const name of SCREENS) {
    feats.push(effects[name]?.turn ?? 0)
  }

  feats.push(...encodeDelayedAttack(delayedAttack))
  feats.push(wish ? 2 - wish : 0)

  return feats
}

function encodeSideLookup({ active, team }: Side) {
  return {
    active: active.species,
    tera: Object.keys(team).find((k) => team[k].tera)
  }
}

function encodeBattle({ fields, weather }: { fields: Fields; weather?: Weather }) {
  const feats: number[] = []

  for (const name of WEATHER_NAMES) {
    feats.push(weather?.name === name ? 5 - weather.turn : 0)
  }

  let terrain: { name: string; turnsLeft: number } | null = null
  let trickRoomTurnsLeft = 0

  for (const name in fields) {
    const turnsLeft = 5 - fields[name]
    if (TERRAIN_NAMES.includes(name)) terrain = { name, turnsLeft }
    else if (name === "Trick Room") trickRoomTurnsLeft = turnsLeft
    else throw Error(name)
  }

  for (const name of TERRAIN_NAMES) {
    feats.push(terrain?.name === name ? terrain.turnsLeft : 0)
  }

  feats.push(trickRoomTurnsLeft)

  return feats
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

      const initialForme = getPresetForme(format, forme)

      xTeam[species] = {
        f: encodeUser({
          revealed,
          stats: { ...stats, hp: hp[1] },
          hpLeft: hp[0],
          flags,
          volatiles,
          boosts,
          status
        }),
        lookup: encodeUserLookup(user),
        moveSet: encodeMoveSet(moveSet),
        ability,
        item,
        types: types.base,
        teraType,
        initialForme
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

      const initialForme = getPresetForme(format, forme)

      xTeam[species] = {
        f: encodeUser({
          revealed: true,
          stats: { ...inferStats(gen, user), hp: hp[1] },
          hpLeft: hp[0],
          flags,
          volatiles,
          boosts,
          status
        }),
        lookup: encodeUserLookup(user),
        moveSet: Object.keys(moveSet).map((k) => encodeMove(moveSet, k)!),
        movepool: [...validMoves].filter((move) => !(move in moveSet)),
        items: item ? [item] : [...validItems],
        abilities: ability ? [ability] : [...validAbilities],
        types: types.base,
        teraTypes: teraType ? [teraType] : [...validTeraTypes],
        initialForme
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
