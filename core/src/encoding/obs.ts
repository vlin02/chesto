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
import { Flags, FoeUser, Status, User, Volatiles } from "../client/user.js"
import { DelayedAttack, Side, SideEffects } from "../client/side.js"
import { getPresetForme, getPotentialPresets, matchesPreset } from "../version.js"
import { Format } from "../format.js"
import {
  encodeDelayedAttack,
  encodeMove,
  encodeMoveSet,
  encodeStats,
  encodeStatus,
  encodeVolatiles,
  FMoveSlot
} from "./features.js"
import { INTERIM_FORMES } from "./forme.js"

type UserLookup = {
  disabled: FMoveSlot | undefined
  choice: FMoveSlot | undefined
  encore: FMoveSlot | undefined
  locked: FMoveSlot | undefined
  lastMove: FMoveSlot | undefined
  lastBerry: string | undefined
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
  feats.push(hpLeft)
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

export type RequestType = "move" | "switch" | "revive" | "wait"

function encodeSide({
  requestType,
  effects,
  wish,
  delayedAttack,
  teraUsed,
  isReviving
}: {
  requestType: RequestType
  effects: SideEffects
  wish?: number
  delayedAttack?: DelayedAttack
  isReviving?: boolean
  teraUsed?: boolean
}) {
  const feats: number[] = []

  for (const x of ["move", "switch", "revive", "wait"]) {
    feats.push(requestType === x ? 1 : 0)
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

  {
    const turnsLeft = wish ? 2 - wish : 0
    feats.push(turnsLeft)
  }

  return feats
}

type SideLookup = {
  active: string
  tera: string | undefined
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

type FAllyUser = {
  features: number[]
  lookup: UserLookup
  moveSet: FMoveSlot[]
  item: string | null
  ability: string | null
  types: string[]
  teraType: string
  initialForme: string
}

type FAlly = {
  features: number[]
  team: { [k: string]: FAllyUser }
  lookup: SideLookup
}

type FFoeUser = {
  features: number[]
  lookup: UserLookup
  moveSet: FMoveSlot[]
  unusedMoves: string[]
  abilities: string[]
  items: string[]
  types: string[]
  teraTypes: string[]
  initialForme: string
}

type FFoe = {
  features: number[]
  team: { [k: string]: FFoeUser }
  lookup: SideLookup
}

export function encodeObserver(format: Format, obs: Observer) {
  const { gen } = format

  const { ally, foe, fields, weather, req } = obs

  let fAlly: FAlly
  {
    const { team, delayedAttack, effects, teraUsed, wish } = ally

    let fTeam: {
      [k: string]: FAllyUser
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

      fTeam[species] = {
        features: encodeUser({
          revealed,
          stats: { ...stats, hp: hp[1] },
          hpLeft: hp[0] / hp[1],
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

    if (ally.isReviving && req.type !== "switch") throw Error()

    fAlly = {
      features: encodeSide({
        requestType: ally.isReviving ? "revive" : req.type,
        delayedAttack,
        teraUsed,
        effects,
        wish
      }),
      lookup: encodeSideLookup(ally),
      team: fTeam
    }
  }

  let fFoe: FFoe
  {
    const { team, delayedAttack, effects, teraUsed, wish } = foe

    let fTeam: {
      [k: string]: FFoeUser
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

      fTeam[species] = {
        features: encodeUser({
          revealed: true,
          stats: inferStats(gen, user),
          hpLeft: hp[0] / hp[1],
          flags,
          volatiles,
          boosts,
          status
        }),
        lookup: encodeUserLookup(user),
        moveSet: Object.keys(moveSet).map((k) => encodeMove(moveSet, k)!),
        unusedMoves: [...validMoves].filter((move) => !(move in moveSet)),
        items: item ? [item] : [...validItems],
        abilities: ability ? [ability] : [...validAbilities],
        types: types.base,
        teraTypes: teraType ? [teraType] : [...validTeraTypes],
        initialForme
      }
    }

    let requestType: RequestType = "wait"
    {
      if (req.type === "move") requestType = "move"
      if (req.type === "wait") requestType = "switch"
    }

    fFoe = {
      features: encodeSide({
        requestType,
        delayedAttack,
        teraUsed,
        effects,
        wish
      }),
      lookup: encodeSideLookup(foe),
      team: fTeam
    }
  }

  return {
    features: encodeBattle({ fields, weather }),
    ally: fAlly,
    foe: fFoe
  }
}
