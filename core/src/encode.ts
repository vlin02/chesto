import { Stats as Calc, Generation } from "@pkmn/data"
import { Fields, Observer, Weather } from "./client/observer.js"
import {
  BOOST_IDS,
  Boosts,
  STAT_IDS,
  Stats,
  STATUS_IDS,
  TERRAIN_NAMES,
  WEATHER_NAMES
} from "./battle.js"
import { Format, getPresetForme, getPotentialPresets, matchesPreset } from "./run.js"
import { Flags, FoeUser, MoveSet, Status, User, Volatiles } from "./client/user.js"
import {
  DELAYED_MOVES,
  DelayedAttack,
  HAZARDS,
  Party,
  SCREENS,
  SideEffects
} from "./client/side.js"

const STAT_RANGES = {
  hp: [191, 566],
  atk: [13, 348],
  def: [57, 393],
  spa: [85, 318],
  spd: [71, 402],
  spe: [57, 357]
} as const

function scale(n: number, lo: number, hi: number, neg = false) {
  if (neg) {
    const mid = (hi + lo) / 2
    return scale(n, mid, hi)
  }
  return (n - lo) / (hi - lo)
}

const INTERIM_FORMES = [
  "Minior",
  "Minior-Meteor",
  "Terapagos",
  "Terapagos-Terastal",
  "Shaymin-Sky",
  "Ogerpon",
  "Ogerpon-Cornerstone",
  "Ogerpon-Hearthflame",
  "Ogerpon-Wellspring",
  "Palafin",
  "Eiscue",
  "Eiscue-Noice",
  "Cramorant",
  "Cramorant-Gulping",
  "Cramorant-Gorging",
  "Mimikyu",
  "Mimikyu-Busted",
  "Meloetta",
  "Meloetta-Pirouette",
  "Morpeko",
  "Morpeko-Hangry"
]

function encodeVolatiles(volatiles: Volatiles) {
  const feats: number[] = []

  for (const name in [
    "Leech Seed",
    "Charge",
    "Attract",
    "No Retreat",
    "Salt Cure",
    "Flash Fire",
    "Substitute",
    "Pressure",
    "Transform",
    "Trace",
    "Destiny Bond",
    "Glaive Rush",
    "Roost",
    "Protect",
    "Beak Blast",
    "Focus Punch",
    "Type Change",
    "Taunt",
    "Disable",
    "Encore",
    "Locked Move",
    "Yawn",
    "Throat Chop",
    "Heal Block",
    "Slow Start",
    "Magnet Rise",
    "Confusion",
    "Protosynthesis",
    "Quark Drive",
    "Fallen"
  ]) {
    switch (name) {
      case "Leech Seed":
      case "Charge":
      case "Attract":
      case "No Retreat":
      case "Salt Cure":
      case "Flash Fire":
      case "Substitute":
      case "Pressure":
      case "Transform":
      case "Trace":
      case "Destiny Bond":
      case "Glaive Rush":
      case "Roost":
      case "Protect":
      case "Beak Blast":
      case "Focus Punch":
      case "Type Change":
        feats.push(name in volatiles ? 1 : 0)
        break
      case "Taunt":
      case "Yawn":
      case "Throat Chop":
      case "Heal Block":
      case "Slow Start":
      case "Disable":
      case "Encore":
      case "Magnet Rise": {
        const duration = {
          "Taunt": 3,
          "Yawn": 2,
          "Throat Chop": 2,
          "Heal Block": 5,
          "Slow Start": 5,
          "Magnet Rise": 5,
          "Disable": 4,
          "Encore": 3
        }[name]
        if (name in volatiles) {
          const { turn } = volatiles[name]!
          const turnsLeft = Math.max(duration! - turn!, 1)
          feats.push(turnsLeft)
        } else {
          feats.push(0)
        }
        break
      }
      case "Locked Move":
      case "Confusion":
        const duration = {
          "Locked Move": [2, 3],
          "Confusion": [2, 5]
        }[name]

        if (name in volatiles) {
          const { turn } = volatiles[name]!
          const [lo, hi] = duration!
          feats.push(...[Math.max(lo - turn!, 1), hi - turn!])
        } else {
          feats.push(...[0, 0])
        }
        break
      case "Protosynthesis":
      case "Quark Drive":
        for (const k of ["atk", "def", "spa", "spd", "spe"] as const) {
          feats.push(volatiles[name as "Protosynthesis" | "Quark Drive"]?.statId === k ? 1 : 0)
        }
        break
      case "Fallen":
        feats.push(volatiles[name as "Fallen"]?.count ?? 0)
        break

      default:
        throw Error(name)
    }
  }

  return feats
}

function encodeStats(stats: Stats) {
  const feats: number[] = []
  for (const statId of STAT_IDS) {
    const [lo, hi] = STAT_RANGES[statId]
    feats.push(scale(stats[statId], lo, hi))
  }
  return feats
}

function encodeStatus(status: Status | undefined) {
  let sleepAttemptsLeft = [0, 0]
  let toxicTurns = 0

  if (status?.id === "tox") toxicTurns = status.turn!
  if (status?.id === "slp")
    sleepAttemptsLeft = [Math.max(1 - status.attempt!, 1), 3 - status.attempt!]

  const feats = []
  for (const id of STATUS_IDS) {
    feats.push(id === status?.id ? 1 : 0)
  }
  feats.push(toxicTurns, ...sleepAttemptsLeft)

  return feats
}

function encodeMoveSet(moveSet: MoveSet) {
  const feats: { [k: string]: number[] } = {}
  for (const name in moveSet) {
    const { max, used } = moveSet[name]
    const left = Math.max(0, max - used) / max
    feats[name] = [left, scale(max, 0, 64)]
  }

  return feats
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

function encodeDelayedAttack(delayedAttack: DelayedAttack | undefined) {
  const encoded: number[] = []

  const turnsLeft = delayedAttack ? 2 - delayedAttack.turn : 0
  encoded.push(turnsLeft)

  for (const name of DELAYED_MOVES) {
    encoded.push(delayedAttack?.move === name ? 1 : 0)
  }

  return encoded
}

function encodeSide({
  effects,
  wish,
  delayedAttack,
  teraUsed
}: {
  effects: SideEffects
  wish?: number
  delayedAttack?: DelayedAttack
  teraUsed: boolean
}) {
  const feats: number[] = []

  feats.push(teraUsed ? 1 : 0)

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

function encodeMemberRefs({ active, team }: Party) {
  return [active.species, Object.keys(team).find((k) => team[k].tera)]
}

function encodeMoveRefs({ volatiles, lastMove, moveSet }: User) {
  return [
    volatiles["Disable"]?.move,
    volatiles["Choice Locked"]?.move,
    volatiles["Encore"]?.move,
    volatiles["Locked Move"]?.move,
    lastMove
  ].map((m) => {
    if (!m) return null
    if (!(m in moveSet)) {
      console.warn(m, moveSet)
      return null
    }
    return m
  })
}

function encodeItemRefs({ lastBerry }: User) {
  return [lastBerry?.name ?? null]
}

function encodeBattle({
  fields,
  weather,
  forceSwitch
}: {
  fields: Fields
  weather?: Weather
  forceSwitch: boolean
}) {
  const feats: number[] = []

  feats.push(forceSwitch ? 1 : 0)

  for (const name of WEATHER_NAMES) {
    feats.push(weather?.name === name ? 5 - weather.turn : 0)
  }

  let terrain: { name: string; turnsLeft: number } | null = null
  let trickRoomTurnsLeft = 0

  for (const name in fields) {
    const turnsLeft = 5 - fields[name]
    if (TERRAIN_NAMES.includes(name)) terrain = { name, turnsLeft }
    if (name === "Trick Room") trickRoomTurnsLeft = turnsLeft
  }

  for (const name of TERRAIN_NAMES) {
    feats.push(terrain?.name === name ? terrain.turnsLeft : 0)
  }

  feats.push(trickRoomTurnsLeft)

  return feats
}

export function encodeObserver(format: Format, obs: Observer) {
  const { gen } = format

  const { ally, foe, fields, weather, request } = obs

  let encodedAlly

  {
    const { team, delayedAttack, effects, teraUsed, wish } = ally

    let encodedTeam: {
      [k: string]: {
        features: number[]
        moveRefs: (string | null)[]
        itemRefs: (string | null)[]
        moveSet: {
          [k: string]: number[]
        }
        item: string | null
        ability: string | null
        types: string[]
        teraType: string
        initialForme: string
      }
    } = {}
    for (const species in team) {
      const user = team[species]

      const {
        revealed,
        forme,
        hp,
        types,
        ability,
        item,
        stats,
        status,
        moveSet,
        teraType,
        boosts,
        flags,
        volatiles
      } = user

      const initialForme = getPresetForme(format, forme)

      encodedTeam[species] = {
        features: encodeUser({
          revealed,
          stats: { ...stats, hp: hp[1] },
          hpLeft: hp[0] / hp[1],
          flags,
          volatiles,
          boosts,
          status
        }),
        moveRefs: encodeMoveRefs(user),
        itemRefs: encodeItemRefs(user),
        moveSet: encodeMoveSet(moveSet),
        item,
        ability,
        types,
        teraType,
        initialForme
      }
    }

    encodedAlly = {
      features: encodeSide({
        delayedAttack,
        teraUsed,
        effects,
        wish
      }),
      memberRefs: encodeMemberRefs(ally),
      team: encodedTeam
    }
  }

  let encodedFoe
  {
    const { team, delayedAttack, effects, teraUsed, wish } = foe

    let encodedTeam: {
      [k: string]: {
        features: number[]
        moveRefs: (string | null)[]
        itemRefs: (string | null)[]
        moveSet: {
          [k: string]: number[]
        }
        unusedMoves: string[]
        abilities: string[]
        items: string[]
        types: string[]
        teraTypes: string[]
        initialForme: string
      }
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

      encodedTeam[species] = {
        features: encodeUser({
          revealed: true,
          stats: inferStats(gen, user),
          hpLeft: hp[0] / hp[1],
          flags,
          volatiles,
          boosts,
          status
        }),
        moveRefs: encodeMoveRefs(user),
        itemRefs: encodeItemRefs(user),
        moveSet: encodeMoveSet(moveSet),
        unusedMoves: [...validMoves].filter((move) => !(move in moveSet)),
        items: item ? [item] : [...validItems],
        abilities: ability ? [ability] : [...validAbilities],
        types,
        teraTypes: teraType ? [teraType] : [...validTeraTypes],
        initialForme
      }
    }

    encodedFoe = {
      features: encodeSide({
        delayedAttack,
        teraUsed,
        effects,
        wish
      }),
      memberRefs: encodeMemberRefs(foe),
      team: encodedTeam
    }
  }

  return {
    features: encodeBattle({ fields, weather, forceSwitch: request.type === "switch" }),
    ally: encodedAlly,
    foe: encodedFoe
  }
}
