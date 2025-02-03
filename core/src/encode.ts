import { Stats as Calc, Generation } from "@pkmn/data"
import { Fields, Observer, Weather } from "./client/observer.js"
import { BOOST_IDS, Boosts, STAT_IDS, Stats, STATUS_IDS, TERRAIN_NAMES, WEATHER_NAMES } from "./battle.js"
import { Format, getPresetForme, getPotentialPresets, matchesPreset } from "./run.js"
import { Flags, FoeUser, MoveSet, Status, Volatiles } from "./client/user.js"
import { DELAYED_MOVES, DelayedAttack, HAZARDS, SCREENS, SideEffects } from "./client/side.js"
import { Dex } from "@pkmn/dex"

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
  const encoded: number[] = []

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
        encoded.push(name in volatiles ? 1 : 0)
        break
      case "Taunt":
      case "Yawn":
      case "Throat Chop":
      case "Heal Block":
      case "Slow Start":
      case "Magnet Rise": {
        if (name in volatiles) {
          const duration = {
            "Taunt": 3,
            "Yawn": 2,
            "Throat Chop": 2,
            "Heal Block": 5,
            "Slow Start": 5,
            "Magnet Rise": 5
          }[name]!
          const { turn } = volatiles[name]!
          const turnsLeft = Math.max(duration - turn!, 1)

          encoded.push(turnsLeft)
        } else {
          encoded.push(0)
        }
        break
      }
      case "Confusion":
        if (name in volatiles) {
          const { turn } = volatiles[name]!
          encoded.push(...[Math.max(2 - turn!, 1), Math.max(5 - turn!)])
        }
        break
      case "Protosynthesis":
      case "Quark Drive":
        for (const k of ["atk", "def", "spa", "spd", "spe"] as const) {
          encoded.push(volatiles[name as "Protosynthesis" | "Quark Drive"]?.statId === k ? 1 : 0)
        }
        break
      case "Fallen":
        encoded.push(volatiles[name as "Fallen"]?.count ?? 0)
        break

      default:
        throw Error(name)
    }
  }

  return encoded
}

function encodeStats(stats: Stats) {
  const feats: number[] = []
  for (const statId of STAT_IDS) {
    feats.push(stats[statId])
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
  for (const name in moveSet) {
    const { max, used } = moveSet[name]
    const left = Math.max(0, max - used) / max
    return {
      name,
      slot: [left, max]
    }
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
}

function inferStats(gen: Generation, { forme, lvl }: FoeUser): Stats {
  const calc = new Calc(Dex)
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

function encodeBattle({ fields, weather }: { fields: Fields; weather?: Weather }) {
  const feats: number[] =  []
  for (const name of WEATHER_NAMES) {
    feats.push(weather?.name === name ? (5 - weather.turn) : 0)
  }
  
  let terrain: {name: string, turnsLeft: number} | null = null
  let trickRoomTurnsLeft = 0

  for (const name in fields) {
    const turnsLeft = 5 - fields[name]
    if (TERRAIN_NAMES.includes(name)) terrain = {name, turnsLeft}
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

  let encodedWeather: number[] = []
  let encodedTerrain = null
  let trickRoom = null

  if (weather) {
    const { name, turn } = weather
    encodedWeather = { name, turnsLeft: 5 - turn }
  }

  for (const name in fields) {
    const turn = fields[name]
    const turnsLeft = 5 - turn

    if (TERRAIN_NAMES.includes(name)) encodedTerrain = { name, turnsLeft }
    else if (name === "Trick Room") {
      trickRoom = { turnsLeft }
    } else {
      throw Error()
    }
  }

  let encodedAlly

  {
    const { team, delayedAttack, effects, active, teraUsed, wish } = ally

    let encodedTeam: any = {}
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
        lastMove,
        flags,
        lastBerry,
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
        ability,
        item,
        types,
        teraType,
        initialForme,
        lastBerry,
        moveSet: encodeMoveSet(moveSet)
      }
    }

    encodedAlly = {
      features: encodeSide({
        delayedAttack,
        teraUsed,
        effects,
        wish,

      })
      team: encodedTeam,
      active: active.species,
    }
  }

  let encodedFoe
  {
    const { team, delayedAttack, effects, active, teraUsed, wish } = foe

    let encodedTeam: any = {}
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
        lastBerry,
        lastMove,
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
        stats: inferStats(gen, user),
        hpLeft: hp[0] / hp[1],
        types,
        moveSet: encodeMoveSet(moveSet),
        unusedMoves: [...validMoves].filter((move) => !(move in moveSet)),
        items: item ? [item] : [...validItems],
        abilities: ability ? [ability] : [...validAbilities],
        status: status ? encodeStatus(status) : null,
        teraType,
        flags,
        initialForme,
        isInterim: INTERIM_FORMES.includes(initialForme),
        lastBerry,
        lastMove,
        volatiles: encodeVolatiles(volatiles),
        boosts
      }
    }

    encodedFoe = {
      team: encodedTeam,
      delayedAttack: delayedAttack ? encodeDelayedAttack(delayedAttack) : null,
      teraUsed,
      effects: encodeSide(effects),
      active: active.species,
      wish: encodeWish(wish)
    }
  }

  return {
    weather: encodedWeather,
    terrain: encodedTerrain,
    trickRoom,
    ally: encodedAlly,
    foe: encodedFoe,
    forceSwitch: request.type === "switch"
  }
}
