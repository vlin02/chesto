import { Stats as Calc, Generation, StatID } from "@pkmn/data"
import { Observer } from "./client/observer.js"
import { STAT_IDS, Stats, TERRAIN_NAMES } from "./battle.js"
import { Format, getPresetForme, getPotentialPresets, matchesPreset } from "./run.js"
import { FoeUser, MoveSet, Status, Volatiles } from "./client/user.js"
import { DelayedAttack, Hazard, HAZARDS, SideEffects } from "./client/side.js"
import { Dex } from "@pkmn/dex"

function scale(n: number, lo: number, hi: number, neg = false) {
  if (neg) {
    const mid = (hi + lo) / 2
    return scale(n, mid, hi)
  }
  return (n - lo) / (hi - lo)
}

function encodeStatus({ id, turn, attempt }: Status) {
  let sleepAttemptsLeft = [0, 0]
  let toxicTurns = 0

  if (id === "tox") toxicTurns = turn!
  if (id === "slp") sleepAttemptsLeft = [Math.max(1 - attempt!, 1), 3 - attempt!]

  return {
    id,
    toxicTurns,
    sleepAttemptsLeft
  }
}

function encodeStats(stats: Stats): Stats {
  const encoded: any = {}
  for (const id in stats) {
    encoded[id] = scale(stats[id as StatID], 0, 200)
  }
  return encoded
}

function encodeMoveSet(moveSet: MoveSet) {
  const encoded: { [k: string]: { left: number; max: number } } = {}
  for (const move in moveSet) {
    const { used, max } = moveSet[move]
    encoded[move] = {
      left: Math.max(0, max - used) / max,
      max: scale(max, 0, 64)
    }
  }

  return encoded
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

type EncodedVolaties =
  // flags
  | {
      [K in
        | "Charge"
        | "Attract"
        | "No Retreat"
        | "Salt Cure"
        | "Flash Fire"
        | "Leech Seed"
        | "Substitute"
        | "Pressure"
        | "Transform"
        | "Trace"
        | "Destiny Bond"
        | "Glaive Rush"
        | "Roost"
        | "Protect"
        | "Beak Blast"
        | "Focus Punch"]?: boolean
    } & {
      // turns
      [K in "Taunt" | "Yawn" | "Throat Chop" | "Heal Block" | "Slow Start" | "Magnet Rise"]?: {
        turnsLeft: number
      }
    } & {
      "Encore"?: { move: string }
      "Choice Locked"?: { move: string }
      "Protosynthesis"?: { statId: StatID }
      "Quark Drive"?: { statId: StatID }
      "Fallen"?: { count: number }
      "Confusion"?: {
        turnsLeft: [number, number]
      }
      "Disable"?: {
        turnsLeft: number
        move: string
      }
      "Locked Move"?: {
        turnsLeft: [number, number]
        move: string
      }
    }

function encodeVolatiles(volatiles: Volatiles) {
  const encoded: EncodedVolaties = {}

  for (const name in volatiles) {
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
        encoded[name] = true
        break
      case "Taunt":
      case "Yawn":
      case "Throat Chop":
      case "Heal Block":
      case "Slow Start":
      case "Magnet Rise": {
        const duration = {
          "Taunt": 3,
          "Yawn": 2,
          "Throat Chop": 2,
          "Heal Block": 5,
          "Slow Start": 5,
          "Magnet Rise": 5
        }[name]
        const { turn } = volatiles[name]!

        encoded[name] = {
          turnsLeft: Math.max(duration - turn, 1)
        }
        break
      }
      case "Confusion":
        const { turn } = volatiles[name]!

        encoded[name] = {
          turnsLeft: [Math.max(2 - turn, 1), Math.max(5 - turn)]
        }
        break
      case "Encore":
      case "Choice Locked":
        encoded[name] = volatiles[name]
        break
      case "Protosynthesis":
      case "Quark Drive":
        encoded[name] = volatiles[name]
        break
      case "Fallen":
        encoded[name] = volatiles[name]!
        break
      case "Disable": {
        const { turn, move } = volatiles[name]!
        encoded[name] = {
          turnsLeft: 4 - turn,
          move
        }
        break
      }
      case "Locked Move": {
        const { turn, move } = volatiles[name]!
        encoded[name] = {
          turnsLeft: [Math.max(2 - turn, 1), Math.max(3 - turn)],
          move
        }
        break
      }
      default:
        throw Error(name)
    }
  }

  return encoded
}

function encodeDelayedAttack({ move, turn }: DelayedAttack) {
  return {
    move,
    turnsLeft: 2 - turn
  }
}

function encodeWish(wish?: number) {
  return { turnsLeft: wish ? 2 - wish : 0 }
}

function encodeSideEffects(effects: SideEffects) {
  const encoded: any = {}
  for (const name in effects) {
    if (HAZARDS.includes(name as Hazard)) {
      encoded[name] = effects[name]
    } else {
      const { turn } = effects[name]
      encoded[name] = {
        turnsLeft: (name === "Tailwind" ? 4 : 5) - turn!
      }
    }
  }
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

export function encode(format: Format, obs: Observer) {
  const { gen } = format

  const { ally, foe, fields, weather, request } = obs

  let encodedWeather = null
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

      const baseForme = getPresetForme(format, forme)

      encodedTeam[species] = {
        revealed,
        stats: encodeStats({ ...stats, hp: hp[1] }),
        hpLeft: hp[0] / hp[1],
        moveSet: encodeMoveSet(moveSet),
        ability,
        item,
        status: status ? encodeStatus(status) : null,
        teraType,
        flags,
        baseForme,
        isInterim: INTERIM_FORMES.includes(baseForme),
        lastBerry,
        lastMove,
        volatiles: encodeVolatiles(volatiles),
        boosts
      }
    }

    encodedAlly = {
      team: encodedTeam,
      delayedAttack: delayedAttack ? encodeDelayedAttack(delayedAttack) : null,
      teraUsed,
      effects: encodeSideEffects(effects),
      active: active.species,
      wish: encodeWish(wish)
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

      const stats = inferStats(gen, user)

      encodedTeam[species] = {
        stats: encodeStats(stats),
        hpLeft: hp[0] / hp[1],
        moveSet: encodeMoveSet(moveSet),
        remainingMoves: [...validMoves].filter((move) => !(move in moveSet)),
        ability,
        items: item ? [item] : [...validItems],
        abilities: ability ? [ability] : [...validAbilities],
        status: status ? encodeStatus(status) : null,
        teraType,
        flags,
        baseForme: getPresetForme(format, forme),
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
      effects: encodeSideEffects(effects),
      active: active.species,
      wish: encodeWish(wish)
    }
  }

  return {
    weather: encodedWeather,
    terrain: encodedTerrain,
    trickRoom,
    fields,
    ally: encodedAlly,
    foe: encodedFoe,
    forceSwitch: request.type === "switch"
  }
}
