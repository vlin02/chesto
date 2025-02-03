import { Stats as Calc, Generation, StatID } from "@pkmn/data"
import { Observer } from "./client/observer.js"
import { STAT_IDS, Stats, TERRAIN_NAMES } from "./battle.js"
import { Format, getBaseForme, getPotentialPresets, matchesPreset } from "./run.js"
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
  let sleepAttemptsLeft = { min: 0, max: 0 }
  let toxicTurns = 0

  if (id === "tox") toxicTurns = turn!
  if (id === "slp")
    sleepAttemptsLeft = {
      min: Math.max(1 - attempt!, 1),
      max: 3 - attempt!
    }

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

function encodeVolatiles(volatiles: Volatiles) {
  const encoded: any = {}

  for (const name in volatiles) {
    switch (name) {
      case "Leech Seed":
      case "Charge":
      case "Attract":
      case "No Retreat":
      case "Salt Cure":
      case "Flash Fire":
      case "Leech Seed":
      case "Substitute":
      case "Pressure":
      case "Transform":
      case "Trace":
      case "Destiny Bond":
      case "Roost":
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
          turnsLeft: {
            min: Math.max(2 - turn, 1),
            max: Math.max(5 - turn)
          }
        }
        break
      case "Encore":
      case "Type Change":
      case "Choice Locked":
      case "Protosynthesis":
      case "Quark Drive":
      case "Fallen":
        encoded[name] = volatiles[name]
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
          turnsLeft: {
            min: Math.max(2 - turn, 1),
            max: Math.max(3 - turn)
          },
          move
        }
        break
      }
    }
  }
}

function encodeDelayedAttack({ move, turn }: DelayedAttack) {
  return {
    move,
    turnsLeft: 2 - turn
  }
}

function encodeWish(wish?: number) {
  return wish ? 2 - wish : 0
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

function inferStats(gen: Generation, { forme, lvl }: FoeUser) {
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

      const baseForme = getBaseForme(format, forme)

      encodedTeam[species] = {
        revealed,
        stats: encodeStats(stats),
        hp: {
          left: hp[0] / hp[1],
          max: scale(hp[1], 100, 300)
        },
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
      const { hp } = stats

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
        baseForme: getBaseForme(format, forme),
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
    fields,
    ally: encodedAlly,
    foe: encodedFoe,
    forceSwitch: request.type === "switch"
  }
}
