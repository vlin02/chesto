import { Stats as StatCalc } from "@pkmn/data"
import { Observer } from "./client/observer.js"
import { Stats, TERRAIN_NAMES } from "./battle.js"
import { Format, getBaseForme } from "./run.js"
import { MoveSet, Status, Volatiles } from "./client/user.js"
import { DelayedAttack, HAZARDS, SideEffects } from "./client/side.js"

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
    encoded[id] = scale(stats[id], 0, 200)
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

function encodeSideEffects(effects: SideEffects) {
  const encoded = {}
  for (const name in effects) {
    if (HAZARDS.includes(name)) {
      encoded[name] = {
        layers: 
      }
    }
  }
}

export function encode(format: Format, obs: Observer) {
  const { gen } = format
  const statCalc = new StatCalc(gen.dex)

  let weather = null
  let terrain = null
  const fields: { [k: string]: number } = {}

  if (obs.weather) {
    const { name, turn } = obs.weather
    weather = { name, turnsLeft: scale(5 - turn, 0, 5) }
  }

  for (const name in obs.fields) {
    const turn = obs.fields[name]
    const turnsLeft = scale(5 - turn, 0, 5)

    if (TERRAIN_NAMES.includes(name)) terrain = { name, turnsLeft }
    else fields[name] = turnsLeft
  }

  const { ally } = obs

  let encodedAlly

  {
    const { team, delayedAttack, teraUsed } = ally

    let encodedTeam: any = {}
    for (const species in ally.team) {
      const user = ally.team[species]

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
        baseForme: getBaseForme(format, forme),
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
      hazards: 
    }
  }

  {
  }

  return {
    weather,
    terrain,
    fields
  }
}
