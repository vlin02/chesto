import { STAT_IDS, STATUS_IDS, Stats, DELAYED_MOVES, MOD_STAT_IDS } from "../battle.js"
import { DelayedAttack } from "../client/side.js"
import { Status, Volatiles } from "../client/user.js"
import { scaleStat } from "./norm.js"

export function encodeStats(stats: Stats) {
  return STAT_IDS.map((id) => scaleStat(id, stats[id]))
}

export function encodeStatus(status: Status | undefined) {
  let toxicTurns = 0
  let sleepAttemptsLeft = [0, 0]

  if (status?.id === "tox") toxicTurns = status.turn!
  if (status?.id === "slp")
    sleepAttemptsLeft = [Math.max(1 - status.attempt!, 1), 3 - status.attempt!]

  const feats = []
  feats.push(...STATUS_IDS.map((id) => feats.push(id === status?.id ? 1 : 0)))
  feats.push(toxicTurns, ...sleepAttemptsLeft)

  return feats
}

export function encodeDelayedAttack(delayedAttack: DelayedAttack | undefined) {
  const encoded: number[] = []

  const turnsLeft = delayedAttack ? 2 - delayedAttack.turn : 0
  encoded.push(turnsLeft)

  for (const name of DELAYED_MOVES) {
    encoded.push(delayedAttack?.move === name ? 1 : 0)
  }

  return encoded
}

export function encodeVolatiles(volatiles: Volatiles) {
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
        for (const k of MOD_STAT_IDS) {
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
