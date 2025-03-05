import { Generation } from "@pkmn/data"
import { Fields, Observer, Weather } from "../client/observer.js"
import {
  BOOST_IDS,
  Boosts,
  DELAYED_MOVES,
  HAZARDS,
  MOD_STAT_IDS,
  PSEUDO_WEATHER_NAMES,
  SCREENS,
  STAT_IDS,
  Stats,
  STATUS_IDS,
  TERRAIN_NAMES,
  WEATHER_NAMES
} from "../battle.js"
import { Flags, MoveSet, Status, User, Volatiles } from "../client/user.js"
import { DelayedAttack, SideEffects } from "../client/side.js"
import { inferInitialForme, getPotentialPresets, matchesPreset } from "../version.js"
import { Format } from "../run.js"
import { INTERIM_FORMES } from "./forme.js"
import { scalePP, scaleStat } from "./norm.js"
import { inferMaxPP } from "../client/move.js"

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
  feats.push(...STATUS_IDS.map((id) => (id === status?.id ? 1 : 0)))
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

  for (const name of [
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
          const turnsLeft = duration! - turn!
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

export type MoveSlotFeature = {
  move: string
  x: number[]
}

type UserFeature = {
  x: number[]
  moveSet: MoveSlotFeature[]
  movePool: MoveSlotFeature[]
  abilities: string[]
  items: string[] | null
  types: string[]
  teraTypes: string[]
  disabled: MoveSlotFeature | null
  choice: MoveSlotFeature | null
  encore: MoveSlotFeature | null
  locked: MoveSlotFeature | null
  lastMove: MoveSlotFeature | null
  lastBerry: string | null
}

type SideFeature = {
  x: number[]
  team: { [k: string]: UserFeature }
  active: string
}

export type Observation = {
  x: number[]
  ally: SideFeature
  foe: SideFeature
}

export const CHOICE_MODES = ["move", "switch", "revive", "wait"]
export type ChoiceMode = (typeof CHOICE_MODES)[number]

export function extractMoveSlot(moveSet: MoveSet, move: string): MoveSlotFeature {
  if (move in moveSet) {
    const { used, max } = moveSet[move]
    return { move, x: [scalePP(Math.max(0, max - used)), scalePP(max)] }
  }

  return {
    move,
    x: [1, 1]
  }
}

export function extractMoveSet(moveSet: MoveSet) {
  return Object.keys(moveSet).map((k) => extractMoveSlot(moveSet, k)!)
}

function extractUserLookup({ moveSet, volatiles, lastBerry, lastMove }: User) {
  return {
    disabled: volatiles["Disable"] ? extractMoveSlot(moveSet, volatiles["Disable"].move) : null,
    choice: volatiles["Choice Locked"]
      ? extractMoveSlot(moveSet, volatiles["Choice Locked"].move)
      : null,
    encore: volatiles["Encore"] ? extractMoveSlot(moveSet, volatiles["Encore"].move) : null,
    locked: volatiles["Locked Move"]
      ? extractMoveSlot(moveSet, volatiles["Locked Move"].move)
      : null,
    lastMove: lastMove ? extractMoveSlot(moveSet, lastMove) : null,
    lastBerry: lastBerry?.name ?? null
  }
}

function extractUser({
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
  const x: number[] = []

  x.push(revealed ? 1 : 0)
  x.push(tera ? 1 : 0)
  x.push(scaleStat("hp", hpLeft))
  x.push(...encodeStats(stats))
  x.push(...encodeStatus(status))

  x.push(
    ...(["battleBond", "intrepidSword", "illusionRevealed"] as const).map((k) => (flags[k] ? 1 : 0))
  )

  x.push(...INTERIM_FORMES.map((k) => (k === forme ? 1 : 0)))
  x.push(...BOOST_IDS.map((id) => boosts[id] ?? 0))

  x.push(...encodeVolatiles(volatiles))

  return x
}

function inferStats(gen: Generation, forme: string, lvl: number): Stats {
  const { baseStats } = gen.species.get(forme)!

  const stats: any = {}
  for (const id of STAT_IDS) {
    stats[id] = gen.stats.calc(id, baseStats[id], 31, 85, lvl)
  }

  return stats
}

function extractSide({
  mode,
  effects,
  wish,
  delayedAttack,
  teraUsed
}: {
  mode: ChoiceMode
  effects: SideEffects
  wish?: number
  delayedAttack?: DelayedAttack
  teraUsed?: boolean
}) {
  const x: number[] = []

  x.push(wish ? 2 - wish : 0)
  x.push(...encodeDelayedAttack(delayedAttack))
  x.push(teraUsed ? 1 : 0)

  x.push(...CHOICE_MODES.map((x) => (mode === x ? 1 : 0)))
  x.push(...HAZARDS.map((name) => effects[name]?.layers ?? 0))
  x.push(...SCREENS.map((name) => effects[name]?.turn ?? 0))

  return x
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

export function extractObservation(format: Format, obs: Observer): Observation {
  const { gen } = format

  const { ally, foe, fields, weather, req } = obs

  let allyF: SideFeature
  {
    const { team, delayedAttack, effects, teraUsed, wish, active } = ally

    let teamF: {
      [k: string]: UserFeature
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

      teamF[species] = {
        x: extractUser({
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
        moveSet: extractMoveSet(moveSet),
        movePool: [],
        abilities: [ability],
        items: item ? [item] : null,
        types,
        teraTypes: [teraType],
        ...extractUserLookup(user)
      }
    }

    allyF = {
      x: extractSide({
        mode: ally.isReviving ? "revive" : req.type,
        delayedAttack,
        teraUsed,
        effects,
        wish
      }),
      active: active.species,
      team: teamF
    }
  }

  let foeF: SideFeature
  {
    const { team, delayedAttack, effects, teraUsed, wish, active } = foe

    let teamF: {
      [k: string]: UserFeature
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
        base,
        volatiles
      } = user

      const initialForme = inferInitialForme(format, base.forme)
      let presets = getPotentialPresets(format, initialForme)
      let filtered = presets.filter((x) => matchesPreset(x, user))

      if (filtered.length) {
        presets = filtered
      } else {
        for (const initialForme of ["Zoroark", "Zoroark-Hisui"]) {
          const filtered = getPotentialPresets(format, initialForme).filter((x) =>
            matchesPreset(x, user)
          )

          if (filtered.length) {
            forme = initialForme
            presets = filtered
            lvl = format.patch[initialForme].level
            break
          }
        }
      }

      const validItems = new Set<string>()
      const validAbilities = new Set<string>()
      const validMoves = new Set<string>()
      const validTeraTypes = new Set<string>()

      for (const {
        agg: { items, abilities, moves, teraTypes }
      } of presets) {
        for (const item of items) if (item) validItems.add(item)
        for (const ability of abilities) validAbilities.add(ability)
        for (const type of teraTypes) validTeraTypes.add(type)
        for (const move of moves) validMoves.add(move)
      }

      teamF[species] = {
        x: extractUser({
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
        moveSet: extractMoveSet(moveSet),
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
        teraTypes: teraType ? [teraType] : [...validTeraTypes],
        ...extractUserLookup(user)
      }
    }

    let mode: ChoiceMode
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

    foeF = {
      x: extractSide({
        mode: mode,
        delayedAttack,
        teraUsed,
        effects,
        wish
      }),
      active: active.species,
      team: teamF
    }
  }

  return {
    x: encodeBattle({ fields, weather }),
    ally: allyF,
    foe: foeF
  }
}
