import { Generation } from "@pkmn/data"
import { Fields, Observer, Weather } from "../client/observer.js"
import {
  BOOST_IDS,
  Boosts,
  HAZARDS,
  PSEUDO_WEATHER_NAMES,
  SCREENS,
  STAT_IDS,
  Stats,
  TERRAIN_NAMES,
  WEATHER_NAMES
} from "../battle.js"
import { Flags, MoveSet, Status, User, Volatiles } from "../client/user.js"
import { DelayedAttack, SideEffects } from "../client/side.js"
import { getInitialForme, getPotentialPresets, matchesPreset } from "../version.js"
import { Format } from "../run.js"
import { encodeDelayedAttack, encodeStats, encodeStatus, encodeVolatiles } from "./features.js"
import { INTERIM_FORMES } from "./forme.js"
import { scalePP, scaleStat } from "./norm.js"
import { inferMaxPP } from "../client/move.js"

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
  disabled: MoveSlotFeature | undefined
  choice: MoveSlotFeature | undefined
  encore: MoveSlotFeature | undefined
  locked: MoveSlotFeature | undefined
  lastMove: MoveSlotFeature | undefined
  lastBerry: string | undefined
}

type SideFeature = {
  x: number[]
  team: { [k: string]: UserFeature }
  active: string
}

export type BattleFeature = {
  x: number[]
  ally: SideFeature
  foe: SideFeature
}

export const CHOICE_MODES = ["move", "switch", "revive", "wait"]
export type ChoiceMode = (typeof CHOICE_MODES)[number]

export function extractMoveSlot(moveSet: MoveSet, move: string): MoveSlotFeature | undefined {
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
    disabled: volatiles["Disable"] ? extractMoveSlot(moveSet, volatiles["Disable"].move) : undefined,
    choice: volatiles["Choice Locked"]
      ? extractMoveSlot(moveSet, volatiles["Choice Locked"].move)
      : undefined,
    encore: volatiles["Encore"] ? extractMoveSlot(moveSet, volatiles["Encore"].move) : undefined,
    locked: volatiles["Locked Move"]
      ? extractMoveSlot(moveSet, volatiles["Locked Move"].move)
      : undefined,
    lastMove: lastMove ? extractMoveSlot(moveSet, lastMove) : undefined,
    lastBerry: lastBerry?.name
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

export function extractBattle(format: Format, obs: Observer): BattleFeature {
  const { gen } = format

  const { ally, foe, fields, weather, req } = obs

  let fAlly: SideFeature
  {
    const { team, delayedAttack, effects, teraUsed, wish, active } = ally

    let fTeam: {
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

      fTeam[species] = {
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

    fAlly = {
      x: extractSide({
        mode: ally.isReviving ? "revive" : req.type,
        delayedAttack,
        teraUsed,
        effects,
        wish
      }),
      active: active.species,
      team: fTeam
    }
  }

  let foeInput: SideFeature
  {
    const { team, delayedAttack, effects, teraUsed, wish, active } = foe

    let fTeam: {
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

      const initialForme = getInitialForme(format, base.forme)
      let presets = getPotentialPresets(format, initialForme)
      let filtered = presets.filter((x) => matchesPreset(x, user))

      if (filtered.length) {
        presets = filtered
      } else {
        let v = 0
        for (const initialForme of ["Zoroark", "Zoroark-Hisui"]) {
          const filtered = getPotentialPresets(format, initialForme).filter((x) =>
            matchesPreset(x, user)
          )

          if (filtered.length) {
            forme = initialForme
            presets = filtered
            lvl = format.patch[initialForme].level
            v = 1
            break
          }
        }

        if (v === 0) {
          console.log(user)
          console.warn("failed")
        }
      }

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

      fTeam[species] = {
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

    foeInput = {
      x: extractSide({
        mode: mode,
        delayedAttack,
        teraUsed,
        effects,
        wish
      }),
      active: active.species,
      team: fTeam
    }
  }

  return {
    x: encodeBattle({ fields, weather }),
    ally: fAlly,
    foe: foeInput
  }
}
