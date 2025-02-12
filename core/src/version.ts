import { FoeUser } from "./client/user.js"
import { Stats } from "fs"
import { TypeName } from "./battle.js"
import { Format } from "./run.js"

export type Preset = {
  role: string
  movepool: string[]
  abilities?: string[]
  teraTypes?: TypeName[]
  agg: {
    teraTypes: TypeName[]
    evs: Stats[]
    ivs: Stats[]
    formes: string[]
    genders: string[]
    items: string[]
    abilities: string[]
    moves: string[]
  }
}

export type Patch = {
  [k: string]: {
    name: string
    level: number
    presets: Preset[]
  }
}

export function getInitialForme({ gen, patch }: Format, forme: string) {
  return forme in patch ? forme : gen.species.get(forme)!.baseSpecies
}

export function getPotentialPresets(format: Format, initialForme: string) {
  const { patch } = format

  const presets = [...patch[initialForme].presets]

  if (initialForme === "Greninja") presets.push(...patch["Greninja-Bond"].presets)

  return presets
}

export function matchesPreset(preset: Preset, user: FoeUser) {
  const {
    base: { ability, item, moveSet },
    teraType
  } = user

  const {
    movepool,
    agg: { moves, teraTypes, abilities, items }
  } = preset

  if (teraTypes && teraType && !teraTypes.includes(teraType)) return false
  if (abilities && ability && !abilities.includes(ability)) return false
  if (item && !items.includes(item)) return false
  if (!Object.keys(moveSet).every((move) => moves.includes(move) || movepool.includes(move)))
    return false

  return true
}
