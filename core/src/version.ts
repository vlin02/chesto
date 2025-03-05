import { Stats } from "fs"
import { TypeName } from "./battle.js"
import { Generation } from "@pkmn/data"
import { User } from "./parser/user.js"

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

export type Version = {
  gen: Generation
  patch: Patch
}

export function inferInitialForme({ gen, patch }: Version, forme: string) {
  return forme in patch ? forme : gen.species.get(forme)!.baseSpecies
}

export function getPotentialPresets({ gen, patch }: Version, initialForme: string) {
  const presets = [...patch[initialForme].presets]

  if (initialForme === "Greninja") presets.push(...patch["Greninja-Bond"].presets)

  return presets
}

export function matchesPreset(preset: Preset, user: User) {
  const {
    init: { ability, item, moveSet },
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
