import { Stats, TypeName } from "./battle.js"

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
  hash: string
  timestamp: number
  patch: Patch
}
