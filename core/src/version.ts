import { TypeName } from "@pkmn/data"

export type Preset = {
  role: string
  abilities: string[]
  movepool: string[]
  teraTypes?: TypeName[]
  derived: {
    moves: string[]
  }
}

export type Version = {
  hash: string
  timestamp: number
  patch: {
    [k: string]: {
      name: string
      level: number
      presets: Preset[]
    }
  }
}
