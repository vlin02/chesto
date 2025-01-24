import { TypeName } from "@pkmn/data"

export type Preset = {
  role: string
  abilities: string[]
  movepool: string[]
  teraTypes?: TypeName[]
}

export type Version = {
  commit: string
  timestamp: number
  patch: {
    [k: string]: {
      name: string
      level: number
      presets: Preset[]
    }
  }
}
