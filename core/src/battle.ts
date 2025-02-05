export const WEATHER_NAMES = ["Snow", "SunnyDay", "SandStorm", "RainDance"] as const
export type WeatherName = (typeof WEATHER_NAMES)[number]

export const TERRAIN_NAMES = [
  "Electric Terrain",
  "Psychic Terrain",
  "Grassy Terrain",
  "Misty Terrain"
]
export type TerrainName = (typeof TERRAIN_NAMES)[number]

export type Gender = "M" | "F" | null

export type TypeName =
  | "Normal"
  | "Fighting"
  | "Flying"
  | "Poison"
  | "Ground"
  | "Rock"
  | "Bug"
  | "Ghost"
  | "Steel"
  | "Fire"
  | "Water"
  | "Grass"
  | "Electric"
  | "Psychic"
  | "Ice"
  | "Dragon"
  | "Dark"
  | "Fairy"
  | "???"
  | "Stellar"

export const STATUS_IDS = ["slp", "psn", "brn", "frz", "par", "tox"] as const
export type StatusId = (typeof STATUS_IDS)[number]

export const CHOICE_ITEMS = ["Choice Band", "Choice Specs", "Choice Scarf"]

export const BOOST_IDS = ["atk", "def", "spa", "spd", "spe", "evasion", "accuracy"]
export type BoostId = (typeof BOOST_IDS)[number]

export type Boosts = {
  [k in BoostId]?: number
}

export const STAT_IDS = ["hp", "atk", "def", "spa", "spd", "spe"] as const
export type StatId = (typeof STAT_IDS)[number]

export type Stats = { [k in StatId]: number }

export const HAZARDS = ["Sticky Web", "Toxic Spikes", "Stealth Rock", "Spikes"] as const
export type Hazard = (typeof HAZARDS)[number]

export const SCREENS = ["Aurora Veil", "Light Screen", "Tailwind", "Reflect"] as const
export type Screen = (typeof SCREENS)[number]

export const DELAYED_MOVES = ["Future Sight", "Doom Desire"]
export type DelayedMove = (typeof DELAYED_MOVES)[number]

export const PARTIALLY_TRAPPED_MOVES = ["Magma Storm", "Infestation", "Whirlpool"]