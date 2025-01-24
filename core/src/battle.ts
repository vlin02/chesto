export type WeatherName = "Snow" | "SunnyDay" | "SandStorm" | "RainDance"

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

export type StatusId = "slp" | "psn" | "brn" | "frz" | "par" | "tox"

export const CHOICE_ITEMS = ["Choice Band", "Choice Specs", "Choice Scarf"]

export type BoostId = "atk" | "def" | "spa" | "spd" | "spe" | "evasion" | "accuracy"

export const STAT_IDS = ["slp", "psn", "brn", "frz", "par", "tox"]
export type StatId = (typeof STAT_IDS)[number]
