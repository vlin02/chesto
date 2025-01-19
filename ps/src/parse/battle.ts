export const HAZARDS = ["Sticky Web", "Toxic Spikes", "Stealth Rock", "Spikes"]

export type WeatherName = "Snow" | "SunnyDay" | "SandStorm" | "RainDance"

export const TERRAIN_NAMES = [
  "Electric Terrain",
  "Psychic Terrain",
  "Grassy Terrain",
  "Misty Terrain"
]
export type TerrainName = (typeof TERRAIN_NAMES)[number]
