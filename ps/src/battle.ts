export type Side = "p1" | "p2"
export const SIDES = ["p1", "p2"] as const

export const FOE = { p1: "p2", p2: "p1" } as const

export type StatId = "atk" | "def" | "spa" | "spd" | "spe"
export type BoostId = "atk" | "def" | "spa" | "spd" | "spe" | "evasion" | "accuracy"

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
  | "Stellar"