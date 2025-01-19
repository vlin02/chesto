import { AllyUser, FoeUser } from "./user.js"

export type POV = "ally" | "foe"

export const POVS = ["ally", "foe"] as const
export const OPP = { ally: "foe", foe: "ally" } as const

export type Fields = {
  [k: string]: {
    turn?: number
    layers?: number
  }
}

export type Ally = {
  fields: Fields
  active: AllyUser
  team: { [k: string]: AllyUser }
  wish?: number
}

export type Foe = {
  fields: Fields
  active: FoeUser
  team: { [k: string]: FoeUser }
  wish?: number
}

export type Party = Ally | Foe
