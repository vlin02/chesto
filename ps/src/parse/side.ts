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
  turnMoves: number
  teraUsed: boolean
  tera?: AllyUser
  fields: Fields
  active: AllyUser
  team: { [k: string]: AllyUser }
  wish?: number
}

export type Foe = {
  turnMoves: number
  teraUsed: boolean
  tera?: FoeUser
  fields: Fields
  active: FoeUser
  team: { [k: string]: FoeUser }
  wish?: number
}

export const HAZARDS = ["Sticky Web", "Toxic Spikes", "Stealth Rock", "Spikes"]

export type Side = Ally | Foe
