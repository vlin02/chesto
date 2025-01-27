import { AllyUser, FoeUser, User } from "./user.js"

export type POV = "ally" | "foe"

export const POVS = ["ally", "foe"] as const
export const OPP = { ally: "foe", foe: "ally" } as const

export type Hazards = {
  [k: string]: {
    turn?: number
    layers?: number
  }
}

export const DELAYED_MOVES = ["Future Sight", "Doom Desire"]
export type DelayedMove = (typeof DELAYED_MOVES)[number]

export type DelayedAttack = {
  move: DelayedMove
  turn: number
  user: User
}

export type Ally = {
  delayedAttack?: DelayedAttack
  turnMoves: number
  teraUsed: boolean
  hazards: Hazards
  active: AllyUser
  team: { [k: string]: AllyUser }
  wish?: number
}

export type Foe = {
  delayedAttack?: DelayedAttack
  turnMoves: number
  teraUsed: boolean
  hazards: Hazards
  active: FoeUser
  team: { [k: string]: FoeUser }
  wish?: number
}

export const HAZARDS = ["Sticky Web", "Toxic Spikes", "Stealth Rock", "Spikes"]

export type Side = Ally | Foe
