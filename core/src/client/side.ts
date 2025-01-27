import { AllyUser, FoeUser, User } from "./user.js"

export type POV = "ally" | "foe"

export const POVS = ["ally", "foe"] as const
export const OPP = { ally: "foe", foe: "ally" } as const

export const HAZARDS = ["Sticky Web", "Toxic Spikes", "Stealth Rock", "Spikes"] as const
export type Hazard = (typeof HAZARDS)[number]

export const SCREENS = ["Aurora Veil", "Light Screen", "Tailwind", "Reflect"] as const
export type Screen = (typeof SCREENS)[number]

export type SideEffects = {
  [k in string]: {
    layers?: number
    turn?: number
  }
}

export type SideEffect = Hazard | Screen

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
  effects: SideEffects
  active: AllyUser
  team: { [k: string]: AllyUser }
  wish?: number
}

export type Foe = {
  delayedAttack?: DelayedAttack
  turnMoves: number
  teraUsed: boolean
  effects: SideEffects
  active: FoeUser
  team: { [k: string]: FoeUser }
  wish?: number
}

export type Side = Ally | Foe
