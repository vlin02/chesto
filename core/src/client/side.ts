import { Hazard, Screen, DelayedMove } from "../battle.js"
import { AllyUser, FoeUser, User } from "./user.js"

export type POV = "ally" | "foe"

export const POVS = ["ally", "foe"] as const
export const OPP = { ally: "foe", foe: "ally" } as const

export type SideEffects = {
  [k in string]: {
    layers?: number
    turn?: number
  }
}

export type SideEffect = Hazard | Screen

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
  slots: AllyUser[]
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

export type Party = Ally | Foe
