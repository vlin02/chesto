import { Hazard, Screen, DelayedMove } from "../battle.js"
import { User } from "./user.js"

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
  effects: SideEffects
  active: User
  team: { [k: string]: User }
  wish?: number
  slots: User[]
  teraUsed?: boolean
  isReviving?: boolean
}

export type Foe = {
  delayedAttack?: DelayedAttack
  effects: SideEffects
  active: User
  team: { [k: string]: User }
  wish?: number
  teraUsed?: boolean
  isReviving?: boolean
}

export type Team = Ally | Foe
