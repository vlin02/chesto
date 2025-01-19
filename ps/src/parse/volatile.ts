import { TypeName } from "@pkmn/data"
import { BoostId, Gender, StatId } from "./dex.js"
import { User } from "./user.js"
import { MoveSlot } from "./move.js"

export type Boosts = {
  [k in BoostId]?: number
}

export type DelayedAttack = {
  turn: number
  user: User
}

export type Volatiles = {
  [k: string]: { turn?: number; singleMove?: boolean; singleTurn?: boolean }
} & {
  "Recharge"?: { turn: number }
  "Yawn"?: {}
  "Taunt"?: {}
  "Type Change"?: {
    types: TypeName[]
  }
  "Disable"?: {
    turn: number
    move: string
  }
  "Transform"?: {
    into: User
    ability: string | null
    moveset: {
      [k: string]: MoveSlot
    }
    boosts: Boosts
    gender: Gender
    species: string
  }
  "Choice Locked"?: {
    move: string
  }
  "Locked Move"?: {
    move: string
    turn: number
  }
  "Protosynthesis"?: {
    statId: StatId
  }
  "Quark Drive"?: {
    statId: StatId
  }
  "Fallen"?: {
    count: number
  }
  "Encore"?: {
    turn: number
    move: string
  }
  "Future Sight"?: DelayedAttack
  "Doom Desire"?: DelayedAttack
}
