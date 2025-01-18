import { TypeName } from "@pkmn/data"
import { StatusId, Gender, StatId, BoostId } from "./dex.js"

export type POV = "ally" | "foe"

export const POVS = ["ally", "foe"] as const
export const OPP = { ally: "foe", foe: "ally" } as const

export type Status = {
  id: StatusId
  turn?: number
  move?: number
}

export type MoveSet = {
  [k: string]: number
}

export type Boosts = {
  [k in BoostId]?: number
}

export type Flags = {
  "Battle Bond"?: boolean
  "Intrepid Sword"?: boolean
  "Illusion revealed"?: boolean
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
    moveset: MoveSet
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

type LastMove = {
  name: string
  tookDamage?: boolean
  missed?: boolean
  failed?: boolean
}

export type AllyUser = {
  pov: "ally"
  species: string
  revealed: boolean
  lvl: number
  forme: string
  gender: Gender
  hp: [number, number]
  ability: string
  item: string | null
  stats: { [k in StatId]: number }
  status?: Status
  moveset: MoveSet
  teraType: TypeName
  flags: Flags
  lastMove?: LastMove
  lastBerry?: {
    name: string
    turn: number
  }
  volatiles: Volatiles
  boosts: Boosts
  tera: boolean
}

export type FoeUser = {
  pov: "foe"
  lvl: number
  species: string
  forme: string
  gender: Gender
  hp: [number, number]
  ability?: string
  item?: string | null
  initial: {
    formeId: string
    ability?: string
    item?: string
  }
  status?: Status
  moveset: MoveSet
  flags: Flags
  lastMove?: LastMove
  lastBerry?: {
    name: string
    turn: number
  }
  volatiles: Volatiles
  boosts: Boosts
  tera: boolean
}

export type User = AllyUser | FoeUser

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
