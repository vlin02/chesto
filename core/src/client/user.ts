import { Generation, TypeName } from "@pkmn/data"
import { Label } from "./protocol.js"
import { getMaxPP } from "./move.js"
import { Boosts, Gender, StatId, StatusId } from "../battle.js"
import { Member } from "./request.js"

export type MoveSlot = {
  used: number
  max: number
}

export type MoveSet = { [k: string]: MoveSlot }

export type LastBerry = {
  name: string
  turn: number
}

export type Volatiles = {
  [k: string]: { turn?: number; singleMove?: boolean; singleTurn?: boolean }
} & {
  [k in
    | "Taunt"
    | "Yawn"
    | "Confusion"
    | "Throat Chop"
    | "Heal Block"
    | "Slow Start"
    | "Magnet Rise"]?: { turn: number }
} & {
  [k in
    | "Leech Seed"
    | "Charge"
    | "Attract"
    | "No Retreat"
    | "Salt Cure"
    | "Flash Fire"
    | "Leech Seed"
    | "Substitute"
    | "Pressure"]?: {}
} & {
  [k in "Destiny Bond" | "GlaiveRush"]?: { singleMove: true }
} & {
  [k in "Roost" | "Protect" | "Beak Blast" | "Focus Punch"]?: { singleTurn: true }
} & {
  "Type Change"?: {
    types: TypeName[]
  }
  "Disable"?: {
    turn: number
    move: string
  }
  "Trace"?: {
    ability: string
  }
  "Transform"?: {
    ability: string
    moveSet: MoveSet
    boosts: Boosts
    gender: Gender
    forme: string
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
}

export type Flags = {
  battleBond?: boolean
  intrepidSword?: boolean
  illusionRevealed?: boolean
}

export type Status = {
  id: StatusId
  turn?: number
  attempt?: number
}

export type FormeChange = {
  forme: string
  whileActiveOnly: boolean
  ability?: string
}

export class AllyUser {
  pov: "ally"
  gen: Generation
  revealed: boolean
  lvl: number
  hp: [number, number]
  formeChange?: FormeChange
  item: string | null
  species: string
  base: {
    forme: string
    moveSet: MoveSet
    gender: Gender
    ability: string
  }
  stats: { [k in StatId]: number }
  status?: Status
  teraType: TypeName
  flags: Flags
  lastMove?: string
  lastBerry?: {
    name: string
    turn: number
  }
  volatiles: Volatiles
  boosts: Boosts
  tera: boolean

  constructor(
    gen: Generation,
    {
      health,
      species,
      label: { forme, gender, lvl },
      stats,
      baseAbility,
      item,
      moves,
      teraType
    }: Member
  ) {
    const { hp } = health!

    if (species === "Ditto") {
      moves = ["transform"]
      baseAbility = "Imposter"
    }

    this.species = species
    this.pov = "ally"
    this.flags = {}
    this.lvl = lvl
    this.revealed = false
    this.teraType = teraType!
    this.base = {
      forme,
      moveSet: Object.fromEntries(
        moves.map((id) => {
          const { name } = gen.moves.get(id)!
          return [name, { used: 0, max: getMaxPP(gen, name) }]
        })
      ),
      gender,
      ability: gen.abilities.get(baseAbility)!.name
    }
    this.gen = gen
    this.item = item ? gen.items.get(item)!.name : null
    this.stats = stats
    this.hp = hp!
    this.volatiles = {}
    this.boosts = {}
    this.tera = false
  }

  get types() {
    const {
      teraType,
      tera,
      volatiles: { "Type Change": typeChange }
    } = this

    const types = new Set([...(typeChange ?? this.gen.species.get(this.forme)!).types])
    if (tera) types.add(teraType)
    return [...types]
  }

  get moveSet() {
    const { volatiles, base } = this
    return (volatiles["Transform"] ?? base).moveSet
  }

  get forme() {
    const { formeChange, base } = this
    return formeChange?.forme ?? base.forme
  }

  get ability() {
    const { volatiles, base, formeChange } = this
    return (
      (volatiles["Trace"] ?? volatiles["Transform"])?.ability ??
      formeChange?.ability ??
      base.ability
    )
  }

  get gender() {
    const { volatiles, base } = this
    return (volatiles["Transform"] ?? base).gender
  }
}

export class FoeUser {
  pov: "foe"
  lvl: number
  hp: [number, number]
  item?: string | null
  formeChange?: FormeChange
  status?: Status
  flags: Flags
  species: string
  gen: Generation
  base: {
    item?: string | null
    ability?: string | null
    forme: string
    moveSet: MoveSet
    gender: Gender
  }
  lastMove?: string
  lastBerry?: LastBerry
  volatiles: Volatiles
  boosts: Boosts
  clone: () => FoeUser
  tera: boolean
  teraType?: TypeName

  constructor(gen: Generation, species: string, traits: Label) {
    this.clone = () => {
      return new FoeUser(gen, species, traits)
    }

    const { forme, lvl, gender } = traits

    this.species = species
    this.pov = "foe"
    this.lvl = lvl
    this.hp = [100, 100]
    this.volatiles = {}
    this.boosts = {}
    this.flags = {}
    this.tera = false
    this.gen = gen
    this.base = {
      forme,
      moveSet: {},
      gender
    }
  }

  clear() {
    const { volatiles, boosts, lastBerry, lastMove, formeChange } = this
    const recover = { volatiles, boosts, lastBerry, lastMove, formeChange }

    this.volatiles = {}
    this.boosts = {}
    delete this.lastBerry
    delete this.lastMove

    if (formeChange?.whileActiveOnly) delete this.formeChange

    return recover
  }

  get types() {
    const {
      teraType,
      tera,
      volatiles: { "Type Change": typeChange }
    } = this

    const types = new Set([...(typeChange ?? this.gen.species.get(this.forme)!).types])
    if (tera) types.add(teraType!)

    return [...types]
  }

  get moveSet() {
    const { volatiles, base } = this
    return (volatiles["Transform"] ?? base).moveSet
  }

  get forme() {
    const { formeChange, base } = this
    return formeChange?.forme ?? base.forme
  }

  get ability() {
    const { volatiles, base, formeChange } = this
    return (
      (volatiles["Trace"] ?? volatiles["Transform"])?.ability ??
      formeChange?.ability ??
      base.ability
    )
  }

  get gender() {
    const { volatiles, base } = this
    return (volatiles["Transform"] ?? base).gender
  }
}

export type User = AllyUser | FoeUser

type Temp = {
  volatiles: Volatiles
  boosts: Boosts
  lastBerry: undefined
  lastMove: string | undefined
  formeChange: FormeChange | undefined
}

export function clear(user: User) {
  const { volatiles, boosts, lastBerry, lastMove, formeChange } = user
  const recover = { volatiles, boosts, lastBerry, lastMove, formeChange }

  user.volatiles = {}
  user.boosts = {}
  delete user.lastBerry
  delete user.lastMove

  if (formeChange?.whileActiveOnly) delete user.formeChange

  return recover
}

export function recover(user: User, { volatiles, boosts, lastBerry, lastMove, formeChange }: Temp) {
  user.volatiles = volatiles
  user.boosts = boosts
  user.lastBerry = lastBerry
  user.lastMove = lastMove
  user.formeChange = formeChange
}
