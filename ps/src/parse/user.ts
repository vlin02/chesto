import { Generation, TypeName } from "@pkmn/data"
import { StatusId, Gender, BoostId, StatId } from "./species.js"
import { Member, parseCondition, parseLabel, parseTraits, Traits } from "./protocol.js"
import { getMaxPP } from "./move.js"

export type Boosts = {
  [k in BoostId]?: number
}

export type MoveSlot = {
  used: number
  max: number
}

export type MoveSet = { [k: string]: MoveSlot }

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
  "Pressure"?: {}
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
  reverts: boolean
}

export class AllyUser {
  pov: "ally"
  gen: Generation
  revealed: boolean
  lvl: number
  hp: [number, number]
  formeChange?: FormeChange
  item: string | null
  firstItem?: string | null
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
    { ident, details, condition, stats, item, moves, baseAbility, teraType }: Member
  ) {
    const { species } = parseLabel(ident)
    const { gender, lvl, forme } = parseTraits(details)
    const { hp } = parseCondition(condition)

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
    const { volatiles, base } = this
    return (volatiles["Trace"] ?? volatiles["Transform"] ?? base).ability
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
  firstItem?: string | null
  formeChange?: FormeChange
  status?: Status
  flags: Flags
  species: string
  gen: Generation
  base: {
    forme: string
    moveSet: MoveSet
    gender: Gender
    ability?: string
  }
  lastMove?: string
  lastBerry?: {
    name: string
    turn: number
  }
  volatiles: Volatiles
  boosts: Boosts
  clone: () => FoeUser
  tera: boolean
  teraType?: TypeName

  constructor(gen: Generation, species: string, traits: Traits) {
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
    const { volatiles, base } = this
    return (volatiles["Trace"] ?? volatiles["Transform"] ?? base).ability
  }

  get gender() {
    const { volatiles, base } = this
    return (volatiles["Transform"] ?? base).gender
  }
}

export type User = AllyUser | FoeUser
