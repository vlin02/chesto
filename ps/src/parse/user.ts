import { Generation, Move, TypeName } from "@pkmn/data"
import { StatusId, Gender, BoostId, StatId } from "./species.js"
import { Member, parseHp, parseLabel, parseTraits, Traits } from "./protocol.js"

export function getMaxPP({ noPPBoosts, pp }: Move) {
  return noPPBoosts ? pp : Math.floor(pp * 1.6)
}

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
    species: string
  }
  "Choice Locked"?: {
    name: string
  }
  "Locked Move"?: {
    move: string
    attempt: number
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
  "Battle Bond"?: boolean
  "Intrepid Sword"?: boolean
  "Illusion revealed"?: boolean
}

export type Status = {
  id: StatusId
  turn?: number
  attempt?: number
}

export type FormeChange = {
  name: string
  reverts: boolean
}

export class AllyUser {
  pov: "ally"
  species: string
  revealed: boolean
  lvl: number
  gender: Gender
  hp: [number, number]
  formeChange?: FormeChange
  forme: string
  ability: string
  moveSet: MoveSet
  item: string | null
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

    if (species === "Ditto") {
      moves = ["transform"]
      baseAbility = "Imposter"
    }

    this.pov = "ally"
    this.species = species
    this.flags = {}
    this.gender = gender
    this.lvl = lvl
    this.revealed = false
    this.teraType = teraType!

    this.ability = gen.abilities.get(baseAbility)!.name
    this.forme = forme
    this.moveSet = Object.fromEntries(
      moves.map((id) => {
        const move = gen.moves.get(id)!
        return [move.name, { used: 0, max: getMaxPP(move) }]
      })
    )

    this.item = item ? gen.items.get(item)!.name : "Leftovers"
    this.stats = stats
    this.hp = parseHp(condition)!
    this.volatiles = {}
    this.boosts = {}
    this.tera = false
  }

  get current() {
    let { volatiles, ability, gender, species, forme, moveSet } = this
    const { Transform: transform, Trace: trace } = volatiles

    const base = transform ?? {
      ability,
      moveSet,
      gender,
      forme,
      species
    }

    if (trace) base.ability = trace.ability

    return base
  }
}

export class FoeUser {
  pov: "foe"
  lvl: number
  species: string
  gender: Gender
  hp: [number, number]
  item?: string | null
  ability?: string
  forme: string
  moveSet: MoveSet
  formeChange?: FormeChange
  status?: Status
  flags: Flags
  lastMove?: string
  lastBerry?: {
    name: string
    turn: number
  }
  volatiles: Volatiles
  boosts: Boosts
  clone: () => FoeUser
  tera: boolean

  constructor(gen: Generation, species: string, traits: Traits) {
    this.clone = () => {
      return new FoeUser(gen, species, traits)
    }

    const { forme, lvl, gender } = traits

    this.pov = "foe"
    this.species = species
    this.lvl = lvl
    this.gender = gender
    this.hp = [100, 100]
    this.forme = forme
    this.volatiles = {}
    this.boosts = {}
    this.flags = {}
    this.tera = false
    this.moveSet = {}
  }

  get current() {
    let { volatiles, ability, gender, species, forme, moveSet } = this
    const { Transform: transform, Trace: trace } = volatiles

    const base = transform ?? {
      ability,
      moveSet,
      gender,
      forme,
      species
    }

    if (trace) base.ability = trace.ability

    return base
  }
}

export type User = AllyUser | FoeUser
