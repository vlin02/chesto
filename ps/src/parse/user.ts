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
    into: User
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
  base: {
    forme: string
    ability: string
    moveSet: MoveSet
  }
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
    { ident, details, condition, stats, item, moves: moveIds, baseAbility, teraType }: Member
  ) {
    const { species } = parseLabel(ident)
    const { gender, lvl, forme } = parseTraits(details)

    if (species === "Ditto") {
      moveIds = ["transform"]
      baseAbility = "Imposter"
    }

    const moveSet: MoveSet = {}
    for (const id of moveIds) {
      moveSet[gen.moves.get(id)!.name] = {
        used: 0,
        max: getMaxPP(gen.moves.get(id)!)
      }
    }

    this.pov = "ally"
    this.species = species
    this.flags = {}
    this.gender = gender
    this.lvl = lvl
    this.revealed = false
    this.teraType = teraType!
    this.base = {
      ability: gen.abilities.get(baseAbility)!.name,
      forme,
      moveSet
    }
    this.item = item ? gen.items.get(item)!.name : "Leftovers"
    this.stats = stats
    this.hp = parseHp(condition)!
    this.volatiles = {}
    this.boosts = {}
    this.tera = false
  }

  get forme() {
    const {
      formeChange,
      base: { forme }
    } = this
    return formeChange?.name ?? forme
  }

  get ability() {
    const {
      volatiles,
      base: { ability }
    } = this
    return volatiles["Trace"]?.ability ?? volatiles["Transform"]?.ability ?? ability
  }

  get moveSet() {
    const {
      volatiles,
      base: { moveSet }
    } = this
    return volatiles["Transform"]?.moveSet ?? moveSet
  }
}

export class FoeUser {
  pov: "foe"
  lvl: number
  species: string
  gender: Gender
  hp: [number, number]
  item?: string | null
  formeChange?: FormeChange
  base: {
    forme: string
    moveSet: MoveSet
    ability?: string
    item?: string
  }
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
    this.base = {
      forme,
      moveSet: {}
    }
    this.volatiles = {}
    this.boosts = {}
    this.flags = {}
    this.tera = false
  }

  get ability() {
    const {
      volatiles,
      base: { ability }
    } = this

    return volatiles["Trace"]?.ability ?? volatiles["Transform"]?.ability ?? ability
  }

  get forme() {
    const {
      formeChange,
      base: { forme }
    } = this
    return formeChange?.name ?? forme
  }

  get moveSet() {
    const {
      volatiles,
      base: { moveSet }
    } = this
    return volatiles["Transform"]?.moveSet ?? moveSet
  }
}

export type User = AllyUser | FoeUser
