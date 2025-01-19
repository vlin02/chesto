import { Generation, TypeName } from "@pkmn/data"
import { StatusId, Gender, StatId } from "./dex.js"
import { getMaxPP, MoveSet } from "./move.js"
import { Boosts, Volatiles } from "./volatile.js"
import { Member, parseHp, parseLabel, parseTraits, Traits } from "./protocol.js"

export type Flags = {
  "Battle Bond"?: boolean
  "Intrepid Sword"?: boolean
  "Illusion revealed"?: boolean
}

export type Status = {
  id: StatusId
  turn?: number
  move?: number
}

export class AllyUser {
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
  moveSet: MoveSet
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
    { ident, details, condition, stats, item, moves, ability, teraType }: Member
  ) {
    const { species } = parseLabel(ident)
    const { gender, lvl, forme } = parseTraits(details)

    if (species === "Ditto") {
      moves = ["Transform"]
      ability = "Imposter"
    }

    this.moveSet = {}
    for (const name of moves) {
      this.moveSet[name] = {
        used: 0,
        max: getMaxPP(gen.moves.get(name)!)
      }
    }

    this.pov = "ally"
    this.species = species
    this.flags = {}
    this.forme = forme
    this.gender = gender
    this.lvl = lvl
    this.revealed = false
    this.teraType = teraType
    this.ability = gen.abilities.get(ability)!.name
    this.item = item ? gen.items.get(item)!.name : "Leftovers"
    this.stats = stats
    this.hp = parseHp(condition)!
    this.volatiles = {}
    this.boosts = {}
    this.tera = false
  }

  setAbility(v: string) {
    this.ability = v
  }

  setItem(v: string | null) {
    this.item = v
  }
}

export class FoeUser {
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
  moveSet: MoveSet
  flags: Flags
  lastMove?: string
  lastBerry?: {
    name: string
    turn: number
  }
  volatiles: Volatiles
  boosts: Boosts
  tera: boolean
  clone: () => FoeUser

  constructor(gen: Generation, species: string, traits: Traits) {
    this.clone = () => {
      return new FoeUser(gen, species, traits)
    }

    const { forme, lvl, gender } = traits

    this.pov = "foe"
    this.species = species
    this.forme = forme
    this.lvl = lvl
    this.gender = gender
    this.hp = [100, 100]
    this.moveSet = {}
    this.initial = {
      formeId: gen.species.get(forme)!.id
    }
    this.volatiles = {}
    this.boosts = {}
    this.tera = false
    this.flags = {}
  }

  setAbility(v: string) {
    const { initial } = this

    this.ability = v
    initial.ability = initial.ability ?? v
  }

  setItem(v: string | null) {
    const { initial } = this

    this.item = v
    initial.item = initial.item ?? v ?? undefined
  }
}

export type User = AllyUser | FoeUser
