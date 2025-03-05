import { Generation, TypeName } from "@pkmn/data"
import { inferMaxPP } from "./move.js"
import { Boosts, Gender, PARTIAL_TRAPPING_MOVES, StatId, StatusId } from "../battle.js"
import { Member } from "./request.js"
import { Label } from "./protocol.js"

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
    | "Recharge"
    | "Magnet Rise"
    | "Magma Storm"
    | "Infestation"
    | "Whirlpool"]?: { turn: number }
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
    | "Trapped"
    | "Pressure"]?: {}
} & {
  [k in "Destiny Bond" | "GlaiveRush"]?: { singleMove: true }
} & {
  [k in "Roost" | "Protect" | "Beak Blast" | "Focus Punch"]?: { singleTurn: true }
} & {
  "Partially Trapped"?: {
    move: "Magma Storm" | "Whirlpool" | "Infestation"
    turn: number
  }
  "Prepare"?: {
    move: string
    turn: number
  }
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
    firstTurn: number
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
  temporary: boolean
  ability?: string
}

export class User {
  pov: "ally" | "foe"
  lvl: number
  hp: [number, number]
  item: string | null
  gen: Generation
  revealed: boolean
  status?: Status
  formeChange?: FormeChange
  species: string
  stats?: {
    [k in "atk" | "def" | "spa" | "spd" | "spe"]: number
  }
  init: {
    item?: string | null
    ability?: string
    forme: string
    moveSet: MoveSet
    gender: Gender
  }
  teraType?: TypeName
  flags: Flags
  lastMove?: string
  lastBerry?: LastBerry
  volatiles: Volatiles
  boosts: Boosts
  tera: boolean
  clone: () => User

  constructor(
    gen: Generation,
    option: { pov: "ally"; member: Member } | { pov: "foe"; species: string; label: Label }
  ) {
    this.clone = () => {
      return new User(gen, option)
    }

    if (option.pov === "ally") {
      const { member } = option

      let {
        health,
        species,
        label: { forme, gender, lvl },
        stats,
        baseAbility,
        item,
        moves,
        teraType
      } = member
      item = item ? gen.items.get(item)!.name : null

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
      this.item = item
      this.init = {
        item,
        forme,
        moveSet: Object.fromEntries(
          moves.map((id) => {
            const { name } = gen.moves.get(id)!
            return [name, { used: 0, max: inferMaxPP(gen, name) }]
          })
        ),
        gender,
        ability: gen.abilities.get(baseAbility)!.name
      }
      this.gen = gen
      this.hp = hp!
      this.volatiles = {}
      this.boosts = {}
      this.tera = false
      this.stats = stats
    } else {
      const { species, label } = option
      const { forme, lvl, gender } = label

      this.revealed = true
      this.item = null
      this.species = species
      this.pov = "foe"
      this.lvl = lvl
      this.hp = [100, 100]
      this.volatiles = {}
      this.boosts = {}
      this.flags = {}
      this.tera = false
      this.gen = gen
      this.init = {
        forme,
        moveSet: {},
        gender,
        ability: {
          "Calyrex-Ice": "As One (Glastrier)",
          "Calyrex-Shadow": "As One (Spectrier)"
        }[forme]
      }
    }
  }

  get moveSet() {
    const { volatiles, init: base } = this
    return (volatiles["Transform"] ?? base).moveSet
  }

  get forme() {
    const { formeChange, init: base } = this
    return formeChange?.forme ?? base.forme
  }

  get ability() {
    const { volatiles, init: base, formeChange } = this
    return (
      (volatiles["Trace"] ?? volatiles["Transform"])?.ability ??
      formeChange?.ability ??
      base.ability
    )
  }

  get gender() {
    const { volatiles, init: base } = this
    return (volatiles["Transform"] ?? base).gender
  }

  get trapped() {
    const { volatiles, defensiveTyping: types } = this

    if (volatiles["Recharge"] || volatiles["Prepare"] || volatiles["Locked Move"]) return true

    if (types.includes("Ghost")) return false

    if (
      volatiles["Trapped"] ||
      volatiles["No Retreat"] ||
      PARTIAL_TRAPPING_MOVES.some((k) => volatiles[k])
    )
      return true

    return false
  }

  get types() {
    const {
      volatiles: { "Type Change": typeChange },
      gen,
      forme
    } = this
    const { types } = typeChange ?? gen.species.get(forme)!

    return types
  }

  get defensiveTyping() {
    const { tera, teraType, types } = this
    return tera ? [teraType!] : types
  }

  get offensiveTyping() {
    const { tera, types, teraType } = this

    const typing: { [k in TypeName]?: number } = Object.fromEntries(types.map((t) => [t, 1]))
    if (tera) {
      typing[teraType!] = (typing[teraType!] ?? 0) + 1
    }

    return typing
  }
}
