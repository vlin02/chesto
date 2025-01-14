import { Generation, TypeName } from "@pkmn/data"
import {
  BoostId,
  Gender,
  parseEffect,
  parseHp,
  parseLabel,
  parseTags,
  parseTraits,
  parseTypes,
  piped,
  Side,
  StatId,
  StatusId,
  WeatherName
} from "./proto.js"

export const POVS = ["ally", "foe"] as const

type Status = {
  id: StatusId
  turn?: number
  moves?: number
}

type MoveSet = {
  [k: string]: number
}

type Eaten = {
  name: string
  turn: number
}

type Used = {
  "Battle Bond"?: boolean
  "Intrepid Sword"?: boolean
}

type AllyMember = {
  pov: "ally"
  species: string
  revealed: boolean
  lvl: number
  forme: string
  gender: Gender
  hp: [number, number]
  ability: string | null
  item: string | null
  stats: { [k in StatId]: number }
  status?: Status
  moveset: MoveSet
  teraType: TypeName
  used: Used
}

export type FoeMember = {
  pov: "foe"
  lvl: number
  species: string
  forme: string
  gender: Gender
  hp: [number, number]
  ability?: string | null
  item?: string | null
  initial: {
    formeId: string
    ability?: string
    item?: string
  }
  status?: Status
  moveset: MoveSet
  used: Used
}

type DelayedAttack = {
  turn: number
  member: Member
}

type Volatiles = { [k: string]: { turn?: number; singleMove?: boolean; singleTurn?: boolean } } & {
  "Type Change"?: {
    types: TypeName[]
  }
  "Disable"?: {
    turn: number
    move: string
  }
  "Locked Move"?: {
    turn: number
    move: string
  }
  "Protosynthesis"?: {
    stat: StatId
  }
  "Quark Drive"?: {
    stat: StatId
  }
  "Fallen"?: {
    count: number
  }
  "Future Sight"?: DelayedAttack
  "Doom Desire"?: DelayedAttack
}

type Tera = {
  species: string
  type: TypeName
}

type Condition = {
  turn?: number
  layers?: number
}

type Ally = {
  tera: Tera | null
  conditions: { [k: string]: Condition }
  active?:  {
    member: AllyMember
    volatiles: Volatiles
    lastBerry?: Eaten
    boosts: {
      [k in BoostId]?: number
    }
  }
  team: { [k: string]: AllyMember }
  wish?: number
}

type Foe = {
  tera: Tera | null
  conditions: { [k: string]: Condition }
  active?: {
    member: FoeMember
    volatiles: Volatiles
    lastBerry?: Eaten
    boosts: {
      [k in BoostId]?: number
    }
  }
  team: { [k: string]: FoeMember }
  wish?: number
}

const OPP = { ally: "foe", foe: "ally" } as const

type POV = "ally" | "foe"

const HAZARDS = new Set(["Sticky Web", "Toxic Spikes", "Stealth Rock", "Spikes"])
function isHazard(name: string) {
  return name in HAZARDS
}

type Member = AllyMember | FoeMember

function hasItem(memb: Member, item: string | null) {
  memb.item = item
  if (memb.pov === "foe" && item) {
    const { initial } = memb
    initial.item = initial.item ?? item
  }
}

function hasAbility(memb: Member, ability: string | null) {
  memb.ability = ability
  if (memb.pov === "foe" && ability) {
    const { initial } = memb
    initial.ability = initial.ability ?? ability
  }
}

export type MemberId = {
  pov: POV
  species: string
}

export class Observer {
  side: Side

  ally: Ally
  foe: Foe
  name: string
  gen: Generation

  turn: number
  fields: { [k: string]: number }
  weather: { name: WeatherName; turn: number } | null
  winner?: POV

  constructor(side: Side, gen: Generation) {
    this.gen = gen
    this.side = side
    this.ally = { tera: null, team: {}, conditions: {} }
    this.foe = { tera: null, team: {}, conditions: {} }
    this.weather = null
    this.fields = {}
    this.turn = 0
  }

  pov(side: Side): POV {
    return this.side === side ? "ally" : "foe"
  }

  active(pov: POV) {
    return this[pov].active!
  }

  member(s: string) {
    const { side, species } = parseLabel(s)
    return this[side === this.side ? "ally" : "foe"].team[species]
  }

  read(line: string) {
    if (line[0] !== "|") return

    let p
    p = piped(line, 0)
    const msgType = p.args[0]

    switch (msgType) {
      case "request": {
        if (!this.name) {
          const {
            side: { pokemon: team, name }
          }: {
            side: {
              name: string
              pokemon: {
                ident: string
                details: string
                condition: string
                active: boolean
                stats: { [k in StatId]: number }
                item: string
                ability: string
                moves: string[]
                teraType: TypeName
              }[]
            }
          } = JSON.parse(line.slice(p.i + 1))

          this.name = name

          const { ally } = this
          for (const { ident, details, condition, stats, item, moves, ability, teraType } of team) {
            const { species } = parseLabel(ident)
            const { gender, lvl, forme } = parseTraits(details)

            const moveset: MoveSet = {}
            for (const move of moves) moveset[this.gen.moves.get(move)!.name] = 0

            ally.team[species] = {
              used: {},
              pov: "ally",
              species: this.gen.species.get(species)!.name,
              forme,
              gender,
              lvl,
              revealed: false,
              teraType,
              ability: this.gen.abilities.get(ability)!.name,
              item: this.gen.items.get(item)!.name,
              stats,
              hp: parseHp(condition)!,
              moveset
            }
          }
        }

        break
      }
      case "-ability": {
        p = piped(line, p.i, 2)
        const dest = this.member(p.args[0])
        const ability = p.args[1]

        {
          p = piped(line, p.i, -1)
          const { from, of } = parseTags(p.args)
          const { ability: prev } = parseEffect(from)

          if (prev === "Trace") {
            hasAbility(dest, prev)
            hasAbility(this.member(of), ability)
          }
        }

        if (ability === "Intrepid Sword") {
          dest.used["Intrepid Sword"] = true
        }

        hasAbility(dest, ability)
        break
      }
      /*
      [from] Shed Tail: transfers substitute
      */
      case "switch":
      case "drag": {
        p = piped(line, p.i, 2)
        const { side, species } = parseLabel(p.args[0])
        const pov = this.pov(side)

        const traits = parseTraits(p.args[1])
        p = piped(line, p.i, -1)

        const { from } = parseTags(p.args)

        if (pov === "ally") {
          this[pov].team[species].revealed = true
        }

        if (pov === "foe") {
          const { team } = this[pov]
          const { forme, lvl, gender } = traits
          if (!(species in team)) {
            team[species] = {
              used: {},
              pov: "foe",
              species,
              forme,
              lvl,
              gender,
              hp: [100, 100],
              moveset: {},
              initial: {
                formeId: this.gen.species.get(forme)!.id
              }
            }
          }
        }

        const member: any = this[pov].team[species]
        const { status } = member
        if (status?.id === "tox") status.turn! = 0

        const prev = this[pov].active!
        const volatiles: Volatiles = {}

        if (from === "Shed Tail") {
          volatiles["Substitute"] = prev.volatiles["Substitute"]
        }

        this[pov].active = {member, volatiles, boosts: {}}

        break
      }
      case "-weather": {
        p = piped(line, p.i)
        const name = p.args[0] as WeatherName | "none"

        if (name === "none") {
          this.weather = null
          break
        }

        p = piped(line, p.i, -1)
        const { upkeep, from, of } = parseTags(p.args)
        const { ability } = parseEffect(from)

        if (upkeep === "") {
          this.weather!.turn++
          break
        }

        this.weather = { name, turn: 0 }
        if (ability) hasAbility(this.member(of), ability)

        break
      }
      case "-fieldstart": {
        p = piped(line, p.i)
        const { move: name } = parseEffect(p.args[0])

        this.fields[name!] = 0

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)

        const { ability } = parseEffect(from)
        if (ability) hasAbility(this.member(of), ability)
        break
      }
      case "-fieldend": {
        p = piped(line, p.i)
        const { move: field } = parseEffect(p.args[0])

        delete this.fields[field!]
        break
      }
      case "-status": {
        p = piped(line, p.i, 2)
        const dest = this.member(p.args[0])
        const id = p.args[1] as StatusId

        dest.status = {
          id,
          turn: id === "tox" ? 0 : undefined,
          moves: id === "slp" ? 0 : undefined
        }

        {
          p = piped(line, p.i, -1)

          const { from, of } = parseTags(p.args)

          const source = of ? this.member(of) : dest
          const { ability, item } = parseEffect(from)

          if (item) hasItem(source, item)
          if (ability) hasAbility(source, ability)
        }
        break
      }
      case "-curestatus": {
        p = piped(line, p.i, 2)
        const target = this.member(p.args[0])

        delete target.status

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)

        const { ability } = parseEffect(from)
        if (ability) hasAbility(target, ability)

        break
      }
      case "move": {
        p = piped(line, p.i, 3)
        const src = this.member(p.args[0])
        const { pov } = src
        const move = p.args[1]
        const opp = this[OPP[pov]]

        const { volatiles } = this.active(pov)
        const { moveset, status } = src

        {
          p = piped(line, p.i, -1)
          const { from, notarget, miss } = parseTags(p.args)

          for (const name in volatiles) {
            if (volatiles[name].singleMove) delete volatiles[name]
          }

          if (status?.moves) status.moves++

          const { ability } = parseEffect(from)
          if (ability) hasAbility(src, ability)

          if (miss === undefined) {
            switch (move) {
              case "Outrage": {
                if (from !== "lockedmove" && notarget === undefined) {
                  volatiles["Locked Move"] = { turn: 0, move: "Outrage" }
                }
                break
              }
              case "Wish": {
                this[pov].wish = 0
              }
            }
          } else {
            if (volatiles["Locked Move"]) delete volatiles["Locked Move"]
          }
        }

        moveset[move] =
          (moveset[move] ?? 0) + (opp.active!.member.ability === "Pressure" ? 2 : 1)
        break
      }
      case "-heal":
      case "-sethp": {
        p = piped(line, p.i, 2)
        const target = this.member(p.args[0])
        const hp = parseHp(p.args[1])

        target.hp = hp!

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)

        const { ability, item } = parseEffect(from)
        if (ability) hasAbility(target, ability)

        // berries already include an -enditem
        if (item === "Leftovers") hasItem(target, item)

        break
      }
      case "-damage": {
        p = piped(line, p.i, 2)

        const target = this.member(p.args[0])
        const hp = parseHp(p.args[1])

        if (hp) target.hp = hp
        else target.hp[0] = 0

        {
          p = piped(line, p.i, -1)

          const { from, of } = parseTags(p.args)

          const { item, ability } = parseEffect(from)
          const src = of ? this.member(of) : target

          if (ability) hasAbility(src, ability)
          if (item) hasItem(src, item)
        }

        break
      }
      case "-boost":
      case "-unboost": {
        p = piped(line, p.i, 3)
        const dest = this.member(p.args[0])
        const { boosts } = this.active(dest.pov)

        const id = p.args[1] as BoostId
        const n = Number(p.args[2])
        boosts[p.args[1] as BoostId] = Math.min(
          Math.max(-6, (boosts[id] ?? 0) + (msgType === "-boost" ? n : -n)),
          6
        )

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)
        const { item } = parseEffect(from)

        // boosts from item consume it
        if (item) {
          hasItem(dest, item)
          hasItem(dest, null)
        }

        break
      }
      case "-clearboost": {
        p = piped(line, p.i)
        const { pov } = this.member(p.args[0])
        this.active(pov).boosts = {}
        break
      }
      case "-clearallboost": {
        for (const pov of POVS) {
          this.active(pov).boosts = {}
        }
        break
      }
      case "-clearnegativeboost": {
        p = piped(line, p.i)
        const { pov } = this.member(p.args[0])
        const { boosts } = this.active(pov)

        for (const k in boosts) {
          const id = k as BoostId
          boosts[id] = Math.max(0, boosts[id]!)
        }
        break
      }
      case "-item": {
        p = piped(line, p.i, 2)
        const dest = this.member(p.args[0])
        const item = p.args[1]

        hasItem(dest, item)

        p = piped(line, p.i, -1)
        const { from, of, identify } = parseTags(p.args)
        const { ability } = parseEffect(from)

        const src = of ? this.member(of) : undefined

        if (identify) {
          hasAbility(src!, ability!)
          break
        }

        if (ability) hasAbility(dest, ability)

        // magician doesnt emit an -enditem
        if (ability === "Magician") {
          hasItem(src!, item)
          hasItem(src!, null)
        }
        break
      }
      case "-enditem": {
        p = piped(line, p.i, 2)
        const target = this.member(p.args[0])
        const { pov } = target
        const item = p.args[1]

        hasItem(target, item)
        hasItem(target, null)

        p = piped(line, p.i, -1)
        const { eat } = parseTags(p.args)

        if (eat != null) {
          this.active(pov).lastBerry = {
            name: item,
            turn: 0
          }
        }

        break
      }
      case "-start": {
        p = piped(line, p.i, 2)
        const dest = this.member(p.args[0])
        let { pov } = dest
        let { stripped: name } = parseEffect(p.args[1])

        const active = this.active(pov)
        const { volatiles } = active

        if (name.startsWith("quarkdrive")) {
          volatiles["Quark Drive"] = { stat: name.slice(-3) as StatId }
        } else if (name.startsWith("protosynthesis")) {
          volatiles["Protosynthesis"] = { stat: name.slice(-3) as StatId }
        } else if (name.startsWith("fallen")) {
          volatiles["Fallen"] = {
            count: Number(name.slice(-1)[0])
          }
        } else {
          name = { confusion: "Confusion", typechange: "Type Change" }[name] ?? name

          switch (name) {
            case "Type Change": {
              p = piped(line, p.i)

              volatiles[name] = {
                types: parseTypes(p.args[0])
              }
              break
            }
            case "Disable": {
              p = piped(line, p.i)
              const [move] = p.args

              volatiles[name] = {
                move,
                turn: 0
              }
              break
            }
            case "Future Sight":
            case "Doom Desire": {
              const {member, volatiles} = this[OPP[pov]].active!

              volatiles[name] = {
                turn: 0,
                member
              }
              break
            }
            case "Charge": {
              p = piped(line, p.i)
              volatiles[name] = {}
              break
            }
            case "Salt Cure":
            case "Flash Fire":
            case "Leech Seed":
            case "Substitute": {
              volatiles[name] = {}
              break
            }
            default: {
              volatiles[name] = { turn: 0 }
            }
          }
        }

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)

        const { ability, item } = parseEffect(from)
        const src = of ? this.member(of) : dest

        if (ability) hasAbility(src, ability)
        if (item) hasItem(src, item)

        break
      }
      case "-terastallize": {
        p = piped(line, p.i, 2)
        const { pov, species } = this.member(p.args[0])
        const type = p.args[1] as TypeName

        this[pov].tera = { species, type }
        break
      }
      case "-formechange": {
        p = piped(line, p.i, 2)
        const dest = this.member(p.args[0])
        const forme = p.args[1]

        dest.forme = forme

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)
        const { ability } = parseEffect(from)

        if (ability) hasAbility(dest, ability)

        break
      }
      case "detailschange": {
        p = piped(line, p.i, 2)
        const dest = this.member(p.args[0])
        const { forme } = parseTraits(p.args[1])

        dest.forme = forme
        break
      }
      case "-activate": {
        p = piped(line, p.i, 2)
        const dest = this.member(p.args[0])
        const { pov } = dest

        let { ability, item, move, stripped } = parseEffect(p.args[1])

        if (stripped === "Orichalcum Pulse") ability = stripped

        if (item) {
          switch (item) {
            case "Leppa Berry": {
              p = piped(line, p.i, 1)
              dest.moveset[p.args[0]] = 0
              break
            }
          }
        } else if (move) {
          switch (move) {
            case "Poltergeist": {
              p = piped(line, p.i)
              hasItem(dest, p.args[0])
              break
            }
            case "Magma Storm":
            case "Infestation":
            case "Whirlpool": {
              this.active(pov).volatiles["Partially Trapped"] = { turn: 0 }
              break
            }
          }
        } else if (ability) {
          if (ability === "Battle Bond") {
            dest.used["Battle Bond"] = true
          }

          hasAbility(dest, ability)
        } else {
          stripped = { trapped: "Trapped" }[stripped] ?? stripped

          switch (stripped) {
            case "Trapped":
              this[pov].active!.volatiles[stripped] = {}
              break
          }
        }

        break
      }
      case "-swapsideconditions": {
        const { ally, foe } = this
        ;[ally.conditions, foe.conditions] = [foe.conditions, ally.conditions]
        break
      }
      case "replace": {
        p = piped(line, p.i, 2)
        const { side, species } = parseLabel(p.args[0])
        const { forme, lvl, gender } = parseTraits(p.args[1])

        const {active, team} = this[this.pov(side)]
        const {member} = active!

        member.species = species
        member.forme = forme
        member.lvl = lvl
        member.gender = gender
        
        team[species] = member
        
        delete team[]

        break
      }
      case "-end": {
        p = piped(line, p.i, 2)
        const dest = this.member(p.args[0])
        let { stripped: name } = parseEffect(p.args[1])
        const { volatiles } = this.active(dest.pov)

        if (name.startsWith("fallen")) {
          name = "Fallen"
        }

        delete volatiles[name]
        break
      }
      case "-singleturn": {
        p = piped(line, p.i, 2)
        const { pov } = this.member(p.args[0])
        const { stripped: name } = parseEffect(p.args[1])

        this.active(pov).volatiles[name] = { singleTurn: true }
        break
      }
      case "-singlemove": {
        p = piped(line, p.i, 2)
        const { pov } = this.member(p.args[0])
        const { stripped: name } = parseEffect(p.args[1])

        this.active(pov).volatiles[name] = { singleMove: true }
        break
      }
      case "faint": {
        p = piped(line, p.i)
        const dest = this.member(p.args[0])
        dest.hp[0] = 0

        break
      }
      case "-sidestart": {
        p = piped(line, p.i, 2)
        const { side } = parseLabel(p.args[0])
        const { stripped: name } = parseEffect(p.args[1])

        const { conditions } = this[this.pov(side)]
        if (isHazard(name)) (conditions[name] ?? { layers: 0 }).layers!++
        else conditions[name] = { turn: 0 }

        break
      }
      case "-sideend": {
        p = piped(line, p.i, 2)
        const { side } = parseLabel(p.args[0])
        const { stripped: name } = parseEffect(p.args[1])

        const { conditions } = this[this.pov(side)]
        delete conditions[name]

        break
      }
      case "upkeep": {
        const { fields } = this

        for (const name in fields) {
          fields[name]++
        }

        break
      }
      case "turn": {
        p = piped(line, p.i)
        this.turn = Number(p.args[0])

        for (const pov of POVS) {
          const side = this[pov]
          const { conditions } = side

          const { volatiles, member, lastBerry } = side.active!

          if (lastBerry) lastBerry.turn++

          const { status } = member
          if (status?.turn !== undefined) status.turn++

          for (const name in volatiles) {
            if (volatiles[name]?.turn !== undefined) volatiles[name].turn++
            if (volatiles[name].singleTurn) delete volatiles[name]
          }

          for (const name in conditions) {
            if (conditions[name]?.turn !== undefined) conditions[name].turn++
          }

          if (side.wish) {
            if (side.wish === 0) side.wish++
            delete side.wish
          }
        }

        return "turn"
      }
      case "tie": {
        return "tie"
      }
      case "win": {
        p = piped(line, p.i)
        this.winner = p.args[0] === this.name ? "ally" : "foe"

        return "win"
      }
    }

    return null
  }
}
