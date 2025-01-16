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
  move?: number
}

type MoveSet = {
  [k: string]: number
}

type Eaten = {
  name: string
  turn: number
}

type OneTime = {
  "Battle Bond"?: boolean
  "Intrepid Sword"?: boolean
}

export type AllyMember = {
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
  oneTime: OneTime
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
  oneTime: OneTime
}

export type DelayedAttack = {
  turn: number
  member: Member
}

export type Volatiles = {
  [k: string]: { turn?: number; singleMove?: boolean; singleTurn?: boolean }
} & {
  "Type Change"?: {
    types: TypeName[]
  }
  "Disable"?: {
    turn: number
    move: string
  }
  "Transform"?: {
    into: Member
    complete: boolean
  }
  "Locked Move"?: {
    turn: number
    move: string
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

type Condition = {
  turn?: number
  layers?: number
}

type Member = AllyMember | FoeMember

type Active = {
  lastMove?: string
  member: Member
  volatiles: Volatiles
  lastBerry?: Eaten
  boosts: {
    [k in BoostId]?: number
  }
}

type Ally = {
  tera: {
    member: Member
    type: TypeName
  } | null
  conditions: { [k: string]: Condition }
  active?: Active
  team: { [k: string]: Member }
  wish?: number
}

type Foe = {
  tera: {
    member: Member
    type: TypeName
  } | null
  conditions: { [k: string]: Condition }
  active?: Active
  team: { [k: string]: Member }
  wish?: number
}

const OPP = { ally: "foe", foe: "ally" } as const

type POV = "ally" | "foe"

const HAZARDS = new Set(["Sticky Web", "Toxic Spikes", "Stealth Rock", "Spikes"])
function isHazard(name: string) {
  return name in HAZARDS
}

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

type Label = {
  species: string
  pov: POV
}

export class Observer {
  side: Side

  ally: Ally
  foe: Foe
  name?: string
  private gen: Generation

  turn: number
  fields: { [k: string]: number }
  weather?: { name: WeatherName; turn: number }
  winner?: POV

  constructor(side: Side, gen: Generation) {
    this.gen = gen
    this.side = side
    this.ally = { tera: null, team: {}, conditions: {} }
    this.foe = { tera: null, team: {}, conditions: {} }
    this.weather = undefined
    this.fields = {}
    this.turn = 0
  }

  active(pov: POV) {
    return this[pov].active!
  }

  label(s: string): Label {
    const { side, species } = parseLabel(s)
    return { pov: side === this.side ? "ally" : "foe", species }
  }

  member({ pov, species }: Label) {
    return this[pov].team[species]
  }

  read(line: string) {
    if (line[0] !== "|") return

    let p
    p = piped(line, 0)
    const msgType = p.args[0]

    switch (msgType) {
      case "request": {
        const { Transform: transform } = this.ally.active?.volatiles ?? {}

        if (this.name && transform?.complete !== false) break

        const {
          side: { pokemon, name }
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

        if (transform?.complete === false) {
          transform.complete = true
          const { into } = transform
          const { member: from } = this.ally.active!

          const { moves, ability } = pokemon.find(
            ({ ident }) => parseLabel(ident).species === from.species
          )!

          into.ability = ability
          const { moveset } = into
          for (const move of moves) {
            const { name } = this.gen.moves.get(move)!
            moveset[name] = moveset[name] ?? 0
          }
        }

        if (!this.name) {
          this.name = name

          const {
            ally: { team }
          } = this

          for (const {
            ident,
            details,
            condition,
            stats,
            item,
            moves,
            ability,
            teraType
          } of pokemon) {
            const { species } = parseLabel(ident)
            const { gender, lvl, forme } = parseTraits(details)

            const moveset: MoveSet = {}
            for (const move of moves) moveset[this.gen.moves.get(move)!.name] = 0

            team[species] = {
              pov: "ally",
              species,
              oneTime: {},
              forme,
              gender,
              lvl,
              revealed: false,
              teraType,
              ability: this.gen.abilities.get(ability)!.name,
              item: item ? this.gen.items.get(item)!.name : "Leftovers",
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
        const dest = this.member(this.label(p.args[0]))
        const ability = p.args[1]

        {
          p = piped(line, p.i, -1)
          const { from, of } = parseTags(p.args)
          const { ability: prev } = parseEffect(from)

          if (prev === "Trace") {
            hasAbility(dest, prev)
            hasAbility(this.member(this.label(of)), ability)
          }
        }

        if (ability === "Intrepid Sword") {
          dest.oneTime[ability] = true
        }

        hasAbility(dest, ability)
        break
      }
      case "switch":
      case "drag": {
        p = piped(line, p.i, 3)
        let label = this.label(p.args[0])
        const { pov, species } = label
        const traits = parseTraits(p.args[1])
        const hp = parseHp(p.args[2])!

        let dest: Member

        if (pov === "ally") {
          dest = this.member(label)
        } else {
          const { team } = this[pov]
          const { forme, lvl, gender } = traits

          dest = team[species] = team[species] ?? {
            pov: "foe",
            oneTime: {},
            species,
            forme,
            lvl,
            gender,
            hp,
            moveset: {},
            initial: {
              formeId: this.gen.species.get(forme)!.id
            }
          }
        }

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)

        dest.hp = hp

        const { status } = dest
        if (status?.id === "tox") status.turn! = 0

        const prev = this[pov].active!

        const volatiles: Volatiles = {}
        if (from === "Shed Tail") {
          volatiles["Substitute"] = prev.volatiles["Substitute"]
        }

        this[pov].active = { member: dest, volatiles, boosts: {} }
        break
      }
      case "-weather": {
        p = piped(line, p.i)
        const name = p.args[0] as WeatherName | "none"

        if (name === "none") {
          this.weather = undefined
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
        if (ability) hasAbility(this.member(this.label(of)), ability)

        break
      }
      case "-fieldstart": {
        p = piped(line, p.i)
        const { move: name } = parseEffect(p.args[0])

        this.fields[name!] = 0

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)

        const { ability } = parseEffect(from)
        if (ability) hasAbility(this.member(this.label(of)), ability)
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
        const dest = this.member(this.label(p.args[0]))
        const id = p.args[1] as StatusId

        dest.status = {
          id,
          turn: id === "tox" ? 0 : undefined,
          move: id === "slp" ? 0 : undefined
        }

        {
          p = piped(line, p.i, -1)

          const { from, of } = parseTags(p.args)

          const src = of ? this.member(this.label(of)) : dest
          const { ability, item } = parseEffect(from)

          if (item) hasItem(src, item)
          if (ability) hasAbility(src, ability)
        }
        break
      }
      case "-curestatus": {
        p = piped(line, p.i, 2)
        const target = this.member(this.label(p.args[0]))

        delete target.status

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)

        const { ability } = parseEffect(from)
        if (ability) hasAbility(target, ability)

        break
      }
      case "move": {
        p = piped(line, p.i, 3)
        const dest = this.member(this.label(p.args[0]))
        const { pov } = dest

        const move = p.args[1]
        const opp = this[OPP[pov]]

        const { volatiles } = this.active(pov)
        const { moveset, status } = dest

        {
          p = piped(line, p.i, -1)
          const { from, notarget, miss } = parseTags(p.args)

          this.active(pov).lastMove = move

          for (const name in volatiles) {
            if (volatiles[name].singleMove) delete volatiles[name]
          }

          if (status?.move) status.move++

          const { ability } = parseEffect(from)
          if (ability) hasAbility(dest, ability)

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

        moveset[move] = (moveset[move] ?? 0) + (opp.active!.member.ability === "Pressure" ? 2 : 1)
        break
      }
      case "-heal":
      case "-sethp": {
        p = piped(line, p.i, 2)
        const dest = this.member(this.label(p.args[0]))
        const hp = parseHp(p.args[1])

        dest.hp = hp!

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)

        const { ability, item } = parseEffect(from)
        if (ability) hasAbility(dest, ability)

        // berries already include an -enditem
        if (item === "Leftovers") hasItem(dest, item)

        break
      }
      case "-damage": {
        p = piped(line, p.i, 2)

        const dest = this.member(this.label(p.args[0]))
        const hp = parseHp(p.args[1])

        if (hp) dest.hp = hp
        else dest.hp[0] = 0

        {
          p = piped(line, p.i, -1)

          const { from, of } = parseTags(p.args)

          const { item, ability } = parseEffect(from)
          const src = of ? this.member(this.label(of)) : dest

          if (ability) hasAbility(src, ability)
          if (item) hasItem(src, item)
        }

        break
      }
      case "-boost":
      case "-unboost": {
        p = piped(line, p.i, 3)
        const dest = this.member(this.label(p.args[0]))
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
        const { pov } = this.label(p.args[0])

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
        const { pov } = this.label(p.args[0])
        const { boosts } = this.active(pov)

        for (const k in boosts) {
          const id = k as BoostId
          boosts[id] = Math.max(0, boosts[id]!)
        }
        break
      }
      case "-item": {
        p = piped(line, p.i, 2)
        const dest = this.member(this.label(p.args[0]))
        const item = p.args[1]

        hasItem(dest, item)

        p = piped(line, p.i, -1)
        const { from, of, identify } = parseTags(p.args)
        const { ability } = parseEffect(from)

        const src = of ? this.member(this.label(of)) : undefined

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
        const dest = this.member(this.label(p.args[0]))
        const { pov } = dest
        const item = p.args[1]

        hasItem(dest, item)
        hasItem(dest, null)

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
      case "-transform": {
        p = piped(line, p.i, 2)
        const { pov } = this.label(p.args[0])

        this.active(pov).volatiles["Transform"] = {
          into: this.member(this.label(p.args[1])),
          complete: false
        }

        break
      }
      case "-start": {
        p = piped(line, p.i, 2)
        const dest = this.member(this.label(p.args[0]))
        const { pov } = dest

        let { stripped: name } = parseEffect(p.args[1])

        const active = this.active(pov)
        const { volatiles } = active

        if (name.startsWith("quarkdrive")) {
          volatiles["Quark Drive"] = { statId: name.slice(-3) as StatId }
        } else if (name.startsWith("protosynthesis")) {
          volatiles["Protosynthesis"] = { statId: name.slice(-3) as StatId }
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
              const { member, volatiles } = this[OPP[pov]].active!

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
            case "Encore": {
              volatiles[name] = {
                turn: 0,
                move: active.lastMove!
              }
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
        const src = of ? this.member(this.label(of)) : dest

        if (ability) hasAbility(src, ability)
        if (item) hasItem(src, item)

        break
      }
      case "-terastallize": {
        p = piped(line, p.i, 2)
        const dest = this.member(this.label(p.args[0]))
        const { pov } = dest
        const type = p.args[1] as TypeName

        this[pov].tera = { member: dest, type }
        break
      }
      case "-formechange": {
        p = piped(line, p.i, 2)
        const dest = this.member(this.label(p.args[0]))
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
        const dest = this.member(this.label(p.args[0]))
        const { forme } = parseTraits(p.args[1])

        dest.forme = forme
        break
      }
      case "-activate": {
        p = piped(line, p.i, 2)
        const dest = this.member(this.label(p.args[0]))
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
            dest.oneTime[ability] = true
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
        p = piped(line, p.i, 3)
        const { pov, species } = this.label(p.args[0])
        const { forme, lvl, gender } = parseTraits(p.args[1])

        const party = this[pov]
        const { team, active } = party
        const { member } = active!

        if (member.pov === "foe") {
          team[member.species] = {
            pov: "foe",
            oneTime: {},
            species: member.species,
            forme: member.forme,
            lvl: member.lvl,
            gender: member.gender,
            hp: [100, 100],
            moveset: {},
            initial: {
              formeId: member.initial.formeId
            }
          }

          member.species = species
          member.forme = forme
          member.lvl = lvl
          member.gender = gender
          member.initial.formeId = this.gen.species.get(forme)!.id

          team[species] = member
        }

        active!.member = team[species]
        break
      }
      case "-end": {
        p = piped(line, p.i, 2)
        const { pov } = this.label(p.args[0])

        let { stripped: name } = parseEffect(p.args[1])

        const { volatiles } = this.active(pov)

        if (name.startsWith("fallen")) {
          name = "Fallen"
        }

        delete volatiles[name]
        break
      }
      case "-singleturn": {
        p = piped(line, p.i, 2)
        const { pov } = this.label(p.args[0])
        const { stripped: name } = parseEffect(p.args[1])

        this.active(pov).volatiles[name] = { singleTurn: true }
        break
      }
      case "-singlemove": {
        p = piped(line, p.i, 2)
        const { pov } = this.label(p.args[0])
        const { stripped: name } = parseEffect(p.args[1])

        this.active(pov).volatiles[name] = { singleMove: true }
        break
      }
      case "faint": {
        p = piped(line, p.i)
        const dest = this.member(this.label(p.args[0]))
        dest.hp[0] = 0

        break
      }
      case "-sidestart": {
        p = piped(line, p.i, 2)
        const { pov } = this.label(p.args[0])
        const { stripped: name } = parseEffect(p.args[1])

        const { conditions } = this[pov]
        if (isHazard(name)) (conditions[name] ?? { layers: 0 }).layers!++
        else conditions[name] = { turn: 0 }

        break
      }
      case "-sideend": {
        p = piped(line, p.i, 2)
        const { pov } = this.label(p.args[0])
        const { stripped: name } = parseEffect(p.args[1])

        const { conditions } = this[pov]
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
