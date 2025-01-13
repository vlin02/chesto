import { Generations, TypeName } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
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

const POVS = ["ally", "foe"] as const

type Status = {
  id: StatusId
  turns?: number
  moves?: number
}

type MoveSet = {
  [k: string]: number
}

type Eaten = {
  name: string
  turns: number
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
  fnt: boolean
  hp: [number, number]
  ability: string | null
  item: string | null
  stats: { [k in StatId]: number }
  status: Status | null
  moveset: MoveSet
  teraType: TypeName
  used: Used
}

type FoeMember = {
  pov: "foe"
  lvl: number
  species: string
  forme: string
  gender: Gender
  fnt: boolean
  hp: [number, number]
  ability?: string | null
  item?: string | null
  initial: {
    forme: string
    ability?: string
    item?: string
  }
  status: Status | null
  moveset: MoveSet
  used: Used
}

type Active = {
  name: string
  volatiles: Volatiles
  lastBerry?: Eaten
  boosts: {
    [k in BoostId]?: number
  }
}

type DelayedAttack = {
  turn: number
  name: string
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
  name: string
  type: TypeName
}

type Condition = {
  turn?: number
  layers?: number
}

type Ally = {
  tera: Tera | null
  conditions: { [k: string]: Condition }
  active?: Active
  team: { [k: string]: AllyMember }
}

type Foe = {
  tera: Tera | null
  conditions: { [k: string]: Condition }
  active?: Active
  team: { [k: string]: FoeMember }
}

const SINGLE_TURN = new Set(["outrage", "glaverush"])
const SINGLE_MOVE = new Set(["roost", "protect"])

const OPP = { ally: "foe", foe: "ally" } as const

const gen = new Generations(Dex).get(9)

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

export class Observer {
  side: Side

  ally: Ally
  foe: Foe
  name: string

  turn: number
  fields: { [k: string]: number }
  weather: { name: WeatherName; turns: number } | null
  winner?: POV

  constructor(side: Side) {
    this.side = side
    this.ally = { tera: null, team: {}, conditions: {} }
    this.foe = { tera: null, team: {}, conditions: {} }
    this.weather = null
    this.fields = {}
    this.turn = 0
  }

  activeV1({ pov }: { pov: POV }) {
    return this[pov].active!
  }

  active(pov: POV) {
    return this[pov].active!
  }

  parseLabel(s: string) {
    const i = s.indexOf(": ")
    const side = s.slice(0, 2) as Side
    const name = s.slice(i + 2)
    return { pov: this.side === side ? "ally" : "foe", name } as const
  }

  memberV1({ pov, name }: { pov: POV; name: string }) {
    return this[pov].team[name]
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
            const { name: species } = this.parseLabel(ident)
            const { gender, lvl, forme } = parseTraits(details)

            const moveset: MoveSet = {}
            for (const move of moves) moveset[move] = 0

            ally.team[name] = {
              used: {},
              pov: "ally",
              species,
              forme,
              gender,
              lvl,
              fnt: false,
              revealed: false,
              teraType,
              ability,
              item,
              stats,
              hp: parseHp(condition)!,
              status: null,
              moveset
            }
          }
        }

        break
      }
      /*
      [from] ability: target's copying ability
      [of]: ability source
      */
      case "-ability": {
        p = piped(line, p.i, 2)
        const target = this.member(p.args[0])
        const ability = p.args[1]

        {
          p = piped(line, p.i, -1)
          const { from, of } = parseTags(p.args)
          const { ability: copyAbility } = parseEffect(from)

          if (copyAbility) hasAbility(target, copyAbility)
          if (of) hasAbility(this.member(of), ability)
        }

        if (ability === "Intrepid Sword") {
          target.used["Intrepid Sword"] = true
        }

        hasAbility(target, ability)

        break
      }
      /*
      none
      */
      case "switch":
      case "drag": {
        p = piped(line, p.i, 2)
        const { pov, name: species } = this.parseLabel(p.args[0])

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
              status: null,
              initial: {
                forme
              },
              fnt: false
            }
          }
        }

        const { status } = this[pov].team[species]
        if (status?.id === "tox") status.turns! = 0

        const prev = this[pov].active!
        const curr: Active = { name: species, volatiles: {}, boosts: {} }

        if (from === "Shed Tail") {
          curr.volatiles.substitute = prev.volatiles.substitute
        }

        this[pov].active = curr

        break
      }
      /*
      [from] ability: initiates weather
      [of]: owner of ability
      */
      case "-weather": {
        p = piped(line, p.i)
        const name = p.args[0] as WeatherName | "none"

        if (name === "none") {
          this.weather = null
          break
        }

        p = piped(line, p.i, -1)
        const { upkeep, from, of } = parseTags(p.args)
        const { ability } = parseEffect(from ?? "")

        if (upkeep === "") {
          this.weather!.turns++
          break
        }

        this.weather = { name, turns: 0 }
        if (ability) hasAbility(this.member(of), ability)

        break
      }
      case "-fieldstart": {
        p = piped(line, p.i)
        const { move: field } = parseEffect(p.args[0])

        this.fields[field!] = 0

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)

        const { ability } = parseEffect(from ?? "")
        if (ability) this.memberV1(this.parseLabel(of)).ability = ability
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
        const target = this.parseLabel(p.args[0])
        const id = p.args[1] as StatusId

        this.memberV1(target).status = {
          id,
          turns: id === "tox" ? 0 : undefined,
          moves: id === "slp" ? 0 : undefined
        }

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)

        const source = of ? this.parseLabel(of) : target
        const { ability, item } = parseEffect(from ?? "")

        if (item) this.memberV1(source).item = item
        if (ability) this.memberV1(source).ability = ability
        break
      }
      case "-curestatus": {
        p = piped(line, p.i, 2)
        const { pov, name } = this.parseLabel(p.args[0])
        const memb = this[pov].team[name]

        memb.status = null

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)

        const { ability } = parseEffect(from ?? "")
        if (ability) memb.ability = ability

        break
      }
      case "move": {
        p = piped(line, p.i, 3)
        const src = this.member(p.args[0])
        const { pov } = src
        const move = p.args[1]
        const target = this.member(p.args[2])

        const { volatiles } = this.active(pov)
        const { moveset, status } = src

        {
          p = piped(line, p.i, -1)
          const { from, notarget, miss } = parseTags(p.args)

          for (const k in volatiles) if (k in SINGLE_MOVE) delete volatiles[k]
          if (status?.moves) status.moves++

          const { ability } = parseEffect(from ?? "")
          if (ability) hasAbility(src, ability)

          if (miss === undefined) {
            switch (move) {
              case "Outrage": {
                if (from !== "lockedmove" && notarget === undefined) {
                  volatiles["Locked Move"] = { turn: 0, move: "Outrage" }
                }
                break
              }
            }
          }
        }

        moveset[move] = (moveset[move] ?? 0) + (target.ability === "Pressure" ? 2 : 1)
        break
      }
      /*
      [from] item: target's item
      [from] ability: target's healing ability
      [of]: drain source
      */
      case "-heal":
      case "-sethp": {
        p = piped(line, p.i, 2)
        const target = this.member(p.args[0])
        const hp = parseHp(p.args[1])

        target.hp = hp!

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)

        const { ability, item } = parseEffect(from ?? "")
        if (ability) hasAbility(target, ability)

        // berries already include an -enditem
        if (item === "Leftovers") hasItem(target, item)

        break
      }
      /*
      [from] item: source of damage
      [from] ability: source of damage
      [of]: owner of effect, defaulting to target
      */
      case "-damage": {
        p = piped(line, p.i, 2)

        const target = this.member(p.args[0])
        const hp = parseHp(p.args[1])

        if (hp) target.hp = hp
        else {
          target.hp[0] = 0
          target.fnt = true
        }

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)
        const { item, ability } = parseEffect(from ?? "")
        const owner = of ? this.member(of) : target

        if (ability) hasAbility(owner, ability)
        if (item) hasAbility(owner, item)

        break
      }
      case "-boost":
      case "-unboost": {
        p = piped(line, p.i, 3)
        const { pov } = this.parseLabel(p.args[0])
        const { boosts } = this[pov].active!

        const id = p.args[1] as BoostId
        const n = Number(p.args[2])
        boosts[p.args[1] as BoostId] = Math.min(
          Math.max(-6, (boosts[id] ?? 0) + (msgType === "-boost" ? n : -n)),
          6
        )
        break
      }
      case "-clearboost": {
        p = piped(line, p.i)
        const target = this.parseLabel(p.args[0])
        this.activeV1(target).boosts = {}
        break
      }
      case "-clearallboost": {
        for (const pov of POVS) {
          this.activeV1({ pov }).boosts = {}
        }
        break
      }
      case "-clearnegativeboost": {
        p = piped(line, p.i)
        const target = this.parseLabel(p.args[0])
        const { boosts } = this.activeV1(target)
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

        // magician doesnt emit an -enditem msg so we have to do it here
        if (ability === "Magician") {
          hasItem(src!, item)
          hasItem(src!, null)
        }

        if (ability) {
          hasAbility(dest, ability)
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
            turns: 0
          }
        }

        break
      }
      case "-start": {
        p = piped(line, p.i, 2)
        const target = this.parseLabel(p.args[0])
        let { pov } = target
        let { stripped: name } = parseEffect(p.args[1])

        const active = this[pov].active!
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
              const { name: species } = active
              this[OPP[pov]].active!.volatiles[name] = {
                turn: 0,
                name: species
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

        if (from) {
          const { ability, item } = parseEffect(from)
          const { pov, name } = of ? this.parseLabel(of) : target
          const memb = this[pov].team[name]

          if (ability) memb.ability = ability
          if (item) memb.item = item
        }

        break
      }
      case "-terastallize": {
        p = piped(line, p.i, 2)
        const { pov, name } = this.parseLabel(p.args[0])
        const type = p.args[1] as TypeName

        this[pov].tera = { name, type }
        break
      }
      case "-formechange": {
        p = piped(line, p.i, 2)
        const { pov, name } = this.parseLabel(p.args[0])
        const forme = p.args[1]

        const memb = this[pov].team[name]
        memb.forme = forme

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)

        if (from) {
          const { ability } = parseEffect(from)
          if (ability) memb.ability = ability
        }

        break
      }
      case "detailschange": {
        p = piped(line, p.i, 2)
        const { pov, name } = this.parseLabel(p.args[0])
        const { forme } = parseTraits(p.args[1])

        const memb = this[pov].team[name]
        memb.forme = forme

        break
      }
      case "-activate": {
        p = piped(line, p.i, 2)
        const target = this.member(p.args[0])
        const { pov } = target

        let { ability, item, move, stripped } = parseEffect(p.args[1])

        if (stripped === "Orichalcum Pulse") ability = stripped

        if (item) {
          switch (item) {
            case "Leppa Berry": {
              p = piped(line, p.i, 1)
              const [move] = p.args
              target.moveset[move] = 0
              break
            }
          }
        } else if (move) {
          switch (move) {
            case "Poltergeist": {
              p = piped(line, p.i)
              target.item = p.args[0]
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
          }

          p = piped(line, p.i, -1)
          target.ability = ability
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
      case "-end": {
        p = piped(line, p.i, 2)
        const { pov } = this.parseLabel(p.args[0])
        let { stripped: name } = parseEffect(p.args[1])

        if (name.startsWith("fallen")) {
          name = "Fallen"
        }

        delete this[pov].active!.volatiles[name]
        break
      }
      case "-status": {
        p = piped(line, p.i, 2)
        const { pov, name } = this.parseLabel(p.args[0])
        this[pov].team[name].status = { id: p.args[1] as StatusId, turns: 0 }
        break
      }
      case "-singleturn": {
        p = piped(line, p.i, 2)
        const { pov } = this.parseLabel(p.args[0])

        this[pov].active!.volatiles[p.args[1]] = {}
        break
      }
      case "-singlemove": {
        p = piped(line, p.i)
        const { pov } = this.parseLabel(p.args[0])

        this[pov].active!.volatiles[p.args[1]] = {}
        break
      }
      case "faint": {
        p = piped(line, p.i)
        const { pov, name } = this.parseLabel(p.args[0])
        this[pov].team[name].fnt = true
        break
      }
      case "-sidestart": {
        p = piped(line, p.i, 2)
        const { pov } = this.parseLabel(p.args[0])
        const { stripped: name } = parseEffect(p.args[1])
        const { conditions } = this[pov]

        if (isHazard(name)) {
          ;(conditions[name] ?? { layers: 0 }).layers!++
        } else {
          conditions[name] = { turn: 0 }
        }

        break
      }
      case "-sideend": {
        p = piped(line, p.i, 2)
        const { pov } = this.parseLabel(p.args[0])
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
          const { conditions, team } = side

          const { volatiles, name, lastBerry } = side.active!

          if (lastBerry) lastBerry.turns++

          const { status } = team[name]
          if (status?.turns !== undefined) status.turns++

          for (const name in volatiles) {
            if (volatiles[name]?.turn !== undefined) volatiles[name].turn++
            if (name in SINGLE_TURN) delete volatiles[name]
          }

          for (const name in conditions) {
            if (conditions[name]?.turn !== undefined) conditions[name].turn++
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
