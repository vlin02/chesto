import { Generations, TypeName } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
import {
  BoostId,
  Gender,
  parseEntity,
  parseHp,
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
  turns: number
}

type MoveSet = {
  [k: string]: {
    ppUsed: number
  }
}

type AllyMember = {
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
  moves: MoveSet
  teraType: TypeName
}

type FoeMember = {
  lvl: number
  forme: string
  gender: Gender
  fnt: boolean
  hp: [number, number]
  ability?: string | null
  item?: string | null
  status: Status | null
  moves: MoveSet
}

type Active = {
  name: string
  volatiles: Volatiles
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

type Ally = {
  tera: Tera | null
  hazards: { [k: string]: number }
  screens: { [k: string]: number }
  active?: Active
  team: { [k: string]: AllyMember }
}

type Foe = {
  tera: Tera | null
  hazards: { [k: string]: number }
  screens: { [k: string]: number }
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

export class Observer {
  side: Side

  ally: Ally
  foe: Foe
  name: string

  turn: number
  fields: { [k: string]: number }
  weather: { name: WeatherName; turns: number } | null
  winner?: POV

  history: {
    battleBond: boolean
  }

  constructor(side: Side) {
    this.side = side
    this.ally = { tera: null, team: {}, hazards: {}, screens: {} }
    this.foe = { tera: null, team: {}, hazards: {}, screens: {} }
    this.weather = null
    this.fields = {}
    this.turn = 0
    this.history = { battleBond: false }
  }

  parseLabel(s: string) {
    const i = s.indexOf(": ")
    const side = s.slice(0, 2) as Side
    const name = s.slice(i + 2)
    return { pov: this.side === side ? "ally" : "foe", name } as const
  }

  active({ pov }: { pov: POV }) {
    return this[pov].active!
  }

  member({ pov, name }: { pov: POV; name: string }) {
    return this[pov].team[name]
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
            const { name } = this.parseLabel(ident)
            const { gender, lvl, forme: species } = parseTraits(details)

            ally.team[name] = {
              forme: species,
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
              moves: Object.fromEntries(
                moves.map((id) => {
                  return [gen.moves.get(id)!.name, { ppUsed: 0 }]
                })
              )
            }
          }
        }

        break
      }
      case "-ability": {
        p = piped(line, p.i, 2)
        const { pov, name } = this.parseLabel(p.args[0])
        this[pov].team[name].ability = p.args[1]

        break
      }
      case "switch":
      case "drag": {
        p = piped(line, p.i, 2)
        const { pov, name } = this.parseLabel(p.args[0])

        const traits = parseTraits(p.args[1])
        p = piped(line, p.i, -1)

        const { from } = parseTags(p.args)

        if (pov === "ally") {
          this[pov].team[name].revealed = true
        }

        if (pov === "foe") {
          const { team } = this[pov]
          const { forme, lvl, gender } = traits
          if (!(name in team)) {
            team[name] = {
              forme,
              lvl,
              gender,
              hp: [100, 100],
              moves: {},
              status: null,
              fnt: false
            }
          }
        }

        const { status } = this[pov].team[name]
        if (status?.id === "tox") status.turns! = 0

        const prev = this[pov].active!
        const curr: Active = { name: name, volatiles: {}, boosts: {} }

        if (from === "Shed Tail") {
          curr.volatiles.substitute = prev.volatiles.substitute
        }

        this[pov].active = curr

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

        if (upkeep !== undefined) break

        this.weather = { name, turns: 0 }

        if (from && of) {
          const { ability } = parseEntity(from)
          const { pov, name } = this.parseLabel(of)

          this[pov].team[name].ability = ability!
        }

        break
      }
      case "-fieldstart": {
        p = piped(line, p.i)
        const { move: field } = parseEntity(p.args[0])

        this.fields[field!] = 0

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)

        const { ability } = parseEntity(from ?? "")
        if (ability) this.member(this.parseLabel(of)).ability = ability
        break
      }

      case "-curestatus": {
        p = piped(line, p.i, 2)
        const { pov, name } = this.parseLabel(p.args[0])
        const memb = this[pov].team[name]

        memb.status = null

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)
        const { ability } = parseEntity(from ?? "")
        if (ability) memb.ability = ability

        break
      }
      case "move": {
        p = piped(line, p.i, 2)
        const label = this.parseLabel(p.args[0])
        const side = this[label.pov]
        const { volatiles } = side.active!
        const memb = side.team[label.name]
        const { moves, status } = memb

        const name = p.args[1]

        p = piped(line, p.i, -1)
        const { from, notarget, miss } = parseTags(p.args)

        for (const k in volatiles) if (k in SINGLE_MOVE) delete volatiles[k]

        if (status?.id === "slp") status.turns++

        if (from) {
          const { ability } = parseEntity(from)
          if (ability) side.team[name].ability = ability
        }

        if (miss === undefined) {
          switch (name) {
            case "Outrage": {
              if (from !== "lockedmove" && notarget === undefined) {
                volatiles["Locked Move"] = { turn: 0, move: "Outrage" }
              }
              break
            }
          }
        }

        ;(moves[name] = moves[name] ?? { ppUsed: 0 }).ppUsed++
        break
      }
      case "-heal": {
        p = piped(line, p.i, 2)
        const { pov, name } = this.parseLabel(p.args[0])
        const hp = parseHp(p.args[1])

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)

        this[pov].team[name].hp = hp!

        if (from) {
          let memb = this[pov].team[name]

          const { item, ability } = parseEntity(from)
          if (item) memb.item = item
          if (ability) memb.ability = ability
        }

        break
      }
      case "-damage": {
        p = piped(line, p.i, 2)

        const target = this.parseLabel(p.args[0])
        const hp = parseHp(p.args[1])

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)

        if (from) {
          const { ability, item } = parseEntity(from)
          const memb = this.member(of ? this.parseLabel(of) : target)

          if (ability) memb.ability = ability
          if (item) memb.item = item
        }

        const { pov, name } = target

        const memb = this[pov].team[name]
        if (hp) memb.hp = hp
        else {
          memb.hp[0] = 0
          memb.fnt = true
        }

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
      case "-enditem": {
        p = piped(line, p.i, 2)
        const target = this.parseLabel(p.args[0])
        this.member(target).item = null

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)
        const { ability } = parseEntity(from ?? "")

        if (ability) this.member(of ? this.parseLabel(of) : target).ability = ability

        break
      }
      case "-start": {
        p = piped(line, p.i, 2)
        const target = this.parseLabel(p.args[0])
        let { pov } = target
        let { stripped: name } = parseEntity(p.args[1])

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
          const { ability, item } = parseEntity(from)
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
          const { ability } = parseEntity(from)
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
        const target = this.parseLabel(p.args[0])
        const { pov } = target

        const { ability, item, move, stripped } = parseEntity(p.args[1])

        if (item) {
          switch (item) {
            case "Leppa Berry": {
              p = piped(line, p.i, 1)
              const [move] = p.args
              this.member(target).moves[move].ppUsed = 0
              break
            }
          }
        } else if (move) {
          switch (move) {
            case "Poltergeist": {
              p = piped(line, p.i)
              this.member(target).item = p.args[0]
              break
            }
          }
        } else if (ability) {
          if (ability === "Battle Bond") {
            this.history.battleBond = true
          }

          p = piped(line, p.i, -1)
          const { of } = parseTags(p.args)

          this.member(of ? this.parseLabel(of) : target).ability = ability
        } else {
          if (stripped === "Orichalcum Pulse") {
            this[pov].active!.volatiles["Orichalcum Pulse"] = {}
          }
        }

        break
      }
      case "-end": {
        p = piped(line, p.i, 2)
        const { pov } = this.parseLabel(p.args[0])
        let { stripped: name } = parseEntity(p.args[1])

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
        const { stripped: name } = parseEntity(p.args[1])
        const { hazards, screens } = this[pov]

        if (isHazard(name)) {
          hazards[name] = (hazards[name] ?? 0) + 1
        } else {
          screens[name] = 0
        }

        break
      }
      case "-sideend": {
        p = piped(line, p.i, 2)
        const { pov } = this.parseLabel(p.args[0])
        const { stripped: name } = parseEntity(p.args[1])
        const { hazards, screens } = this[pov]

        if (isHazard(name)) {
          delete hazards[name]
        } else {
          delete screens[name]
        }

        break
      }
      case "upkeep": {
        const { fields, weather } = this

        for (const name in fields) {
          fields[name]++
        }

        if (weather) {
          weather.turns += 1
        }

        break
      }
      case "turn": {
        p = piped(line, p.i)
        this.turn = Number(p.args[0])

        for (const pov of POVS) {
          const side = this[pov]
          const { screens } = side
          const { volatiles, name: name } = this[pov].active!

          const { status } = side.team[name]
          if (status?.id === "tox") status.turns!++

          for (const k in volatiles) {
            if (volatiles[k].turn !== undefined) volatiles[k].turn += 1
            if (k in SINGLE_TURN) delete volatiles[k]
          }

          for (const k in screens) {
            screens[k] += 1
          }
        }

        for (const field in this.fields) {
          this.fields[field]++
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
