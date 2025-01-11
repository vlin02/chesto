import { Generations } from "@pkmn/data"
import { Dex } from "@pkmn/dex"

export type Log = ["update", string[]] | ["sideupdate", string] | ["end", string]

type Side = "p1" | "p2"

const SIDES = ["p1", "p2"] as const
const FOE = { p1: "p2", p2: "p1" } as const

export function split(log: Log) {
  let p

  const chs: { p1: string[]; p2: string[] } = { p1: [], p2: [] }

  const [type] = log

  switch (type) {
    case "update": {
      const [, lines] = log

      let i = 0
      while (i < lines.length) {
        const line = lines[i]

        p = piped(line, 0)

        if (p.args[0] === "split") {
          p = piped(line, p.i)
          const side = p.args[0] as Side

          chs[side].push(lines[i + 1])
          chs[FOE[side]].push(lines[i + 2])

          i += 3
        } else {
          for (const side of SIDES) chs[side].push(line)
          i += 1
        }
      }
      break
    }
    case "sideupdate": {
      const [, line] = log
      const side = line.slice(0, 2) as Side
      const msg = line.slice(3)

      chs[side].push(msg)
      break
    }
    case "end": {
      const [, line] = log
      const msg = `|end|${line}`
      for (const side of SIDES) chs[side].push(msg)
      break
    }
  }

  return chs
}

function parseTraits(s: string) {
  const [species, lvl, gender] = s.split(", ") as [
    string,
    string | undefined,
    "M" | "F" | undefined
  ]
  return { species, lvl: lvl ? Number(lvl.slice(1)) : 100, gender: gender ?? null }
}

function parseTags(strs: string[]) {
  const tags: { [k: string]: string } = {}

  for (const s of strs) {
    const i = s.indexOf("]")
    const name = s.slice(1, i)
    const value = s.slice(i + 2)
    tags[name] = value
  }

  return tags
}

function piped(s: string, i: number, n = 1) {
  const args = []
  for (let j = 0; j !== n; j++) {
    let k = s.indexOf("|", i + 1)
    if (k === -1) k = s.length
    args.push(s.slice(i + 1, k))
    i = k
    if (i === s.length) break
  }
  return { args, i }
}

type Stat = "atk" | "def" | "spa" | "spd" | "spe"
type BoostName = "atk" | "def" | "spa" | "spd" | "spe" | "evasion" | ""

type TypeName =
  | "Normal"
  | "Fighting"
  | "Flying"
  | "Poison"
  | "Ground"
  | "Rock"
  | "Bug"
  | "Ghost"
  | "Steel"
  | "Fire"
  | "Water"
  | "Grass"
  | "Electric"
  | "Psychic"
  | "Ice"
  | "Dragon"
  | "Dark"
  | "Fairy"
  | "Stellar"

type StatusName = "slp" | "psn" | "brn" | "frz" | "par" | "tox"
type WeatherName = "Snow" | "SunnyDay" | "SandStorm" | "RainDance"

const POVS = ["ally", "foe"] as const
type AllyMember = {
  revealed: boolean
  lvl: number
  species: string
  gender: "M" | "F" | null
  fnt: boolean
  hp: [number, number]
  ability: string | null
  item: string | null
  boosts: {
    [k in BoostName]?: number
  }
  stats: { [k in Stat]: number }
  status: {
    name: StatusName
    turns: number
  } | null
  moves: {
    [k: string]: {
      ppUsed: number
    }
  }
  teraType: TypeName
}

type FoeMember = {
  lvl: number
  species: string
  gender: "M" | "F" | null
  fnt: boolean
  hp: [number, number]
  ability?: string | null
  item?: string | null
  boosts: {
    [k in BoostName]?: number
  }
  status: {
    name: StatusName
    turns: number
  } | null
  moves: {
    [k: string]: {
      ppUsed: number
    }
  }
}

type Volatile = { turn?: number; singleMove?: boolean; singleTurn?: boolean }

type Ally = {
  tera: {
    name: string
  } | null
  hazards: { [k: string]: number }
  screens: { [k: string]: number }
  active?: {
    name: string
    volatiles: { [k: string]: Volatile }
  }
  team: { [k: string]: AllyMember }
}

type Foe = {
  hazards: { [k: string]: number }
  screens: { [k: string]: number }
  tera: {
    name: string
  } | null
  active?: {
    name: string
    volatiles: { [k: string]: Volatile }
  }
  team: { [k: string]: FoeMember }
}

function parseHp(s: string): [number, number] {
  if (s.slice(-3) === "fnt") return [0, 0]
  const [a, b] = s.split("/")
  return [Number(a), Number(b)]
}

const SINGLE_TURN = new Set(["outrage", "glaverush"])
const SINGLE_MOVE = new Set(["roost", "protect"])

function parseEntity(s: string) {
  let i = s.indexOf(": ")
  let item = null
  let ability = null
  let move = null

  switch (s.slice(0, i)) {
    case "item":
      item = s.slice(i + 2)
      break
    case "ability":
      ability = s.slice(i + 2)
      break
    case "move":
      move = s.slice(i + 2)
      break
  }

  return { item, ability, move, stripped: ability || move || item || s }
}

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

  constructor(side: Side) {
    this.side = side
    this.ally = { tera: null, team: {}, hazards: {}, screens: {} }
    this.foe = { tera: null, team: {}, hazards: {}, screens: {} }
    this.weather = null
    this.fields = {}
    this.turn = 0
  }

  parseLabel(s: string) {
    const i = s.indexOf(": ")
    const side = s.slice(0, 2) as Side
    const name = s.slice(i + 2)
    return { pov: this.side === side ? "ally" : "foe", name } as const
  }

  getMember({ pov, name }: { pov: POV; name: string }) {
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
                stats: { [k in Stat]: number }
                item: string
                ability: string
                moves: string[]
                teraType: TypeName
              }[]
            }
          } = JSON.parse(line.slice(p.i + 1))

          this.name = name
          const { ally } = this
          for (const {
            ident,
            details,
            condition,
            active,
            stats,
            item,
            moves,
            ability,
            teraType
          } of team) {
            const { name } = this.parseLabel(ident)
            const { gender, lvl, species } = parseTraits(details)

            if (active) ally.active = { name, volatiles: {} }

            ally.team[name] = {
              species,
              gender,
              lvl,
              fnt: false,
              revealed: false,
              teraType,
              ability,
              item,
              stats,
              hp: parseHp(condition)!,
              boosts: {},
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

        if (pov === "ally") {
          this[pov].team[name].revealed = true
        }

        if (pov === "foe") {
          const { team } = this[pov]
          if (!(name in team)) {
            team[name] = {
              ...traits,
              hp: [100, 100],
              boosts: {},
              moves: {},
              status: null,
              fnt: false
            }
          }
        }

        const { status } = this[pov].team[name]

        if (status?.name === "tox") status.turns! = 0

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
        const tags = parseTags(p.args)

        if ("upkeep" in tags) break

        this.weather = { name, turns: 0 }

        if ("from" in tags) {
          const { from, of } = tags
          const { ability } = parseEntity(from)
          const { pov, name } = this.parseLabel(of)
          this[pov].team[name].ability = ability!
        }

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
        const tags = parseTags(p.args)
        const missed = "miss" in tags

        {
          for (const k in volatiles) {
            if (k in SINGLE_MOVE) delete volatiles[k]
          }
        }

        {
          if (status?.name === "slp") status.turns++
        }

        {
          const { from } = tags
          if (from) {
            const { ability } = parseEntity(from)
            if (ability) side.team[name].ability = ability
          }
        }

        if (!missed) {
          switch (name) {
            case "Outrage": {
              if (tags["from"] !== "lockedmove" && tags["notarget"] != "") {
                volatiles["Locked Move"] = { turn: 0 }
              }
              break
            }
          }
        }

        if (!(name in moves)) moves[name] = { ppUsed: 0 }
        moves[name].ppUsed += 1
        break
      }
      case "-heal": {
        p = piped(line, p.i, 2)
        const { pov, name } = this.parseLabel(p.args[0])
        const hp = parseHp(p.args[1])

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)

        this[pov].team[name].hp = hp

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

        const { pov, name } = this.parseLabel(p.args[0])
        const hp = parseHp(p.args[1])

        this[pov].team[name].hp = hp
        break
      }
      case "-boost":
      case "-unboost": {
        p = piped(line, p.i, 3)
        const { pov, name } = this.parseLabel(p.args[0])
        const { boosts } = this[pov].team[name]

        const id = p.args[1] as BoostName
        const n = Number(p.args[2])
        boosts[p.args[1] as BoostName] = (boosts[id] ?? 0) + (msgType === "-boost" ? n : -n)
        break
      }
      case "-enditem": {
        p = piped(line, p.i, 2)
        const { pov, name } = this.parseLabel(p.args[0])
        this[pov].team[name].item = null

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)
        if (from && of) {
          const { ability } = parseEntity(from)
          const { pov, name } = this.parseLabel(of)
          if (ability) this[pov].team[name].ability = ability
        }
        break
      }
      case "-status": {
        p = piped(line, p.i, 2)
        const { pov, name } = this.parseLabel(p.args[0])
        this[pov].team[name].status = { name: p.args[1] as StatusName, turns: 0 }
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
          const { volatiles, name } = this[pov].active!

          const { status } = side.team[name]
          if (status?.name === "tox") status.turns!++

          for (const k in volatiles) {
            if (volatiles[k].turn !== undefined) volatiles[k].turn += 1
            if (k in SINGLE_TURN) delete volatiles[k]
          }

          for (const k in screens) {
            screens[k] += 1
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
