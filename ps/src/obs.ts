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
  weather: { name: WeatherName; turn: number } | null
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
      gen9randombattle-2003212319 |-ability|p2a: Zacian|Intrepid Sword|boost
      gen9randombattle-2003212967 |-ability|p2a: Gardevoir|Water Absorb|[from] ability: Trace|[of] p1a: Volcanion
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
      [from] Shed Tail: transfers substitute
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
        if (status?.id === "tox") status.turn! = 0

        const prev = this[pov].active!
        const curr: Active = { name: species, volatiles: {}, boosts: {} }

        if (from === "Shed Tail") {
          curr.volatiles.substitute = prev.volatiles.substitute
        }

        this[pov].active = curr

        break
      }
      /*
      gen9randombattle-2015042421 |-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Tyranitar
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
          this.weather!.turn++
          break
        }

        this.weather = { name, turn: 0 }
        if (ability) hasAbility(this.member(of), ability)

        break
      }
      /*
      gen9randombattle-2003211376 |-fieldstart|move: Psychic Terrain|[from] ability: Psychic Surge|[of] p1a: Indeedee
      */
      case "-fieldstart": {
        p = piped(line, p.i)
        const { move: name } = parseEffect(p.args[0])

        this.fields[name!] = 0

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)

        const { ability } = parseEffect(from ?? "")
        if (ability) hasAbility(this.member(of), ability)
        break
      }
      /*
       */
      case "-fieldend": {
        p = piped(line, p.i)
        const { move: field } = parseEffect(p.args[0])

        delete this.fields[field!]
        break
      }
      /*
      |-status|p2a: Pachirisu|tox|[from] ability: Toxic Chain|[of] p1a: Okidogi
      |-status|p2a: Flareon|tox|[from] item: Toxic Orb
      */
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
          const { ability, item } = parseEffect(from ?? "")

          if (item) hasItem(source, item)
          if (ability) hasAbility(source, ability)
        }
        break
      }
      /*
      gen9randombattle-2019244979 |-curestatus|p2a: Altaria|par|[from] ability: Natural Cure
      */
      case "-curestatus": {
        p = piped(line, p.i, 2)
        const target = this.member(p.args[0])

        target.status = null

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)

        const { ability } = parseEffect(from ?? "")
        if (ability) hasAbility(target, ability)

        break
      }
      /*
      gen9randombattle-2088477834 |move|p1a: Espeon|Will-O-Wisp|p2a: Misdreavus|[from]ability: Magic Bounce
      gen9randombattle-2085825209 |move|p2a: Oricorio|Quiver Dance|p2a: Oricorio|[from]ability: Dancer
      */
      case "move": {
        p = piped(line, p.i, 3)
        const src = this.member(p.args[0])
        const { pov } = src
        const move = p.args[1]
        const dest = this.member(p.args[2])

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

        moveset[move] = (moveset[move] ?? 0) + (dest.ability === "Pressure" ? 2 : 1)
        break
      }
      /*
      |-heal|p2a: Sandy Shocks|22/267|[from] item: Leftovers
      |-heal|p2a: Gothitelle|204/272|[from] item: Sitrus Berry
      |-heal|p1a: Greedent|205/351|[from] ability: Cheek Pouch
      |-heal|p2a: Pawmot|243/243|[from] ability: Volt Absorb|[of] p1a: Pikachu (of is the attacker of Volt Absorb)
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
      |-damage|p2a: Ludicolo|188/290|[from] item: Life Orb
      |-damage|p2a: Chesnaught|181/285|[from] item: Rocky Helmet|[of] p1a: Garchomp
      |-damage|p1a: Toxicroak|42/280|[from] ability: Dry Skin|[of] p1a: Toxicroak
      |-damage|p2a: Morpeko|198/243|[from] ability: Rough Skin|[of] p1a: Garchomp
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
        
        {
          p = piped(line, p.i, -1)

          const { from, of } = parseTags(p.args)

          const { item, ability } = parseEffect(from ?? "")
          const src = of ? this.member(of) : target

          if (ability) hasAbility(src, ability)
          if (item) hasAbility(src, item)
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

        // magician doesnt emit an -enditem
        if (ability === "Magician") {
          hasItem(src!, item)
          hasItem(src!, null)
        }

        if (ability) {
          hasAbility(dest, ability)
        }

        break
      }
      /*
      */
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
      /*
      |-start|p2a: Meowscarada|typechange|Ice|[from] ability: Protean
      |-start|p1a: Primarina|confusion|[from] ability: Poison Puppeteer|[of] p2a: Pecharunt
      |-start|p2a: Bellibolt|Charge|Super Fang|[from] ability: Electromorphosis
      |-start|p2a: Mightyena|Disable|Poison Fang|[from] ability: Cursed Body|[of] p1a: Froslass
      */
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
              const { name: species } = active
              this[OPP[pov]].active!.volatiles[name] = {
                turn: 0,
                name: species
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
        this[pov].team[name].status = { id: p.args[1] as StatusId, turn: 0 }
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

          if (lastBerry) lastBerry.turn++

          const { status } = team[name]
          if (status?.turn !== undefined) status.turn++

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
