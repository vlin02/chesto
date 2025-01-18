import { Generation, Move } from "@pkmn/data"
import {
  ChoiceRequest,
  parseEffect,
  parseHp,
  parseLabel,
  parseTags,
  parseTraits,
  parseTypes,
  piped,
  Side
} from "./protocol.js"
import { WeatherName } from "@pkmn/client"
import { StatusId, StatId, BoostId, CHOICE_ITEMS, HAZARDS } from "./dex.js"
import { POV, MoveSet, AllyUser, OPP, POVS, User, FoeUser, Ally, Foe } from "./party.js"

type Label = {
  species: string
  pov: POV
}

function isPressureMove({ target, flags: { mustpressure } }: Move) {
  return (
    [
      "adjacentFoe",
      "all",
      "allAdjacent",
      "allAdjacentFoes",
      "any",
      "normal",
      "randomNormal",
      "scripted"
    ].includes(target) || mustpressure
  )
}

export class Observer {
  side!: Side
  name!: string
  ally!: Ally
  foe!: Foe
  request!: ChoiceRequest

  illusion?: {
    from: AllyUser
    to: AllyUser
  }

  ready: boolean
  private gen: Generation

  turn: number
  fields: { [k: string]: number }
  weather?: { name: WeatherName; turn: number }
  winner?: POV

  constructor(gen: Generation) {
    this.gen = gen
    this.fields = {}
    this.turn = 0
  }

  moveset(user: User) {
    const transform = user.volatiles["Transform"]
    return transform ? transform.moveset : user.moveset
  }

  label(s: string): Label {
    const { side, species } = parseLabel(s)
    return { pov: side === this.side ? "ally" : "foe", species }
  }

  member({ pov, species }: Label) {
    const { illusion } = this
    const user = this[pov].team[species]

    if (illusion?.to === user) return illusion.from
    return user
  }

  setAbility(user: User, ability: string) {
    if (user.ability === ability) return

    user.ability = ability
    if (user.pov === "foe" && ability) {
      const { initial } = user
      initial.ability = initial.ability ?? ability
    }
  }

  setItem(user: User, item: string | null) {
    const { volatiles } = user
    if (item === null) delete volatiles["Choice Locked"]

    user.item = item
    if (user.pov === "foe" && item) {
      const { initial } = user
      initial.item = initial.item ?? item
    }
  }

  clear(user: User) {
    user.volatiles = {}
    user.boosts = {}
    delete user.lastBerry
    delete user.lastMove
  }

  read(line: string) {
    if (line[0] !== "|") return

    let p: { args: string[]; i: number }
    p = piped(line, 0)
    const msgType = p.args[0]

    switch (msgType) {
      case "request": {
        this.request = JSON.parse(line.slice(p.i + 1))

        const {
          active,
          side: { id, name, pokemon: pokemons }
        } = this.request

        if (!this.ready) {
          this.side = id
          this.name = name

          const team: { [k: string]: AllyUser } = {}
          let active: AllyUser

          for (let {
            ident,
            details,
            condition,
            stats,
            item,
            moves,
            ability,
            active: isActive,
            teraType
          } of pokemons) {
            const { species } = parseLabel(ident)
            const { gender, lvl, forme } = parseTraits(details)

            if (species === "Ditto") {
              moves = ["Transform"]
              ability = "Imposter"
            }

            const moveset: MoveSet = {}
            for (const move of moves) moveset[this.gen.moves.get(move)!.name] = 0

            const member = (team[species] = {
              pov: "ally",
              species,
              flags: {},
              forme,
              gender,
              lvl,
              revealed: false,
              teraType,
              ability: this.gen.abilities.get(ability)!.name,
              item: item ? this.gen.items.get(item)!.name : "Leftovers",
              stats,
              hp: parseHp(condition)!,
              moveset,
              volatiles: {},
              boosts: {},
              tera: false
            })

            if (isActive) active = member
          }

          this.ally = { team, fields: {}, active: active! }
          this.ready = true
        }

        {
          if (
            active &&
            this.ally.active.volatiles["Locked Move"] &&
            !(
              active[0].moves.length === 1 &&
              active[0].moves[0].move === this.ally.active.volatiles["Locked Move"].move
            )
          ) {
            delete this.ally.active.volatiles["Locked Move"]
          }
        }
        break
      }
      case "-ability": {
        p = piped(line, p.i, 2)
        const user = this.member(this.label(p.args[0]))
        const ability = p.args[1]

        {
          p = piped(line, p.i, -1)
          const { from, of } = parseTags(p.args)
          const { ability: prev } = parseEffect(from)

          if (prev === "Trace") {
            this.setAbility(user, prev)
            this.setAbility(this.member(this.label(of)), ability)
          }
        }

        if (ability === "Intrepid Sword") {
          user.flags[ability] = true
        }

        this.setAbility(user, ability)
        break
      }
      case "switch":
      case "drag": {
        {
          const active = this.request.side.pokemon.find((x) => x.active)!
          const { species } = this.label(active.ident)
          const user = this.ally.team[species]
          const {
            ability,
            flags: { "Illusion revealed": revealed }
          } = user

          if (ability === "Illusion" && !revealed) {
            const target = [...this.request.side.pokemon]
              .reverse()
              .find((x) => parseHp(x.condition) !== null && !x.active)
            if (target) {
              const to = this.ally.team[this.label(target.ident).species]
              this.illusion = { from: user, to }
            }
          } else {
            delete this.illusion
          }
        }

        p = piped(line, p.i, 3)
        let label = this.label(p.args[0])
        const { pov, species } = label

        const traits = parseTraits(p.args[1])
        const hp = parseHp(p.args[2])!

        let user: User

        if (pov === "ally") {
          user = this.member(label)
        } else {
          const { forme, lvl, gender } = traits

          const team = this.foe?.team ?? {}

          user = team[species] = team[species] ?? {
            pov: "foe",
            flags: {},
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

          if (!this.foe) this.foe = { fields: {}, team, active: user }
        }

        const { active } = this[pov]

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)

        this.clear(user)
        const { status } = user
        if (status?.id === "tox") status.turn! = 0

        if (from === "Shed Tail") {
          user.volatiles["Substitute"] = active.volatiles["Substitute"]
        }

        this[pov].active = user
        break
      }
      case "-mustrecharge": {
        p = piped(line, p.i)
        const user = this.member(this.label(p.args[0]))
        user.volatiles["Recharge"] = { turn: 0 }
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
        if (ability) this.setAbility(this.member(this.label(of)), ability)

        break
      }
      case "-fieldstart": {
        p = piped(line, p.i)
        const { move: name } = parseEffect(p.args[0])

        this.fields[name!] = 0

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)

        const { ability } = parseEffect(from)
        if (ability) this.setAbility(this.member(this.label(of)), ability)
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
        const user = this.member(this.label(p.args[0]))
        const id = p.args[1] as StatusId

        user.status = {
          id,
          turn: id === "tox" ? 0 : undefined,
          move: id === "slp" ? 0 : undefined
        }

        {
          p = piped(line, p.i, -1)

          const { from, of } = parseTags(p.args)

          const src = of ? this.member(this.label(of)) : user
          const { ability, item } = parseEffect(from)

          if (item) this.setItem(src, item)
          if (ability) this.setAbility(src, ability)
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
        if (ability) this.setAbility(target, ability)

        break
      }
      case "move": {
        p = piped(line, p.i, 3)
        const user = this.member(this.label(p.args[0]))
        const move = p.args[1]
        const target = this.foe.active

        const { pov } = user

        const { volatiles, status } = user
        const moveset = this.moveset(user)

        let deductPP = true

        {
          p = piped(line, p.i, -1)
          const { from, notarget, miss } = parseTags(p.args)

          user.lastMove = { name: move }

          for (const name in volatiles) {
            if (volatiles[name].singleMove) delete volatiles[name]
          }

          if (status?.move) status.move++

          const effect = parseEffect(from)

          if (effect.move) {
            deductPP = false
          }

          if (effect.ability) {
            this.setAbility(user, effect.ability)
            deductPP = false
          }

          if (miss === undefined) {
            switch (move) {
              case "Petal Dance":
              case "Outrage": {
                if (from !== "lockedmove" && notarget === undefined) {
                  volatiles["Locked Move"] = { turn: 0, move }
                }
                if (from === "lockedmove") deductPP = false
                break
              }
              case "Wish": {
                this[pov].wish = 0
              }
            }
          } else {
            user.lastMove.missed = true
            if (volatiles["Locked Move"]) delete volatiles["Locked Move"]
          }
        }

        if (!volatiles["Choice Locked"] && user.item && CHOICE_ITEMS.includes(user.item)) {
          volatiles["Choice Locked"] = { move }
        }
        if (deductPP)
          moveset[move] =
            (moveset[move] ?? 0) +
            (target?.ability === "Pressure" && isPressureMove(this.gen.moves.get(move)!) ? 2 : 1)

        break
      }
      case "cant": {
        p = piped(line, p.i, 2)
        const { lastMove } = this.member(this.label(p.args[0]))
        if (lastMove) lastMove.failed = true
        break
      }
      case "-heal":
      case "-sethp": {
        p = piped(line, p.i, 2)
        const user = this.member(this.label(p.args[0]))

        user.hp = parseHp(p.args[1])!

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)

        const { ability, item } = parseEffect(from)
        if (ability) this.setAbility(user, ability)

        // berries already include an -enditem
        if (item === "Leftovers") this.setItem(user, item)

        break
      }
      case "-immune": {
        p = piped(line, p.i)
        const { pov } = this.member(this.label(p.args[0]))
        const { active } = this[OPP[pov]]

        if (active.lastMove) active.lastMove.failed = true
        break
      }
      case "-damage": {
        p = piped(line, p.i, 2)

        const user = this.member(this.label(p.args[0]))
        const hp = parseHp(p.args[1])

        if (hp) user.hp = hp
        else user.hp[0] = 0

        {
          p = piped(line, p.i, -1)

          const { from, of } = parseTags(p.args)

          const { item, ability } = parseEffect(from)
          const target = of ? this.member(this.label(of)) : user

          if (ability) this.setAbility(target, ability)
          if (item) this.setItem(target, item)
        }

        break
      }
      case "-boost":
      case "-unboost": {
        p = piped(line, p.i, 3)
        const user = this.member(this.label(p.args[0]))

        const id = p.args[1] as BoostId
        const n = Number(p.args[2])
        user.boosts[p.args[1] as BoostId] = Math.min(
          Math.max(-6, (user.boosts[id] ?? 0) + (msgType === "-boost" ? n : -n)),
          6
        )

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)
        const { item } = parseEffect(from)

        // boosts from item consume it
        if (item) {
          this.setItem(user, item)
          this.setItem(user, null)
        }

        break
      }
      case "-clearboost": {
        p = piped(line, p.i)
        const { pov } = this.label(p.args[0])

        this[pov].active!.boosts = {}
        break
      }
      case "-clearallboost": {
        for (const pov of POVS) {
          this[pov].active!.boosts = {}
        }
        break
      }
      case "-clearnegativeboost": {
        p = piped(line, p.i)
        const { boosts } = this.member(this.label(p.args[0]))

        for (const k in boosts) {
          const id = k as BoostId
          boosts[id] = Math.max(0, boosts[id]!)
        }
        break
      }
      case "-item": {
        p = piped(line, p.i, 2)
        const user = this.member(this.label(p.args[0]))
        const item = p.args[1]

        p = piped(line, p.i, -1)
        const { from, of, identify } = parseTags(p.args)
        const { ability } = parseEffect(from)

        // treat as replacing existing item, important for choice items
        if (identify === undefined) this.setItem(user, null)
        this.setItem(user, item)

        const src = of ? this.member(this.label(of)) : undefined

        if (identify !== undefined) {
          this.setAbility(src!, ability!)
          break
        }

        if (ability) this.setAbility(user, ability)

        // magician doesnt emit an -enditem
        if (ability === "Magician") {
          this.setItem(src!, item)
          this.setItem(src!, null)
        }
        break
      }
      case "-enditem": {
        p = piped(line, p.i, 2)
        const user = this.member(this.label(p.args[0]))
        const item = p.args[1]

        this.setItem(user, item)
        this.setItem(user, null)

        p = piped(line, p.i, -1)
        const { eat } = parseTags(p.args)

        if (eat != null) {
          user.lastBerry = {
            name: item,
            turn: 0
          }
        }

        break
      }
      case "-transform": {
        p = piped(line, p.i, 2)
        const user = this.member(this.label(p.args[0]))
        const into = this.member(this.label(p.args[1]))

        const { pov, volatiles } = user

        let ability
        let moves
        if (pov === "ally") {
          const {
            side: { pokemon }
          } = this.request

          const pkmn = pokemon.find((x) => this.label(x.ident).species === user.species)!
          ability = this.gen.abilities.get(pkmn.ability)!.name
          moves = pkmn.moves.map((x) => this.gen.moves.get(x)!.name)

          {
            const { moveset } = into

            into.ability = ability
            for (const move of moves) moveset[move] = moveset[move] ?? 0
          }
        } else {
          const user = into as AllyUser
          ability = user.ability
          moves = Object.keys(user.moveset)
        }

        const { species, gender, boosts } = into

        volatiles["Transform"] = {
          into,
          species,
          gender,
          moveset: Object.fromEntries(moves.map((x) => [x, 0])),
          ability,
          boosts: { ...boosts }
        }

        {
          p = piped(line, p.i, -1)
          const { from } = parseTags(p.args)
          const { ability } = parseEffect(from)
          if (ability) this.setAbility(user, ability)
        }

        break
      }
      case "-start": {
        p = piped(line, p.i, 2)

        const user = this.member(this.label(p.args[0]))
        let { stripped: name } = parseEffect(p.args[1])

        const { pov, volatiles } = user
        const opp = OPP[pov]

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
              this[opp].active.volatiles[name] = {
                turn: 0,
                user
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
                move: user.lastMove!.name
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
        const { from, of, fatigue } = parseTags(p.args)

        const { ability, item } = parseEffect(from)
        const src = of ? this.member(this.label(of)) : user

        if (ability) this.setAbility(src, ability)
        if (item) this.setItem(src, item)
        if (fatigue !== undefined) delete volatiles["Locked Move"]

        break
      }
      case "-terastallize": {
        p = piped(line, p.i, 2)
        const user = this.member(this.label(p.args[0]))
        user.tera = true
        break
      }
      case "-formechange": {
        p = piped(line, p.i, 2)
        const user = this.member(this.label(p.args[0]))
        const forme = p.args[1]

        user.forme = forme

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)
        const { ability } = parseEffect(from)

        if (ability) this.setAbility(user, ability)

        break
      }
      case "detailschange": {
        p = piped(line, p.i, 2)
        const user = this.member(this.label(p.args[0]))
        const { forme } = parseTraits(p.args[1])

        user.forme = forme
        break
      }
      case "-activate": {
        p = piped(line, p.i, 2)
        const user = this.member(this.label(p.args[0]))
        const { pov } = user
        const opp = OPP[pov]

        let { ability, item, move, stripped } = parseEffect(p.args[1])

        if (stripped === "Orichalcum Pulse") ability = stripped

        if (item) {
          switch (item) {
            case "Leppa Berry": {
              p = piped(line, p.i, 1)
              user.moveset[p.args[0]] = 0
              break
            }
          }
        } else if (move) {
          switch (move) {
            case "Poltergeist": {
              p = piped(line, p.i)
              this.setItem(user, p.args[0])
              break
            }
            case "Magma Storm":
            case "Infestation":
            case "Whirlpool": {
              user.volatiles["Partially Trapped"] = { turn: 0 }
              break
            }
            case "Protect": {
              this[opp].active.lastMove!.failed = true
              break
            }
          }
        } else if (ability) {
          if (ability === "Battle Bond") {
            user.flags[ability] = true
          }

          this.setAbility(user, ability)
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
        ;[ally.fields, foe.fields] = [foe.fields, ally.fields]
        break
      }
      case "replace": {
        p = piped(line, p.i, 3)
        const { pov, species } = this.label(p.args[0])
        const { forme, lvl, gender } = parseTraits(p.args[1])

        const src = this[pov]
        const { team } = src

        if (pov === "foe") {
          const active = src.active as FoeUser

          {
            const {
              species,
              forme,
              lvl,
              gender,
              initial: { formeId }
            } = active

            team[species] = {
              pov: "foe",
              flags: {},
              species,
              forme,
              lvl,
              gender,
              hp: [100, 100],
              moveset: {},
              volatiles: {},
              boosts: {},
              tera: false,
              initial: {
                formeId
              }
            }
          }

          active.species = species
          active.forme = forme
          active.lvl = lvl
          active.gender = gender
          active.initial.formeId = this.gen.species.get(forme)!.id
          team[species] = active
        } else {
          delete this.illusion
        }

        break
      }
      case "-end": {
        p = piped(line, p.i, 2)
        const user = this.member(this.label(p.args[0]))

        let { stripped: name } = parseEffect(p.args[1])

        const { volatiles } = user

        if (name.startsWith("fallen")) {
          name = "Fallen"
        }

        delete volatiles[name]
        break
      }
      case "-singleturn": {
        p = piped(line, p.i, 2)
        const user = this.member(this.label(p.args[0]))
        const { stripped: name } = parseEffect(p.args[1])

        user.volatiles[name] = { singleTurn: true }
        break
      }
      case "-singlemove": {
        p = piped(line, p.i, 2)
        const user = this.member(this.label(p.args[0]))
        const { stripped: name } = parseEffect(p.args[1])

        user.volatiles[name] = { singleMove: true }
        break
      }
      case "faint": {
        p = piped(line, p.i)
        const user = this.member(this.label(p.args[0]))
        user.hp[0] = 0

        break
      }
      case "-sidestart": {
        p = piped(line, p.i, 2)
        const { pov } = this.label(p.args[0])
        const { stripped: name } = parseEffect(p.args[1])

        const { fields } = this[pov]
        if (HAZARDS.includes(name)) (fields[name] ?? { layers: 0 }).layers!++
        else fields[name] = { turn: 0 }

        break
      }
      case "-sideend": {
        p = piped(line, p.i, 2)
        const { pov } = this.label(p.args[0])
        const { stripped: name } = parseEffect(p.args[1])

        const { fields: conditions } = this[pov]
        delete conditions[name]

        break
      }
      case "upkeep": {
        const { fields } = this
        for (const name in fields) fields[name]++

        break
      }
      case "turn": {
        p = piped(line, p.i)
        this.turn = Number(p.args[0])

        for (const pov of POVS) {
          const side = this[pov]
          const { fields: conditions } = side

          const {
            active: { lastBerry, lastMove, volatiles, status }
          } = side

          if (lastBerry) lastBerry.turn++

          if (status?.turn !== undefined) status.turn++

          if (volatiles["Recharge"]?.turn === 1) delete volatiles["Recharge"]

          for (const name in volatiles) {
            if (volatiles[name]?.turn !== undefined) volatiles[name].turn++
            if (volatiles[name].singleTurn) delete volatiles[name]

            if (name === "Locked Move" && lastMove?.failed) delete volatiles[name]
          }

          for (const name in conditions) {
            if (conditions[name]?.turn !== undefined) conditions[name].turn++
          }

          if (side.wish) {
            if (side.wish === 0) side.wish++
            else delete side.wish
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
