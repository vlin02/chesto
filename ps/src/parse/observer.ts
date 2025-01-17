import { Generation } from "@pkmn/data"
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
import { POV, Ally, Foe, MoveSet, AllyUser, OPP, POVS, User, FoeUser } from "./pov.js"

function clear(user: User) {
  user.volatiles = {}
  user.boosts = {}
  delete user.lastBerry
  delete user.lastMove
}

function setItem(user: User, item: string | null) {
  user.item = item

  const { volatiles, pov } = user
  if (volatiles["Choice Locked"]) delete volatiles["Choice Locked"]

  if (pov === "foe" && item) {
    const { initial } = user
    initial.item = initial.item ?? item
  }
}

function setAbility(user: User, ability: string | null) {
  user.ability = ability
  if (user.pov === "foe" && ability) {
    const { initial } = user
    initial.ability = initial.ability ?? ability
  }
}

type Label = {
  species: string
  pov: POV
}

export class Observer {
  side!: Side
  name!: string
  ally!: Ally
  foe!: Foe

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

  label(s: string): Label {
    const { side, species } = parseLabel(s)
    return { pov: side === this.side ? "ally" : "foe", species }
  }

  member({ pov, species }: Label) {
    return this[pov].team[species]
  }

  setItem(memb: User, item: string | null) {
    if (memb.volatiles["Choice Locked"]) {
      delete memb.volatiles["Choice Locked"]
    }

    memb.item = item
    if (memb.pov === "foe" && item) {
      const { initial } = memb
      initial.item = initial.item ?? item
    }
  }

  read(line: string) {
    if (line[0] !== "|") return

    let p: { args: string[]; i: number }
    p = piped(line, 0)
    const msgType = p.args[0]

    switch (msgType) {
      case "request": {
        const request = JSON.parse(line.slice(p.i + 1)) as ChoiceRequest

        const {
          side: { id, name, pokemon: pokemons }
        } = request

        if (!this.ready) {
          this.side = id
          this.name = name

          const team: { [k: string]: AllyUser } = {}
          let active: AllyUser

          for (const {
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

            const moveset: MoveSet = {}
            for (const move of moves) moveset[this.gen.moves.get(move)!.name] = 0

            const member = (team[species] = {
              pov: "ally",
              species,
              once: {},
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
        }

        const { active } = this.ally
        const { volatiles } = active

        if (volatiles["Transform"]?.copied === false) {
          const { into } = volatiles["Transform"]

          const { moves, ability } = pokemons.find(
            ({ ident }) => parseLabel(ident).species === active.species
          )!

          setAbility(into, ability)

          const moveset: MoveSet = {}
          for (const move of moves) {
            const { name } = this.gen.moves.get(move)!
            into.moveset[name] = into.moveset[name] ?? 0
            moveset[name] = 0
          }

          const { gender, species } = into

          volatiles["Transform"] = {
            copied: true,
            into,
            species,
            gender,
            moveset,
            ability,
            boosts: { ...this.foe.active!.boosts }
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
            setAbility(user, prev)
            setAbility(this.member(this.label(of)), ability)
          }
        }

        if (ability === "Intrepid Sword") {
          user.once[ability] = true
        }

        setAbility(user, ability)
        break
      }
      case "switch":
      case "drag": {
        p = piped(line, p.i, 3)
        let label = this.label(p.args[0])

        const { pov, species } = label
        const party = this[pov]
        const { active, team } = party

        const traits = parseTraits(p.args[1])
        const hp = parseHp(p.args[2])!

        let user: User

        if (pov === "ally") {
          user = this.member(label)
        } else {
          const { forme, lvl, gender } = traits

          user = team[species] = team[species] ?? {
            pov: "foe",
            once: {},
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

        clear(user)
        const { status } = user
        if (status?.id === "tox") status.turn! = 0

        if (from === "Shed Tail") {
          user.volatiles["Substitute"] = active.volatiles["Substitute"]
        }

        party.active = user
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
        if (ability) setAbility(this.member(this.label(of)), ability)

        break
      }
      case "-fieldstart": {
        p = piped(line, p.i)
        const { move: name } = parseEffect(p.args[0])

        this.fields[name!] = 0

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)

        const { ability } = parseEffect(from)
        if (ability) setAbility(this.member(this.label(of)), ability)
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

          if (item) setItem(src, item)
          if (ability) setAbility(src, ability)
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
        if (ability) setAbility(target, ability)

        break
      }
      case "move": {
        p = piped(line, p.i, 3)
        const user = this.member(this.label(p.args[0]))
        const move = p.args[1]

        const { pov } = user
        const src = this[pov]
        const opp = this[OPP[pov]]

        const { volatiles, moveset, status } = user

        if (user.item && CHOICE_ITEMS.includes(user.item)) {
          volatiles["Choice Locked"] = { move }
        }

        let deductPP = true

        {
          p = piped(line, p.i, -1)
          const { from, notarget, miss } = parseTags(p.args)

          user.lastMove = move

          for (const name in volatiles) {
            if (volatiles[name].singleMove) delete volatiles[name]
          }

          if (status?.move) status.move++

          const { ability } = parseEffect(from)
          if (ability) setAbility(user, ability)

          if (miss === undefined) {
            switch (move) {
              case "Outrage": {
                if (from !== "lockedmove" && notarget === undefined) {
                  volatiles["Locked Move"] = { turn: 0, move: "Outrage" }
                }
                if (from === "lockedmove") deductPP = false
                break
              }
              case "Wish": {
                src.wish = 0
              }
            }
          } else {
            if (volatiles["Locked Move"]) delete volatiles["Locked Move"]
          }
        }

        if (deductPP)
          moveset[move] = (moveset[move] ?? 0) + (opp.active.ability === "Pressure" ? 2 : 1)

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
        if (ability) setAbility(user, ability)

        // berries already include an -enditem
        if (item === "Leftovers") setItem(user, item)

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

          if (ability) setAbility(target, ability)
          if (item) setItem(target, item)
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
          setItem(user, item)
          setItem(user, null)
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

        setItem(user, item)

        p = piped(line, p.i, -1)
        const { from, of, identify } = parseTags(p.args)
        const { ability } = parseEffect(from)

        const src = of ? this.member(this.label(of)) : undefined

        if (identify) {
          setAbility(src!, ability!)
          break
        }

        if (ability) setAbility(user, ability)

        // magician doesnt emit an -enditem
        if (ability === "Magician") {
          setItem(src!, item)
          setItem(src!, null)
        }
        break
      }
      case "-enditem": {
        p = piped(line, p.i, 2)
        const user = this.member(this.label(p.args[0]))
        const item = p.args[1]

        setItem(user, item)
        setItem(user, null)

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
        const from = this.member(this.label(p.args[0]))
        const into = this.member(this.label(p.args[1]))

        from.volatiles["Transform"] = {
          into,
          copied: false
        }

        break
      }
      case "-start": {
        p = piped(line, p.i, 2)

        const user = this.member(this.label(p.args[0]))
        let { stripped: name } = parseEffect(p.args[1])

        const { volatiles } = user
        const opp = this[OPP[user.pov]]

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
              opp.active.volatiles[name] = {
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
                move: user.lastMove!
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

        if (ability) setAbility(src, ability)
        if (item) setItem(src, item)
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

        if (ability) setAbility(user, ability)

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
              setItem(user, p.args[0])
              break
            }
            case "Magma Storm":
            case "Infestation":
            case "Whirlpool": {
              user.volatiles["Partially Trapped"] = { turn: 0 }
              break
            }
          }
        } else if (ability) {
          if (ability === "Battle Bond") {
            user.once[ability] = true
          }

          setAbility(user, ability)
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
        const { team, active } = src

        if (pov === "foe") {
          {
            const {
              species,
              forme,
              lvl,
              gender,
              initial: { formeId }
            } = active as FoeUser

            team[species] = {
              pov: "foe",
              once: {},
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
          ;(active as FoeUser).initial.formeId = this.gen.species.get(forme)!.id
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
      case "-siuserart": {
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
          const { fields: conditions } = side

          const {
            active: { lastBerry, volatiles, status }
          } = side

          if (lastBerry) lastBerry.turn++

          if (status?.turn !== undefined) status.turn++

          if (volatiles["Recharge"]?.turn === 1) delete volatiles["Recharge"]

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
