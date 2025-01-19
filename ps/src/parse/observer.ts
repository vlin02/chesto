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
import { StatusId, StatId, BoostId, CHOICE_ITEMS } from "./species.js"
import { Ally, Foe, HAZARDS, OPP, POV, POVS } from "./side.js"
import { AllyUser, FoeUser, getMaxPP, MoveSet, User } from "./user.js"
import { isLocked, isPressured } from "./move.js"

type Label = {
  species: string
  pov: POV
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
    const { illusion } = this
    const user = this[pov].team[species]

    if (illusion?.to === user) return illusion.from
    return user
  }

  switchOut(user: User) {
    user.volatiles = {}
    user.boosts = {}
    delete user.lastBerry
    delete user.lastMove
  }

  setAbility({ forme, volatiles, base }: User, ability: string) {
    // As One is treated as two abilities, with separate messages
    if (["Calyrex-Shadow", "Calyrex-Ice"].includes(forme)) return
    if (volatiles["Trace"] || volatiles["Transform"]) return

    base.ability = ability
  }

  setItem(user: User, item: string | null) {
    user.item = item
    if (user.pov === "ally") return

    const { base } = user
    base.item = base.item ?? item ?? undefined
  }

  disrupt(user: User) {
    delete user.volatiles["Locked Move"]
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
          active: choice,
          side: { id, name, pokemon: members }
        } = this.request

        if (!this.ally) {
          this.side = id
          this.name = name

          this.ally = { fields: {}, team: {}, teraUsed: false } as Ally

          for (const member of members) {
            const user = new AllyUser(this.gen, member)
            if (member.active) this.ally.active = user
            const { species } = user

            this.ally.team[species] = user
          }
        }

        const {
          active: { volatiles }
        } = this.ally

        if (choice && volatiles["Locked Move"]) {
          const [{ moves }] = choice
          const { move: name } = volatiles["Locked Move"]
          if (!moves.every((x) => x.disabled || x.move === name)) delete volatiles["Locked Move"]
        }

        break
      }
      case "-ability": {
        p = piped(line, p.i, 2)
        const user = this.member(this.label(p.args[0]))
        const ability = p.args[1]

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)

        const { ability: fromAbility } = parseEffect(from)
        if (ability === "Intrepid Sword") {
          user.flags[ability] = true
        }

        if (fromAbility === "Trace") {
          const target = this.member(this.label(of))

          user.volatiles["Trace"] = { ability }
          user.base.ability = "Trace"
          
          this.setAbility(target, ability)
        } else {
          this.setAbility(user, ability)
        }

        break
      }
      case "faint": {
        p = piped(line, p.i)
        const user = this.member(this.label(p.args[0]))

        user.tera = false
        user.hp[0] = 0
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

        let user: User

        if (pov === "ally") {
          user = this.member(label)
        } else {
          const team = this.foe?.team ?? {}
          user = team[species]
          if (!user) {
            user = team[species] = new FoeUser(this.gen, species, traits)

            const { forme, base } = user
            base.ability = {
              "Calyrex-Ice": "As One (Glastrier)",
              "Calyrex-Shadow": "As One (Spectrier)"
            }[forme]
          }

          if (!this.foe) {
            this.foe = { fields: {}, team: { [species]: user }, active: user, teraUsed: true }
          }
        }

        const { active: prev } = this[pov]

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)

        const { status } = user
        if (status?.id === "tox") status.turn! = 0

        if (prev.hp[0] && prev.ability === "Regenerator") {
          const { hp } = prev
          const heal = Math.floor(hp[1] / 3)
          hp[0] = Math.min(hp[0] + heal, hp[1])
        }

        if (from === "Shed Tail" && "Substitute" in prev.volatiles) {
          user.volatiles["Substitute"] = prev.volatiles["Substitute"]
        }

        this.switchOut(prev)

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
        if (ability) {
          const user = this.member(this.label(of))
          this.setAbility(user, ability)
        }

        break
      }
      case "-fieldstart": {
        p = piped(line, p.i)
        const { move: name } = parseEffect(p.args[0])

        this.fields[name!] = 0

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)

        const { ability } = parseEffect(from)

        const user = this.member(this.label(of))
        if (ability) this.setAbility(user, ability)
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
          attempt: id === "slp" ? 0 : undefined
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
        const name = p.args[1]

        const { pov, volatiles, status } = user
        const move = this.gen.moves.get(name)!

        p = piped(line, p.i, -1)
        const { from, notarget, miss } = parseTags(p.args)
        const effect = parseEffect(from)

        let deductPP = true
        user.lastMove = name

        for (const name in volatiles) {
          if (volatiles[name].singleMove) delete volatiles[name]
        }

        if (status?.attempt) status.attempt++

        if (effect.move) deductPP = false
        if (effect.ability) {
          this.setAbility(user, effect.ability)
          deductPP = false
        }

        if (isLocked(move)) {
          if (from === "lockedmove" && volatiles["Locked Move"]) {
            const n = volatiles["Locked Move"]!.attempt++
            if (n === 2) delete volatiles["Locked Move"]
          } else {
            volatiles["Locked Move"] = { attempt: 0, move: name }
          }
        }

        if (user.item && CHOICE_ITEMS.includes(user.item)) {
          volatiles["Choice Locked"] = { name }
        }

        if (notarget != null || miss != null) this.disrupt(user)
        if (name === "Wish") this[pov].wish = 0

        if (deductPP) {
          const slot = (user.moveSet[name] = user.moveSet[name] ?? {
            used: 0,
            max: getMaxPP(move)
          })

          slot.used += isPressured(move) ? 2 : 1
        }

        break
      }
      case "-immune":
        p = piped(line, p.i)
        const { pov } = this.member(this.label(p.args[0]))

        this.disrupt(this[OPP[pov]].active)
        break
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
      case "-damage": {
        p = piped(line, p.i, 2)

        const user = this.member(this.label(p.args[0]))
        const hp = parseHp(p.args[1])

        if (hp) user.hp = hp
        else user.hp[0] = 0

        p = piped(line, p.i, -1)

        const { from, of } = parseTags(p.args)

        const { item, ability } = parseEffect(from)
        const target = of ? this.member(this.label(of)) : user

        if (ability) this.setAbility(target, ability)
        if (item) this.setItem(target, item)

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

        p = piped(line, p.i, -1)
        const { eat } = parseTags(p.args)

        this.setItem(user, item)
        this.setItem(user, null)

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

        p = piped(line, p.i, -1)

        let ability
        let moveNames: string[] = []
        if (pov === "ally") {
          const {
            side: { pokemon: members }
          } = this.request

          const member = members.find((x) => this.label(x.ident).species === user.species)!
          ability = this.gen.abilities.get(member.ability)!.name

          const { moveSet } = into
          const moves = member.moves.map((id) => this.gen.moves.get(id)!)

          this.setAbility(into, ability)
          for (const move of moves) {
            const { name } = move
            moveSet[name] = moveSet[name] ?? {
              used: 0,
              max: getMaxPP(move)
            }
          }
        } else {
          const user = into as AllyUser
          ability = user.ability
          moveNames = Object.keys(user.moveSet)
        }

        const moveSet: MoveSet = {}
        for (const name of moveNames) {
          moveSet[name] = {
            used: 0,
            max: 5
          }
        }

        const { species, gender, boosts } = into

        volatiles["Transform"] = {
          into,
          species,
          gender,
          moveSet,
          ability,
          boosts: { ...boosts }
        }

        this.setAbility(user, "Imposter")

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

        if (ability) this.setAbility(src, ability)
        if (item) this.setItem(src, item)
        if (fatigue !== undefined) delete volatiles["Locked Move"]
        break
      }
      case "-terastallize": {
        p = piped(line, p.i, 2)
        const user = this.member(this.label(p.args[0]))
        const { pov } = user

        this[pov].teraUsed = false
        user.tera = true
        break
      }
      case "-formechange": {
        p = piped(line, p.i, 2)
        const user = this.member(this.label(p.args[0]))
        const forme = p.args[1]

        // Shaymin emits both a forme & detailchange. ignore forme.
        if (forme !== "Shaymin") {
          user.formeChange = {
            name: forme,
            reverts: true
          }
        }

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

        user.formeChange = {
          name: forme,
          reverts: false
        }

        if (forme === "Terapagos-Terastal") {
          user.base.ability = "Tera Shell"
        } else if (forme === "Terapagos-Stellar") {
          user.base.ability = "Teraform Zero"
        }

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
              user.moveSet[p.args[0]].used = 0
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

        if (pov === "foe") {
          const { forme, lvl, gender } = parseTraits(p.args[1])

          const { team } = this.foe
          const { active: user } = this.foe
          team[user.species] = user.clone()

          user.species = species
          user.base.forme = forme
          user.lvl = lvl
          user.gender = gender
          user.base.forme = this.gen.species.get(forme)!.id
          team[species] = user
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

        if (name.startsWith("fallen")) name = "Fallen"

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
            active: { lastBerry, volatiles, status }
          } = side

          if (lastBerry) lastBerry.turn++
          if (status?.turn !== undefined) status.turn++
          if (side.wish) side.wish++

          for (const name in volatiles) {
            if (volatiles[name].turn !== undefined) volatiles[name].turn++
            if (volatiles[name].singleTurn) delete volatiles[name]
          }

          for (const name in conditions) {
            if (conditions[name].turn !== undefined) conditions[name].turn++
          }

          if (volatiles["Recharge"]?.turn === 2) delete volatiles["Recharge"]
          if (side.wish === 2) delete side.wish
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
