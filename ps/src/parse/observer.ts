import { Generation, TypeName } from "@pkmn/data"
import {
  ChoiceRequest,
  parseEntity,
  parseCondition,
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
import { AllyUser, FoeUser, MoveSet, User } from "./user.js"
import { getMaxPP, isLocking, isPressured } from "./move.js"
import { checkLocked } from "./request.js"

type Label = {
  species: string
  pov: POV
}

type Line = {
  dancer?: boolean
  sleepTalk?: boolean
}

export class Observer {
  side!: Side
  name!: string
  ally!: Ally
  foe!: Foe
  request!: ChoiceRequest
  outrageActive: boolean

  illusion?: {
    from: AllyUser
    to: AllyUser
  }

  private gen: Generation

  turn: number
  fields: { [k: string]: number }
  weather?: { name: WeatherName; turn: number }
  winner?: POV
  prevLine?: Line

  constructor(gen: Generation) {
    this.gen = gen
    this.fields = {}
    this.turn = 0
    this.outrageActive = false
  }

  label(s: string): Label {
    const { side, species } = parseLabel(s)
    return { pov: side === this.side ? "ally" : "foe", species }
  }

  user({ pov, species }: Label) {
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

  setAbility(user: User, ability: string) {
    const { volatiles, base } = user
    // As One is treated as two abilities, with separate messages
    if (
      user.ability?.startsWith("As One") &&
      ["Unnerve", "Chilling Neigh", "Grim Neigh", "As One"].includes(ability)
    )
      return
    if (volatiles["Trace"] || volatiles["Transform"]) return

    base.ability = ability
  }

  setItem(user: User, item: string | null) {
    const { volatiles } = user

    if (item === null && volatiles["Choice Locked"]) delete volatiles["Choice Locked"]
    user.item = item

    if (user.pov === "ally") return

    user.firstItem = user.firstItem ?? item ?? undefined
  }

  disrupt(user: User) {
    delete user.volatiles["Locked Move"]
  }

  allocateSlot(moveSet: MoveSet, move: string) {
    return (moveSet[move] = moveSet[move] ?? {
      used: 0,
      max: getMaxPP(this.gen, move)
    })
  }

  read(line: string) {
    if (line[0] !== "|") return

    let p: { args: string[]; i: number }
    p = piped(line, 0)
    const msgType = p.args[0]

    const currLine: Line = {}

    switch (msgType) {
      case "request": {
        this.request = JSON.parse(line.slice(p.i + 1))

        const {
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

        const { volatiles } = this.ally.active
        const { "Locked Move": lockedMove } = volatiles

        if (lockedMove && checkLocked(this.request, lockedMove.move)) {
          delete volatiles["Locked Move"]
        }

        break
      }
      case "-ability": {
        p = piped(line, p.i, 2)
        const user = this.user(this.label(p.args[0]))
        const ability = p.args[1]

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)
        const cause = parseEntity(from)

        if (ability === "Intrepid Sword") user.flags[ability] = true
        if (ability === "Pressure") user.volatiles["Pressure"] = {}

        if (cause.ability === "Trace") {
          const target = this.user(this.label(of))

          this.setAbility(user, "Trace")
          user.volatiles["Trace"] = { ability }

          this.setAbility(target, ability)
        } else {
          this.setAbility(user, ability)
        }

        break
      }
      case "faint": {
        p = piped(line, p.i)
        const user = this.user(this.label(p.args[0]))

        user.tera = false
        user.hp[0] = 0
        user.status = undefined
        break
      }
      case "switch":
      case "drag": {
        p = piped(line, p.i, 3)
        let label = this.label(p.args[0])
        const { pov, species } = label

        if (pov === "ally") {
          const { ident } = this.request.side.pokemon.find((x) => x.active)!
          const { species } = this.label(ident)

          const user = this.ally.team[species]

          const {
            ability,
            flags: { "Illusion revealed": revealed }
          } = user

          if (ability === "Illusion" && !revealed) {
            const target = [...this.request.side.pokemon]
              .reverse()
              .find((x) => parseCondition(x.condition) !== null && !x.active)
            if (target) {
              const to = this.ally.team[this.label(target.ident).species]
              this.illusion = { from: user, to }
            }
          } else {
            delete this.illusion
          }
        }

        const traits = parseTraits(p.args[1])

        let user: User

        if (pov === "ally") {
          user = this.user(label)
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
        const user = this.user(this.label(p.args[0]))
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
        const { ability } = parseEntity(from)

        if (upkeep === "") {
          this.weather!.turn++
          break
        }

        this.weather = { name, turn: 0 }
        if (ability) {
          const user = this.user(this.label(of))
          this.setAbility(user, ability)
        }

        break
      }
      case "-fieldstart": {
        p = piped(line, p.i)
        const { move: name } = parseEntity(p.args[0])

        this.fields[name!] = 0

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)

        const { ability } = parseEntity(from)

        const user = this.user(this.label(of))
        if (ability) this.setAbility(user, ability)
        break
      }
      case "-fieldend": {
        p = piped(line, p.i)
        const { move: field } = parseEntity(p.args[0])

        delete this.fields[field!]
        break
      }
      case "-status": {
        p = piped(line, p.i, 2)
        const user = this.user(this.label(p.args[0]))
        const id = p.args[1] as StatusId

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)

        user.status = {
          id,
          turn: id === "tox" ? 0 : undefined,
          attempt: id === "slp" ? 0 : undefined
        }

        const src = of ? this.user(this.label(of)) : user
        const { ability, item } = parseEntity(from)

        if (item) this.setItem(src, item)
        if (ability) this.setAbility(src, ability)
        break
      }
      case "-curestatus": {
        p = piped(line, p.i, 2)
        const target = this.user(this.label(p.args[0]))

        delete target.status

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)

        const { ability } = parseEntity(from)
        if (ability) this.setAbility(target, ability)

        break
      }
      case "move": {
        p = piped(line, p.i, 3)
        const user = this.user(this.label(p.args[0]))
        const move = p.args[1]

        const { pov, volatiles, status } = user

        p = piped(line, p.i, -1)
        const { from, notarget, miss } = parseTags(p.args)
        const cause = parseEntity(from)

        for (const name in volatiles) {
          if (volatiles[name].singleMove) delete volatiles[name]
        }

        if (status?.attempt) status.attempt++
        if (cause.ability) this.setAbility(user, cause.ability)
        if (notarget != null || miss != null) this.disrupt(user)

        let deductFrom: string | null = move
        let choiceLockable = true

        switch (move) {
          case "Sleep Talk":
            currLine.sleepTalk = true
            deductFrom = null
            break
          case "Wish":
            this[pov].wish = 0
            break
          case "Struggle":
            deductFrom = null
            break
        }

        if (cause.move === "Sleep Talk") deductFrom = "Sleep Talk"
        if (cause.ability === "Magic Bounce") {
          deductFrom = null
          choiceLockable = false
        }
        if (this.prevLine?.dancer) deductFrom = null

        if (from === "lockedmove") {
          deductFrom = null
          const { "Locked Move": lockedMove } = volatiles

          if (lockedMove) {
            const n = lockedMove.attempt++
            if (n === 2) delete volatiles["Locked Move"]
          }
        }

        if (choiceLockable && user.item && CHOICE_ITEMS.includes(user.item)) {
          const locked = volatiles["Choice Locked"]?.move
          if (locked && locked !== move) deductFrom = null
          else volatiles["Choice Locked"] = { move }
        }

        if (deductFrom) {
          const slot = this.allocateSlot(user.moveSet, deductFrom)

          slot.used +=
            this[OPP[pov]].active.volatiles["Pressure"] &&
            this[OPP[pov]].active.hp[0] !== 0 &&
            (move === "Curse" ? user.types.includes("Ghost") : isPressured(this.gen, move))
              ? 2
              : 1

          user.lastMove = deductFrom
        }

        if (isLocking(this.gen, move) && (pov === "foe" || checkLocked(this.request, move))) {
          volatiles["Locked Move"] = { attempt: 0, move: move }
        }

        break
      }
      case "cant":
      case "-fail":
        p = piped(line, p.i)
        const { moveSet } = this.user(this.label(p.args[0]))
        if (this.prevLine?.sleepTalk) this.allocateSlot(moveSet, "Sleep Talk").used++
        break
      case "-immune":
        p = piped(line, p.i)
        const { pov } = this.user(this.label(p.args[0]))

        this.disrupt(this[OPP[pov]].active)
        break
      case "-heal":
      case "-sethp": {
        p = piped(line, p.i, 2)
        const user = this.user(this.label(p.args[0]))
        const { hp } = parseCondition(p.args[1])

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)

        user.hp = hp!

        const { ability, item, move } = parseEntity(from)
        if (ability) this.setAbility(user, ability)
        if (move === "Lunar Dance") {
          for (const move in user.moveSet) {
            user.moveSet[move].used = 0
          }
        }

        // berries already include an -enditem
        if (item === "Leftovers") this.setItem(user, item)

        break
      }
      case "-damage": {
        p = piped(line, p.i, 2)

        const user = this.user(this.label(p.args[0]))
        const { hp } = parseCondition(p.args[1])

        if (hp) user.hp = hp
        else user.hp[0] = 0

        p = piped(line, p.i, -1)

        const { from, of } = parseTags(p.args)

        const { item, ability } = parseEntity(from)
        const target = of ? this.user(this.label(of)) : user

        if (ability) this.setAbility(target, ability)
        if (item) this.setItem(target, item)

        break
      }
      case "-boost":
      case "-unboost": {
        p = piped(line, p.i, 3)
        const user = this.user(this.label(p.args[0]))

        const id = p.args[1] as BoostId
        const n = Number(p.args[2])
        user.boosts[p.args[1] as BoostId] = Math.min(
          Math.max(-6, (user.boosts[id] ?? 0) + (msgType === "-boost" ? n : -n)),
          6
        )

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)
        const { item } = parseEntity(from)

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
        const { boosts } = this.user(this.label(p.args[0]))

        for (const k in boosts) {
          const id = k as BoostId
          boosts[id] = Math.max(0, boosts[id]!)
        }
        break
      }
      case "-item": {
        p = piped(line, p.i, 2)
        const user = this.user(this.label(p.args[0]))
        const item = p.args[1]

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)
        const { ability } = parseEntity(from)

        const src = of ? this.user(this.label(of)) : undefined

        if (ability === "Frisk") {
          this.setItem(user, item)
          this.setAbility(src!, ability!)
          break
        }

        // treat as replacing existing item, important for choice items
        this.setItem(user, null)
        this.setItem(user, item)

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
        const user = this.user(this.label(p.args[0]))
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
        const from = this.user(this.label(p.args[0]))
        const to = this.user(this.label(p.args[1]))

        const { pov, volatiles } = from

        p = piped(line, p.i, -1)

        let ability
        let moves: string[] = []

        if (pov === "ally") {
          const {
            side: { pokemon: members }
          } = this.request

          const member = members.find((x) => this.label(x.ident).species === from.species)!
          ability = this.gen.abilities.get(member.ability)!.name

          const { moveSet } = to
          moves = member.moves.map((id) => this.gen.moves.get(id)!.name)

          for (const move of moves) {
            moveSet[move] = moveSet[move] ?? {
              used: 0,
              max: getMaxPP(this.gen, move)
            }
          }

          this.setAbility(to, ability)
        } else {
          const user = to as AllyUser
          ability = user.ability
          moves = Object.keys(user.moveSet)
        }

        const { boosts } = to
        const { forme, gender } = to.base

        this.setAbility(from, "Imposter")

        volatiles["Transform"] = {
          forme,
          gender,
          ability,
          moveSet: Object.fromEntries(
            moves.map((x) => [x, { used: 0, max: x === "Revival Blessing" ? 1 : 5 }])
          ),
          boosts: { ...boosts }
        }

        break
      }
      case "-start": {
        p = piped(line, p.i, 2)

        const user = this.user(this.label(p.args[0]))
        let { stripped: name } = parseEntity(p.args[1])

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

        const { ability, item } = parseEntity(from)
        const src = of ? this.user(this.label(of)) : user

        if (ability) this.setAbility(src, ability)
        if (item) this.setItem(src, item)

        const { "Locked Move": lockedMove } = volatiles
        if (fatigue != null && lockedMove) {
          const { move } = lockedMove
          if (pov === "foe" || checkLocked(this.request, move)) delete volatiles["Locked Move"]
        }
        break
      }
      case "-terastallize": {
        p = piped(line, p.i, 2)
        const user = this.user(this.label(p.args[0]))
        const teraType = p.args[1] as TypeName
        const { pov } = user

        this[pov].teraUsed = true
        user.tera = true
        user.teraType = teraType
        break
      }
      case "-formechange": {
        p = piped(line, p.i, 2)
        const user = this.user(this.label(p.args[0]))
        const forme = p.args[1]

        // Shaymin emits both a forme & detailchange. ignore forme.
        if (forme !== "Shaymin") {
          user.formeChange = {
            forme: forme,
            reverts: true
          }
        }

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)
        const { ability } = parseEntity(from)

        if (ability) this.setAbility(user, ability)

        break
      }
      case "detailschange": {
        p = piped(line, p.i, 2)
        const user = this.user(this.label(p.args[0]))
        const { forme } = parseTraits(p.args[1])

        user.formeChange = {
          forme: forme,
          reverts: false
        }

        const ability = {
          "Terapagos-Terastal": "Tera Shell",
          "Terapagos-Stellar": "Teraform Zero",
          "Shaymin": "Natural Cure"
        }[forme]

        if (ability) this.setAbility(user, ability)

        break
      }
      case "-activate": {
        p = piped(line, p.i, 2)
        const user = this.user(this.label(p.args[0]))
        const { pov } = user

        let { ability, item, move, stripped } = parseEntity(p.args[1])

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
          if (ability === "Battle Bond") user.flags[ability] = true
          if (ability === "Dancer") currLine.dancer = true

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

          const { team, active: from } = this.foe
          const { base } = from

          team[from.species] = from.clone()

          from.lvl = lvl
          from.species = species
          base.forme = forme
          base.gender = gender

          team[species] = from
        } else {
          delete this.illusion
        }

        break
      }
      case "-end": {
        p = piped(line, p.i, 2)
        const user = this.user(this.label(p.args[0]))
        let { stripped: name } = parseEntity(p.args[1])

        const { volatiles } = user

        if (name.startsWith("fallen")) name = "Fallen"

        delete volatiles[name]
        break
      }
      case "-singleturn": {
        p = piped(line, p.i, 2)
        const user = this.user(this.label(p.args[0]))
        const { stripped: name } = parseEntity(p.args[1])

        user.volatiles[name] = { singleTurn: true }
        break
      }
      case "-singlemove": {
        p = piped(line, p.i, 2)
        const user = this.user(this.label(p.args[0]))
        const { stripped: name } = parseEntity(p.args[1])

        user.volatiles[name] = { singleMove: true }
        break
      }

      case "-sidestart": {
        p = piped(line, p.i, 2)
        const { pov } = this.label(p.args[0])
        const { stripped: name } = parseEntity(p.args[1])

        const { fields } = this[pov]
        if (HAZARDS.includes(name)) (fields[name] ?? { layers: 0 }).layers!++
        else fields[name] = { turn: 0 }

        break
      }
      case "-sideend": {
        p = piped(line, p.i, 2)
        const { pov } = this.label(p.args[0])
        const { stripped: name } = parseEntity(p.args[1])

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

        break
      }
      case "tie": {
        break
      }
      case "win": {
        p = piped(line, p.i)
        this.winner = p.args[0] === this.name ? "ally" : "foe"
        break
      }
    }

    this.prevLine = currLine
    return null
  }
}
