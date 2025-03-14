import { Generation, TypeName } from "@pkmn/data"
import {
  parseEntity,
  parseHealth,
  parseReference,
  parseTags,
  parseLabel,
  parseTypes
} from "./protocol.js"
import { Side } from "../battle.js"
import { parseRequest, RawRequest, Request } from "./request.js"
import { Ally, Foe, OPP, POV, POVS } from "./side.js"
import { User, MoveSet } from "./user.js"
import { getMaxPP, isLocking, triggersPressure } from "../move.js"
import {
  StatusId,
  CHOICE_ITEMS,
  BoostId,
  StatId,
  WeatherName,
  DELAYED_MOVES,
  Hazard,
  HAZARDS,
  Screen,
  PARTIAL_TRAPPING_MOVES
} from "../battle.js"
import { piped } from "../parse.js"
import { InputChoice } from "../log.js"
import { Option, Choice, Selection } from "./action.js"

type Ref = {
  species: string
  pov: POV
}

type Line = {
  revivalBlessing?: POV
  dancer?: boolean
  sleepTalk?: boolean
  stealEat?: boolean
}

export function assertLocked(req: Request, move: string) {
  if (req.type !== "move") return undefined
  const {
    choices: [{ moveSlots }]
  } = req

  if (moveSlots.length !== 1) return false

  const [{ name }] = moveSlots
  return name === move
}

function resolveSwaps(a: string[], b: string[]) {
  let i = 0

  const switches = []
  while (a[i] != b[0]) {
    i = b.findIndex((v) => v === a[i])!
    switches.push(a[i])
  }

  return switches
}

function canTera(gen: Generation, { volatiles }: User) {
  if (volatiles["Transform"]) {
    const { forme } = volatiles["Transform"]
    if (["Ogerpon", "Terapagos"].includes(gen.species.get(forme)!.baseSpecies)) return false
  }

  if (volatiles["Locked Move"]) return false

  return true
}

export type Fields = { [k: string]: number }
export type Weather = { name: WeatherName; turn: number }

type Event = {
  winner?: Side | "tie"
  turn?: number
  pending?: boolean
  error?: string
}

export class Observer {
  private swaps: string[]

  private illusion?: {
    from: User
    to: User
  }

  gen: Generation
  private prevLine?: Line

  req!: Request
  side!: Side
  name!: string
  ally!: Ally
  foe!: Foe
  turn: number
  fields: Fields
  weather?: Weather
  names: { [k: string]: Side }

  constructor(gen: Generation) {
    this.gen = gen
    this.fields = {}
    this.turn = 0
    this.swaps = []
    this.names = {}
  }

  ppCost(move: string, src: User, dest: User) {
    return dest.volatiles["Pressure"] &&
      dest.hp[0] !== 0 &&
      (move === "Curse" ? "Ghost" in src.offensiveTyping : triggersPressure(this.gen, move))
      ? 2
      : 1
  }

  toRef(s: string): Ref {
    const { side, species } = parseReference(s)
    return { pov: side === this.side ? "ally" : "foe", species }
  }

  clear(user: User) {
    const { volatiles, boosts, lastBerry, lastMove, formeChange, pov } = user
    const recover = { volatiles, boosts, lastBerry, lastMove, formeChange }

    const opp = this[OPP[pov]]
    if (opp) {
      const { volatiles } = opp.active

      delete volatiles["Trapped"]
      for (const move of PARTIAL_TRAPPING_MOVES) {
        delete volatiles[move]
      }
    }

    user.volatiles = {}
    user.boosts = {}
    delete user.lastBerry
    delete user.lastMove

    if (formeChange?.temporary) delete user.formeChange

    return recover
  }

  onSwitchIn(user: User) {
    if (user.pov === "ally") user.revealed = true
  }

  deref({ pov, species }: Ref) {
    const { illusion } = this
    const user = this[pov].team[species]
    if (illusion?.from === user) return illusion.to
    return user
  }

  setAbility(user: User, ability: string) {
    const { volatiles } = user
    // As One is treated as two abilities, with separate messages
    if (
      user.ability?.startsWith("As One") &&
      ["Unnerve", "Chilling Neigh", "Grim Neigh", "As One"].includes(ability)
    )
      return

    if (volatiles["Trace"] || volatiles["Transform"] || user.formeChange?.ability) return

    const { init: base } = user
    base.ability = ability
  }

  setItem(user: User, item: string | null) {
    const { volatiles } = user

    if (item === null && volatiles["Choice Locked"]) delete volatiles["Choice Locked"]
    user.item = item

    if (user.pov === "ally") return

    const { init } = user
    if (init.item === undefined) init.item = item
  }

  allocateSlot(moveSet: MoveSet, move: string) {
    return (moveSet[move] = moveSet[move] ?? {
      used: 0,
      max: getMaxPP(this.gen, move)
    })
  }

  read(line: string): Event {
    let p: { args: string[]; i: number }
    p = piped(line, 1)
    const msgType = p.args[0]

    const currLine: Line = {}
    let event: Event = {}

    switch (msgType) {
      case "player": {
        p = piped(line, p.i, 2)
        const [side, name] = p.args as [Side, string]
        this.names[name] = side
        break
      }
      case "error": {
        event.error = line.slice(p.i)
        break
      }
      case "request": {
        this.req = parseRequest(this.gen, JSON.parse(line.slice(p.i)) as RawRequest)

        if (this.req.type !== "wait") event.pending = true

        if (this.ally) {
          this.swaps.push(
            ...resolveSwaps(
              this.ally.slots.map((x) => x.species),
              this.req.team.map((x) => x.species)
            )
          )
        } else {
          this.swaps.push(this.req.team.find((x) => x.active)!.species)
        }

        if (!this.ally) {
          this.side = this.req.side
          this.name = this.req.name

          let active: User | undefined = undefined
          let team: { [k: string]: User } = {}
          let slots: User[] = []

          for (const member of this.req.team) {
            const user = new User(this.gen, { pov: "ally", member })
            if (member.active) {
              this.onSwitchIn(user)
              active = user
            }

            team[user.species] = user
            slots.push(user)
          }

          this.ally = {
            active: active!,
            effects: {},
            team,
            isReviving: false,
            teraUsed: false,
            slots,
            turnMoves: 0
          }
        }

        for (const { species, stats } of this.req.team) {
          this.ally.team[species].stats = stats
        }

        const { volatiles } = this.ally.active
        const { "Locked Move": lockedMove } = volatiles

        if (lockedMove && assertLocked(this.req, lockedMove.move) === false) {
          delete volatiles["Locked Move"]
        }

        break
      }
      case "-ability": {
        p = piped(line, p.i, 2)
        const user = this.deref(this.toRef(p.args[0]))
        const ability = p.args[1]

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)
        const cause = parseEntity(from)

        if (ability === "Intrepid Sword") user.flags.intrepidSword = true
        if (ability === "Pressure") user.volatiles["Pressure"] = {}

        if (cause.ability === "Trace") {
          const target = this.deref(this.toRef(of))

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
        const user = this.deref(this.toRef(p.args[0]))

        user.isTera = false
        user.hp[0] = 0
        user.status = undefined

        this.clear(user)
        break
      }
      case "switch":
      case "drag": {
        p = piped(line, p.i, 3)
        let ref = this.toRef(p.args[0])
        const { pov, species } = ref

        if (pov === "ally" && this.swaps.length) {
          const { team, slots } = this.ally
          const to = team[this.swaps.shift()!]

          {
            const i = slots.findIndex((x) => x === to)!
            ;[slots[i], slots[0]] = [slots[0], slots[i]]
          }

          if (to.species !== species) this.illusion = { from: team[species], to }
          else delete this.illusion
        }

        const label = parseLabel(p.args[1])

        let user: User

        if (pov === "ally") {
          user = this.deref(ref)
        } else {
          const team = this.foe?.team ?? {}
          user = team[species]
          if (!user) {
            user = team[species] = new User(this.gen, { pov: "foe", species, label })
          }

          if (!this.foe) {
            this.foe = {
              effects: {},
              team: { [species]: user },
              active: user,
              turnMoves: 0
            }
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

        this.clear(prev)
        this.onSwitchIn(user)
        this[pov].active = user
        break
      }
      case "-mustrecharge": {
        p = piped(line, p.i)
        const user = this.deref(this.toRef(p.args[0]))
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
          const user = this.deref(this.toRef(of))
          this.setAbility(user, ability)
        }

        break
      }
      case "-prepare": {
        p = piped(line, p.i, 2)
        const user = this.deref(this.toRef(p.args[0]))

        const move = p.args[1]
        if (move === "Solar Beam" && this.weather?.name === "SunnyDay") break

        user.volatiles["Prepare"] = { move: p.args[1], turn: 0 }
        break
      }
      case "-fieldstart": {
        p = piped(line, p.i)
        const { move: field } = parseEntity(p.args[0])
        this.fields[field!] = 0

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)

        const { ability } = parseEntity(from)

        const user = this.deref(this.toRef(of))
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
        const user = this.deref(this.toRef(p.args[0]))
        const id = p.args[1] as StatusId

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)

        user.status = {
          id,
          turn: id === "tox" ? 0 : undefined,
          attempt: id === "slp" ? 0 : undefined
        }

        const src = of ? this.deref(this.toRef(of)) : user
        const { ability, item } = parseEntity(from)

        if (item) this.setItem(src, item)
        if (ability) this.setAbility(src, ability)
        break
      }
      case "-curestatus": {
        p = piped(line, p.i, 2)
        const target = this.deref(this.toRef(p.args[0]))

        delete target.status

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)

        const { ability } = parseEntity(from)
        if (ability) this.setAbility(target, ability)

        break
      }
      case "move": {
        p = piped(line, p.i, 3)
        const user = this.deref(this.toRef(p.args[0]))
        const move = p.args[1]

        const { pov, volatiles, status } = user
        const { active: opp } = this[OPP[pov]]

        p = piped(line, p.i, -1)
        const { from, notarget, miss } = parseTags(p.args)
        const cause = parseEntity(from)

        for (const name in volatiles) {
          if (volatiles[name].singleMove) delete volatiles[name]
        }

        if (status?.attempt) status.attempt++
        if (cause.ability) this.setAbility(user, cause.ability)

        const failed = notarget != null || miss != null

        let isDirect: boolean = true

        const lockChoice = (move: string) => {
          const { item } = user
          const curr = volatiles["Choice Locked"]

          if (
            !(
              item &&
              CHOICE_ITEMS.includes(item) &&
              // choice lock can be overriden if it is a different turn and the choice move is not in the move set (due to dancer)
              (!curr || (!(curr.move in user.moveSet) && curr.firstTurn !== this.turn))
            )
          )
            return

          volatiles["Choice Locked"] = { move, firstTurn: this.turn }
        }

        switch (move) {
          case "Wish":
            if (!failed) this[pov].wish = 0
            break
          case "Struggle":
            isDirect = false
            user.lastMove = move
            break
          case "Revival Blessing": {
            currLine.revivalBlessing = pov
            if (!failed) this[pov].isReviving = true
            break
          }
        }

        if (cause.move === "Sleep Talk") {
          isDirect = false

          // revert sleep talk deduction, replace with move deduction
          user.moveSet["Sleep Talk"].used += -1 + this.ppCost(move, user, opp)
        }

        if (cause.ability === "Magic Bounce") {
          isDirect = false
        }

        // dancer sometimes only shows up in the previous -active line and not in the tag
        if (cause.ability === "Dancer" || this.prevLine?.dancer) {
          isDirect = false

          // dancing to a move counts as a choice lock
          lockChoice(move)
        }

        if (from === "lockedmove") {
          isDirect = false

          if (volatiles["Locked Move"]?.turn === 2) delete volatiles["Locked Move"]
          // outrage locked turns still can choice lock (if due to trick)
          lockChoice(move)
        }

        if (isDirect) {
          user.lastMove = move

          lockChoice(move)

          const slot = this.allocateSlot(user.moveSet, move)
          const choiceMove = volatiles["Choice Locked"]?.move
          // if the selected move is not the chosen one, don't deduct pp (dancer)
          if (!(choiceMove && choiceMove !== move)) slot.used += this.ppCost(move, user, opp)
        }

        if (
          isLocking(this.gen, move) &&
          (pov === "foe" || assertLocked(this.req, move) !== false)
        ) {
          volatiles["Locked Move"] = { turn: 0, move }
        }

        break
      }
      case "-fail":
        if (this.prevLine?.revivalBlessing) this[this.prevLine.revivalBlessing].isReviving = false
        break
      case "-heal":
      case "-sethp": {
        p = piped(line, p.i, 2)
        const user = this.deref(this.toRef(p.args[0]))
        const { hp } = parseHealth(p.args[1])!
        const { pov } = user

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)

        user.hp = hp

        const { ability, item, move } = parseEntity(from)
        if (ability) this.setAbility(user, ability)

        if (move === "Lunar Dance") {
          delete user.status

          const { moveSet } = user
          for (const move in moveSet) {
            moveSet[move].used = 0
          }
        }
        if (move === "Revival Blessing") this[pov].isReviving = false
        if (move === "Healing Wish") delete user.status

        // berries already include an -enditem
        if (item === "Leftovers") this.setItem(user, item)
        break
      }
      case "-damage": {
        p = piped(line, p.i, 2)

        const user = this.deref(this.toRef(p.args[0]))
        const health = parseHealth(p.args[1])

        if (health) user.hp = health.hp
        else user.hp[0] = 0

        p = piped(line, p.i, -1)

        const { from, of } = parseTags(p.args)

        const { item, ability } = parseEntity(from)
        const target = of ? this.deref(this.toRef(of)) : user

        if (ability) this.setAbility(target, ability)
        if (item) this.setItem(target, item)

        break
      }
      case "-boost":
      case "-unboost": {
        p = piped(line, p.i, 3)
        const user = this.deref(this.toRef(p.args[0]))

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
        if (item && !this.prevLine?.stealEat) {
          this.setItem(user, item)
          this.setItem(user, null)
        }

        break
      }
      case "-clearboost": {
        p = piped(line, p.i)
        const { pov } = this.toRef(p.args[0])

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
        const { boosts } = this.deref(this.toRef(p.args[0]))

        for (const k in boosts) {
          const id = k as BoostId
          boosts[id] = Math.max(0, boosts[id]!)
        }
        break
      }
      case "-item": {
        p = piped(line, p.i, 2)
        const user = this.deref(this.toRef(p.args[0]))
        const item = p.args[1]

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)
        const { ability } = parseEntity(from)

        const src = of ? this.deref(this.toRef(of)) : undefined

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
        const user = this.deref(this.toRef(p.args[0]))
        const item = p.args[1]

        p = piped(line, p.i, -1)
        const { from, eat, of } = parseTags(p.args)

        delete user.volatiles["Prepare"]

        if (from === "stealeat") currLine.stealEat = true

        this.setItem(user, item)
        this.setItem(user, null)

        let eater = eat != null && user
        if (from === "stealeat") eater = this.deref(this.toRef(of))

        if (eater) {
          eater.lastBerry = {
            name: item,
            turn: 0
          }
        }

        break
      }
      case "-transform": {
        p = piped(line, p.i, 2)
        const src = this.deref(this.toRef(p.args[0]))
        const dest = this.deref(this.toRef(p.args[1]))

        const { pov, volatiles } = src

        p = piped(line, p.i, -1)

        let ability: string
        let moves: string[] = []

        if (pov === "ally") {
          const { type, team } = this.req

          const user = team.find((x) => x.species === src.species)!

          // immediately fainted / switched out
          if (!user.active || type === "switch" || !user.health) {
            // TODO:
            volatiles["Transform"] = {} as any
            break
          }

          ability = this.gen.abilities.get(user.ability!)!.name

          const { moveSet } = dest
          moves = user.moves.map((id) => this.gen.moves.get(id)!.name)

          for (const move of moves) {
            moveSet[move] = moveSet[move] ?? {
              used: 0,
              max: getMaxPP(this.gen, move)
            }
          }

          this.setAbility(dest, ability)
        } else {
          const user = dest as User
          ability = user.ability!
          moves = Object.keys(user.moveSet)
        }

        const { boosts } = dest
        const { forme, gender } = dest.init

        this.setAbility(src, "Imposter")

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

        const user = this.deref(this.toRef(p.args[0]))
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
              this[opp].delayedAttack = {
                move: name,
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
            case "Taunt":
            case "Yawn":
            case "Confusion":
            case "Throat Chop":
            case "Heal Block":
            case "Slow Start":
            case "Magnet Rise": {
              volatiles[name] = { turn: 0 }
              break
            }
            case "Leech Seed":
            case "Charge":
            case "Attract":
            case "No Retreat":
            case "Salt Cure":
            case "Flash Fire":
            case "Leech Seed":
            case "Substitute": {
              volatiles[name] = {}
              break
            }
            default:
              throw Error(name)
          }
        }

        p = piped(line, p.i, -1)
        const { from, of, fatigue } = parseTags(p.args)

        const { ability, item } = parseEntity(from)
        const src = of ? this.deref(this.toRef(of)) : user

        if (ability) this.setAbility(src, ability)
        if (item) this.setItem(src, item)

        const { "Locked Move": lockedMove } = volatiles

        if (fatigue != null && lockedMove) {
          const { move } = lockedMove
          if (pov === "foe" || assertLocked(this.req, move) !== true)
            delete volatiles["Locked Move"]
        }
        break
      }
      case "-terastallize": {
        p = piped(line, p.i, 2)
        const user = this.deref(this.toRef(p.args[0]))
        const teraType = p.args[1] as TypeName
        const { pov } = user

        this[pov].teraUsed = true
        user.isTera = true
        user.teraType = teraType
        break
      }
      case "-formechange": {
        p = piped(line, p.i, 2)
        const user = this.deref(this.toRef(p.args[0]))
        const forme = p.args[1]

        // Shaymin emits both a forme & detailchange. ignore forme.
        if (forme !== "Shaymin") {
          user.formeChange = {
            forme: forme,
            temporary: true
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
        const user = this.deref(this.toRef(p.args[0]))
        const { forme } = parseLabel(p.args[1])

        user.formeChange = {
          forme: forme,
          temporary: false,
          ability: {
            "Ogerpon-Teal-Tera": "Embody Aspect (Teal)",
            "Ogerpon-Hearthflame-Tera": "Embody Aspect (Hearthflame)",
            "Ogerpon-Cornerstone-Tera": "Embody Aspect (Cornerstone)",
            "Ogerpon-Wellspring-Tera": "Embody Aspect (Wellspring)",
            "Shaymin": "Natural Cure",
            "Terapagos-Terastal": "Tera Shell",
            "Terapagos-Stellar": "Teraform Zero"
          }[forme]
        }

        break
      }
      case "-activate": {
        p = piped(line, p.i, 2)
        const user = this.deref(this.toRef(p.args[0]))
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
              user.volatiles[move] = { turn: 0 }
              break
            }
          }
        } else if (ability) {
          if (ability === "Battle Bond") user.flags.battleBond = true
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
        ;[ally.effects, foe.effects] = [foe.effects, ally.effects]
        break
      }
      case "replace": {
        p = piped(line, p.i, 3)
        const { pov, species } = this.toRef(p.args[0])

        if (pov === "foe") {
          const { team, active: user } = this.foe
          team[user.species] = user.clone()

          const { forme, lvl, gender } = parseLabel(p.args[1])

          user.lvl = lvl
          user.species = species
          user.init.forme = forme
          user.init.gender = gender

          team[species] = user
        } else {
          delete this.illusion
        }

        this[pov].active.flags.illusionRevealed = true

        break
      }
      case "-end": {
        p = piped(line, p.i, 2)
        const user = this.deref(this.toRef(p.args[0]))
        const { pov } = user
        let { stripped: name } = parseEntity(p.args[1])

        if (DELAYED_MOVES.includes(name)) {
          delete this[pov].delayedAttack
          break
        }

        const { volatiles } = user
        if (name.startsWith("fallen")) name = "Fallen"
        delete volatiles[name]
        break
      }
      case "-singleturn": {
        p = piped(line, p.i, 2)
        const user = this.deref(this.toRef(p.args[0]))
        const { stripped: name } = parseEntity(p.args[1])

        user.volatiles[name] = { singleTurn: true }
        break
      }
      case "-singlemove": {
        p = piped(line, p.i, 2)
        const user = this.deref(this.toRef(p.args[0]))
        const { stripped: name } = parseEntity(p.args[1])

        user.volatiles[name] = { singleMove: true }
        break
      }

      case "-sidestart": {
        p = piped(line, p.i, 2)
        const { pov } = this.toRef(p.args[0])
        const { stripped: name } = parseEntity(p.args[1])

        const { effects } = this[pov]
        if (HAZARDS.includes(name as Hazard)) {
          const hazard = effects[name as Hazard] ?? { layers: 0 }
          hazard.layers = Math.min(
            hazard.layers! + 1,
            { "Sticky Web": 1, "Toxic Spikes": 2, "Stealth Rock": 1, "Spikes": 3 }[name as Hazard]
          )
        } else {
          effects[name as Screen] = { turn: 0 }
        }

        break
      }
      case "-sideend": {
        p = piped(line, p.i, 2)
        const { pov } = this.toRef(p.args[0])
        const { stripped: name } = parseEntity(p.args[1])

        const { effects } = this[pov]
        delete effects[name]

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
        event.turn = this.turn

        for (const pov of POVS) {
          const side = this[pov]
          const { effects: conditions } = side

          side.turnMoves = 0

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
          if (volatiles["Prepare"]?.turn === 2) delete volatiles["Prepare"]
          if (side.wish === 2) delete side.wish
        }

        break
      }
      case "tie": {
        event.winner = "tie"
        break
      }
      case "win": {
        p = piped(line, p.i)
        event.winner = this.names[p.args[0]]
        break
      }
    }

    this.prevLine = currLine
    return event
  }

  getSelection(): Selection {
    const {
      gen,
      ally: { active, teraUsed }
    } = this

    const tera = !teraUsed && canTera(this.gen, active)

    let {
      volatiles: {
        "Encore": encore,
        "Taunt": taunt,
        "Heal Block": healBlock,
        "Locked Move": locked,
        "Disable": disable,
        "Throat Chop": throatChop,
        "Recharge": recharge,
        "Choice Locked": choiceLocked
      },
      item,
      lastMove
    } = active

    if (recharge)
      return {
        type: "recharge"
      }

    const { moveSet } = active
    const moves = []

    if (locked?.move) return { type: "default", tera, moves: [locked.move] }

    let stuck = [choiceLocked?.move, encore?.move].some((x) => x && !(x in moveSet))

    for (const move in moveSet) {
      const {
        category,
        flags: { heal, sound }
      } = gen.moves.get(move)!

      const { used, max } = moveSet[move]
      if (used >= max) continue

      switch (move) {
        case "Stuff Cheeks": {
          if (!item?.endsWith("Berry")) continue
          break
        }
        case "Gigaton Hammer":
        case "Blood Moon": {
          if (lastMove === move) continue
        }
      }

      if (!stuck) {
        if (choiceLocked && choiceLocked.move !== move) continue
        if (encore && encore.move !== move) continue
      }

      if (disable?.move === move) continue
      if (taunt && category === "Status") continue
      if (healBlock && heal) continue
      if (throatChop && sound) continue
      if (item === "Assault Vest" && category === "Status") continue

      moves.push(move)
    }

    if (!moves.length)
      return {
        type: "struggle"
      }

    return { type: "default", moves, tera, stuck }
  }

  listRevivable() {
    const {
      ally: { team }
    } = this

    const opts: string[] = []
    for (const species in team) {
      if (team[species].hp[0] !== 0) continue
      opts.push(species)
    }
    return opts
  }

  listSwitches() {
    const {
      ally: { team, active, isReviving }
    } = this

    const opts: string[] = []

    for (const species in team) {
      const member = team[species]
      if (isReviving) {
        if (team[species].hp[0] !== 0) continue
      } else {
        if (member === active || member.hp[0] === 0) continue
      }
      opts.push(species)
    }

    return opts
  }

  getOption(): Option | null {
    let switches: string[] = []
    let select: Selection | null = null

    const {
      req,
      ally: { isReviving, active }
    } = this

    if (req.type === "wait") return null

    switch (req.type) {
      case "move": {
        // use request trapped over active.trapped due to arena-trap abilities
        const { trapped } = req.choices[0]
        select = this.getSelection()

        if (!trapped) switches = this.listSwitches()
        break
      }
      case "switch":
        switches = isReviving ? this.listRevivable() : this.listSwitches()
        break
    }

    return { select, switches }
  }

  formatChoice(choice: Choice) {
    switch (choice.type) {
      case "move":
        const pfx = `move ${choice.move}`
        return choice.tera ? `${pfx} terastallize` : pfx
      case "switch":
        const slots = this.ally.slots.map((x) => x.species)
        const i = slots.indexOf(choice.species) + 1
        return `switch ${i}`
    }
  }

  resolveInputChoice(input: InputChoice): Choice {
    const { gen, ally } = this

    switch (input.type) {
      case "move": {
        const { move, tera } = input
        return {
          type: "move",
          move: move === "recharge" ? "Recharge" : gen.moves.get(move)!.name,
          tera
        }
      }
      case "switch": {
        const { i } = input
        return { type: "switch", species: ally.slots[i - 1].species }
      }
    }
  }
}
