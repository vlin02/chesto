import { Generation, TypeName } from "@pkmn/data"
import {
  parseEntity,
  parseHealth,
  parseReference,
  parseTags,
  parseLabel,
  parseTypes,
  Side
} from "./protocol.js"
import { parseRequest, RawRequest, Request } from "./request.js"
import { Ally, Foe, OPP, POV, POVS } from "./side.js"
import { AllyUser, FoeUser, getOffTyping, MoveSet, User } from "./user.js"
import { inferMaxPP, isLockingMove, isPressuredMove } from "./move.js"
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
  PARTIALLY_TRAPPED_MOVES
} from "../battle.js"
import { piped } from "../parse.js"

type Ref = {
  species: string
  pov: POV
}

type Line = {
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

export type Fields = { [k: string]: number }
export type Weather = { name: WeatherName; turn: number }

export class Observer {
  private swaps: string[]

  private illusion?: {
    from: AllyUser
    to: AllyUser
  }

  private gen: Generation
  private prevLine?: Line

  req!: Request
  side!: Side
  name!: string
  ally!: Ally
  foe!: Foe
  turn: number
  fields: Fields
  weather?: Weather
  winner?: Side | null

  constructor(gen: Generation) {
    this.gen = gen
    this.fields = {}
    this.turn = 0
    this.swaps = []
  }

  ppCost(move: string, src: User, dest: User) {
    return dest.volatiles["Pressure"] &&
      dest.hp[0] !== 0 &&
      (move === "Curse" ? "Ghost" in getOffTyping(src) : isPressuredMove(this.gen, move))
      ? 2
      : 1
  }

  ref(s: string): Ref {
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
      for (const move of PARTIALLY_TRAPPED_MOVES) {
        delete volatiles[move]
      }
    }

    user.volatiles = {}
    user.boosts = {}
    delete user.lastBerry
    delete user.lastMove

    if (formeChange?.whileActiveOnly) delete user.formeChange

    return recover
  }

  onSwitchIn(user: User) {
    if (user.pov === "ally") user.revealed = true
  }

  user({ pov, species }: Ref) {
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

    const { base } = user
    base.ability = ability
  }

  setItem(user: User, item: string | null) {
    const { volatiles } = user

    if (item === null && volatiles["Choice Locked"]) delete volatiles["Choice Locked"]
    user.item = item

    if (user.pov === "ally") return

    const { base } = user
    if (base.item === undefined) base.item = item
  }

  disrupt(user: User) {
    delete user.volatiles["Locked Move"]
  }

  allocateSlot(moveSet: MoveSet, move: string) {
    return (moveSet[move] = moveSet[move] ?? {
      used: 0,
      max: inferMaxPP(this.gen, move)
    })
  }

  read(line: string) {
    let p: { args: string[]; i: number }
    p = piped(line, 1)
    const msgType = p.args[0]

    const currLine: Line = {}
    let event: "request" | "turn" | "end" | null = null

    switch (msgType) {
      case "request": {
        event = "request"

        this.req = parseRequest(this.gen, JSON.parse(line.slice(p.i)) as RawRequest)

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

          let active: AllyUser | undefined = undefined
          let team: { [k: string]: AllyUser } = {}
          let slots: AllyUser[] = []

          for (const member of this.req.team) {
            const user = new AllyUser(this.gen, member)
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
        const user = this.user(this.ref(p.args[0]))
        const ability = p.args[1]

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)
        const cause = parseEntity(from)

        if (ability === "Intrepid Sword") user.flags.intrepidSword = true
        if (ability === "Pressure") user.volatiles["Pressure"] = {}

        if (cause.ability === "Trace") {
          const target = this.user(this.ref(of))

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
        const user = this.user(this.ref(p.args[0]))

        user.tera = false
        user.hp[0] = 0
        user.status = undefined

        this.clear(user)
        break
      }
      case "switch":
      case "drag": {
        p = piped(line, p.i, 3)
        let ref = this.ref(p.args[0])
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
          user = this.user(ref)
        } else {
          const team = this.foe?.team ?? {}
          user = team[species]
          if (!user) {
            user = team[species] = new FoeUser(this.gen, species, label)

            const { forme, base } = user
            base.ability = {
              "Calyrex-Ice": "As One (Glastrier)",
              "Calyrex-Shadow": "As One (Spectrier)"
            }[forme]
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
        const user = this.user(this.ref(p.args[0]))
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
          const user = this.user(this.ref(of))
          this.setAbility(user, ability)
        }

        break
      }
      case "-prepare": {
        p = piped(line, p.i, 2)
        const user = this.user(this.ref(p.args[0]))

        const move = p.args[1]
        if (move === "Solar Beam" && this.weather?.name === "SunnyDay") break

        user.volatiles["Prepare"] = { move: p.args[1], turn: 0 }
        break
      }
      case "-fieldstart": {
        p = piped(line, p.i)
        const { move: name } = parseEntity(p.args[0])

        this.fields[name!] = 0

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)

        const { ability } = parseEntity(from)

        const user = this.user(this.ref(of))
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
        const user = this.user(this.ref(p.args[0]))
        const id = p.args[1] as StatusId

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)

        user.status = {
          id,
          turn: id === "tox" ? 0 : undefined,
          attempt: id === "slp" ? 0 : undefined
        }

        const src = of ? this.user(this.ref(of)) : user
        const { ability, item } = parseEntity(from)

        if (item) this.setItem(src, item)
        if (ability) this.setAbility(src, ability)
        break
      }
      case "-curestatus": {
        p = piped(line, p.i, 2)
        const target = this.user(this.ref(p.args[0]))

        delete target.status

        p = piped(line, p.i, -1)
        const { from } = parseTags(p.args)

        const { ability } = parseEntity(from)
        if (ability) this.setAbility(target, ability)

        break
      }
      case "move": {
        p = piped(line, p.i, 3)
        const user = this.user(this.ref(p.args[0]))
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
        if (failed) this.disrupt(user)

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
          isLockingMove(this.gen, move) &&
          (pov === "foe" || assertLocked(this.req, move) !== false)
        ) {
          volatiles["Locked Move"] = { turn: 0, move }
        }

        break
      }
      case "-immune":
        p = piped(line, p.i)
        const { pov } = this.user(this.ref(p.args[0]))

        this.disrupt(this[OPP[pov]].active)
        break
      case "-heal":
      case "-sethp": {
        p = piped(line, p.i, 2)
        const user = this.user(this.ref(p.args[0]))
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

        const user = this.user(this.ref(p.args[0]))
        const health = parseHealth(p.args[1])

        if (health) user.hp = health.hp
        else user.hp[0] = 0

        p = piped(line, p.i, -1)

        const { from, of } = parseTags(p.args)

        const { item, ability } = parseEntity(from)
        const target = of ? this.user(this.ref(of)) : user

        if (ability) this.setAbility(target, ability)
        if (item) this.setItem(target, item)

        break
      }
      case "-boost":
      case "-unboost": {
        p = piped(line, p.i, 3)
        const user = this.user(this.ref(p.args[0]))

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
        const { pov } = this.ref(p.args[0])

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
        const { boosts } = this.user(this.ref(p.args[0]))

        for (const k in boosts) {
          const id = k as BoostId
          boosts[id] = Math.max(0, boosts[id]!)
        }
        break
      }
      case "-item": {
        p = piped(line, p.i, 2)
        const user = this.user(this.ref(p.args[0]))
        const item = p.args[1]

        p = piped(line, p.i, -1)
        const { from, of } = parseTags(p.args)
        const { ability } = parseEntity(from)

        const src = of ? this.user(this.ref(of)) : undefined

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
        const user = this.user(this.ref(p.args[0]))
        const item = p.args[1]

        p = piped(line, p.i, -1)
        const { from, eat, of } = parseTags(p.args)

        delete user.volatiles["Prepare"]

        if (from === "stealeat") currLine.stealEat = true

        this.setItem(user, item)
        this.setItem(user, null)

        let eater = eat != null && user
        if (from === "stealeat") eater = this.user(this.ref(of))

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
        const from = this.user(this.ref(p.args[0]))
        const to = this.user(this.ref(p.args[1]))

        const { pov, volatiles } = from

        p = piped(line, p.i, -1)

        let ability
        let moves: string[] = []

        if (pov === "ally") {
          const { type, team } = this.req

          const user = team.find((x) => x.species === from.species)!

          // immediately fainted / switched out
          if (!user.active || type === "switch" || !user.health) {
            // TODO:
            volatiles["Transform"] = {} as any
            break
          }

          ability = this.gen.abilities.get(user.ability!)!.name

          const { moveSet } = to
          moves = user.moves.map((id) => this.gen.moves.get(id)!.name)

          for (const move of moves) {
            moveSet[move] = moveSet[move] ?? {
              used: 0,
              max: inferMaxPP(this.gen, move)
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

        const user = this.user(this.ref(p.args[0]))
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
        const src = of ? this.user(this.ref(of)) : user

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
        const user = this.user(this.ref(p.args[0]))
        const teraType = p.args[1] as TypeName
        const { pov } = user

        this[pov].teraUsed = true
        user.tera = true
        user.teraType = teraType
        break
      }
      case "-formechange": {
        p = piped(line, p.i, 2)
        const user = this.user(this.ref(p.args[0]))
        const forme = p.args[1]

        // Shaymin emits both a forme & detailchange. ignore forme.
        if (forme !== "Shaymin") {
          user.formeChange = {
            forme: forme,
            whileActiveOnly: true
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
        const user = this.user(this.ref(p.args[0]))
        const { forme } = parseLabel(p.args[1])

        user.formeChange = {
          forme: forme,
          whileActiveOnly: false,
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
        const user = this.user(this.ref(p.args[0]))
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
        const { pov, species } = this.ref(p.args[0])

        if (pov === "foe") {
          const { team, active: from } = this.foe
          team[from.species] = from.clone()

          const { forme, lvl, gender } = parseLabel(p.args[1])
          const { base } = from

          from.lvl = lvl
          from.species = species
          base.forme = forme
          base.gender = gender

          team[species] = from
        } else {
          delete this.illusion
        }

        this[pov].active.flags.illusionRevealed = true

        break
      }
      case "-end": {
        p = piped(line, p.i, 2)
        const user = this.user(this.ref(p.args[0]))
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
        const user = this.user(this.ref(p.args[0]))
        const { stripped: name } = parseEntity(p.args[1])

        user.volatiles[name] = { singleTurn: true }
        break
      }
      case "-singlemove": {
        p = piped(line, p.i, 2)
        const user = this.user(this.ref(p.args[0]))
        const { stripped: name } = parseEntity(p.args[1])

        user.volatiles[name] = { singleMove: true }
        break
      }

      case "-sidestart": {
        p = piped(line, p.i, 2)
        const { pov } = this.ref(p.args[0])
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
        const { pov } = this.ref(p.args[0])
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
        event = "turn"

        p = piped(line, p.i)
        this.turn = Number(p.args[0])

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
        this.winner = null
        event = "end"
        break
      }
      case "win": {
        p = piped(line, p.i)
        this.winner = p.args[0] as Side
        event = "end"
        break
      }
    }

    this.prevLine = currLine
    return event
  }
}
