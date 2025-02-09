import { getDefTyping, User } from "./user.js"
import { Run } from "../run.js"
import { PARTIALLY_TRAPPED_MOVES } from "../battle.js"
import { Observer } from "./observer.js"

type MoveOption =
  | {
      type: "struggle" | "recharge"
    }
  | {
      type: "default"
      moves: string[]
      stuck?: boolean
    }

export function getMoveOption({ fmt: { gen } }: Run, user: User): MoveOption {
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
  } = user

  if (recharge)
    return {
      type: "recharge"
    }

  const { moveSet } = user
  const moves = []

  if (locked?.move) return { type: "default", moves: [locked.move] }

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

    if (!stuck && choiceLocked && choiceLocked.move !== move) continue
    if (!stuck && encore && encore.move !== move) continue
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

  return { type: "default", moves, stuck }
}

export function getValidMoves({ struggle, recharge, moves = [] }: Options) {
  const all = [...moves]
  if (struggle) all.push("Struggle")
  if (recharge) all.push("Recharge")
  return all
}

export function isTrapped(user: User) {
  const { volatiles } = user
  if (volatiles["Recharge"] || volatiles["Prepare"] || volatiles["Locked Move"]) return true

  if (getDefTyping(user).includes("Ghost")) return false

  if (
    volatiles["Trapped"] ||
    volatiles["No Retreat"] ||
    PARTIALLY_TRAPPED_MOVES.some((k) => volatiles[k])
  )
    return true

  return false
}

export function getValidRevives({ ally: { team } }: Observer) {
  const opts: string[] = []
  for (const species in team) {
    if (team[species].hp[0] !== 0) continue
    opts.push(species)
  }
  return opts
}

export function getValidSwitches({ ally: { team, active, isReviving } }: Observer) {
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

type Options = {
  canTera?: boolean
  struggle?: boolean
  recharge?: boolean
  moves?: string[]
  switches?: string[]
}

export function getOptions(run: Run): Options {
  const { obs } = run

  const opt: Options = {}

  const {
    req,
    ally: { active, isReviving, teraUsed }
  } = obs

  switch (req.type) {
    case "move":
      const trapped = isTrapped(active)

      const moveOpt = getMoveOption(run, active)
      if (moveOpt.type === "struggle") opt.struggle = true
      if (moveOpt.type === "recharge") opt.recharge = true
      if (moveOpt.type === "default") opt.moves = moveOpt.moves
      opt.canTera = !teraUsed

      if (!trapped) {
        opt.switches = getValidSwitches(obs)
      }
      break
    case "switch":
      opt.switches = isReviving ? getValidRevives(obs) : getValidSwitches(obs)
      break
  }

  return opt
}
