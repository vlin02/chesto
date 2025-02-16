import { Generation } from "@pkmn/data"
import { Patch } from "./version.js"
import { Observer } from "./client/observer.js"
import { PARTIALLY_TRAPPED_MOVES } from "./battle.js"
import { User, getDefTyping } from "./client/user.js"
import { Choice as RawChoice } from "./log.js"
import { BattleFeature } from "./encoding/obs.js"

export type Format = {
  gen: Generation
  patch: Patch
}

export type Run = {
  fmt: Format
  obs: Observer
}

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

export function getValidRevives({
  obs: {
    ally: { team }
  }
}: Run) {
  const opts: string[] = []
  for (const species in team) {
    if (team[species].hp[0] !== 0) continue
    opts.push(species)
  }
  return opts
}

export function getValidSwitches({
  obs: {
    ally: { team, active, isReviving }
  }
}: Run) {
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

export type Choice =
  | {
      type: "move"
      move: string
      tera: boolean
    }
  | {
      type: "switch"
      species: string
    }

export function toChoice({ fmt: { gen }, obs }: Run, raw: RawChoice): Choice {
  switch (raw.type) {
    case "move": {
      const { move, tera } = raw
      return {
        type: "move",
        move: move === "recharge" ? "Recharge" : gen.moves.get(move)!.name,
        tera
      }
    }
    case "switch": {
      const { i } = raw
      return { type: "switch", species: obs.ally.slots[i - 1].species }
    }
  }
}
