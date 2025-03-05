import { Generation } from "@pkmn/data"
import { Patch } from "./version.js"
import { Observer } from "./parser/observer.js"
import { InputChoice as RawChoice } from "./log.js"

export type Format = {
  gen: Generation
  patch: Patch
}

export type Run = {
  gen: Generation
  obs: Observer
}

type MoveSelection =
  | {
      type: "struggle" | "recharge"
    }
  | {
      type: "default"
      moves: string[]
      stuck?: boolean
    }

export function getValidMoves({ gen, obs }: Run): MoveSelection {
  const {
    ally: { active }
  } = obs

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

  return { type: "default", moves, stuck }
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

export function toChoice({ gen, obs }: Run, raw: RawChoice): Decision {
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

export type Decision =
  | {
      type: "move"
      move: string
      tera: boolean
    }
  | {
      type: "switch"
      species: string
    }

export type Actions = {
  tera: boolean
  move: MoveSelection | null
  switch: string[]
}

export function getValidActions(run: Run): Actions {
  const { obs } = run

  let canTera = false
  let switches: string[] = []
  let moves: MoveSelection | null = null

  const {
    req,
    ally: { active, isReviving, teraUsed }
  } = obs

  switch (req.type) {
    case "move":
      moves = getValidMoves(run)

      if (!teraUsed && moves.type === "default") canTera = true
      if (!active.trapped) switches = getValidSwitches(run)
      break
    case "switch":
      switches = isReviving ? getValidRevives(run) : getValidSwitches(run)
      break
  }

  return { tera: canTera, move: moves, switch: switches }
}
