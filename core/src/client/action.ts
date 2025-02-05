import { User } from "./user.js"
import { Format } from "../format.js"

type MoveOptions =
  | {
      type: "struggle" | "recharge"
    }
  | {
      type: "default"
      stuck?: boolean
      moves: string[]
    }

export function getMoveOptions({ gen }: Format, user: User): MoveOptions {
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

export function toMoves(choice: MoveOptions) {
  switch (choice.type) {
    case "struggle":
      return ["Struggle"]
    case "recharge":
      return ["Recharge"]
    case "default":
      const { moves } = choice
      return moves
  }
}

export function isTrapped({ volatiles }: User) {
  return !!(
    volatiles["Trapped"] ||
    volatiles["Prepare"] ||
    volatiles["Partially Trapped"] ||
    volatiles["No Retreat"] ||
    volatiles["Locked Move"] ||
    volatiles["Recharge"]
  )
}
