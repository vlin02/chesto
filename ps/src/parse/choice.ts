import { Generation } from "@pkmn/data"
import { Observer } from "./observer.js"

export function availableMoves(gen: Generation, obs: Observer): string[] {
  const { active } = obs.ally

  let {
    volatiles: {
      "Encore": encore,
      "Taunt": taunt,
      "Heal Block": healBlock,
      "Locked Move": lockedMove,
      "Disable": disable,
      "Throat Chop": throatChop,
      "Recharge": recharge,
      "Choice Locked": choiceLocked
    },
    item,
    lastMove
  } = active

  if (recharge) return ["Recharge"]
  const { moveSet } = active

  const available = []

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

    if (choiceLocked && choiceLocked.move in moveSet && choiceLocked.move !== move) continue

    if (lockedMove && lockedMove.move !== move) continue

    if (disable?.move === move && !lockedMove) continue

    if (encore && move !== encore.move) {
      continue
    }
    if (taunt && category === "Status") {
      continue
    }
    if (healBlock && heal) {
      continue
    }
    if (throatChop && sound) {
      continue
    }
    if (item === "Assault Vest" && category === "Status") {
      continue
    }

    available.push(move)
  }

  if (!available.length) return ["Struggle"]

  return available
}
